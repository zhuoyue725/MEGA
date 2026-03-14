import os
import argparse
import torch
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import trimesh
from omegaconf import OmegaConf
import mesh_vq_vae

from mega import hrnet_w48
from mega.model.cvqdiffusion import CVQDiffusion
from mega.data.dataset_demo import DemoDataset
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
)


def main():
    parser = argparse.ArgumentParser(description="CVQDiffusion Demo: save mesh as .obj")
    parser.add_argument("--input_path",  type=str, default="demo_data/",
                        help="图像目录或单张图像路径")
    parser.add_argument("--output_path", type=str, default="demo_out_vqdiffusion",
                        help="obj 输出目录")
    parser.add_argument("--checkpoint",  type=str, default="",
                        help="CVQDiffusion checkpoint 路径（留空则读配置文件中的 resume.path）")
    parser.add_argument("--config",      type=str,
                        default="configs/config_cvqdiffusion/config_hrnet.yaml",
                        help="yaml 配置文件路径")
    args = parser.parse_args()

    # 切换到项目根目录，保证相对路径正确
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    cfg = OmegaConf.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ #
    #  YOLOv8 人体检测器                                                  #
    # ------------------------------------------------------------------ #
    weights_path = "body_models/"
    if not os.path.exists(os.path.join(weights_path, "yolov8x.pt")):
        from ultralytics.utils.downloads import download
        download(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
            weights_path,
        )
    yolo_model = YOLO(os.path.join(weights_path, "yolov8x.pt"))

    # ------------------------------------------------------------------ #
    #  Backbone + CVQDiffusion 模型                                       #
    # ------------------------------------------------------------------ #
    backbone = hrnet_w48(
        pretrained_ckpt_path=cfg.backbone.pretrained,
        downsample=True,
        use_conv=True,
    )

    model_cfg = cfg.model
    model = CVQDiffusion(
        backbone=backbone,
        backbone_feat_dim=model_cfg.backbone_feat_dim,
        cond_emb_dim=model_cfg.cond_emb_dim,
        num_tok=model_cfg.num_tok,
        seq_len=model_cfg.seq_len,
        n_emb=model_cfg.n_emb,
        cond_dim=model_cfg.cond_dim,
        cond_len=model_cfg.cond_len,
        n_head=model_cfg.n_head,
        n_layer=model_cfg.n_layer,
        diff_step=model_cfg.diff_step,
        auxiliary_loss_weight=model_cfg.auxiliary_loss_weight,
        adaptive_auxiliary_loss=model_cfg.adaptive_auxiliary_loss,
        mask_weight=list(model_cfg.mask_weight),
    )

    ckpt_path = args.checkpoint if args.checkpoint else cfg.resume.path
    model.load(ckpt_path)
    model.to(device).eval()

    # ------------------------------------------------------------------ #
    #  MeshVQVAE（decode indices -> mesh_canonical）                      #
    # ------------------------------------------------------------------ #
    convmesh_model = mesh_vq_vae.FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = mesh_vq_vae.MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    mesh_vqvae.to(device)
    convmesh_model.init_test_mode()

    # SMPL faces
    ref_bm = np.load("body_models/smplh/neutral/model.npz")
    faces = ref_bm["f"].astype(np.int32)

    # ------------------------------------------------------------------ #
    #  输入/输出路径                                                       #
    # ------------------------------------------------------------------ #
    input_path  = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    if os.path.isdir(input_path):
        files = sorted(
            f for f in os.listdir(input_path)
            if f.lower().endswith((".jpg", ".png"))
        )
    else:
        files = [os.path.basename(input_path)]
        input_path = os.path.dirname(input_path)

    # ------------------------------------------------------------------ #
    #  逐图处理                                                            #
    # ------------------------------------------------------------------ #
    for file in tqdm(files, desc="Processing"):
        file_path = os.path.join(input_path, file)
        img_cv2 = cv2.imread(file_path)
        if img_cv2 is None:
            print(f"[WARN] cannot read {file_path}, skip.")
            continue

        boxes = (
            yolo_model.predict(
                img_cv2,
                device="cuda",
                classes=0,
                conf=0.5,
                save=False,
                verbose=False,
            )[0]
            .boxes.xyxy.detach()
            .cpu()
            .numpy()
        )

        if len(boxes) == 0:
            print(f"[WARN] no person detected in {file}, skip.")
            continue

        dataset = DemoDataset(img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        img_fn, _ = os.path.splitext(os.path.basename(file_path))

        with torch.no_grad():
            for person_idx, batch in enumerate(dataloader):
                img_tensor = batch["img"].to(device)  # [1, 3, 224, 224]

                # CVQDiffusion 采样 -> mesh token indices
                sample_out = model.sample(
                    img_tensor[:1],
                    filter_ratio=0.0,
                    temperature=1.0,
                )
                mesh_indices = sample_out["content_token"].to(device)  # [B, 54]

                # Decode indices -> mesh_canonical: [B, V, 3]
                mesh_canonical = mesh_vqvae.decode(mesh_indices)[: img_tensor.shape[0]]
                pred_rot = sample_out['pred_rot']
                rotmat   = rotation_6d_to_matrix(pred_rot)  
                pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

                vertices = pred_mesh[0].cpu().numpy()  # [V, 3]

                # 命名与图像文件名相同，多人时加后缀
                # if len(dataloader) > 1:
                #     obj_name = f"{img_fn}_person{person_idx}.obj"
                # else:
                #     obj_name = f"{img_fn}.obj"
                obj_name = f"{img_fn}.obj"

                obj_path = os.path.abspath(os.path.join(output_path, obj_name))
                mesh = trimesh.Trimesh(vertices, faces, process=False)
                mesh.export(obj_path)
                print(f"Saved: {obj_path}")
                break


if __name__ == "__main__":
    main()
