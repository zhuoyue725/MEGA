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
from mega.utils.demo_renderer import Renderer, cam_crop_to_full
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
)


parser = argparse.ArgumentParser(description="CVQDiffusion Demo: save mesh as .obj")
parser.add_argument("--input_path",  type=str, default="demo_data/",
                    help="图像目录或单张图像路径")
parser.add_argument("--output_path", type=str, default="demo_out_vqdiffusion",
                    help="obj 输出目录")
parser.add_argument("--checkpoint",  type=str, default="",
                    help="CVQDiffusion checkpoint 路径（留空则读配置文件中的 resume.path）")
args, _ = parser.parse_known_args()


def main():
    cfg = OmegaConf.load('configs/config_cvqdiffusion/config_hrnet_large.yaml')

    # 从 args 获取参数
    input_path = args.input_path
    output_path = args.output_path
    checkpoint = args.checkpoint
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

    ckpt_path = checkpoint if checkpoint else cfg.resume.path
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

    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
    renderer = Renderer(1000, 224, faces=faces)

    # ------------------------------------------------------------------ #
    #  输入/输出路径                                                       #
    # ------------------------------------------------------------------ #
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
                pred_cam = sample_out['pred_cam']
                rotmat = rotation_6d_to_matrix(pred_rot)
                pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

                vertices = pred_mesh[0].cpu().numpy()  # [V, 3]

                # 保存 obj
                if len(dataloader) > 1:
                    obj_name = f"{img_fn}_person{person_idx}.obj"
                    out_img_name = f"{img_fn}_person{person_idx}.png"
                else:
                    obj_name = f"{img_fn}.obj"
                    out_img_name = f"{img_fn}.png"

                obj_path = os.path.abspath(os.path.join(output_path, obj_name))
                # mesh = trimesh.Trimesh(vertices, faces, process=False)
                # mesh.export(obj_path)
                # print(f"Saved: {obj_path}")

                # 可视化并保存覆盖图像
                box_center = torch.tensor(batch["box_center"]).to(device).float()
                box_size = torch.tensor(batch["box_size"]).to(device).float()
                img_size = torch.tensor(batch["img_size"]).to(device).float()
                scaled_focal_length = 1000.0 / 224.0 * img_size.max()
                cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                cam_t = cam_t.squeeze(0).detach().cpu().numpy()

                render_h, render_w = img_cv2.shape[:2]
                cam_view = renderer.render_rgba_multiple(
                    [vertices],
                    [cam_t],
                    render_res=[render_w, render_h],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )

                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                if cam_view.shape[0] != render_h or cam_view.shape[1] != render_w:
                    cam_view = cv2.resize(cam_view, (render_w, render_h), interpolation=cv2.INTER_LINEAR)

                if cam_view.shape[2] == 4:
                    alpha = cam_view[:, :, 3:4]
                else:
                    alpha = np.ones((render_h, render_w, 1), dtype=np.float32)

                input_img_overlay = input_img[:, :, :3] * (1 - alpha) + cam_view[:, :, :3] * alpha
                cv2.imwrite(
                    os.path.join(output_path, out_img_name),
                    (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8),
                )
                print(f"Saved image: {os.path.join(output_path, out_img_name)}")

    print(f"All outputs saved in: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
