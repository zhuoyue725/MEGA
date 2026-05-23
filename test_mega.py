from mega import (
    CVQMAE,
    CVQMAE_Train,
    MixedDataset,
    set_seed,
    hrnet_w48,
    vit,
)
import mesh_vq_vae
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import torch
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-p", "--path", type=str, help="Path to H5", default="datasets")

args = parser.parse_args()
path = args.path
print(os.listdir(args.path))


@hydra.main(
    config_path="configs/config_cvqmae",
    config_name="config_hrnet",
    version_base=None,
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    set_seed()

    ref_bm_path = "body_models/smplh/neutral/model.npz"
    ref_bm = np.load(ref_bm_path)

    """Data"""
    test_data = MixedDataset(
        cfg.test_data.file,
        augment=False,
        flip=False,
        proportion=1,
    )

    """ Backbone """

    if cfg.backbone.type == "resnet":
        resnet_checkpoints = models.ResNet50_Weights.DEFAULT
        resnet_model = models.resnet50(weights=resnet_checkpoints)
        backbone = torch.nn.Sequential(*list(resnet_model.children())[:-2])
    elif cfg.backbone.type == "hrnet":
        pretrained_ckpt_path = cfg.backbone.pretrained
        backbone = hrnet_w48(
            pretrained_ckpt_path=pretrained_ckpt_path,
            downsample=True,
            use_conv=True,
        )
    else:
        backbone = vit()
        backbone.load_state_dict(
            torch.load(cfg.backbone.pretrained, map_location="cpu")["state_dict"]
        )

    # 计算backbone参数量
    backbone_params = sum(p.numel() for p in backbone.parameters())
    print(f'Backbone params: {backbone_params:,} ({backbone_params/1_000_000:.2f}M)')

    """ ConvMesh VQVAE model """
    convmesh_model = mesh_vq_vae.FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = mesh_vq_vae.MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    convmesh_model.init_test_mode()
    mesh_vqvae_params = sum(
        p.numel() for p in mesh_vqvae.parameters() if p.requires_grad
    )
    print(f"Mesh-VQVAE: {mesh_vqvae_params}")

    """ MeshRegressor model """
    mesh_regressor = CVQMAE(
        backbone=backbone,
        **cfg.model,
    )
    mesh_regressor_params = sum(
        p.numel() for p in mesh_regressor.parameters() if p.requires_grad
    )

    # 计算不包含backbone的CVQMAE参数量
    mesh_regressor_without_backbone_params = 0
    for name, param in mesh_regressor.named_parameters():
        if param.requires_grad and not name.startswith('backbone.'):
            mesh_regressor_without_backbone_params += param.numel()

    # Load the VQMAE pretrained on motion capture data
    resume_path = cfg.get("resume", {}).get("path", "checkpoint/CVQMAE/mega_hrnet")
    mesh_regressor.load(resume_path)
    print(f'Backbone params: {backbone_params:,} ({backbone_params/1_000_000:.2f}M)')
    print(f'CVQMAE (without backbone) params: {mesh_regressor_without_backbone_params:,} ({mesh_regressor_without_backbone_params/1_000_000:.2f}M)')
    print(f'CVQMAE (total with backbone) params: {mesh_regressor_params:,} ({mesh_regressor_params/1_000_000:.2f}M) (loaded from {resume_path})')

    # 计算总参数量
    total_params_with_backbone = mesh_regressor_params + mesh_vqvae_params
    total_params_without_backbone = mesh_regressor_without_backbone_params + mesh_vqvae_params

    print(f"\n======= MEGA Model Parameters =======")
    print(f"1. Backbone: {backbone_params:,} ({backbone_params/1_000_000:.2f}M)")
    print(f"2. CVQMAE (without backbone): {mesh_regressor_without_backbone_params:,} ({mesh_regressor_without_backbone_params/1_000_000:.2f}M)")
    print(f"3. CVQMAE (total with backbone): {mesh_regressor_params:,} ({mesh_regressor_params/1_000_000:.2f}M)")
    print(f"4. Mesh-VQVAE: {mesh_vqvae_params:,} ({mesh_vqvae_params/1_000_000:.2f}M)")
    print(f"-----------------------------------------")
    print(f"Total (without backbone + Mesh-VQVAE): {total_params_without_backbone:,} ({total_params_without_backbone/1_000_000:.2f}M)")
    print(f"Total (with backbone + Mesh-VQVAE): {total_params_with_backbone:,} ({total_params_with_backbone/1_000_000:.2f}M)")
    print(f"====================================\n")

    """Joint regressor"""
    J_regressor = torch.from_numpy(np.load("body_models/J_regressor_h36m.npy")).float()

    J_regressor_24 = torch.from_numpy(np.load("body_models/J_regressor_24.npy")).float()

    """ Training """
    pretrain_mesh_regressor = CVQMAE_Train(
        mesh_regressor,
        mesh_vqvae,
        test_data,
        test_data,
        cfg.train,
        faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
        joints_regressor=J_regressor,
        joints_regressor_smpl=J_regressor_24,
    )

    # 执行评估并获取推理时间
    # v2v_result, avg_inference_time = pretrain_mesh_regressor.eval_deterministic(vis_all_sample=True)
    v2v_result, avg_inference_time = pretrain_mesh_regressor.eval_stochastic(sample_size=1, steps=10, temp=1, visualise=False)

    print(f"\n======= 最终统计结果 =======")
    print(f"总参数量 (包含backbone): {total_params_with_backbone:,} ({total_params_with_backbone/1_000_000:.2f}M)")
    print(f"总参数量 (不包含backbone): {total_params_without_backbone:,} ({total_params_without_backbone/1_000_000:.2f}M)")
    print(f"单次推理平均时间: {avg_inference_time:.4f}s")
    print(f"V2V误差: {v2v_result}")
    print(f"================================")
    # pretrain_mesh_regressor.eval_stochastic(sample_size=25)


if __name__ == "__main__":
    main()
