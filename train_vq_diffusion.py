"""
train_vq_diffusion.py

参照 train_mega.py，使用 Hydra 读取配置文件，训练 CVQDiffusion 模型。

用法（从项目根目录）：
    python train_vq_diffusion.py

指定配置文件：
    python train_vq_diffusion.py --config-name config_hrnet

覆盖单个配置项：
    python train_vq_diffusion.py train.batch=8 train.total_epoch=100
"""
import sys
import os
import numpy as np

# ---- 加入项目根目录 ----
target_path = os.path.abspath(os.path.dirname(__file__))
if target_path not in sys.path:
    sys.path.insert(0, target_path)

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import mesh_vq_vae

from mega import set_seed, hrnet_w48, MixedDataset
from mega.model.cvqdiffusion import CVQDiffusion
from mega.train.train_cvqdiffusion.train_class import CVQDiffusion_Train


@hydra.main(
    config_path='configs/config_cvqdiffusion',
    config_name='config_hrnet_large',
    version_base=None,
)
def main(cfg: DictConfig):
    # Hydra 会切换工作目录，这里切回项目根目录
    os.chdir(hydra.utils.get_original_cwd())

    print('=' * 60)
    print('CVQDiffusion Training')
    print(OmegaConf.to_yaml(cfg))
    print('=' * 60)

    set_seed()

    # ---------------------------------------------------------------- #
    #  网格 faces（用于可视化）                                         #
    # ---------------------------------------------------------------- #
    ref_bm = np.load('body_models/smplh/neutral/model.npz')
    faces  = torch.from_numpy(ref_bm['f'].astype(np.int32))

    if cfg.backbone.type == 'hrnet':
        backbone = hrnet_w48(
            pretrained_ckpt_path=cfg.backbone.pretrained,
            downsample=True,
            use_conv=True,
        )
        print(f'Backbone: HRNet-W48  ({cfg.backbone.pretrained})')
    else:
        raise ValueError(f'Unsupported backbone type: {cfg.backbone.type}')

    # ---------------------------------------------------------------- #
    #  CVQDiffusion 模型                                                #
    # ---------------------------------------------------------------- #
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f'CVQDiffusion params: {total_params:,}')

    # ---------------------------------------------------------------- #
    #  MeshVQVAE（用于动态获取 codebook indices，与 train_mega.py 一致）#
    # ---------------------------------------------------------------- #
    convmesh_model = mesh_vq_vae.FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = mesh_vq_vae.MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model='checkpoint/MESH_VQVAE/mesh_vqvae_54')
    convmesh_model.init_test_mode()
    vqvae_params = sum(p.numel() for p in mesh_vqvae.parameters() if p.requires_grad)
    print(f'MeshVQVAE params: {vqvae_params:,}')

    # ---------------------------------------------------------------- #
    #  数据集（与 train_mega.py 一致，使用 MixedDataset）               #
    # ---------------------------------------------------------------- #
    training_data = MixedDataset(
        cfg.training_data.file,
        augment=False,
        flip=False,
        proportion=1,
    )
    validation_data = MixedDataset(
        cfg.validation_data.file,
        augment=False,
        flip=False,
        proportion=0.5,
    )
    print(f'Training   samples: {len(training_data)}')
    print(f'Validation samples: {len(validation_data)}')

    # ---------------------------------------------------------------- #
    #  Joint regressor（用于评估和重投影损失）                           #
    # ---------------------------------------------------------------- #
    J_regressor = torch.from_numpy(np.load('body_models/J_regressor_h36m.npy')).float()
    J_regressor_24 = torch.from_numpy(np.load('body_models/J_regressor_24.npy')).float()

    # ---------------------------------------------------------------- #
    #  Trainer                                                          #
    # ---------------------------------------------------------------- #
    trainer = CVQDiffusion_Train(
        model=model,
        vqvae=mesh_vqvae,
        training_data=training_data,
        validation_data=validation_data,
        config_training=OmegaConf.to_container(cfg.train, resolve=True),
        faces=faces,
        joints_regressor=J_regressor,
        joints_regressor_smpl=J_regressor_24,
    )

    trainer.load_rotcam_weights('checkpoint/CVQMAE/rotcam_weights.pth')
    # ---- 可选：从预训练的无条件扩散模型加载权重 ----
    pretrain_path = cfg.get('pretrain', {}).get('path', '')
    if pretrain_path:
        trainer.load_from_pretrain(path=pretrain_path)
        print(f'Loaded pretrained unconditional weights from {pretrain_path}')
    # ---- 可选：从 checkpoint 恢复 ----
    resume_path = cfg.get('resume', {}).get('path', '')
    if resume_path:
        load_optimizer = cfg.get('resume', {}).get('optimizer', True)
        trainer.load(path=resume_path, optimizer=load_optimizer)
        # 加载相机和旋转网络参数
        trainer.load_rotcam_weights('checkpoint/CVQMAE/rotcam_weights.pth')
        print(f'Resumed from {resume_path}')

    # ---------------------------------------------------------------- #
    #  开始训练                                                         #
    # ---------------------------------------------------------------- #
    trainer.fit()


if __name__ == '__main__':
    main()
