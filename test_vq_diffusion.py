"""
test_vq_diffusion.py

参照 test_mega.py，使用 Hydra 读取配置文件，调用 CVQDiffusion_Train.eval_deterministic
对验证集进行确定性评估（diffusion 采样，filter_ratio=0）。

用法（从项目根目录）：
    python test_vq_diffusion.py -p <datasets_path> resume.path=<checkpoint_path>

覆盖配置项示例：
    python test_vq_diffusion.py train.batch=4 resume.path=checkpoint/CVQDIFFUSION/xxx/model_best_loss
"""
import sys
import os
import numpy as np
import argparse

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


parser = argparse.ArgumentParser(description='Test CVQDiffusion (eval_deterministic)')
parser.add_argument('-p', '--path', type=str, default='datasets',
                    help='Path to dataset root (H5 files)')
args, _ = parser.parse_known_args()
print(os.listdir(args.path))


@hydra.main(
    config_path='configs/config_cvqdiffusion',
    config_name='config_hrnet_large',
    version_base=None,
)
def main(cfg: DictConfig):
    # Hydra 会切换工作目录，切回项目根目录
    os.chdir(hydra.utils.get_original_cwd())

    print('=' * 60)
    print('CVQDiffusion  –  eval_deterministic')
    print(OmegaConf.to_yaml(cfg))
    print('=' * 60)

    set_seed()

    # ---------------------------------------------------------------- #
    #  网格 faces（用于可视化）                                         #
    # ---------------------------------------------------------------- #
    ref_bm = np.load('body_models/smplh/neutral/model.npz')
    faces  = torch.from_numpy(ref_bm['f'].astype(np.int32))

    # ---------------------------------------------------------------- #
    #  Backbone                                                         #
    # ---------------------------------------------------------------- #
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
    #  MeshVQVAE                                                        #
    # ---------------------------------------------------------------- #
    # 如果使用 eval_stochastic，需要将 modelconv.batch 乘以 sample_size
    sample_size = 1  # 与下面 eval_stochastic 的 sample_size 保持一致
    cfg.modelconv.batch = cfg.modelconv.batch * sample_size
    
    convmesh_model = mesh_vq_vae.FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = mesh_vq_vae.MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model='checkpoint/MESH_VQVAE/mesh_vqvae_54')
    convmesh_model.init_test_mode()
    vqvae_params = sum(p.numel() for p in mesh_vqvae.parameters() if p.requires_grad)
    print(f'MeshVQVAE params: {vqvae_params:,}')

    # ---------------------------------------------------------------- #
    #  关节点回归矩阵                                                   #
    # ---------------------------------------------------------------- #
    J_regressor = torch.from_numpy(
        np.load('body_models/J_regressor_h36m.npy')
    ).float()

    # ---------------------------------------------------------------- #
    #  测试数据集（使用 test_data，与 test_mega.py 一致）               #
    # ---------------------------------------------------------------- #
    test_data = MixedDataset(
        cfg.test_data.file,
        augment=False,
        flip=False,
        proportion=1,
    )
    print(f'Test samples: {len(test_data)}')

    # ---------------------------------------------------------------- #
    #  Trainer                                                          #
    # ---------------------------------------------------------------- #
    trainer = CVQDiffusion_Train(
        model=model,
        vqvae=mesh_vqvae,
        training_data=test_data,
        validation_data=test_data,
        config_training=OmegaConf.to_container(cfg.train, resolve=True),
        faces=faces,
        joints_regressor=J_regressor,
    )

    # ---------------------------------------------------------------- #
    #  加载 checkpoint（必须指定）                                      #
    # ---------------------------------------------------------------- #
    resume_path = cfg.get('resume', {}).get('path', '')
    if resume_path:
        trainer.load(path=resume_path, optimizer=False)
        print(f'Loaded checkpoint: {resume_path}')
    else:
        print('[WARNING] No checkpoint specified (resume.path not set). '
              'Running with random weights.')

    # ---------------------------------------------------------------- #
    #  确定性评估                                                       #
    # ---------------------------------------------------------------- #
    # trainer.eval_deterministic(visualize=True)
    # V2V: 31.89  MPJPE: 28.88  PA-MPJPE: 19.38  (mm)
    # sto:
    # V2V: 21.20  MPJPE: 19.06  PA-MPJPE: 8.76  (mm)
    # trainer.eval_stochastic(sample_size=sample_size, temperature=1.0, visualize=True, vis_idx=2) # 采样1.0，设置其他值结果不行
    # diffusion_steps = list(range(99, -1, -5))
    diffusion_steps = [99, 79, 59, 39, 19, 9,8,7,6,5,4,3,2,1,0]
    trainer.eval_stochastic_diffusion_step(diffusion_steps=diffusion_steps, temperature=1.0, vis_idx=1) # 99开始 0结束

if __name__ == '__main__':
    main()