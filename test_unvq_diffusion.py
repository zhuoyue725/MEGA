"""test_unconditional_diffusion.py

测试无条件 VQ-Diffusion 模型：从测试集中取一批样本，
通过 visualize_diffusion_steps() 可视化扩散各步骤的 mesh 重建结果。

用法:
    python test_unconditional_diffusion.py
    python test_unconditional_diffusion.py --config-name config
    python test_unconditional_diffusion.py train.device=cpu
"""
import sys
import os

_root = os.path.abspath(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import argparse
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mega.utils import set_seed
from mega.model.unconditional_diffusion import UnconditionalDiffusion
from mega.train.train_unconditional_diffusion import UnconditionalDiffusion_Train

from mesh_vq_vae import (
    MeshVQVAE,
    FullyConvAE,
    DatasetMeshTest,
)


@hydra.main(
    config_path='configs/config_unconditional_diffusion',
    config_name='config',
    version_base=None,
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    set_seed()

    # ------------------------------------------------------------------ #
    #  命令行参数（通过 argparse 补充，hydra 之外的选项）                  #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint', type=str,
                        default='',
                        help='模型 checkpoint 路径，覆盖 cfg.resume.path')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='从测试集中取几个 batch 进行可视化')
    parser.add_argument('--vis_idx', type=int, default=0,
                        help='batch 内第几个样本进行 mesh 可视化')
    parser.add_argument('--output_dir', type=str, default='demo_out/diffusion_steps',
                        help='可视化结果保存目录')
    parser.add_argument('--save_steps', type=int, nargs='+',
                        default=[10, 30, 50, 70, 90],
                        help='要可视化的扩散步骤列表')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度')
    parser.add_argument('--filter_ratio', type=float, default=0.0,
                        help='掩码保留比例（0.0 = 从完全掩码开始）')
    # 忽略 hydra 注入的参数
    args, _ = parser.parse_known_args()

    # checkpoint 优先级：命令行 > cfg.resume.path
    ckpt_path = args.checkpoint or cfg.get('resume', {}).get('path', '')
    if not ckpt_path:
        raise ValueError(
            '请通过 --checkpoint <path> 或在 config.yaml 的 resume.path 中指定模型路径。'
        )

    # ------------------------------------------------------------------ #
    #  测试数据集                                                          #
    # ------------------------------------------------------------------ #
    test_data = DatasetMeshTest(
        dataset_file=cfg.validation_data.file, proportion=0.1
    )
    print(f'Test samples: {len(test_data)}')

    test_loader = DataLoader(
        test_data,
        batch_size=cfg.train.batch,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ------------------------------------------------------------------ #
    #  MeshVQVAE  (frozen)                                                #
    # ------------------------------------------------------------------ #
    convmesh_model = FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model='checkpoint/MESH_VQVAE/mesh_vqvae_54')
    convmesh_model.init_test_mode()

    vqvae_params = sum(p.numel() for p in mesh_vqvae.parameters())
    print(f'MeshVQVAE params: {vqvae_params:,}  (frozen)')

    # ------------------------------------------------------------------ #
    #  UnconditionalDiffusion 模型                                        #
    # ------------------------------------------------------------------ #
    model_cfg = cfg.model
    model = UnconditionalDiffusion(
        num_tok=model_cfg.num_tok,
        seq_len=model_cfg.seq_len,
        n_emb=model_cfg.n_emb,
        n_head=model_cfg.n_head,
        n_layer=model_cfg.n_layer,
        diff_step=model_cfg.diff_step,
        auxiliary_loss_weight=model_cfg.auxiliary_loss_weight,
        adaptive_auxiliary_loss=model_cfg.adaptive_auxiliary_loss,
        mask_weight=list(model_cfg.mask_weight),
    )
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'UnconditionalDiffusion params: {model_params:,}')

    # ------------------------------------------------------------------ #
    #  Body model faces (for visualization)                               #
    # ------------------------------------------------------------------ #
    ref_bm_path = 'body_models/smplh/neutral/model.npz'
    ref_bm = np.load(ref_bm_path)
    faces = torch.from_numpy(ref_bm['f'].astype(np.int32))

    # ------------------------------------------------------------------ #
    #  构建 Trainer（仅用于推理/可视化，不执行 fit）                       #
    # ------------------------------------------------------------------ #
    config_training = dict(cfg.train)
    config_training['save_every_n_steps'] = None
    # 测试时 batch 大小可以小一些（和 loader 保持一致即可）
    config_training['batch'] = cfg.train.batch

    trainer = UnconditionalDiffusion_Train(
        model=model,
        vqvae=mesh_vqvae,
        training_data=test_data,   # 占位，测试阶段不会调用 fit
        validation_data=test_data,
        config_training=config_training,
        faces=faces,
    )

    # ------------------------------------------------------------------ #
    #  加载 checkpoint                                                    #
    # ------------------------------------------------------------------ #
    load_optimizer = cfg.get('resume', {}).get('optimizer', False)
    trainer.load(path=ckpt_path, optimizer=load_optimizer)
    print(f'Loaded checkpoint: {ckpt_path}')

    # ------------------------------------------------------------------ #
    #  逐 batch 调用 visualize_diffusion_steps                            #
    # ------------------------------------------------------------------ #
    device = trainer.device
    os.makedirs(args.output_dir, exist_ok=True)

    for batch_idx, mesh in enumerate(test_loader):
        if batch_idx >= args.num_samples:
            break

        # DatasetMeshTest 直接返回 mesh tensor
        if isinstance(mesh, dict):
            mesh = mesh['local_mesh']
        mesh = mesh.to(device)

        # 从 VQVAE 获取 GT tokens
        with torch.no_grad():
            mesh_vqvae.to(device)
            gt_tokens = mesh_vqvae.get_codebook_indices(mesh)  # [B, seq_len]

        token_text_path = os.path.join(
            args.output_dir,
            f'mask_step_text_batch{batch_idx}.png',
        )

        print(f'\n[Batch {batch_idx}] Running visualize_diffusion_steps ...')
        # diffusion_steps = [99, 79, 59, 39, 19, 9,8,7,6,5,4,3,2,1,0]
        diffusion_steps = [50, 49, 48, 47, 19,10,0]
        results = trainer.visualize_diffusion_steps(
            gt_tokens=gt_tokens,
            save_steps=diffusion_steps,
            output_dir=args.output_dir,
            temperature=args.temperature,
            filter_ratio=0.5,
            vis_idx=args.vis_idx,
            batch_idx=batch_idx,
            token_text_path=token_text_path,
        )
        print(f'[Batch {batch_idx}] Done. Steps visualized: {sorted(results.keys())}')

    print(f'\nAll results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
