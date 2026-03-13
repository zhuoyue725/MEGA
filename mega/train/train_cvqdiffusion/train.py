"""CVQDiffusion 训练 + 过拟合验证脚本

用法（从项目根目录运行）：
    python mega/train/train_cvqdiffusion/train.py

流程：
  1. 读取 MixedDataset 数据集（img + local_mesh）
  2. 加载 HRNet-W48 backbone
  3. 构建 CVQDiffusion 并训练
  4. 保存权重，再加载权重进行 eval 推理，验证可以过拟合
"""
import sys
import os

# 将项目根目录加入 sys.path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from mega.model.backbone import hrnet_w48
from mega.model.cvqdiffusion import CVQDiffusion
from mega.data import MixedDataset
import mesh_vq_vae

# ------------------------------------------------------------------ #
#  超参                                                               #
# ------------------------------------------------------------------ #
CKPT_DIR       = 'mega/train/train_cvqdiffusion/checkpoints'
CKPT_PATH      = os.path.join(CKPT_DIR, 'cvqdiffusion_overfit.pth')
CFG_PATH       = 'configs/config_cvqdiffusion/config_hrnet.yaml'

BATCH_SIZE     = 4
NUM_EPOCHS     = 200       # 足够多的 epoch 以过拟合 100 个样本
LR             = 1e-4
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_INTERVAL   = 20        # 每隔多少 epoch 打印一次 loss


# ------------------------------------------------------------------ #
#  主流程                                                             #
# ------------------------------------------------------------------ #
def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ---- 读取配置 ----
    cfg = OmegaConf.load(CFG_PATH)
    model_cfg = cfg.model
    print('Config loaded.')

    # ---- 数据集（与 train_mega.py 一致） ----
    training_data = MixedDataset(
        cfg.training_data.file,
        augment=False,
        flip=False,
        proportion=1,
    )
    dataloader = DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print(f'Dataset size: {len(training_data)}')

    # ---- MeshVQVAE（用于动态获取 codebook indices） ----
    convmesh_model = mesh_vq_vae.FullyConvAE(cfg.modelconv, test_mode=True)
    vqvae = mesh_vq_vae.MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    vqvae.load(path_model='checkpoint/MESH_VQVAE/mesh_vqvae_54')
    convmesh_model.init_test_mode()
    vqvae = vqvae.to(DEVICE)
    vqvae.eval()
    print('MeshVQVAE loaded.')

    # ---- Backbone ----
    backbone = hrnet_w48(
        pretrained_ckpt_path=cfg.backbone.pretrained,
        downsample=True,
        use_conv=True,
    ).to(DEVICE)
    print(f'Backbone loaded from {cfg.backbone.pretrained}')

    # ---- 模型 ----
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
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'CVQDiffusion total params: {total_params:,}')

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # ================================================================ #
    #  训练循环                                                         #
    # ================================================================ #
    # print(f'\n--- Training on {DEVICE} for {NUM_EPOCHS} epochs ---')
    # model.train()
    # for epoch in range(1, NUM_EPOCHS + 1):
    #     epoch_loss = 0.0
    #     for data in dataloader:
    #         # 与 train_class.py one_epoch 一致：从 local_mesh 动态获取 tokens
    #         imgs = data['img'].to(DEVICE)          # [B, 3, 224, 224]
    #         with torch.no_grad():
    #             tokens = vqvae.get_codebook_indices(
    #                 data['local_mesh'].to(DEVICE)
    #             )                                  # [B, 54]

    #         optimizer.zero_grad()
    #         out  = model(tokens, imgs, return_loss=True, return_logits=False)
    #         loss = out['loss']
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()

    #         epoch_loss += loss.item()

    #     scheduler.step()

    #     if epoch % LOG_INTERVAL == 0 or epoch == 1:
    #         avg = epoch_loss / len(dataloader)
    #         print(f'Epoch [{epoch:4d}/{NUM_EPOCHS}]  loss={avg:.4f}  '
    #               f'lr={scheduler.get_last_lr()[0]:.2e}')

    # # ---- 保存权重 ----
    # torch.save({'model': model.state_dict(), 'loss': epoch_loss / len(dataloader)},
    #            CKPT_PATH)
    # print(f'\nCheckpoint saved to {CKPT_PATH}')

    # ================================================================ #
    #  测试：加载权重，在训练集上推理，验证过拟合                         #
    # ================================================================ #
    print('\n--- Evaluation (overfit check) ---')
    model_eval = CVQDiffusion(
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
    ).to(DEVICE)
    model_eval.load(CKPT_PATH)
    model_eval.eval()

    # 逐批次计算 token 预测准确率
    eval_loader = DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    total_correct = 0
    total_tokens  = 0

    with torch.no_grad():
        for data in tqdm(eval_loader, desc='Eval'):
            imgs = data['img'].to(DEVICE)
            tokens = vqvae.get_codebook_indices(
                data['local_mesh'].to(DEVICE)
            )                                      # [B, 54]  ground-truth

            # sample: 从条件图像生成 token 序列
            sample_out = model_eval.sample(
                imgs,
                filter_ratio=0.0,
                temperature=1.0,
                return_logits=False,
            )
            if isinstance(sample_out, dict):
                pred_tokens = sample_out.get(
                    'content_token',
                    sample_out.get('tokens', None)
                )
            else:
                pred_tokens = sample_out

            if pred_tokens is not None:
                pred_tokens = pred_tokens.to(DEVICE)
                correct = (pred_tokens == tokens).sum().item()
                total_correct += correct
                total_tokens  += tokens.numel()

    if total_tokens > 0:
        acc = total_correct / total_tokens * 100
        print(f'Token accuracy on train set: {acc:.2f}%  '
              f'({total_correct}/{total_tokens})')
    else:
        print('No predictions returned from sample(); '
              'check DiffusionTransformer.sample output keys.')

    print('Done.')


if __name__ == '__main__':
    main()
