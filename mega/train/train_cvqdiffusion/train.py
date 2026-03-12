"""
CVQDiffusion 训练 + 过拟合验证脚本

用法（从项目根目录运行）：
    python mega/train/train_cvqdiffusion/train.py

流程：
  1. 读取 npz 数据集（图像路径 + token 索引）
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
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from omegaconf import OmegaConf
from tqdm import tqdm

from mega.model.backbone import hrnet_w48
from mega.model.cvqdiffusion import CVQDiffusion
from mega.utils.augmentations import crop

# ------------------------------------------------------------------ #
#  超参                                                               #
# ------------------------------------------------------------------ #
NPZ_PATH       = 'mega/train/train_cvqdiffusion/data/train_cvqdiffusion.npz'
CKPT_DIR       = 'mega/train/train_cvqdiffusion/checkpoints'
CKPT_PATH      = os.path.join(CKPT_DIR, 'cvqdiffusion_overfit.pth')
CFG_PATH       = 'configs/config_cvqmae/config_hrnet.yaml'

BATCH_SIZE     = 4
NUM_EPOCHS     = 200       # 足够多的 epoch 以过拟合 100 个样本
LR             = 1e-4
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_INTERVAL   = 20        # 每隔多少 epoch 打印一次 loss


# ------------------------------------------------------------------ #
#  Dataset                                                            #
# ------------------------------------------------------------------ #
class CVQDiffusionDataset(Dataset):
    """
    读取 npz 文件中的图像路径和 token 索引。
    图像处理方式与 DatasetHMRSquare 完全一致：
      cv2.imread -> crop(img, center, scale, [224, 224]) -> normalize_resnet
    center 取图像中心，scale = min(H, W) / 200。
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.img_paths = data['img_paths'].tolist()   # list of str
        self.tokens    = data['tokens']               # [N, 54]  int32
        # 与 DatasetHMRSquare 保持一致的归一化参数
        self.normalize_resnet = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        # center 取图像中心，scale = min(H,W)/200（与 DatasetHMRSquare 的 scale 含义一致）
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale  = min(h, w) / 200.0

        # 与 DatasetHMRSquare.rgb_processing 一致：crop 到 [224, 224]
        img = crop(img, center, scale, [224, 224], rot=0)
        img = np.transpose(img.astype('float32'), (2, 0, 1)) / 255.0  # [3, 224, 224]
        img = torch.from_numpy(img).float()
        img = self.normalize_resnet(img)                               # [3, 224, 224]

        token = torch.from_numpy(self.tokens[idx].astype(np.int64))   # [54]
        return img, token


# ------------------------------------------------------------------ #
#  主流程                                                             #
# ------------------------------------------------------------------ #
def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ---- 读取配置 ----
    cfg = OmegaConf.load(CFG_PATH)
    model_cfg = cfg.model
    print('Config loaded.')

    # ---- 数据集 ----
    dataset    = CVQDiffusionDataset(NPZ_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=False)
    print(f'Dataset size: {len(dataset)}')

    # ---- Backbone ----
    backbone = hrnet_w48(
        pretrained_ckpt_path=cfg.backbone.pretrained,  # body_models/pose_hrnet_w48.pth
        downsample=True,
        use_conv=True,
    ).to(DEVICE)
    print(f'Backbone loaded from {cfg.backbone.pretrained}')

    # ---- 模型 ----
    model = CVQDiffusion(
        backbone=backbone,
        backbone_feat_dim=model_cfg.cond_dim,    # 720
        cond_emb_dim=1024,
        num_tok=model_cfg.num_embeddings,        # 512
        seq_len=model_cfg.seq_length,            # 54
        n_emb=512,
        cond_dim=1024,
        cond_len=model_cfg.cond_length,          # 49
        n_head=8,
        n_layer=12,
        diff_step=100,
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
    #     for imgs, tokens in dataloader:
    #         imgs   = imgs.to(DEVICE)    # [B, 3, 224, 224]  (已裁剪)
    #         tokens = tokens.to(DEVICE)  # [B, 54]

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
        backbone_feat_dim=model_cfg.cond_dim,
        cond_emb_dim=1024,
        num_tok=model_cfg.num_embeddings,
        seq_len=model_cfg.seq_length,
        n_emb=512,
        cond_dim=1024,
        cond_len=model_cfg.cond_length,
        n_head=8,
        n_layer=12,
        diff_step=100,
    ).to(DEVICE)
    model_eval.load(CKPT_PATH)
    model_eval.eval()

    # 逐批次计算 token 预测准确率
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    total_correct = 0
    total_tokens  = 0

    with torch.no_grad():
        for imgs, tokens in tqdm(eval_loader, desc='Eval'):
            imgs   = imgs.to(DEVICE)
            tokens = tokens.to(DEVICE)      # [B, 54]  ground-truth

            # sample: 从条件图像生成 token 序列
            sample_out = model_eval.sample(
                imgs,
                filter_ratio=0.0,    # filter_ratio=0 表示从头开始完整生成
                temperature=1.0,
                return_logits=False,
            )
            # DiffusionTransformer.sample 返回 dict，content_token 在 'content_token' 键
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
