import sys
import os

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt

from mega.base import Train
from mega.model.cvqdiffusion import CVQDiffusion
from mega.data import MixedDataset
from mega.utils.eval import pa_mpjpe, mpjpe, v2v
import pandas as pd


# ------------------------------------------------------------------ #
#  简单的 Follow（参照 train_conditional_vqmae/follow_up_mae.py）      #
# ------------------------------------------------------------------ #
class FollowDiff:
    """轻量级训练记录器，保存 checkpoint 并绘制 loss 曲线。"""

    def __init__(self, name: str = 'cvqdiffusion', dir_save: str = 'checkpoint'):
        from datetime import datetime
        from pathlib import Path

        self.name = name
        now = datetime.now()
        base = Path(dir_save) / name.upper()
        base.mkdir(parents=True, exist_ok=True)
        tag  = now.strftime('%Y-%m-%d/%H-%M')
        self.path = base / tag
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / 'samples').mkdir(exist_ok=True)
        (self.path / 'samples_train').mkdir(exist_ok=True)
        self.path_samples       = self.path / 'samples'
        self.path_samples_train = self.path / 'samples_train'

        self.best_loss = 1e8
        self.train_losses: list = []
        self.val_losses:   list = []

    def save(self, parameters: dict, epoch: int, every_step: int = 10):
        if epoch % every_step == 0:
            torch.save(parameters, str(self.path / 'model_checkpoint'))
            print(f'\t [FollowDiff] checkpoint saved (epoch={epoch}, '
                  f'loss={parameters["loss"]:.4f})')
        if parameters['loss'] <= self.best_loss:
            self.best_loss = parameters['loss']
            torch.save(parameters, str(self.path / 'model_best_loss'))
            print(f'\t [FollowDiff] best model saved (loss={self.best_loss:.4f})')

    def plot(self):
        if not self.train_losses:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.train_losses, label='train')
        if self.val_losses:
            plt.plot(self.val_losses, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Diffusion Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(self.path / 'loss.png'))
        plt.close()


# ------------------------------------------------------------------ #
#  CVQDiffusion_Train                                                 #
# ------------------------------------------------------------------ #
class CVQDiffusion_Train(Train):
    """
    参照 CVQMAE_Train，但损失函数只使用 DiffusionTransformer 内部的
    VB-stochastic diffusion loss：

        out  = model(tokens, imgs, return_loss=True, return_logits=False)
        loss = out['loss']
        loss.backward()

    数据集使用 MixedDataset（与 train_mega.py 一致）。
    token 索引通过 MeshVQVAE.get_codebook_indices(local_mesh) 动态获取，
    与 CVQMAE_Train.one_epoch 保持一致。
    """

    def __init__(
        self,
        model: CVQDiffusion,
        vqvae,                   # mesh_vq_vae.MeshVQVAE，用于获取 codebook indices
        training_data: Dataset,
        validation_data: Dataset,
        config_training: dict = None,
        faces=None,              # 网格三角面，用于可视化
        joints_regressor=None,   # 关节点回归矩阵，用于评估指标
    ):
        super().__init__()

        self.device = torch.device(config_training.get('device', 'cuda'))

        # ---- 模型 ----
        self.model = model.to(self.device)
        self.vqvae = vqvae.to(self.device)
        self.vqvae.eval()  # vqvae 只做推理，不参与训练
        self.f = faces         # 网格三角面片，用于可视化
        self.joints_reg = joints_regressor

        # ---- DataLoader ----
        self.training_loader = DataLoader(
            training_data,
            batch_size=config_training['batch'],
            shuffle=True,
            num_workers=config_training.get('workers', 0),
            drop_last=True,
        )
        self.validation_loader = DataLoader(
            validation_data,
            batch_size=config_training['batch'],
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )

        # ---- 优化器（小数据集过拟合：直接使用配置 lr，不做 batch 缩放）----
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config_training['lr'],
            betas=(0.9, 0.95),
            weight_decay=config_training.get('weight_decay', 0.0),
        )
        total_epoch  = config_training['total_epoch']
        warmup_epoch = config_training.get('warmup_epoch', 20)
        lr_func = lambda epoch: min(
            (epoch + 1) / (warmup_epoch + 1e-8),
            0.5 * (math.cos(epoch / total_epoch * math.pi) + 1),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_func, verbose=True
        )

        # ---- 记录器 ----
        self.follow = FollowDiff(
            name='cvqdiffusion',
            dir_save='checkpoint',
        )

        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0    # 全局迭代步数，用于控制可视化频率
        self.train_losses: list = []
        self.val_losses:   list = []

    # ---------------------------------------------------------------- #
    #  one_epoch                                                        #
    # ---------------------------------------------------------------- #
    def one_epoch(self, epoch: int):
        self.model.train()
        losses = []
        for data in tqdm(self.training_loader, desc=f'Train epoch {epoch}'):
            # 与 CVQMAE_Train.one_epoch 一致：用 vqvae 动态获取 token 索引
            imgs = data['img'].to(self.device)          # [B, 3, 224, 224]
            with torch.no_grad():
                tokens = self.vqvae.get_codebook_indices(
                    data['local_mesh'].to(self.device)
                )                                       # [B, 54]

            self.optimizer.zero_grad()

            # ---- 唯一的损失：diffusion VB loss ----
            out  = self.model(tokens, imgs,
                              return_loss=True, return_logits=False)
            loss = out['loss']

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())
            self.train_losses.append(loss.item())
            self.step_count += 1

            # if self.step_count % 10 == 0:
            #     break
            # ---- 每 50 步绘制一次重建网格（参照 CVQMAE_Train）----
            if self.step_count % 50 == 0 and self.f is not None:
                with torch.no_grad():
                    # 用当前 img 采样生成 token，解码为网格
                    sample_out  = self.model.sample(
                        imgs[:4], filter_ratio=0.0, temperature=1.0
                    )
                    pred_tokens = sample_out['content_token']          # [B, 54]
                    pred_mesh   = self.vqvae.decode(pred_tokens).cpu() # [B, V, 3]
                    real_mesh   = data['mesh'][:4].cpu()
                    real_mesh   = data['local_mesh'][:4].cpu()

                    gt_tokens = self.vqvae.get_codebook_indices( # [B, 54]
                        data['local_mesh'].to(self.device)
                    )     
                    gt_mesh   = self.vqvae.decode(gt_tokens).cpu() # [B, V, 3]
                self.plot_meshes_(
                    pred_mesh,
                    show=False,
                    rot=True,
                    save=f'{self.follow.path_samples_train}/'
                         f'epoch{epoch}_step{self.step_count}-reconstruction.png',
                )
                self.plot_meshes_(
                    real_mesh,
                    show=False,
                    rot=True,
                    save=f'{self.follow.path_samples_train}/'
                         f'epoch{epoch}_step{self.step_count}-real.png',
                )
                self.plot_meshes_(
                    gt_mesh,
                    show=False,
                    rot=True,
                    save=f'{self.follow.path_samples_train}/'
                         f'epoch{epoch}_step{self.step_count}-gt_mesh.png',
                )
        with torch.no_grad():
            # 用当前 img 采样生成 token，解码为网格
            sample_out  = self.model.sample(
                imgs[:4], filter_ratio=0.0, temperature=1.0
            )
            pred_tokens = sample_out['content_token']          # [B, 54]
            pred_mesh   = self.vqvae.decode(pred_tokens).cpu() # [B, V, 3]
            real_mesh   = data['local_mesh'][:4].cpu()
        self.plot_meshes_(
            pred_mesh,
            show=False,
            rot=True,
            save=f'{self.follow.path_samples_train}/'
                    f'epoch{epoch}_step{self.step_count}-reconstruction.png',
        )
        self.plot_meshes_(
            real_mesh,
            show=False,
            rot=True,
            save=f'{self.follow.path_samples_train}/'
                    f'epoch{epoch}_step{self.step_count}-real.png',
        )
        return losses

    # ---------------------------------------------------------------- #
    #  eval                                                             #
    # ---------------------------------------------------------------- #
    def eval(self, epoch: int):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in tqdm(self.validation_loader, desc=f'Val   epoch {epoch}'):
                imgs   = data['img'].to(self.device)
                tokens = self.vqvae.get_codebook_indices( # 这个返回的bs是mesh-vq-vae的
                    data['local_mesh'].to(self.device)
                )

                out  = self.model(tokens, imgs,
                                  return_loss=True, return_logits=False)
                loss = out['loss']
                losses.append(loss.item())
                self.val_losses.append(loss.item())

        return losses

    # ---------------------------------------------------------------- #
    #  fit                                                              #
    # ---------------------------------------------------------------- #
    def fit(self):
        total_epoch = self.config_training['total_epoch']
        for e in range(self.load_epoch, total_epoch):
            train_losses = self.one_epoch(epoch=e)
            val_losses   = self.eval(epoch=e)
            self.lr_scheduler.step()

            avg_train = mean(train_losses)
            avg_val   = mean(val_losses) if val_losses else float('nan')
            print(f'Epoch [{e+1:4d}/{total_epoch}]  '
                  f'train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  '
                  f'lr={self.lr_scheduler.get_last_lr()[0]:.2e}')

            parameters = dict(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.lr_scheduler.state_dict(),
                epoch=e,
                loss=avg_val,
            )
            self.follow.save(parameters, epoch=e, every_step=10)
            self.follow.train_losses.append(avg_train)
            self.follow.val_losses.append(avg_val)
            self.follow.plot()

    # ---------------------------------------------------------------- #
    #  eval_deterministic                                               #
    # ---------------------------------------------------------------- #
    def eval_deterministic(self, visualize: bool = True):
        """用 diffusion 采样（确定性 filter_ratio=0）评估验证集指标。

        指标：PA-MPJPE、MPJPE、V2V（mm），结果保存为 results.csv。
        可视化：GT 网格 + 重建网格（每个 batch 存一张对比图）。
        """
        self.model.eval()
        with torch.no_grad():
            lpampjpe = []
            lmpjpe   = []
            lv2v     = []
            limgname = []
            count    = 0

            for data in tqdm(iter(self.validation_loader), desc='eval_deterministic'):
                imgs = data['img'].to(self.device)           # [B, 3, 224, 224]
                limgname.append(data['imgname'])

                # 通过 diffusion 采样得到 content token
                sample_out  = self.model.sample(
                    imgs, filter_ratio=0.0, temperature=1.0
                )
                pred_tokens = sample_out['content_token']            # [B, 54]
                pred_mesh   = self.vqvae.decode(pred_tokens).cpu()   # [B, V, 3]

                gt_mesh = data['mesh']   # [B, V, 3]，世界坐标系真值

                if self.joints_reg is not None:
                    pa_mpjpe_err = pa_mpjpe(gt_mesh, pred_mesh, self.joints_reg)
                    mpjpe_err    = mpjpe(gt_mesh, pred_mesh, self.joints_reg)
                else:
                    pa_mpjpe_err = torch.tensor(0.0)
                    mpjpe_err    = torch.tensor(0.0)

                v2v_err = v2v(gt_mesh, pred_mesh)

                lpampjpe.append(1000 * pa_mpjpe_err.item())
                lmpjpe.append(1000 * mpjpe_err.item())
                lv2v.append(1000 * v2v_err.item())

                if visualize:
                    count += 1
                    self.plot_meshes_(
                        gt_mesh[:4],
                        show=False,
                        rot=True,
                        save=f'{self.follow.path_samples}/{count}_gt.png',
                    )
                    self.plot_meshes_(
                        pred_mesh[:4],
                        show=False,
                        rot=True,
                        save=f'{self.follow.path_samples}/{count}_reconstructed.png',
                    )

            print(
                f'V2V: {mean(lv2v):.2f}  '
                f'MPJPE: {mean(lmpjpe):.2f}  '
                f'PA-MPJPE: {mean(lpampjpe):.2f}  (mm)'
            )

            dict_results = {
                'imgname':  limgname,
                'pampjpe':  lpampjpe,
                'mpjpe':    lmpjpe,
                'v2v':      lv2v,
            }
            df = pd.DataFrame(dict_results)
            df.to_csv(f'{self.follow.path}/results.csv', index=False)

        return lv2v

    # ---------------------------------------------------------------- #
    #  load                                                             #
    # ---------------------------------------------------------------- #
    def load(self, path: str = '', optimizer: bool = True):
        print('LOAD [', end='')
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        if optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.load_epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 'N/A')
        print(f'model: ok  | optimizer:{optimizer}  '
              f'|  loss: {loss}  |  epoch: {self.load_epoch}]')

    # ---------------------------------------------------------------- #
    #  plot_meshes_（参照 CVQMAE_Train）                                #
    # ---------------------------------------------------------------- #
    def plot_meshes_(
        self,
        meshes,
        show: bool = False,
        save: str = None,
        rot: bool = False,
    ):
        from mega.utils.mesh_render import renderer
        from matplotlib.gridspec import GridSpec

        images = renderer(meshes, self.f, 'cpu', rot=rot)
        n = min(len(meshes), 4)
        fig = plt.figure(figsize=(5 * n, 5))
        gs  = GridSpec(ncols=n, nrows=1)
        for i in range(n):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(images[i].cpu().detach().numpy())
            ax.axis('off')
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
        plt.close()
