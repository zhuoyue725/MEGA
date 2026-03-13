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
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
)


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

        self.mse = nn.MSELoss()

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

            # ---- 损失：diffusion VB loss + rotation regression loss ----
            out  = self.model(tokens, imgs,
                              return_loss=True, return_logits=False)
            loss = out['loss']

            # rotation loss: 6D pred -> rotation matrix vs GT axis-angle -> matrix
            pred_rot = out['pred_rot']                              # [B, 6]
            rotmat   = rotation_6d_to_matrix(pred_rot)             # [B, 3, 3]
            rot_loss = self.mse(
                rotmat,
                axis_angle_to_matrix(data['rotation'].to(self.device)),
            ).mean()
            loss = loss + rot_loss

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())
            self.train_losses.append(loss.item())
            self.step_count += 1

            # if self.step_count % 10 == 0:
            #     break
            # ---- 每 50 步绘制一次重建网格（参照 CVQMAE_Train）----
            if self.step_count % 10 == 0 and self.f is not None:
                with torch.no_grad():
                    # 用当前 img 采样生成 token，解码为网格
                    sample_out  = self.model.sample(
                        imgs[:4], filter_ratio=0.0, temperature=1.0
                    )
                    pred_rot = out['pred_rot']                              # [B, 6]
                    rotmat   = rotation_6d_to_matrix(pred_rot).cpu()             # [B, 3, 3]

                    pred_tokens = sample_out['content_token']          # [B, 54]
                    mesh_canonical   = self.vqvae.decode(pred_tokens).cpu() # [B, V, 3]
                    real_mesh   = data['mesh'][:4].cpu()
                    pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

                    # 保存 pred_mesh 为 obj 文件
                    import trimesh
                    from pathlib import Path
                    obj_dir = Path('demo_data/output_vqd/3dpw_train')
                    obj_dir.mkdir(parents=True, exist_ok=True)
                    for i, verts in enumerate(pred_mesh):
                        mesh_obj = trimesh.Trimesh(
                            vertices=verts.numpy(),
                            faces=self.f,
                            process=False,
                        )
                        mesh_obj.export(str(obj_dir / f'epoch{epoch}_step{self.step_count}_{i}.obj'))


                save_prefix = (f'{self.follow.path_samples_train}/'
                               f'epoch{epoch}_step{self.step_count}_cmp')
                self.plot_compare_(pred_mesh, real_mesh, save_prefix)
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

                # rotation loss
                pred_rot = out['pred_rot']                              # [B, 6]
                rotmat   = rotation_6d_to_matrix(pred_rot)             # [B, 3, 3]
                rot_loss = self.mse(
                    rotmat,
                    axis_angle_to_matrix(data['rotation'].to(self.device)),
                ).mean()
                loss = loss + rot_loss

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
                mesh_canonical   = self.vqvae.decode(pred_tokens).cpu()   # [B, V, 3]

                pred_rot = sample_out['pred_rot']
                rotmat = rotation_6d_to_matrix(pred_rot).cpu()
                pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

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
                    save_prefix = (f'{self.follow.path_samples_train}/'
                               f'eval_det_step{self.step_count}_cmp_{count}')
                    self.plot_compare_(pred_mesh, gt_mesh, save_prefix)

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
    #  eval_stochastic                                                  #
    # ---------------------------------------------------------------- #
    def eval_stochastic(
        self,
        sample_size: int = 25,
        temperature: float = 1.0,
        visualize: bool = True,
        vis_idx: int = 0,
    ):
        """随机多次采样评估验证集，取每个样本误差最小的采样结果。

        Args:
            sample_size : 每个样本的采样次数。
            temperature : diffusion 采样温度。
            visualize   : 是否保存可视化图像。
            vis_idx     : 每个 batch 中用于可视化的样本索引。
        """
        self.model.eval()
        import PIL.Image as PILImage

        with torch.no_grad():
            lpampjpe = []
            lmpjpe   = []
            lv2v     = []
            limgname = []
            count    = 0

            for data in tqdm(iter(self.validation_loader), desc='eval_stochastic'):
                imgs    = data['img'].to(self.device)   # [B, 3, 224, 224]
                gt_mesh = data['mesh']                  # [B, V, 3]
                B       = imgs.size(0)
                limgname.append(data['imgname'])

                # 重复 sample_size 次: [B*S, 3, 224, 224]
                imgs_rep = imgs.repeat(sample_size, 1, 1, 1)

                sample_out      = self.model.sample(
                    imgs_rep, filter_ratio=0.0, temperature=temperature
                )
                pred_tokens     = sample_out['content_token']              # [B*S, 54]
                mesh_canonical  = self.vqvae.decode(pred_tokens).cpu()    # [B*S, V, 3]
                pred_rot        = sample_out['pred_rot']                   # [B*S, 6]
                rotmat          = rotation_6d_to_matrix(pred_rot).cpu()   # [B*S, 3, 3]
                pred_mesh_all   = (
                    rotmat @ mesh_canonical.transpose(2, 1)
                ).transpose(2, 1)                                          # [B*S, V, 3]

                # reshape -> [S, B, V, 3]
                pred_mesh_all   = pred_mesh_all.view(sample_size, B, *pred_mesh_all.shape[1:])
                mesh_can_all    = mesh_canonical.view(sample_size, B, *mesh_canonical.shape[1:])

                gt_rep = gt_mesh.repeat(sample_size, 1, 1)  # [S*B, V, 3]
                pred_flat      = pred_mesh_all.reshape(sample_size * B, *pred_mesh_all.shape[2:])
                mesh_can_flat  = mesh_can_all.reshape(sample_size * B, *mesh_can_all.shape[2:])

                if self.joints_reg is not None:
                    pa_err = pa_mpjpe(
                        gt_rep, mesh_can_flat, self.joints_reg,
                        reduction=False, dim=1,
                    )  # [S*B]
                    list_pampjpe = (1000 * pa_err).tolist()
                    # 每隔 B 取一个样本的 S 个值，取 min
                    for b in range(B):
                        lpampjpe.append(min(list_pampjpe[b::B]))

                    mp_err = mpjpe(
                        gt_rep, pred_flat, self.joints_reg,
                        reduction=False, dim=1,
                    )  # [S*B]
                    list_mpjpe = (1000 * mp_err).tolist()
                    for b in range(B):
                        lmpjpe.append(min(list_mpjpe[b::B]))
                else:
                    for b in range(B):
                        lpampjpe.append(0.0)
                        lmpjpe.append(0.0)

                v2v_err = v2v(
                    gt_rep, pred_flat,
                    reduction=False, dim=1,
                )  # [S*B]
                list_v2v = (1000 * v2v_err).tolist()
                for b in range(B):
                    lv2v.append(min(list_v2v[b::B]))

                if visualize and self.f is not None:
                    count += 1
                    # 取 vis_idx 样本的所有 sample_size 个采样结果横向拼接
                    idx = min(vis_idx, B - 1)
                    samples_for_idx = pred_mesh_all[:, idx]               # [S, V, 3]
                    save_path = (f'{self.follow.path_samples_train}/'
                                 f'stoch_step{count}_idx{idx}_samples.png')
                    self.plot_meshes_(samples_for_idx, show=False, rot=True, save=save_path)

            print(
                f'[stochastic S={sample_size} T={temperature}]  '
                f'V2V: {mean(lv2v):.2f}  '
                f'MPJPE: {mean(lmpjpe):.2f}  '
                f'PA-MPJPE: {mean(lpampjpe):.2f}  (mm)'
            )

            # dict_results = {
            #     'imgname': limgname,
            #     'pampjpe': lpampjpe,
            #     'mpjpe':   lmpjpe,
            #     'v2v':     lv2v,
            # }
            # df = pd.DataFrame(dict_results)
            # df.to_csv(f'{self.follow.path}/results_stochastic.csv', index=False)

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
    #  plot_samples_row_                                                #
    # ---------------------------------------------------------------- #
    def plot_samples_row_(
        self,
        meshes,
        save: str,
        rot: bool = True,
    ):
        """将多个网格渲染图横向拼接保存为一张图。

        Args:
            meshes : [S, V, 3] tensor，S 个采样网格。
            save   : 保存路径。
        """
        import PIL.Image as PILImage
        from mega.utils.mesh_render import renderer

        images = renderer(meshes, self.f, 'cpu', rot=rot)  # list of [H, W, 3]
        row = np.concatenate(
            [img.cpu().detach().numpy() for img in images], axis=1
        )  # 横向拼接
        PILImage.fromarray(row.astype(np.uint8)).save(save)

    # ---------------------------------------------------------------- #
    #  plot_compare_                                                    #
    # ---------------------------------------------------------------- #
    def plot_compare_(
        self,
        pred_mesh,
        real_mesh,
        save_prefix: str,
        rot: bool = True,
    ):
        """保存 pred 和 real 网格图，纵向拼接为 compare.png，删除中间文件。"""
        import PIL.Image as PILImage

        path_recon = f'{save_prefix}-reconstruction.png'
        path_real  = f'{save_prefix}-real.png'
        path_comb  = f'{save_prefix}-compare.png'

        self.plot_meshes_(pred_mesh, show=False, rot=rot, save=path_recon)
        self.plot_meshes_(real_mesh, show=False, rot=rot, save=path_real)

        img_top = PILImage.open(path_recon)
        img_bot = PILImage.open(path_real)
        combined = PILImage.fromarray(
            np.concatenate([np.array(img_top), np.array(img_bot)], axis=0)
        )
        combined.save(path_comb)
        os.remove(path_recon)
        os.remove(path_real)

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
