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
from mega.utils.loss import reprojection_loss, reprojection_loss_conf
import pandas as pd
from matplotlib.gridspec import GridSpec
from ...utils.img_renderer import visualize_reconstruction_pyrender, PyRender_Renderer
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
        
        # 记录各项损失的历史
        self.history = {
            'epoch': [],
            'loss_train': [],
            'loss_validation': [],
            'loss_diff_train': [],
            'loss_diff_validation': [],
            'loss_rot_train': [],
            'loss_rot_validation': [],
            'loss_2d_train': [],
            'loss_2d_validation': [],
            'v2v_train': [],
            'v2v_validation': [],
            'mpjpe_train': [],
            'mpjpe_validation': [],
            'pampjpe_train': [],
            'pampjpe_validation': [],
        }

    def follow(self, epoch: int, parameters: dict, **kwargs):
        """记录当前 epoch 的各项损失和指标"""
        self.history['epoch'].append(epoch)
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
        
        # 保存历史记录到 CSV
        df = pd.DataFrame(self.history)
        df.to_csv(str(self.path / 'training_history.csv'), index=False)
        
        # 保存 checkpoint
        self.save(parameters, epoch)

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
        """绘制所有损失项的变化曲线"""
        if not self.history['epoch']:
            return
        
        epochs = self.history['epoch']
        
        # 创建多子图布局
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Training Metrics', fontsize=16)
        
        # 1. 总损失
        ax = axes[0, 0]
        ax.plot(epochs, self.history['loss_train'], label='train', marker='o')
        ax.plot(epochs, self.history['loss_validation'], label='val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Diffusion Loss
        ax = axes[0, 1]
        ax.plot(epochs, self.history['loss_diff_train'], label='train', marker='o')
        ax.plot(epochs, self.history['loss_diff_validation'], label='val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Diffusion Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Rotation Loss
        ax = axes[1, 0]
        ax.plot(epochs, self.history['loss_rot_train'], label='train', marker='o')
        ax.plot(epochs, self.history['loss_rot_validation'], label='val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rotation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 2D Reprojection Loss
        ax = axes[1, 1]
        ax.plot(epochs, self.history['loss_2d_train'], label='train', marker='o')
        ax.plot(epochs, self.history['loss_2d_validation'], label='val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('2D Reprojection Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. V2V & MPJPE
        ax = axes[2, 0]
        ax.plot(epochs, self.history['v2v_train'], label='V2V train', marker='o')
        ax.plot(epochs, self.history['v2v_validation'], label='V2V val', marker='s')
        ax.plot(epochs, self.history['mpjpe_train'], label='MPJPE train', marker='^', linestyle='--')
        ax.plot(epochs, self.history['mpjpe_validation'], label='MPJPE val', marker='v', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. PA-MPJPE
        ax = axes[2, 1]
        ax.plot(epochs, self.history['pampjpe_train'], label='train', marker='o')
        ax.plot(epochs, self.history['pampjpe_validation'], label='val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PA-MPJPE (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(self.path / 'loss_metrics.png'), dpi=150)
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
        joints_regressor_smpl=None,  # SMPL 24关节回归矩阵，用于3DPW数据集
    ):
        super().__init__()

        self.device = torch.device(config_training.get('device', 'cuda'))

        # ---- 模型 ----
        self.model = model.to(self.device)
        self.vqvae = vqvae.to(self.device)
        self.vqvae.eval()  # vqvae 只做推理，不参与训练
        self.f = faces         # 网格三角面片，用于可视化
        self.joints_reg = joints_regressor
        self.joints_reg_smpl = joints_regressor_smpl

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
        
        # ---- 损失项记录 ----
        self.train_loss: list = []      # 总损失
        self.val_loss: list = []
        self.train_diff: list = []      # diffusion loss
        self.val_diff: list = []
        self.train_rot: list = []       # rotation loss
        self.val_rot: list = []
        self.train_2d: list = []        # reprojection loss
        self.val_2d: list = []
        self.train_v2v: list = []       # v2v 指标
        self.val_v2v: list = []
        self.train_mpjpe: list = []     # mpjpe 指标
        self.val_mpjpe: list = []
        self.train_pampjpe: list = []   # pa-mpjpe 指标
        self.val_pampjpe: list = []

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

            # ---- 损失：diffusion VB loss + rotation regression loss + reprojection loss ----
            out  = self.model(tokens, imgs,
                              return_loss=True, return_logits=False)
            diff_loss = out['loss']

            # rotation loss: 6D pred -> rotation matrix vs GT axis-angle -> matrix
            pred_rot = out['pred_rot']                              # [B, 6]
            pred_cam = out['pred_cam']                              # [B, 3]
            rotmat   = rotation_6d_to_matrix(pred_rot)             # [B, 3, 3]
            rot_loss = self.mse(
                rotmat,
                axis_angle_to_matrix(data['rotation'].to(self.device)),
            ).mean()

            # reprojection loss: 计算预测网格的重投影损失
            with torch.no_grad():
                # 解码 tokens 得到 canonical mesh
                mesh_canonical = self.vqvae.decode(tokens).cpu()   # [B, V, 3]
            
            # 应用旋转得到预测网格
            pred_mesh = (rotmat.cpu() @ mesh_canonical.transpose(2, 1)).transpose(2, 1)  # [B, V, 3]
            
            # 根据数据集类型计算重投影损失
            is_3dpw = data["is_3dpw"] == True
            not_3dpw = data["is_3dpw"] == False
            reproj_loss = 0
            if is_3dpw.any():
                reproj_loss += reprojection_loss(
                    data["j2d"][is_3dpw][:, :, :2].to(torch.float32),
                    pred_mesh[is_3dpw],
                    pred_cam[is_3dpw].cpu(),
                    self.joints_reg_smpl,
                    # visualize=True,
                    # vis_path='./demo_out/joints2d',
                    # vis_counter=f"epoch{epoch}_{self.step_count}"
                )
            if not_3dpw.any():
                reproj_loss += reprojection_loss_conf(
                    data["j2d"][not_3dpw],
                    pred_mesh[not_3dpw],
                    pred_cam[not_3dpw].cpu(),
                    self.joints_reg,
                )
            
            # 计算评估指标
            gt_mesh = data['mesh'].cpu()
            if self.joints_reg is not None:
                v2v_err = v2v(gt_mesh, pred_mesh).item()
                mpjpe_err = mpjpe(gt_mesh, pred_mesh, self.joints_reg).item()
                pampjpe_err = pa_mpjpe(gt_mesh, pred_mesh, self.joints_reg).item()
            else:
                v2v_err = 0.0
                mpjpe_err = 0.0
                pampjpe_err = 0.0
            
            # 总损失
            loss = diff_loss + rot_loss + reproj_loss

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 记录各项损失
            losses.append(loss.item())
            self.train_loss.append(loss.item())
            self.train_diff.append(diff_loss.item())
            self.train_rot.append(rot_loss.item())
            self.train_2d.append(reproj_loss.item() if isinstance(reproj_loss, torch.Tensor) else reproj_loss)
            self.train_v2v.append(1000 * v2v_err)
            self.train_mpjpe.append(1000 * mpjpe_err)
            self.train_pampjpe.append(1000 * pampjpe_err)
            
            self.step_count += 1

            # if self.step_count % 10 == 0:
            #     break
            # ---- 每 50 步绘制一次重建网格（参照 CVQMAE_Train）----
            if self.step_count % 500 == 0 and self.f is not None:
                with torch.no_grad():
                    # 用当前 img 采样生成 token，解码为网格
                    sample_out  = self.model.sample(
                        imgs[:], filter_ratio=0.0, temperature=1.0
                    )
                    pred_rot = out['pred_rot']                              # [B, 6]
                    rotmat   = rotation_6d_to_matrix(pred_rot).cpu()             # [B, 3, 3]

                    pred_tokens = sample_out['content_token']          # [B, 54]
                    mesh_canonical   = self.vqvae.decode(pred_tokens).cpu() # [B, V, 3]
                    real_mesh   = data['mesh'][:4].cpu()
                    pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

                    pred_v = pred_mesh.detach()
                    cam = pred_cam
                    raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
                    self.plot_reproj_(
                        raw_img[:4],
                        pred_v[:4],
                        cam[:4],
                        show=False,
                        save=f"{self.follow.path_samples_train}/{epoch}_reprojection.png",
                    )
                    # 保存 pred_mesh 为 obj 文件
                    # import trimesh
                    # from pathlib import Path
                    # obj_dir = Path('demo_data/output_vqd/3dpw_train')
                    # obj_dir.mkdir(parents=True, exist_ok=True)
                    # for i, verts in enumerate(pred_mesh):
                    #     mesh_obj = trimesh.Trimesh(
                    #         vertices=verts.numpy(),
                    #         faces=self.f,
                    #         process=False,
                    #     )
                    #     mesh_obj.export(str(obj_dir / f'epoch{epoch}_step{self.step_count}_{i}.obj'))


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
                diff_loss = out['loss']

                # rotation loss
                pred_rot = out['pred_rot']                              # [B, 6]
                pred_cam = out['pred_cam']                              # [B, 3]
                rotmat   = rotation_6d_to_matrix(pred_rot)             # [B, 3, 3]
                rot_loss = self.mse(
                    rotmat,
                    axis_angle_to_matrix(data['rotation'].to(self.device)),
                ).mean()

                # reprojection loss
                mesh_canonical = self.vqvae.decode(tokens).cpu()
                pred_mesh = (rotmat.cpu() @ mesh_canonical.transpose(2, 1)).transpose(2, 1)
                
                is_3dpw = data["is_3dpw"] == True
                not_3dpw = data["is_3dpw"] == False
                reproj_loss = 0
                if is_3dpw.any():
                    reproj_loss += reprojection_loss(
                        data["j2d"][is_3dpw][:, :, :2].to(torch.float32),
                        pred_mesh[is_3dpw],
                        pred_cam[is_3dpw].cpu(),
                        self.joints_reg_smpl,
                        # visualize=True,
                        # vis_path='./demo_out/joints2d',
                        # vis_counter=f"epoch{epoch}_eval_{self.step_count}"
                    )
                if not_3dpw.any():
                    reproj_loss += reprojection_loss_conf(
                        data["j2d"][not_3dpw],
                        pred_mesh[not_3dpw],
                        pred_cam[not_3dpw].cpu(),
                        self.joints_reg,
                    )
                
                # 计算评估指标
                gt_mesh = data['mesh'].cpu()
                if self.joints_reg is not None:
                    v2v_err = v2v(gt_mesh, pred_mesh).item()
                    mpjpe_err = mpjpe(gt_mesh, pred_mesh, self.joints_reg).item()
                    pampjpe_err = pa_mpjpe(gt_mesh, pred_mesh, self.joints_reg).item()
                else:
                    v2v_err = 0.0
                    mpjpe_err = 0.0
                    pampjpe_err = 0.0
                
                # 总损失
                loss = diff_loss + rot_loss + reproj_loss

                losses.append(loss.item())
                self.val_loss.append(loss.item())
                self.val_diff.append(diff_loss.item())
                self.val_rot.append(rot_loss.item())
                self.val_2d.append(reproj_loss.item() if isinstance(reproj_loss, torch.Tensor) else reproj_loss)
                self.val_v2v.append(1000 * v2v_err)
                self.val_mpjpe.append(1000 * mpjpe_err)
                self.val_pampjpe.append(1000 * pampjpe_err)

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

            # 记录当前 epoch 的平均损失到 follow
            parameters = dict(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.lr_scheduler.state_dict(),
                epoch=e,
                loss=avg_val,
            )

            self.follow.follow(
                epoch=e,
                loss_train=mean(self.train_loss[-len(self.training_loader):]),
                loss_validation=mean(self.val_loss[-len(self.validation_loader):]),
                loss_diff_train=mean(self.train_diff[-len(self.training_loader):]),
                loss_diff_validation=mean(self.val_diff[-len(self.validation_loader):]),
                loss_rot_train=mean(self.train_rot[-len(self.training_loader):]),
                loss_rot_validation=mean(self.val_rot[-len(self.validation_loader):]),
                loss_2d_train=mean(self.train_2d[-len(self.training_loader):]),
                loss_2d_validation=mean(self.val_2d[-len(self.validation_loader):]),
                v2v_train=mean(self.train_v2v[-len(self.training_loader):]),
                v2v_validation=mean(self.val_v2v[-len(self.validation_loader):]),
                mpjpe_train=mean(self.train_mpjpe[-len(self.training_loader):]),
                mpjpe_validation=mean(self.val_mpjpe[-len(self.validation_loader):]),
                pampjpe_train=mean(self.train_pampjpe[-len(self.training_loader):]),
                pampjpe_validation=mean(self.val_pampjpe[-len(self.validation_loader):]),
                parameters=parameters,
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
        
        # 使用 strict=False 允许部分加载，忽略缺失或多余的键
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model'], strict=False)
        
        if optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.load_epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 'N/A')
        
        print(f'model: ok  | optimizer:{optimizer}  '
              f'|  loss: {loss}  |  epoch: {self.load_epoch}]')
        
        # 如果有缺失的键，显示警告信息
        if missing_keys:
            print(f'  ⚠ Missing keys (will be randomly initialized): {missing_keys}')
        if unexpected_keys:
            print(f'  ⚠ Unexpected keys (ignored): {unexpected_keys}')

    def load_rotcam_weights(self, path: str = 'checkpoint/CVQMAE/rotcam_weights.pth'):
        """
        加载预训练的旋转和相机预测器权重
        
        Args:
            path: rotcam_weights.pth 文件路径
        """
        print('LOAD ROTCAM WEIGHTS [', end='')
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'rotcam_weights' in checkpoint:
            rotcam_weights = checkpoint['rotcam_weights']
            metadata = checkpoint.get('metadata', {})
            print(f'source: {metadata.get("source_checkpoint", "unknown")} | ', end='')
        else:
            rotcam_weights = checkpoint
        
        # 处理键名差异：CVQMAE 中是 decoder.xxx，CVQDiffusion 中直接是 xxx
        # 移除 'decoder.' 前缀
        from collections import OrderedDict
        adjusted_weights = OrderedDict()
        for key, value in rotcam_weights.items():
            # 移除 'decoder.' 前缀（如果存在）
            if key.startswith('decoder.'):
                new_key = key.replace('decoder.', '', 1)
            else:
                new_key = key
            adjusted_weights[new_key] = value
        
        # 加载权重到模型（strict=False 允许只加载部分权重）
        missing_keys, unexpected_keys = self.model.load_state_dict(adjusted_weights, strict=False)
        
        # 统计成功加载的权重
        loaded_keys = [k for k in adjusted_weights.keys() if k not in unexpected_keys]
        print(f'loaded {len(loaded_keys)} params]')
        
        # 显示加载的模块
        loaded_modules = set()
        for key in loaded_keys:
            if '.' in key:
                module = key.split('.')[0]
            else:
                module = key
            loaded_modules.add(module)
        
        print(f'  Loaded modules: {", ".join(sorted(loaded_modules))}')
        
        if missing_keys:
            print(f'  ⚠ Missing keys (not in checkpoint): {len(missing_keys)} keys')
            # 只显示相关的缺失键
            relevant_missing = [k for k in missing_keys if any(m in k for m in ['rotcam', 'rot_predictor', 'cam_predictor'])]
            if relevant_missing:
                print(f'    Relevant missing: {relevant_missing}')
        
        if unexpected_keys:
            print(f'  ⚠ Unexpected keys (not in model): {len(unexpected_keys)} keys')
            print(f'    Keys: {unexpected_keys}')
        
        return loaded_keys

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

    def plot_reproj_(
        self,
        images,
        meshes,
        cameras,
        show: bool = True,
        save: str = None,
        vertex_colors=None,
    ):
        rendered_img = []
        render_reproj = PyRender_Renderer(faces=self.f)
        if vertex_colors is not None:
            for img, vertices, camera, texture in zip(
                images, meshes, cameras, vertex_colors
            ):
                rendered_img.append(
                    visualize_reconstruction_pyrender(
                        img, vertices, camera, render_reproj, vertex_colors=texture
                    )
                )
        else:
            for img, vertices, camera in zip(images, meshes, cameras):
                rendered_img.append(
                    visualize_reconstruction_pyrender(
                        img, vertices, camera, render_reproj
                    )
                )
        fig = plt.figure(figsize=(10, 10))
        if len(meshes) == 16:
            nrows = 4
            ncols = 4
        elif len(meshes) == 4:
            nrows = 2
            ncols = 2
        else:
            ncols = len(meshes)
            nrows = 1

        gs = GridSpec(ncols=ncols, nrows=nrows)
        i = 0
        for line in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[line, col])
                if rendered_img[i].shape[0] == 1:
                    ax.imshow(rendered_img[i][0, :, :])
                else:
                    ax.imshow(rendered_img[i])
                plt.axis("off")
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()
