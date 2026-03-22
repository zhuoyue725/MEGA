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
from pathlib import Path

from mega.base import Train
from mega.model.unconditional_diffusion import UnconditionalDiffusion
import pandas as pd


class FollowUnconditionalDiff:
    """轻量级训练记录器，保存 checkpoint 并绘制 loss 曲线。"""

    def __init__(self, name: str = 'unconditional_diffusion', dir_save: str = 'checkpoint'):
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
        
        self.history = {
            'epoch': [],
            'loss_train': [],
            'loss_validation': [],
        }

    def follow(self, epoch: int, parameters: dict, **kwargs):
        """记录当前 epoch 的各项损失和指标"""
        self.history['epoch'].append(epoch)
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
        
        df = pd.DataFrame(self.history)
        df.to_csv(str(self.path / 'training_history.csv'), index=False)
        self.save(parameters, epoch)

    def save(self, parameters: dict, epoch: int, every_step: int = 10):
        if epoch % every_step == 0:
            torch.save(parameters, str(self.path / 'model_checkpoint'))
            print(f'\t [FollowUnconditionalDiff] checkpoint saved (epoch={epoch}, '
                  f'loss={parameters["loss"]:.4f})')
        if parameters['loss'] <= self.best_loss:
            self.best_loss = parameters['loss']
            torch.save(parameters, str(self.path / 'model_best_loss'))
            print(f'\t [FollowUnconditionalDiff] best model saved (loss={self.best_loss:.4f})')

    def plot(self):
        """绘制损失曲线"""
        if not self.history['epoch']:
            return
        
        epochs = self.history['epoch']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, self.history['loss_train'], label='train', marker='o')
        ax.plot(epochs, self.history['loss_validation'], label='val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(self.path / 'loss_metrics.png'), dpi=150)
        plt.close()


class UnconditionalDiffusion_Train(Train):
    """
    无条件扩散模型训练类。
    
    损失函数只使用 DiffusionTransformer 内部的 VB-stochastic diffusion loss。
    数据集使用 MixedDataset，token 索引通过 MeshVQVAE.get_codebook_indices 动态获取。
    """

    def __init__(
        self,
        model: UnconditionalDiffusion,
        vqvae,
        training_data: Dataset,
        validation_data: Dataset,
        config_training: dict = None,
        faces=None,
    ):
        super().__init__()

        self.device = torch.device(config_training.get('device', 'cuda'))
        self.model = model.to(self.device)
        self.vqvae = vqvae.to(self.device)
        self.vqvae.eval()
        self.f = faces

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

        self.follow = FollowUnconditionalDiff(
            name='unconditional_diffusion',
            dir_save='checkpoint',
        )

        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.save_every_n_steps = config_training.get('save_every_n_steps', None)
        
        self.train_loss: list = []
        self.val_loss: list = []

    def one_epoch(self, epoch: int):
        self.model.train()
        losses = []
        for mesh in tqdm(self.training_loader, desc=f'Train epoch {epoch}'):
            with torch.no_grad():
                indices = self.vqvae.get_codebook_indices(
                    mesh.to(self.device)
                )
            self.optimizer.zero_grad()
            out  = self.model(indices, return_loss=True, return_logits=False) # [16, 54]
            loss = out['loss']

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())
            self.train_loss.append(loss.item())
            self.step_count += 1

            if self.save_every_n_steps and self.step_count % self.save_every_n_steps == 0:
                parameters = dict(
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    scheduler=self.lr_scheduler.state_dict(),
                    epoch=epoch,
                    step=self.step_count,
                    loss=loss.item(),
                )
                checkpoint_path = str(self.follow.path / f'checkpoint_step_{self.step_count}.pth')
                torch.save(parameters, checkpoint_path)
                print(f'\t [Step {self.step_count}] checkpoint saved (loss={loss.item():.4f})')

            # ---- 每 50 步可视化一次 pred mesh vs real mesh ----
            if self.step_count % 1000 == 0 and self.f is not None:
                with torch.no_grad():
                    self.model.eval()
                    sample_out = self.model.sample(
                        batch_size=indices.shape[0],
                        filter_ratio=0.2, # 掩码比例，但是0表示从完全掩码
                        temperature=1.0,
                        content_token=indices,
                        device=self.device,
                    )
                    self.model.train()

                    pred_tokens = sample_out['content_token']
                    pred_tokens_clipped = torch.clamp(pred_tokens, 0, 511)
                    pred_mesh = self.vqvae.decode(pred_tokens_clipped).cpu()  # [B, V, 3]
                    real_mesh = self.vqvae.decode(indices).cpu()               # [B, V, 3]

                save_prefix = (f'{self.follow.path_samples_train}/'
                               f'epoch{epoch}_step{self.step_count}_cmp')
                self.plot_compare_(pred_mesh, real_mesh, save_prefix)

        return losses

    def eval(self, epoch: int, vis_every: int = 50):
        self.model.eval()
        losses = []
        val_step = 0
        with torch.no_grad():
            for data in tqdm(self.validation_loader, desc=f'Val   epoch {epoch}'):
                mesh = data['local_mesh'] if isinstance(data, dict) else data
                mesh = mesh.to(self.device)
                tokens = self.vqvae.get_codebook_indices(
                    mesh,
                )
                out  = self.model(tokens, return_loss=True, return_logits=False)
                loss = out['loss']
                losses.append(loss.item())
                self.val_loss.append(loss.item())
                val_step += 1

                # ---- 每 vis_every 步可视化一次 pred mesh vs real mesh ----
                if val_step % vis_every == 0 and self.f is not None:
                    sample_out = self.model.sample(
                        batch_size=tokens.shape[0],
                        filter_ratio=0.2, # torch.empty(1).uniform_(0.3, 0.8).item(),
                        temperature=1.0,
                        content_token=tokens,
                        device=self.device,
                    )
                    pred_tokens = sample_out['content_token']
                    pred_tokens_clipped = torch.clamp(pred_tokens, 0, 511)
                    pred_mesh = self.vqvae.decode(pred_tokens_clipped).cpu()  # [B, V, 3]
                    real_mesh = self.vqvae.decode(tokens).cpu()               # [B, V, 3]

                    save_prefix = (f'{self.follow.path_samples}/'
                                   f'val_epoch{epoch}_step{val_step}_cmp')
                    self.plot_compare_(pred_mesh, real_mesh, save_prefix)

        return losses

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

            self.follow.follow(
                epoch=e,
                loss_train=mean(self.train_loss[-len(self.training_loader):]),
                loss_validation=mean(self.val_loss[-len(self.validation_loader):]),
                parameters=parameters,
            )

            self.follow.save(parameters, epoch=e, every_step=10)
            self.follow.train_losses.append(avg_train)
            self.follow.val_losses.append(avg_val)
            self.follow.plot()

    def eval_deterministic(self, visualize: bool = True):
        """用 diffusion 采样评估验证集。"""
        self.model.eval()
        with torch.no_grad():
            count = 0
            for data in tqdm(iter(self.validation_loader), desc='eval_deterministic'):
                sample_out  = self.model.sample(
                    batch_size=data['img'].shape[0],
                    filter_ratio=0.0,
                    temperature=1.0,
                    device=self.device,
                )
                pred_tokens = sample_out['content_token']
                pred_tokens_clipped = torch.clamp(pred_tokens, 0, 511)
                mesh_canonical = self.vqvae.decode(pred_tokens_clipped).cpu()

                if visualize and self.f is not None:
                    count += 1
                    save_prefix = (f'{self.follow.path_samples_train}/'
                               f'eval_det_step{self.step_count}_cmp_{count}')
                    self.plot_compare_(mesh_canonical, save_prefix)

        return count

    def eval_stochastic(
        self,
        sample_size: int = 25,
        temperature: float = 1.0,
        visualize: bool = True,
    ):
        """随机多次采样评估验证集。"""
        self.model.eval()
        with torch.no_grad():
            count = 0
            for batch_idx, data in enumerate(tqdm(iter(self.validation_loader), 
                                                   desc='eval_stochastic')):
                B = data['img'].shape[0]
                for s in range(sample_size):
                    sample_out = self.model.sample(
                        batch_size=B,
                        filter_ratio=0.0,
                        temperature=temperature,
                        device=self.device,
                    )
                    pred_tokens = sample_out['content_token']
                    pred_tokens_clipped = torch.clamp(pred_tokens, 0, 511)
                    mesh_canonical = self.vqvae.decode(pred_tokens_clipped).cpu()

                    if visualize and self.f is not None:
                        count += 1
                        save_path = (f'{self.follow.path_samples}/'
                                    f'stoch_batch{batch_idx}_sample{s}.png')
                        self.plot_meshes_(mesh_canonical, show=False, save=save_path)

        return count

    def load(self, path: str = '', optimizer: bool = True):
        print('LOAD [', end='')
        checkpoint = torch.load(path, map_location='cpu')
        
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model'], strict=False)
        
        if optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.load_epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 'N/A')
        
        print(f'model: ok  | optimizer:{optimizer}  '
              f'|  loss: {loss}  |  epoch: {self.load_epoch}]')
        
        if missing_keys:
            print(f'  ⚠ Missing keys: {missing_keys}')
        if unexpected_keys:
            print(f'  ⚠ Unexpected keys: {unexpected_keys}')

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

    def plot_meshes_(self, meshes, show: bool = False, save: str = None, rot: bool = False):
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
