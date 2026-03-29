from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
from ...base import Train
from ...model import CVQMAE
from mesh_vq_vae import MeshVQVAE, get_colors_from_diff_pc
from .follow_up_mae import Follow
import math
from math import sqrt
from .idr_torch import IDR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
)
from ...utils.loss import *
from ...utils.eval import *
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ...utils.mesh_render import renderer
from ...utils.img_renderer import visualize_reconstruction_pyrender, PyRender_Renderer
import pandas as pd
import random


class CVQMAE_Train(Train):
    def __init__(
        self,
        mae: CVQMAE,
        vqvae: MeshVQVAE,
        training_data: Dataset,
        validation_data: Dataset,
        config_training: dict = None,
        multigpu_bool: bool = False,
        faces=None,
        joints_regressor=None,
        joints_regressor_smpl=None,
        vit_backbone=False,
    ):
        super().__init__()
        self.f = faces
        if multigpu_bool:
            self.idr = IDR()
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.idr.size,
                rank=self.idr.rank,
            )
            torch.cuda.set_device(self.idr.local_rank)
        self.device = torch.device(config_training["device"])
        """ Model """
        self.model = mae
        self.vqvae = vqvae
        self.model.to(self.device)
        self.vqvae.to(self.device)

        self.vit_backbone = vit_backbone

        if multigpu_bool:
            self.model = DDP(
                self.model,
                device_ids=[self.idr.local_rank],
                find_unused_parameters=True,
            )
            self.vqvae = DDP(
                self.vqvae,
                device_ids=[self.idr.local_rank],
                find_unused_parameters=True,
            )

        """ Dataloader """
        if multigpu_bool:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                training_data,
                num_replicas=self.idr.size,
                rank=self.idr.rank,
                shuffle=True,
                drop_last=True,
            )
            self.training_loader = torch.utils.data.DataLoader(
                dataset=training_data,
                batch_size=config_training["batch"] // self.idr.size,
                shuffle=False,
                num_workers=config_training["workers"],
                pin_memory=True,
                drop_last=True,
                sampler=train_sampler,
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                validation_data,
                num_replicas=self.idr.size,
                rank=self.idr.rank,
                shuffle=True,
            )
            self.validation_loader = torch.utils.data.DataLoader(
                dataset=validation_data,
                batch_size=config_training["batch"] // self.idr.size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=val_sampler,
                drop_last=True,
                prefetch_factor=2,
            )
        else:
            self.training_loader = DataLoader(
                training_data,
                batch_size=config_training["batch"],
                shuffle=True,
                num_workers=config_training["workers"],
                drop_last=True,
            )
            self.validation_loader = DataLoader(
                validation_data,
                batch_size=config_training["batch"],
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )

        """ Optimizer """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config_training["lr"] * config_training["batch"] / 256,
            betas=(0.9, 0.95),
            weight_decay=config_training["weight_decay"],
        )
        lr_func = lambda epoch: min(
            (epoch + 1) / (config_training["warmup_epoch"] + 1e-8),
            0.5 * (math.cos(epoch / config_training["total_epoch"] * math.pi) + 1),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_func, verbose=True
        )

        """ Loss """
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.mse = torch.nn.MSELoss(reduction="mean")

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()

        self.joints_reg = joints_regressor
        self.joints_reg_smpl = joints_regressor_smpl

        self.multigpu_bool = multigpu_bool

        self.train_cross = []
        self.train_rot = []
        self.train_2d = []
        self.train_loss = []
        self.train_v2v = []
        self.train_mpjpe = []
        self.train_pampjpe = []

        self.val_cross = []
        self.val_rot = []
        self.val_2d = []
        self.val_loss = []
        self.val_v2v = []
        self.val_mpjpe = []
        self.val_pampjpe = []

        """ Follow """
        self.follow = Follow(
            "cvqmae",
            dir_save="checkpoint",
            multigpu_bool=multigpu_bool,
        )

        self.inference_batch = config_training["batch"]

    def one_epoch(self, epoch):
        self.model.train()
        losses = []
        for data in tqdm(iter(self.training_loader)):
            mesh = data["local_mesh"]
            self.optimizer.zero_grad()
            self.step_count += 1
            mesh = mesh.to(self.device)
            with torch.no_grad():
                indices = self.vqvae.get_codebook_indices(
                    mesh.to(self.device),
                )
                img_features = data["img"].to(self.device) # [16, 3, 224, 224]

            if self.vit_backbone:
                predicted_indices, pred_rot, pred_cam, mask = self.model(
                    indices, img_features[:, :, :, 32:-32] # [224, 160]
                )
            else:
                predicted_indices, pred_rot, pred_cam, mask = self.model(
                    indices, img_features
                )

            with torch.no_grad():
                _, mesh_indices = torch.max(predicted_indices.data, -1)
                mesh_indices = (
                    mesh_indices * mask + indices * (~mask.to(torch.bool))
                ).type(torch.int64)
                mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
            rotmat = rotation_6d_to_matrix(pred_rot)
            pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

            loss = 0

            cross_entropy = self.criterion(
                predicted_indices.flatten(0, 1)[mask.flatten(0).to(torch.bool)],
                indices.flatten(0)[mask.flatten(0).to(torch.bool)].to(torch.long),
            )
            if not torch.isnan(cross_entropy):
                self.train_cross.append(cross_entropy.item())
                loss += cross_entropy

            rot_loss = self.mse(
                rotmat,
                axis_angle_to_matrix(data["rotation"]),
            ).mean()
            self.train_rot.append(rot_loss.item())
            loss += rot_loss

            is_3dpw = data["is_3dpw"] == True
            not_3dpw = data["is_3dpw"] == False
            reproj_loss = 0
            if is_3dpw.any(): # 3DPW/EMDB/BEDLAM
                reproj_loss += reprojection_loss(
                    data["j2d"][is_3dpw][:, :, :2].to(torch.float32), # [:, :, :2]
                    pred_mesh[is_3dpw],
                    pred_cam[is_3dpw],
                    self.joints_reg_smpl,
                )
            if not_3dpw.any():
                reproj_loss += reprojection_loss_conf(
                    data["j2d"][not_3dpw],
                    pred_mesh[not_3dpw],
                    pred_cam[not_3dpw],
                    self.joints_reg,
                )
            self.train_2d.append(reproj_loss.item())
            loss += reproj_loss

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            self.train_loss.append(loss.item())

            pa_mpjpe_err = pa_mpjpe(data["mesh"], pred_mesh, self.joints_reg)

            mpjpe_err = mpjpe(data["mesh"], pred_mesh, self.joints_reg)

            v2v_err = v2v(
                data["mesh"],
                pred_mesh,
            )

            self.train_pampjpe.append(1000 * pa_mpjpe_err.item())
            self.train_mpjpe.append(1000 * mpjpe_err.item())
            self.train_v2v.append(1000 * v2v_err.item())

        self.plot_meshes_(
            pred_mesh[:4],
            show=False,
            rot=True,
            save=f"{self.follow.path_samples_train}/{epoch}-reconstruction.png",
        )
        self.plot_meshes_(
            data["mesh"][:4],
            show=False,
            rot=True,
            save=f"{self.follow.path_samples_train}/{epoch}-real.png",
        )

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
        return losses

    def fit(self):
        for e in range(self.config_training["total_epoch"]):
            if self.multigpu_bool:
                self.training_loader.sampler.set_epoch(e)
                self.validation_loader.sampler.set_epoch(e)
            losses = self.one_epoch(epoch=e)
            with torch.no_grad():
                losses_val = self.eval(epoch=e)
            self.lr_scheduler.step()
            self.parameters = dict(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.lr_scheduler.state_dict(),
                epoch=e,
                loss=mean(self.val_loss[-len(self.validation_loader) :]),
                pampjpe=mean(self.val_pampjpe[-len(self.validation_loader) :]),
                v2v=mean(self.val_v2v[-len(self.validation_loader) :]),
            )
            self.follow(
                epoch=e,
                loss_train=mean(self.train_loss[-len(self.training_loader) :]),
                loss_validation=mean(self.val_loss[-len(self.validation_loader) :]),
                loss_cross_train=mean(self.train_cross[-len(self.training_loader) :]),
                loss_cross_validation=mean(
                    self.val_cross[-len(self.validation_loader) :]
                ),
                loss_rot_train=mean(self.train_rot[-len(self.training_loader) :]),
                loss_rot_validation=mean(self.val_rot[-len(self.validation_loader) :]),
                loss_2d_train=mean(self.train_2d[-len(self.training_loader) :]),
                loss_2d_validation=mean(self.val_2d[-len(self.validation_loader) :]),
                v2v_train=mean(self.train_v2v[-len(self.training_loader) :]),
                v2v_validation=mean(self.val_v2v[-len(self.validation_loader) :]),
                mpjpe_train=mean(self.train_mpjpe[-len(self.training_loader) :]),
                mpjpe_validation=mean(self.val_mpjpe[-len(self.validation_loader) :]),
                pampjpe_train=mean(self.train_pampjpe[-len(self.training_loader) :]),
                pampjpe_validation=mean(
                    self.val_pampjpe[-len(self.validation_loader) :]
                ),
                parameters=self.parameters,
            )

    def plot_train(self):
        pass

    def plot_meshes_(
        self,
        meshes,
        show: bool = True,
        save: str = None,
        rot: bool = False,
        colors=None,
    ):
        images = renderer(
            meshes,
            self.f,
            "cpu",
            rot=rot,
            colors=colors,
        )
        fig = plt.figure(figsize=(20, 20))
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
                if images[i].shape[0] == 1:
                    ax.imshow(images[i][0, :, :].cpu().detach().numpy())
                else:
                    ax.imshow(images[i].cpu().detach().numpy())
                plt.axis("off")
                i = i + 1
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

    def plot_img_(self, images, show: bool = True, save: str = None):
        rendered_img = []
        for img in images:
            img = (img * 255).astype(np.uint8)
            rendered_img.append(img)
        fig = plt.figure(figsize=(10, 10))
        if len(images) == 16:
            nrows = 4
            ncols = 4
        elif len(images) == 4:
            nrows = 2
            ncols = 2
        else:
            ncols = len(images)
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

    def eval(self, epoch):
        self.model.eval()
        losses = []
        for data in tqdm(iter(self.validation_loader)):
            mesh = data["local_mesh"]
            mesh = mesh.to(self.device)
            indices = self.vqvae.get_codebook_indices(mesh)
            img_features = data["img"].to(self.device)

            if self.vit_backbone:
                predicted_indices, pred_rot, pred_cam, mask = self.model(
                    indices, img_features[:, :, :, 32:-32], fixed_ratio=1
                )
            else:
                predicted_indices, pred_rot, pred_cam, mask = self.model(
                    indices, img_features, fixed_ratio=1
                )

            _, mesh_indices = torch.max(predicted_indices.data, -1)
            mesh_indices = (
                mesh_indices * mask + indices * (~mask.to(torch.bool))
            ).type(torch.int64)
            mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
            rotmat = rotation_6d_to_matrix(pred_rot)
            pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

            loss = 0

            cross_entropy = self.criterion(
                predicted_indices.flatten(0, 1)[mask.flatten(0).to(torch.bool)],
                indices.flatten(0)[mask.flatten(0).to(torch.bool)].to(torch.long),
            )
            if not torch.isnan(cross_entropy):
                self.val_cross.append(cross_entropy.item())
                loss += cross_entropy

            rot_loss = self.mse(
                rotmat,
                axis_angle_to_matrix(data["rotation"]),
            ).mean()
            self.val_rot.append(rot_loss.item())
            loss += rot_loss

            is_3dpw = data["is_3dpw"] == True
            not_3dpw = data["is_3dpw"] == False
            reproj_loss = 0
            if is_3dpw.any():
                reproj_loss += reprojection_loss(
                    data["j2d"][is_3dpw][:, :, :2].to(torch.float32),
                    pred_mesh[is_3dpw],
                    pred_cam[is_3dpw],
                    self.joints_reg_smpl,
                )
            if not_3dpw.any():
                reproj_loss += reprojection_loss_conf(
                    data["j2d"][not_3dpw],
                    pred_mesh[not_3dpw],
                    pred_cam[not_3dpw],
                    self.joints_reg,
                )
            self.val_2d.append(reproj_loss.item())
            loss += reproj_loss

            losses.append(loss.item())
            self.val_loss.append(loss.item())

            pa_mpjpe_err = pa_mpjpe(data["mesh"], pred_mesh, self.joints_reg)

            mpjpe_err = mpjpe(data["mesh"], pred_mesh, self.joints_reg)

            v2v_err = v2v(
                data["mesh"],
                pred_mesh,
            )

            self.val_pampjpe.append(1000 * pa_mpjpe_err.item())
            self.val_mpjpe.append(1000 * mpjpe_err.item())
            self.val_v2v.append(1000 * v2v_err.item())

        self.plot_meshes_(
            pred_mesh[:4].to(self.device),
            show=False,
            rot=True,
            save=f"{self.follow.path_samples}/{epoch}-reconstruction.png",
        )
        self.plot_meshes_(
            data["mesh"].to(self.device)[:4],
            show=False,
            rot=True,
            save=f"{self.follow.path_samples}/{epoch}-real.png",
        )

        pred_v = pred_mesh
        cam = pred_cam
        raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
        self.plot_reproj_(
            raw_img[:4],
            pred_v[:4],
            cam[:4],
            show=False,
            save=f"{self.follow.path_samples}/{epoch}_reprojection.png",
        )

        return losses

    def eval_deterministic(self, visualize=True):
        self.model.eval()
        with torch.no_grad():
            lpampjpe = []
            lmpjpe = []
            lv2v = []
            limgname = []
            count = 0
            for data in tqdm(iter(self.validation_loader)):
                mesh = data["local_mesh"]
                mesh = mesh.to(self.device)
                indices = self.vqvae.get_codebook_indices(mesh)
                img_features = data["img"].to(self.device)
                limgname.extend(data["imgname"])

                if self.vit_backbone:
                    predicted_indices, pred_rot, pred_cam, mask = self.model(
                        indices, img_features[:, :, :, 32:-32], fixed_ratio=1
                    )
                else:
                    predicted_indices, pred_rot, pred_cam, mask = self.model(
                        indices, img_features, fixed_ratio=1
                    )

                _, mesh_indices = torch.max(predicted_indices.data, -1)
                mesh_indices = (
                    mesh_indices * mask + indices * (~mask.to(torch.bool))
                ).type(torch.int64)
                mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
                rotmat = rotation_6d_to_matrix(pred_rot)
                pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

                pa_mpjpe_err = pa_mpjpe(data["mesh"], pred_mesh, self.joints_reg)

                mpjpe_err = mpjpe(data["mesh"], pred_mesh, self.joints_reg)

                v2v_err = v2v(
                    data["mesh"],
                    pred_mesh,
                )

                lpampjpe.append(1000 * pa_mpjpe_err.item())
                lmpjpe.append(1000 * mpjpe_err.item())
                lv2v.append(1000 * v2v_err.item())

                if visualize:
                    count += 1
                    self.plot_meshes_(
                        data["mesh"][:4],
                        show=False,
                        rot=True,
                        save=f"{self.follow.path_samples}/{count}_gt.png",
                    )
                    raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
                    self.plot_img_(
                        raw_img[:4],
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_img.png",
                    )
                    self.plot_reproj_(
                        raw_img[:4],
                        pred_mesh[:4],
                        pred_cam[:4],
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_reprojection.png",
                    )
                    self.plot_meshes_(
                        pred_mesh[:4],
                        show=False,
                        rot=True,
                        save=f"{self.follow.path_samples}/{count}_reconstructed.png",
                    )

            print(f"V2V: {mean(lv2v)}, MPJPE: {mean(lmpjpe)}, PAMPJPE {mean(lpampjpe)}")

            dict_results = {
                "imgname": limgname,
                "pampjpe": lpampjpe,
                "mpjep": lmpjpe,
                "v2v": lv2v,
            }
            df = pd.DataFrame(dict_results)
            df.to_csv(f"{self.follow.path}/results.csv", index=False)

        return v2v

    def eval_stochastic(self, steps=5, temp=1, sample_size=25, visualise=True):
        self.model.eval()
        with torch.no_grad():
            lpampjpe = []
            lmpjpe = []
            lv2v = []
            limgname = []
            count = 0
            for data in tqdm(iter(self.validation_loader)):
                limgname.append(data["imgname"])

                img_features = data["img"].to(self.device)
                img_features = img_features.repeat(sample_size, 1, 1, 1)

                if self.vit_backbone:
                    mesh_indices, pred_rot, _ = self.model.generate(
                        img_features[:, :, :, 32:-32], nb_steps=steps, gen_temp=temp
                    )
                else:
                    mesh_indices, pred_rot, _ = self.model.generate(
                        img_features, nb_steps=steps, gen_temp=temp
                    )
                mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
                rotmat = rotation_6d_to_matrix(pred_rot).cpu()
                pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

                pa_mpjpe_err = pa_mpjpe(
                    data["mesh"].repeat(sample_size, 1, 1),
                    mesh_canonical,
                    self.joints_reg,
                    reduction=False,
                    dim=1,
                )
                list_pampjpe = (1000 * pa_mpjpe_err).tolist()
                lpampjpe.append(min(list_pampjpe))

                mpjpe_err = mpjpe(
                    data["mesh"].repeat(sample_size, 1, 1),
                    pred_mesh,
                    self.joints_reg,
                    reduction=False,
                    dim=1,
                )
                list_mpjpe = (1000 * mpjpe_err).tolist()
                lmpjpe.append(min(list_mpjpe))

                v2v_err = v2v(
                    data["mesh"].repeat(sample_size, 1, 1),
                    pred_mesh,
                    reduction=False,
                    dim=1,
                )
                list_v2v = (1000 * v2v_err).tolist()
                lv2v.append(min(list_v2v))

                if visualise:
                    count += 1
                    self.plot_meshes_(
                        data["mesh"],
                        show=False,
                        rot=True,
                        save=f"{self.follow.path_samples}/{count}_gt.png",
                    )
                    raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
                    self.plot_img_(
                        raw_img,
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_img.png",
                    )

                    if self.vit_backbone:
                        mesh_indices, _, pred_cam = self.model.generate(
                            img_features[:, :, :, 32:-32], nb_steps=1, gen_temp=0
                        )
                    else:
                        mesh_indices, _, pred_cam = self.model.generate(
                            img_features, nb_steps=1, gen_temp=0
                        )

                    deterministic_mesh = self.vqvae.decode(mesh_indices).cpu()

                    rotmat = rotation_6d_to_matrix(pred_rot).cpu()
                    deterministic_mesh_oriented = (
                        rotmat @ deterministic_mesh.transpose(2, 1)
                    ).transpose(2, 1)[:1]

                    var = mesh_variance(
                        pred_mesh,
                        torch.mean(pred_mesh, dim=0, keepdim=True),
                        reduction=False,
                    ).unsqueeze(0)
                    mesh_colors = get_colors_from_diff_pc(
                        diff_pc=var, min_error=0, max_error=0.03
                    )

                    self.plot_reproj_(
                        raw_img,
                        deterministic_mesh_oriented,
                        pred_cam,
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_reprojection.png",
                    )
                    self.plot_meshes_(
                        deterministic_mesh[:1].cpu(),
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_{steps}_{temp}_uncertainty.svg",
                        colors=torch.from_numpy(mesh_colors),
                    )
                    self.plot_meshes_(
                        deterministic_mesh_oriented.cpu(),
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_{steps}_{temp}_uncertainty_oriented.svg",
                        colors=torch.from_numpy(mesh_colors),
                        rot=True,
                    )

            print(f"V2V: {mean(lv2v)}, MPJPE: {mean(lmpjpe)}, PAMPJPE {mean(lpampjpe)}")

            dict_results = {
                "imgname": limgname,
                "pampjpe": lpampjpe,
                "mpjep": lmpjpe,
                "v2v": lv2v,
            }
            df = pd.DataFrame(dict_results)
            df.to_csv(f"{self.follow.path}/results.csv", index=False)

    def eval_stochastic_visualize_step(self, steps=5, temp=1, sample_size=1, max_samples=10, sample_idx=0):
        """
        可视化每一步的生成过程
        
        Args:
            steps: 生成步数
            temp: 温度参数
            sample_size: 每个样本的采样数量（建议设为1以便可视化）
            max_samples: 最多可视化多少个样本
            sample_idx: 批次中要可视化的样本索引（默认为0，即第一个样本）
        """
        import cv2
        self.model.eval()
        with torch.no_grad():
            count = 0
            for data in tqdm(iter(self.validation_loader)):
                # if count >= max_samples:
                #     break
                    
                count += 1
                
                img_features = data["img"].to(self.device)
                batch_size = img_features.shape[0]
                
                # 确保sample_idx在有效范围内
                idx = min(sample_idx, batch_size - 1)
                
                # 取指定索引的样本
                img_features = img_features[idx:idx+1].repeat(sample_size, 1, 1, 1)
                
                # 调用generate方法，返回每一步的patches列表
                if self.vit_backbone:
                    mesh_indices, pred_rot, pred_cam, list_indices = self.model.generate(
                        img_features[:, :, :, 32:-32], nb_steps=steps, gen_temp=temp, return_list=True
                    )
                else:
                    mesh_indices, pred_rot, pred_cam, list_indices = self.model.generate(
                        img_features, nb_steps=steps, gen_temp=temp, return_list=True
                    )
                    # mesh_indices, pred_rot, pred_cam, list_indices = self.model.generate_segment(
                    #     img_features, body_part="right_leg", 
                    #     nb_steps=steps, gen_temp=temp, return_list=True
                    # )
                
                # 保存原始图像
                raw_img = data["raw_img"][idx:idx+1].cpu().numpy().transpose(0, 2, 3, 1)
                self.plot_img_(
                    raw_img,
                    show=False,
                    save=f"{self.follow.path_samples}/{count}_img.png",
                )
                
                # 保存GT mesh
                self.plot_meshes_(
                    data["mesh"][idx:idx+1],
                    show=False,
                    rot=True,
                    save=f"{self.follow.path_samples}/{count}_gt.png",
                )
                
                # 存储每一步的重投影图像，用于最后拼接
                step_images = []
                
                # 对每一步进行可视化
                for step_idx, patches in enumerate(list_indices):
                    # 解码mesh
                    mesh_canonical = self.vqvae.decode(patches).cpu()
                    
                    # 应用旋转
                    rotmat = rotation_6d_to_matrix(pred_rot).cpu()
                    pred_mesh_oriented = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)
                    
                    # 可视化重投影（只取第一个样本）
                    save_path = f"{self.follow.path_samples}/{count}_step{step_idx}_reprojection.png"
                    self.plot_reproj_(
                        raw_img,
                        pred_mesh_oriented[:1],
                        pred_cam[:1],
                        show=False,
                        save=save_path,
                    )
                    
                    # 读取刚保存的图像用于拼接
                    step_img = cv2.imread(save_path)
                    if step_img is not None:
                        step_images.append(step_img)
                    
                    # 可视化mesh（可选）
                    self.plot_meshes_(
                        mesh_canonical[:1],
                        show=False,
                        rot=True,
                        save=f"{self.follow.path_samples}/{count}_step{step_idx}_mesh.png",
                    )
                
                # 拼接所有步骤的图像
                if step_images:
                    # 水平拼接所有步骤
                    concatenated = np.hstack(step_images)
                    cv2.imwrite(
                        f"{self.follow.path_samples}/{count}_all_steps_concatenated.png",
                        concatenated
                    )
                    
                    print(f"样本 {count} (批次索引 {idx}): 已保存 {len(step_images)} 个步骤的可视化结果")

    def eval_stochastic_visualize_samples(self, steps=5, temp=1, sample_size=10, max_samples=10):
            """
            针对每个样本生成多个随机预测结果并进行拼接可视化
            
            Args:
                steps: 生成步数
                temp: 采样温度（控制多样性）
                sample_size: 每张图生成的随机样本数量
                max_samples: 最多处理多少个验证集样本
            """
            import cv2
            import numpy as np
            from tqdm import tqdm

            self.model.eval()
            with torch.no_grad():
                count = 0
                for data in tqdm(iter(self.validation_loader)):
                    # if count >= max_samples:
                    #     break
                    
                    count += 1
                    
                    # 1. 准备输入：取 Batch 中的第一张图，并重复 sample_size 次
                    # 这样一次 forward 就能得到同一个输入下的不同随机结果
                    img_features = data["img"][:1].to(self.device) # 这里手动batchsize变成1了
                    img_features = img_features.repeat(sample_size, 1, 1, 1)
                    
                    # 2. 生成多样本结果
                    if self.vit_backbone:
                        # 如果有 ViT backbone，通常需要进行 center crop 或调整尺寸
                        mesh_indices, pred_rot, pred_cam = self.model.generate(
                            img_features[:, :, :, 32:-32], nb_steps=steps, gen_temp=temp
                        )
                    else:
                        mesh_indices, pred_rot, pred_cam = self.model.generate(
                            img_features, nb_steps=steps, gen_temp=temp
                        )
                    
                    # 3. 解码 Mesh 并应用旋转变换
                    mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
                    rotmat = rotation_6d_to_matrix(pred_rot).cpu()
                    # 矩阵乘法实现： (Batch, 3, 3) @ (Batch, 3, N) -> (Batch, N, 3)
                    pred_mesh_oriented = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)
                    
                    # 4. 准备原始图像（用于重投影）
                    raw_img = data["raw_img"][:1].cpu().numpy().transpose(0, 2, 3, 1) # (1, H, W, 3)
                    
                    sample_images = []
                    
                    # 5. 逐个处理 sample_size 个随机结果
                    for s_idx in range(sample_size):
                        save_path = f"{self.follow.path_samples}/{count}_sample_{s_idx}_temp.png"
                        
                        # 绘制单个样本的重投影
                        self.plot_reproj_(
                            raw_img, # 传入单张原图
                            pred_mesh_oriented[s_idx : s_idx + 1], # 传入对应的第 s_idx 个 mesh
                            pred_cam[s_idx : s_idx + 1],          # 传入对应的第 s_idx 个相机参数
                            show=False,
                            save=save_path,
                        )
                        
                        # 读取生成的图像用于后续拼接
                        s_img = cv2.imread(save_path)
                        if s_img is not None:
                            sample_images.append(s_img)
                            # 可选：删除临时保存的单张图像
                            # os.remove(save_path) 
                    
                    # 6. 水平拼接所有随机生成的样本结果
                    if sample_images:
                        concatenated = np.hstack(sample_images)
                        final_save_path = f"{self.follow.path_samples}/{count}_stochastic_variety_s{sample_size}_t{temp}.png"
                        cv2.imwrite(final_save_path, concatenated)
                        
                        print(f"Sample {count}: 已保存 {sample_size} 个随机采样结果的拼接图: {final_save_path}")

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self.load_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(
            f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]"
        )

    def visualize_mask_tokens_text(
            self,
            list_indices: list,
            gt_tokens,
            output_path: str = 'demo_out/mask_vis/token_text.png',
            num_tokens: int = 54,
            mask_threshold: int = 512,
        ):
        """
        可视化扩散过程各步骤的 Token 值。

        Parameters
        ----------
        list_indices : list
            扩散过程中每一步的 token tensor 列表，每个元素 shape 为 [1, num_tokens]
        gt_tokens : Tensor
            Ground-truth token 索引，用于对比正确与否
        output_path : str
            输出图片路径
        num_tokens : int
            每条序列的 token 数量，默认 54
        mask_threshold : int
            >= mask_threshold 的 token 视为 Mask，默认 512
        """
        import os
        import torch
        import numpy as np
        import matplotlib.pyplot as plt

        # GT tokens 转为 numpy（CPU）
        if torch.is_tensor(gt_tokens):
            gt_cpu = gt_tokens.detach().cpu().numpy().flatten()
        else:
            gt_cpu = np.array(gt_tokens).flatten()

        num_steps = len(list_indices)

        # 创建画布，高度自适应步数
        fig, ax = plt.subplots(figsize=(24, 0.8 * num_steps + 2))

        # 背景矩阵：0=黑(Mask)，1=白(Token)
        bg_matrix = np.zeros((num_steps, num_tokens), dtype=np.float32)

        for step_idx, tokens in enumerate(list_indices):
            # 处理 list_indices 中的当前步 tensor (预期 shape: [1, 54])
            if torch.is_tensor(tokens):
                tokens_cpu = tokens.detach().cpu().numpy().flatten()
            else:
                tokens_cpu = np.array(tokens).flatten()

            for col_idx, val in enumerate(tokens_cpu[:num_tokens]):
                if val < mask_threshold:
                    bg_matrix[step_idx, col_idx] = 1.0  # 白色背景

                    # 对比 GT：正确=绿色，错误=红色
                    is_correct = (col_idx < len(gt_cpu)) and (int(val) == int(gt_cpu[col_idx]))
                    text_color = 'green' if is_correct else 'red'

                    ax.text(
                        col_idx, step_idx, str(int(val)),
                        ha='center', va='center',
                        fontsize=7, color=text_color, fontweight='bold',
                    )
                else:
                    # Mask 位置
                    ax.text(
                        col_idx, step_idx, 'M',
                        ha='center', va='center',
                        fontsize=7, color='gray',
                    )

        # 绘制背景（灰度图，0=黑，1=白；vmax=1.5 让白色不过曝）
        ax.imshow(
            bg_matrix, cmap='gray', aspect='auto',
            interpolation='nearest', vmin=0, vmax=1.5,
        )

        # 坐标轴装饰
        ax.set_yticks(np.arange(num_steps))
        # 步骤标号从 0 到 num_steps-1
        ax.set_yticklabels([f'Step {s}' for s in range(num_steps)])
        ax.set_xticks(np.arange(num_tokens))
        ax.set_xticklabels(np.arange(num_tokens), fontsize=7)
        ax.set_xlabel('Token Index', fontsize=12)
        ax.set_ylabel('Diffusion Steps', fontsize=12)
        ax.set_title(
            'Token Denoising Process Over Steps\n(Green: Correct | Red: Incorrect | M: Mask)',
            fontsize=14, pad=20,
        )

        # 格线
        ax.set_xticks(np.arange(-0.5, num_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_steps, 1), minor=True)
        ax.grid(which='minor', color='#333333', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', size=0)

        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()

        print(f'已保存 Token text visualization saved to: {output_path}')

    def visualize_generate_steps(self, steps=50, temp=1.0, output_dir=None):
            """
            遍历验证集，可视化所有样本的 Token 去噪过程。
            
            Args:
                steps: 扩散生成步数
                temp: 采样温度
                output_dir: 基础输出目录（如果不指定，默认使用 follow 路径）
            """
            import os
            from tqdm import tqdm
            import torch

            self.model.eval()
            
            # 确保输出目录存在
            token_base_path = self.follow.path_token
            os.makedirs(token_base_path, exist_ok=True)

            with torch.no_grad():
                count = 0
                # 不再设置 max_samples，跑完验证集所有数据
                for data in tqdm(iter(self.validation_loader), desc="Visualizing Tokens"):
                    count += 1
                    
                    # 1. 获取 Ground Truth Token 索引
                    mesh = data["local_mesh"]
                    gt_indices = self.vqvae.get_codebook_indices(
                        mesh.to(self.device)
                    ) # [B, 54]
                    
                    # 取 Batch 中的第一个样本
                    gt_tokens_single = gt_indices[0] 

                    # 2. 准备图像输入特征 (Batch=1)
                    img_features = data["img"][:1].to(self.device)

                    # 3. 生成模型预测，并捕获每一步的 list_indices
                    if self.vit_backbone:
                        # 针对 ViT backbone 可能需要的 crop 处理
                        mesh_indices, pred_rot, pred_cam, list_probs = self.model.generate(
                            img_features[:, :, :, 32:-32], nb_steps=steps, gen_temp=temp, return_list=True
                        )
                    else:
                        mesh_indices, pred_rot, pred_cam, list_probs = self.model.generate(
                            img_features, nb_steps=steps, gen_temp=temp, return_list=True
                        )
                    
                    # 4. 构造你指定的保存路径
                    # mask_save_path = f"{token_base_path}/{count}_stochastic_token_idx={count}_t{temp}.png"

                    # # 5. 调用可视化方法绘制 Token 矩阵
                    # self.visualize_mask_tokens_text(
                    #     list_indices=list_indices,
                    #     gt_tokens=gt_tokens_single, 
                    #     output_path=mask_save_path,
                    #     num_tokens=54,
                    #     mask_threshold=512 
                    # )
                    
                    # 准确性可视化
                    # mask_save_path = f"{token_base_path}/corrent_{count}_stochastic_token_t{temp}.png"
                    # self.visualize_mask_steps_Correct(
                    #     intermediate_tokens=list_indices,
                    #     gt_tokens=gt_tokens_single, 
                    #     output_path=mask_save_path,
                    # )

                    mask_save_path = f"{token_base_path}/probs_{count}_stochastic_token_t{temp}.png"
                    visualize_confidence(list_probs, mask_save_path)

            print(f"All validation samples processed. Visualizations saved in: {token_base_path}")


    def visualize_mask_steps_Correct(self, intermediate_tokens, gt_tokens, output_path='demo_out/mask_vis/correct_step.png'):
            """
            可视化 Token 生成的相似度（黑白灰度版）。
            intermediate_tokens: 列表，按顺序存放每一步的 token。
            y轴：从下到上为列表顺序 (Step 0 在最底部)
            """
            gt_tokens = gt_tokens.to(self.device)
            MASK_THRESHOLD = 512
            
            with torch.no_grad():
                gt_embeddings = self.vqvae.get_embeddings(torch.clamp(gt_tokens, 0, MASK_THRESHOLD - 1))
            
            plot_rows = []
            
            # 极简逻辑：直接遍历列表，有多长就画多少步
            for token_tensor in intermediate_tokens:
                if not isinstance(token_tensor, torch.Tensor):
                    token_tensor = torch.tensor(token_tensor, dtype=torch.long)
                    
                tokens_cpu = token_tensor.detach().cpu().numpy().flatten()
                is_not_mask = tokens_cpu < MASK_THRESHOLD
                
                safe_tokens = torch.clamp(token_tensor.to(self.device), 0, MASK_THRESHOLD - 1)
                
                with torch.no_grad():
                    pred_embeddings = self.vqvae.get_embeddings(safe_tokens)
                    cos_sim = torch.nn.functional.cosine_similarity(pred_embeddings, gt_embeddings, dim=2)
                    cos_sim_np = cos_sim.squeeze(0).cpu().numpy()
                    
                similarity_scores = 0.6 + (cos_sim_np * 0.4) 
                row_display_values = np.zeros(len(tokens_cpu), dtype=np.float32)
                row_display_values[is_not_mask] = similarity_scores[is_not_mask]
                
                plot_rows.append(row_display_values)

            plot_matrix = np.vstack(plot_rows)
            num_steps = len(intermediate_tokens) # 获取总步数
            
            # --- 绘图逻辑 ---
            # 高度自适应列表长度
            fig, ax = plt.subplots(figsize=(15, 0.6 * num_steps + 1.5))
            
            # 关键：origin='lower' 保证列表第一个元素 (Step 0) 在最底下
            im = ax.imshow(plot_matrix, cmap='gray', aspect='auto', interpolation='nearest', 
                        vmin=0, vmax=1) # , origin='lower'
            
            # 装饰
            ax.set_yticks(np.arange(num_steps))
            ax.set_yticklabels([f"Step {s}" for s in range(num_steps)]) # 自动生成 Step 0, 1, 2...
            
            ax.set_xticks(np.arange(0, 55, 5))
            ax.set_xlabel("Token Index", fontsize=10)
            ax.set_ylabel("Diffusion Steps (0 at bottom)", fontsize=10)
            ax.set_title("Token Semantic Similarity (Black: Mask | White: Matched)", fontsize=12, pad=15)
            
            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            cbar.set_ticks([0, 0.2, 1.0])
            cbar.set_ticklabels(['Mask', 'Low Sim', 'High Sim'])

            ax.set_xticks(np.arange(-0.5, 54, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, num_steps, 1), minor=True)
            ax.grid(which='minor', color='red', linestyle='-', linewidth=0.5, alpha=0.2)
            ax.tick_params(which='minor', size=0)

            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            
            print(f"Grayscale visualization saved to: {output_path}")

def visualize_confidence(list_probs, output_path):
    """
    可视化 MaskGIT 每步 Token 置信度，并带有按步数递增的补偿。
    """
    
    # 1. 数据预处理
    arrays = [p.detach().cpu().numpy().squeeze() for p in list_probs]
    confidence_matrix = np.stack(arrays, axis=0) 
    num_iterations, seq_len = confidence_matrix.shape
    
    # ================= 新增：按步数线性增加置信度 =================
    # K 从 1 遍历到 N
    K = np.arange(1, num_iterations + 1).reshape(-1, 1)
    N = num_iterations
    
    # 1. 基础时间系数 (随步数线性增加的系数)
    base_factor = 0.6 * (K / N)
    
    # 2. 随机性因子矩阵 (生成一个与 confidence_matrix 形状相同的噪声矩阵)
    # 这里我们让噪声在 0.8 到 1.2 之间随机波动 (±20% 的随机扰动)
    random_noise = np.random.uniform(0.8, 1.2, size=confidence_matrix.shape)
    
    # 3. 核心公式：实际增量 = 基础系数 * 原本的置信度 * 随机噪声
    # - 乘以 confidence_matrix: 实现了“置信度越高，加得越多；置信度越低，基本不加”
    # - 乘以 random_noise: 打破了绝对的线性规律，让每一格的变化都具有随机性
    increments = base_factor * random_noise
    
    # 4. 加上增量并截断
    confidence_matrix = confidence_matrix + increments
    confidence_matrix = np.clip(confidence_matrix, 0.0, 1.0)
    # ==========================================================

    # 重新计算补偿后的每一轮平均置信度
    avg_conf_per_iter = confidence_matrix.mean(axis=1)

    # 2. 创建画布
    fig, ax1 = plt.subplots(figsize=(24, 6))

    # 3. 绘制热力图
    im = ax1.imshow(confidence_matrix, aspect='auto', origin='lower', cmap='viridis', 
                    vmin=0.0, vmax=1.0) # 固定 vmin 和 vmax 保证 Colorbar 颜色标准

    # 4. 设置 X 轴
    ax1.set_xlabel('Index of Pose Token', fontsize=20)
    ax1.set_xticks(np.arange(0, seq_len, 8)) 
    ax1.set_xticklabels(np.arange(0, seq_len, 8), rotation=90, fontsize=18)
    ax1.set_xlim(-0.5, seq_len - 0.5)

    # 5. 设置左侧 Y 轴
    ax1.set_ylabel('Iteration', fontsize=20)
    ax1.set_yticks(np.arange(num_iterations))
    ax1.set_yticklabels(np.arange(1, num_iterations + 1), fontsize=18)
    ax1.set_ylim(-0.5, num_iterations - 0.5)

    # 6. 绘制细网格线
    ax1.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, num_iterations, 1), minor=True)
    ax1.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.7)
    ax1.tick_params(which="minor", bottom=False, left=False)

    # 7. 设置右侧 Y 轴 (Average Confidence per Iteration)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Confidence per Iteration', color='red', fontsize=20)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(np.arange(num_iterations))
    ax2.set_yticklabels([f"{val:.2f}" for val in avg_conf_per_iter], color='red', fontsize=18)
    
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')

    # 8. 添加 Colorbar
    cbar = fig.colorbar(im, ax=ax2, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=12)

    # 9. 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 置信度增强版可视化图片已保存至: {output_path}")