"""pretrain_unconditional_diffusion.py

预训练无条件 VQ-Diffusion 模型。

用法:
    python pretrain_unconditional_diffusion.py
    python pretrain_unconditional_diffusion.py --config-name config
"""
import sys
import os

_root = os.path.abspath(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from mega.utils import set_seed
from mega.model.unconditional_diffusion import UnconditionalDiffusion
from mega.train.train_unconditional_diffusion import UnconditionalDiffusion_Train

from mesh_vq_vae import (
    MeshVQVAE,
    FullyConvAE,
    DatasetMeshFromSmpl,
    DatasetMeshTest,
)


@hydra.main(
    config_path="configs/config_unconditional_diffusion",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    set_seed()

    # ------------------------------------------------------------------ #
    #  Data                                                               #
    # ------------------------------------------------------------------ #
    training_data = DatasetMeshFromSmpl(folder=cfg.training_data.file)
    validation_data = DatasetMeshTest(
        dataset_file=cfg.validation_data.file, proportion=0.1
    )
    print(f'Training samples  : {len(training_data)}')
    print(f'Validation samples: {len(validation_data)}')

    # ------------------------------------------------------------------ #
    #  MeshVQVAE  (frozen — only used to obtain codebook indices)         #
    # ------------------------------------------------------------------ #
    convmesh_model = FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model='checkpoint/MESH_VQVAE/mesh_vqvae_54')
    convmesh_model.init_test_mode()

    vqvae_params = sum(p.numel() for p in mesh_vqvae.parameters())
    print(f'MeshVQVAE params  : {vqvae_params:,}  (frozen)')

    # ------------------------------------------------------------------ #
    #  UnconditionalDiffusion model                                       #
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
    #  Trainer                                                            #
    # ------------------------------------------------------------------ #
    config_training = dict(cfg.train)
    config_training['save_every_n_steps'] = cfg.get('save_every_n_steps', None)

    trainer = UnconditionalDiffusion_Train(
        model=model,
        vqvae=mesh_vqvae,
        training_data=training_data,
        validation_data=validation_data,
        config_training=config_training,
        faces=faces,
    )

    # ------------------------------------------------------------------ #
    #  Resume from checkpoint (optional)                                  #
    # ------------------------------------------------------------------ #
    resume_path = cfg.get('resume', {}).get('path', '')
    if resume_path:
        load_optimizer = cfg.get('resume', {}).get('optimizer', False)
        trainer.load(path=resume_path, optimizer=load_optimizer)
        print(f'Resumed from: {resume_path}')

    # ------------------------------------------------------------------ #
    #  Train                                                              #
    # ------------------------------------------------------------------ #
    trainer.fit()


if __name__ == '__main__':
    main()
