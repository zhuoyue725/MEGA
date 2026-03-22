import sys
import os

target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if target_path not in sys.path:
    sys.path.insert(0, target_path)

import torch
import torch.nn as nn
import math
from einops import rearrange
from image_synthesis.modeling.transformers.diffusion_transformer_mesh import DiffusionTransformer


class PositionalEncoding(torch.nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # x: [T, B, C]
        return x + self.pe[: x.size(0)]


class UnconditionalDiffusion(nn.Module):
    """
    Unconditional VQ-Diffusion model.

    forward(img):
        img  : [B, 54]  – content token indices

    Internally:
        1. DiffusionTransformer(content_token=img, condition_embed_token=None)
    """

    # ------------------------------------------------------------------ #
    #  DiffusionTransformer hyper-parameters                              #
    # ------------------------------------------------------------------ #
    _NUM_TOK   = 512   # content codebook size
    _SEQ_LEN   = 54    # content sequence length
    _N_EMB     = 512   # transformer hidden dim
    _N_HEAD    = 8
    _N_LAYER   = 12
    _DIFF_STEP = 100

    def __init__(
        self,
        num_tok: int    = _NUM_TOK,
        seq_len: int    = _SEQ_LEN,
        n_emb: int      = _N_EMB,
        n_head: int     = _N_HEAD,
        n_layer: int    = _N_LAYER,
        diff_step: int  = _DIFF_STEP,
        auxiliary_loss_weight: float = 5e-4,
        adaptive_auxiliary_loss: bool = True,
        mask_weight: list = None,
    ):
        super().__init__()

        if mask_weight is None:
            mask_weight = [1, 1]

        # ---- content embedding config (Sequence1DEmbedding) ----
        content_emb_cfg = {
            'target': 'image_synthesis.modeling.embeddings.sequence_1d_embedding.Sequence1DEmbedding',
            'params': {
                'num_embed': num_tok,
                'seq_len':   seq_len,
                'embed_dim': n_emb,
                'trainable': True,
            },
        }

        # ---- transformer config (no condition) ----
        transformer_cfg = {
            'target': 'image_synthesis.modeling.transformers.transformer_utils.UnCondition2ImageTransformer',
            'params': {
                'attn_type':            'self',
                'n_layer':              n_layer,
                'content_seq_len':      seq_len,
                'content_spatial_size': [seq_len, 1],
                'n_embd':               n_emb,
                'n_head':               n_head,
                'attn_pdrop':           0.0,
                'resid_pdrop':          0.0,
                'block_activate':       'GELU2',
                'timestep_type':        'adalayernorm',
                'mlp_hidden_times':     4,
                'mlp_type':             'fc',
            },
        }
        self.diff_step = diff_step
        self.diffusion = DiffusionTransformer(
            content_emb_config=content_emb_cfg,
            condition_emb_config=None,
            transformer_config=transformer_cfg,
            diffusion_step=diff_step,
            alpha_init_type='alpha1',
            auxiliary_loss_weight=auxiliary_loss_weight,
            adaptive_auxiliary_loss=adaptive_auxiliary_loss,
            mask_weight=mask_weight,
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        img: torch.Tensor,
        return_loss: bool = True,
        return_logits: bool = True,
        is_train: bool = True,
    ):
        """
        img  : [B, 54]  – content token indices (long)

        Returns dict with keys 'loss' (if return_loss), 'logits' (if return_logits).
        """
        # DiffusionTransformer forward (unconditional)
        out = self.diffusion(
            {
                'content_token': img,
                'condition_embed_token': None,
            },
            return_loss=return_loss,
            return_logits=return_logits,
            is_train=is_train,
        )
        return out

    # ------------------------------------------------------------------ #
    #  Sampling                                                           #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        filter_ratio: float = 0.5,
        temperature: float = 1.0,
        return_logits: bool = False,
        content_token=None,
        save_steps=None,
        device='cuda',
    ):
        """
        Unconditional sampling.
        
        Args:
            batch_size: number of samples to generate
            filter_ratio: ratio of tokens to keep during sampling
            temperature: temperature for sampling
            return_logits: whether to return logits
            content_token: optional initial content tokens
            save_steps: list of int or None, specifies which diffusion steps to save
            device: device to generate on
        
        Returns sampled content tokens and optionally logits.
        If save_steps is not None, also returns intermediate_tokens and intermediate_logits.
        """
        # If save_steps is provided, call sample_with_intermediate()
        if save_steps is not None:
            out = self.diffusion.sample_with_intermediate(
                condition_token=None,
                condition_mask=None,
                condition_embed=None,
                content_token=content_token,
                filter_ratio=filter_ratio,
                temperature=temperature,
                return_logits=return_logits,
                save_steps=save_steps,
            )
        else:
            # Otherwise call standard sample()
            out = self.diffusion.sample(
                condition_token=None,
                condition_mask=None,
                condition_embed=None,
                content_token=content_token,
                filter_ratio=filter_ratio,
                temperature=temperature,
                return_logits=return_logits,
                batch_size=batch_size,
            )
        
        return out

    def load(self, path_model: str):
        from collections import OrderedDict
        checkpoint = torch.load(path_model, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict, strict=False)
        loss = checkpoint.get('loss', 'N/A')
        print(f'\t [UnconditionalDiffusion] loaded {path_model}  loss={loss}')


if __name__ == '__main__':
    import sys
    import os

    # Ensure project root is in sys.path
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    os.chdir(_root)

    from omegaconf import OmegaConf

    # Load config
    cfg = OmegaConf.load(
        os.path.join(_root, 'configs/config_cvqmae/config_hrnet.yaml')
    )
    print('Config loaded:')
    print(OmegaConf.to_yaml(cfg))

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build UnconditionalDiffusion
    model_cfg = cfg.model
    model = UnconditionalDiffusion(
        num_tok=model_cfg.num_embeddings,              # 512
        seq_len=model_cfg.seq_length,                  # 54
        n_emb=512,
        n_head=8,
        n_layer=12,
        diff_step=100,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'UnconditionalDiffusion total params: {total_params:,}')

    # Forward test (train mode)
    B = 2
    img = torch.randint(0, model_cfg.num_embeddings, (B, model_cfg.seq_length), device=DEVICE)

    model.train()
    out = model(img, return_loss=True, return_logits=True)
    print(f'  [train] loss   = {out["loss"].item():.4f}')
    print(f'  [train] logits = {out["logits"].shape}')

    # Sampling test (eval mode)
    model.eval()
    sample_out = model.sample(batch_size=1, filter_ratio=0.5, temperature=1.0, device=DEVICE)
    print(f'  [sample] output keys: {list(sample_out.keys()) if isinstance(sample_out, dict) else type(sample_out)}')
    print('UnconditionalDiffusion smoke-test passed.')
