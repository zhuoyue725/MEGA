import sys
import os

target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if target_path not in sys.path:
    sys.path.insert(0, target_path)

import torch
import torch.nn as nn
from einops import rearrange
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.transformers.diffusion_transformer_mesh import DiffusionTransformer


class PositionalEncoding(torch.nn.Module):
    """Sinusoidal positional encoding (from masked_autoencoder.py)."""

    import math as _math

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math
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


class CVQDiffusion(nn.Module):
    """
    Conditional VQ-Diffusion model.

    forward(img, cond):
        img  : [B, 54]          – content token indices
        cond : [B, 3, 224, 224] – raw image condition

    Internally:
        1. cond -> backbone              -> [B, 720, 7, 7]
        2. rearrange + cond_emb + pos_emb -> [B, 49, 1024]  (condition embedding)
        3. DiffusionTransformer(content_token=img,
                                condition_embed_token=cond_emb)
    """

    # ------------------------------------------------------------------ #
    #  DiffusionTransformer hyper-parameters                              #
    # ------------------------------------------------------------------ #
    _NUM_TOK   = 512   # content codebook size
    _SEQ_LEN   = 54    # content sequence length
    _N_EMB     = 512   # transformer hidden dim
    _COND_DIM  = 1024  # condition embedding dim (output of cond_emb projection)
    _COND_LEN  = 49    # condition sequence length  (7×7 spatial tokens)
    _N_HEAD    = 8
    _N_LAYER   = 12
    _DIFF_STEP = 100

    def __init__(
        self,
        backbone: nn.Module,
        # backbone output: [B, backbone_feat_dim, 7, 7]
        backbone_feat_dim: int = 720,
        # condition projection output dim
        cond_emb_dim: int = _COND_DIM,
        # DiffusionTransformer settings (can be overridden)
        num_tok: int    = _NUM_TOK,
        seq_len: int    = _SEQ_LEN,
        n_emb: int      = _N_EMB,
        cond_dim: int   = _COND_DIM,
        cond_len: int   = _COND_LEN,
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

        self.backbone = backbone

        # ---- condition projection: backbone_feat_dim -> cond_emb_dim ----
        # mirrors MAE_Decoder.cond_emb  (Linear) + pos_emb_cond (sinusoidal)
        self.cond_emb     = nn.Linear(backbone_feat_dim, cond_emb_dim)
        self.pos_emb_cond = PositionalEncoding(d_model=cond_emb_dim, max_len=cond_len)

        # ---- rotation and camera regression head (mirrors MAE_Decoder) ----
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rotcam_head = nn.Sequential(
            nn.Linear(backbone_feat_dim, cond_emb_dim),
            nn.Dropout(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.Dropout(),
        )
        self.rot_predictor = nn.Linear(cond_emb_dim, 6)
        self.cam_predictor = nn.Linear(cond_emb_dim, 3)

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

        # ---- transformer config ----
        transformer_cfg = {
            'target': 'image_synthesis.modeling.transformers.transformer_utils.Text2ImageTransformer',
            'params': {
                'attn_type':            'selfcross',
                'n_layer':              n_layer,
                'condition_seq_len':    cond_len,
                'content_seq_len':      seq_len,
                'content_spatial_size': [seq_len, 1],
                'n_embd':               n_emb,
                'n_head':               n_head,
                'condition_dim':        cond_dim,
                'attn_pdrop':           0.0,
                'resid_pdrop':          0.0,
                'block_activate':       'GELU2',
                'timestep_type':        'adalayernorm',
                'mlp_hidden_times':     4,
            },
        }

        self.diffusion = DiffusionTransformer(
            content_emb_config=content_emb_cfg,
            condition_emb_config=None,      # condition provided externally
            transformer_config=transformer_cfg,
            diffusion_step=diff_step,
            alpha_init_type='alpha1',
            auxiliary_loss_weight=auxiliary_loss_weight,
            adaptive_auxiliary_loss=adaptive_auxiliary_loss,
            mask_weight=mask_weight,
        )

    # ------------------------------------------------------------------ #
    #  Condition encoding  (mirrors masked_autoencoder_comment.py 307-310)#
    # ------------------------------------------------------------------ #
    def _encode_cond(self, cond_feat: torch.Tensor) -> torch.Tensor:
        """
        cond_feat : [B, C, H, W]  e.g. [1, 720, 7, 7]
        returns   : [B, H*W, cond_emb_dim]  e.g. [1, 49, 1024]
        """
        # [B, C, W, H] -> [W*H, B, C]
        cond = rearrange(cond_feat, 'b c w h -> (w h) b c')   # [49, B, 720]
        cond = self.cond_emb(cond)                             # [49, B, 1024]
        cond = self.pos_emb_cond(cond)                         # [49, B, 1024]
        cond = rearrange(cond, 't b c -> b t c')               # [B, 49, 1024]
        return cond

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        img: torch.Tensor,
        cond: torch.Tensor,
        return_loss: bool = True,
        return_logits: bool = True,
        is_train: bool = True,
    ):
        """
        img  : [B, 54]           – content token indices (long)
        cond : [B, 3, 224, 224]  – raw image

        Returns dict with keys 'loss' (if return_loss), 'logits' (if return_logits),
        'pred_rot' [B, 6] (6D rotation representation), and 'pred_cam' [B, 3] (camera parameters).
        """
        # 1. backbone: [B, 3, 224, 224] -> [B, 720, 7, 7]
        cond_feat = self.backbone(cond)          # [B, 720, 7, 7]

        # 2. rotation and camera regression from pooled backbone feature
        #    avg_pool: [B, 720, 7, 7] -> [B, 720, 1, 1] -> [B, 1, 720]
        cond_single = self.avg_pool(cond_feat).view(cond_feat.size(0), 1, -1)  # [B, 1, 720]
        rotcam_feature = self.rotcam_head(cond_single)                         # [B, 1, cond_emb_dim]
        pred_rot = self.rot_predictor(rotcam_feature).view(-1, 6)              # [B, 6]
        pred_cam = self.cam_predictor(rotcam_feature).view(-1, 3)              # [B, 3]

        # 3. condition embedding: [B, 720, 7, 7] -> [B, 49, 1024]
        cond_emb = self._encode_cond(cond_feat)  # [B, 49, 1024]

        # 4. DiffusionTransformer forward
        out = self.diffusion(
            {
                'content_token':          img,
                'condition_embed_token':  cond_emb,
            },
            return_loss=return_loss,
            return_logits=return_logits,
            is_train=is_train,
        )
        out['pred_rot'] = pred_rot
        out['pred_cam'] = pred_cam
        return out

    # ------------------------------------------------------------------ #
    #  Sampling                                                           #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        filter_ratio: float = 0.5,
        temperature: float = 1.0,
        return_logits: bool = False,
        content_token=None,
    ):
        """
        cond : [B, 3, 224, 224]
        Returns sampled content tokens, pred_rot [B, 6], pred_cam [B, 3], and optionally logits.
        """
        cond_feat = self.backbone(cond)          # [B, 720, 7, 7]

        # rotation and camera regression
        cond_single = self.avg_pool(cond_feat).view(cond_feat.size(0), 1, -1)  # [B, 1, 720]
        rotcam_feature = self.rotcam_head(cond_single)                         # [B, 1, cond_emb_dim]
        pred_rot = self.rot_predictor(rotcam_feature).view(-1, 6)              # [B, 6]
        pred_cam = self.cam_predictor(rotcam_feature).view(-1, 3)              # [B, 3]

        cond_emb  = self._encode_cond(cond_feat) # [B, 49, 1024]

        out = self.diffusion.sample(
            condition_token=None,
            condition_mask=None,
            condition_embed=cond_emb,
            content_token=content_token,
            filter_ratio=filter_ratio,
            temperature=temperature,
            return_logits=return_logits,
        )
        out['pred_rot'] = pred_rot
        out['pred_cam'] = pred_cam
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
        print(f'\t [CVQDiffusion] loaded {path_model}  loss={loss}')


if __name__ == '__main__':
    import sys
    import os

    # 确保项目根目录在 sys.path 中（从 mega/model/cvqdiffusion/ 往上三级）
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    # 切换工作目录到项目根目录，使相对路径（如 body_models/）可访问
    os.chdir(_root)

    from omegaconf import OmegaConf

    # ---- 读取与 train_mega.py 相同的配置文件 ----
    cfg = OmegaConf.load(
        os.path.join(_root, 'configs/config_cvqmae/config_hrnet.yaml')
    )
    print('Config loaded:')
    print(OmegaConf.to_yaml(cfg))

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 加载 HRNet-W48 backbone（与 train_mega.py 保持一致）----
    from mega.model.backbone import hrnet_w48

    pretrained_ckpt_path = cfg.backbone.pretrained   # body_models/pose_hrnet_w48.pth
    backbone = hrnet_w48(
        pretrained_ckpt_path=pretrained_ckpt_path,
        downsample=True,
        use_conv=True,
    ).to(DEVICE)
    print(f'Backbone (HRNet-W48) loaded from: {pretrained_ckpt_path}')

    # ---- 构建 CVQDiffusion ----
    # 从 config 读取相关超参（与 CVQMAE 共享 cond_dim / num_embeddings / seq_length）
    model_cfg = cfg.model
    model = CVQDiffusion(
        backbone=backbone,
        backbone_feat_dim=model_cfg.cond_dim,          # 720
        cond_emb_dim=1024,
        num_tok=model_cfg.num_embeddings,              # 512
        seq_len=model_cfg.seq_length,                  # 54
        n_emb=512,
        cond_dim=1024,
        cond_len=model_cfg.cond_length,                # 49
        n_head=8,
        n_layer=12,
        diff_step=100,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'CVQDiffusion total params: {total_params:,}')

    # ---- 前向测试（train 模式）----
    B = 2
    img  = torch.randint(0, model_cfg.num_embeddings, (B, model_cfg.seq_length), device=DEVICE)
    cond = torch.randn(B, 3, 224, 224, device=DEVICE)

    model.train()
    out = model(img, cond, return_loss=True, return_logits=True)
    print(f'  [train] loss   = {out["loss"].item():.4f}')
    print(f'  [train] logits = {out["logits"].shape}')

    # ---- 采样测试（eval 模式）----
    model.eval()
    cond_single = torch.randn(1, 3, 224, 224, device=DEVICE)
    sample_out = model.sample(cond_single, filter_ratio=0.5, temperature=1.0)
    print(f'  [sample] output keys: {list(sample_out.keys()) if isinstance(sample_out, dict) else type(sample_out)}')
    print('CVQDiffusion smoke-test passed.')
