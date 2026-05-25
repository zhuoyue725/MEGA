"""
Multi-modal image feature encoder (paper subsection \label{subsec:image_encoder}).

Fuses RGB image features, surface normal maps, and 2D keypoints into a unified
condition embedding z ∈ [B, 49, 1024] for the DiffusionTransformer.

Hot-pluggable: when normal_map and keypoints are both None, acts as a pure RGB
encoder identical to the original _encode_cond + rotcam_head pipeline.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (copied from cvqdiffusion.py)."""

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


class NormalMapViT(nn.Module):
    """Lightweight ViT for surface normal map encoding.

    Encodes a 224×224 surface normal map (3-channel unit-norm vectors) into
    a spatial feature map matching the HRNet backbone output format.

    Input:  [B, 3, 224, 224]
    Output: [B, 720, 7, 7]   (same spatial dims and channel count as HRNet)
    """

    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=256,
                 depth=4, num_heads=4, output_dim=720):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2  # 49
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                       batch_first=True, dropout=0.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, output_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)            # [B, 256, 7, 7]
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)   # [B, 49, 256]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)  # [B, 256, 7, 7]
        x = self.proj(x)                    # [B, 720, 7, 7]
        return x


class KeypointEncoder(nn.Module):
    """Transformer encoder for 2D keypoints (OpenPose body_25 or SMPL 24-joint format).

    Projects (x, y, confidence) to high-dim tokens and refines with self-attention.

    Input:  [B, J, 3]    — 2D keypoints with confidence
    Output: [B, J, 1024] — keypoint token embeddings
    """

    def __init__(self, in_dim=3, embed_dim=1024, num_keypoints=25,
                 depth=2, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_keypoints, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                       batch_first=True, dropout=0.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, J, 3]
        x = self.input_proj(x)                           # [B, J, 1024]
        x = x + self.pos_embed[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x                                          # [B, J, 1024]


class UnifiedConditionEncoder(nn.Module):
    """Transformer that fuses multi-modal tokens into a unified condition.

    Takes concatenated tokens from RGB, normal, and keypoint branches,
    applies self-attention to model cross-modal interactions, and outputs
    only the RGB-spatial tokens [B, 49, 1024].

    Input:  [B, L, 1024]   — concatenated tokens from all available modalities
    Output: [B, 49, 1024]  — first 49 tokens (RGB spatial positions)
    """

    def __init__(self, embed_dim=1024, depth=4, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                       batch_first=True, dropout=0.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, :49, :]   # keep only RGB-spatial tokens


class MultiModalConditionEncoder(nn.Module):
    """Multi-modal image feature encoder.

    Fuses three modalities:
      1. RGB features from HRNet backbone  → Y^I [B, 49, 1024]
      2. Surface normal maps via lightweight ViT → Y^N [B, 49, 1024]
      3. 2D keypoints via transformer → Y^K [B, J, 1024]

    All tokens are concatenated and fused through a shared transformer (UCE),
    producing a unified condition embedding z ∈ [B, 49, 1024].

    Rotation and camera parameters are predicted from RGB features only.

    When normal_map and keypoints are both None, this module acts as a
    pure RGB encoder, producing output *identical* to the original
    CVQDiffusion._encode_cond + rotcam_head pipeline.
    """

    def __init__(
        self,
        backbone_feat_dim: int = 720,
        cond_emb_dim: int = 1024,
        cond_len: int = 49,
        # Normal map ViT
        normal_patch_size: int = 32,
        normal_vit_dim: int = 256,
        normal_vit_depth: int = 4,
        normal_vit_heads: int = 4,
        # Keypoint encoder
        num_keypoints: int = 25,
        kp_embed_dim: int = 1024,
        kp_depth: int = 2,
        kp_heads: int = 8,
        # Unified Condition Encoder
        uce_depth: int = 4,
        uce_heads: int = 8,
    ):
        super().__init__()

        # ---- RGB branch (always active) ----
        # Maps HRNet spatial features → condition tokens (same as _encode_cond)
        self.cond_emb = nn.Linear(backbone_feat_dim, cond_emb_dim)
        self.pos_emb_cond = PositionalEncoding(d_model=cond_emb_dim, max_len=cond_len)

        # ---- Normal map branch ----
        self.normal_vit = NormalMapViT(
            img_size=224,
            patch_size=normal_patch_size,
            embed_dim=normal_vit_dim,
            depth=normal_vit_depth,
            num_heads=normal_vit_heads,
            output_dim=backbone_feat_dim,
        )

        # ---- Keypoint branch ----
        self.kp_encoder = KeypointEncoder(
            in_dim=3,
            embed_dim=kp_embed_dim,
            num_keypoints=num_keypoints,
            depth=kp_depth,
            num_heads=kp_heads,
        )

        # ---- Unified Condition Encoder ----
        self.uce = UnifiedConditionEncoder(
            embed_dim=cond_emb_dim,
            depth=uce_depth,
            num_heads=uce_heads,
        )

        # ---- Rotation & Camera regression (RGB features only) ----
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rotcam_head = nn.Sequential(
            nn.Linear(backbone_feat_dim, cond_emb_dim),
            nn.Dropout(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.Dropout(),
        )
        self.rot_predictor = nn.Linear(cond_emb_dim, 6)
        self.cam_predictor = nn.Linear(cond_emb_dim, 3)

    def _encode_rgb(self, cond_feat):
        """Encode a spatial feature map to condition tokens.

        Mirrors CVQDiffusion._encode_cond exactly:
            [B, C, 7, 7] → rearrange → Linear → PE → rearrange → [B, 49, 1024]
        """
        cond = rearrange(cond_feat, 'b c w h -> (w h) b c')
        cond = self.cond_emb(cond)
        cond = self.pos_emb_cond(cond)
        cond = rearrange(cond, 't b c -> b t c')
        return cond

    def _regress_rotcam(self, cond_feat):
        """Regress rotation [B, 6] and camera [B, 3] from pooled RGB features."""
        cond_single = self.avg_pool(cond_feat).view(cond_feat.size(0), 1, -1)
        rotcam_feature = self.rotcam_head(cond_single)
        pred_rot = self.rot_predictor(rotcam_feature).view(-1, 6)
        pred_cam = self.cam_predictor(rotcam_feature).view(-1, 3)
        return pred_rot, pred_cam

    def forward(self, cond_feat, normal_map=None, keypoints=None):
        """Encode multi-modal conditions into unified embedding.

        Args:
            cond_feat:  [B, 720, 7, 7]  — RGB backbone features (always required)
            normal_map: [B, 3, 224, 224] — surface normal map or None
            keypoints:  [B, J, 3]        — (x, y, confidence) or None

        Returns:
            cond_emb: [B, 49, 1024] — unified condition for DiffusionTransformer
            pred_rot: [B, 6]        — 6D rotation representation
            pred_cam: [B, 3]        — camera translation parameters
        """
        # 1. RGB branch (always active)
        cond_emb_rgb = self._encode_rgb(cond_feat)  # [B, 49, 1024]

        # 2. Rotation/camera from RGB features only (per paper design)
        pred_rot, pred_cam = self._regress_rotcam(cond_feat)

        # 3. No multi-modal inputs → pure RGB mode (identical to original pipeline)
        if normal_map is None and keypoints is None:
            return cond_emb_rgb, pred_rot, pred_cam

        # 4. Build multi-modal token sequence
        tokens = [cond_emb_rgb]  # [B, 49, 1024]

        if normal_map is not None:
            normal_feat = self.normal_vit(normal_map)     # [B, 720, 7, 7]
            cond_emb_normal = self._encode_rgb(normal_feat)  # [B, 49, 1024]
            tokens.append(cond_emb_normal)

        if keypoints is not None:
            cond_emb_kp = self.kp_encoder(keypoints)      # [B, J, 1024]
            tokens.append(cond_emb_kp)

        # 5. Fuse through Unified Condition Encoder
        z = torch.cat(tokens, dim=1)  # [B, 49+N+J, 1024]
        z = self.uce(z)               # [B, 49, 1024]

        return z, pred_rot, pred_cam


if __name__ == '__main__':
    import sys
    import os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    os.chdir(_root)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=== MultiModalConditionEncoder Smoke Test ===\n")

    # ---- Create encoder ----
    encoder = MultiModalConditionEncoder(
        backbone_feat_dim=720,
        cond_emb_dim=1024,
        cond_len=49,
        normal_patch_size=32,
        normal_vit_dim=256,
        normal_vit_depth=4,
        kp_embed_dim=1024,
        kp_depth=2,
        uce_depth=4,
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder total params: {total_params:,}")

    B = 2
    cond_feat = torch.randn(B, 720, 7, 7, device=device)

    # ---- Test 1: RGB only (no multi-modal) ----
    print("\n--- Test 1: RGB only (multi-modal inputs = None) ---")
    cond_emb, pred_rot, pred_cam = encoder(cond_feat)
    print(f"  cond_emb: {cond_emb.shape}  (expected [2, 49, 1024])")
    print(f"  pred_rot: {pred_rot.shape}  (expected [2, 6])")
    print(f"  pred_cam: {pred_cam.shape}  (expected [2, 3])")
    assert cond_emb.shape == (B, 49, 1024), f"cond_emb shape mismatch: {cond_emb.shape}"
    assert pred_rot.shape == (B, 6), f"pred_rot shape mismatch: {pred_rot.shape}"
    assert pred_cam.shape == (B, 3), f"pred_cam shape mismatch: {pred_cam.shape}"
    print("  PASSED")

    # ---- Test 2: RGB + keypoints ----
    print("\n--- Test 2: RGB + keypoints (J=25) ---")
    kps = torch.randn(B, 25, 3, device=device)
    cond_emb2, pred_rot2, pred_cam2 = encoder(cond_feat, keypoints=kps)
    print(f"  cond_emb: {cond_emb2.shape}  (expected [2, 49, 1024])")
    assert cond_emb2.shape == (B, 49, 1024)
    print("  PASSED")

    # ---- Test 3: RGB + normal map ----
    print("\n--- Test 3: RGB + normal map ---")
    normal = torch.randn(B, 3, 224, 224, device=device)
    cond_emb3, pred_rot3, pred_cam3 = encoder(cond_feat, normal_map=normal)
    print(f"  cond_emb: {cond_emb3.shape}  (expected [2, 49, 1024])")
    assert cond_emb3.shape == (B, 49, 1024)
    print("  PASSED")

    # ---- Test 4: RGB + normal + keypoints (full modality) ----
    print("\n--- Test 4: RGB + normal + keypoints (full) ---")
    cond_emb4, pred_rot4, pred_cam4 = encoder(cond_feat, normal_map=normal, keypoints=kps)
    print(f"  cond_emb: {cond_emb4.shape}  (expected [2, 49, 1024])")
    assert cond_emb4.shape == (B, 49, 1024)
    print("  PASSED")

    # ---- Test 5: Keypoints with different J (24 vs 25) ----
    print("\n--- Test 5: Variable number of keypoints (J=24) ---")
    kps24 = torch.randn(B, 24, 3, device=device)
    cond_emb5, _, _ = encoder(cond_feat, keypoints=kps24)
    print(f"  cond_emb: {cond_emb5.shape}  (expected [2, 49, 1024])")
    assert cond_emb5.shape == (B, 49, 1024)
    print("  PASSED")

    print("\n=== All smoke tests passed! ===")
