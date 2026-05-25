import torch
import torch.nn as nn
from .base_embedding import BaseEmbedding


class Sequence1DEmbedding(BaseEmbedding):
    """
    针对非图像序列（如 mesh、点云、序列 token 等）的 content embedding。
    与 DalleMaskImageEmbedding 的区别：
      - 位置编码使用 1D nn.Embedding(seq_len, embed_dim)，不假设输入是 2D 空间网格
      - 适用于任意序列长度，不要求 seq_len = H * W

    Args:
        num_embed  : 码本大小（不含 mask token），内部自动 +1 作为 mask token
        seq_len    : 序列最大长度，用于构造 1D 位置编码表
        embed_dim  : embedding 维度
        trainable  : 是否参与训练
    """

    def __init__(
        self,
        num_embed: int = 512,
        seq_len: int = 54,
        embed_dim: int = 512,
        trainable: bool = True,
    ):
        super().__init__()

        self.seq_len   = seq_len
        self.num_embed = num_embed + 1  # +1 for mask token，与 DalleMaskImageEmbedding 保持一致
        self.embed_dim = embed_dim
        self.trainable = trainable

        # token embedding：[0, num_embed] 共 num_embed+1 个 id
        self.emb = nn.Embedding(self.num_embed, embed_dim)

        # 1D 位置编码：覆盖 [0, seq_len-1]
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        self._set_trainable()

    def forward(self, index, **kwargs):
        """
        Args:
            index: LongTensor, shape (B, L)，token id，范围 [0, num_embed-1]
                   mask token id = num_embed（即最后一个）
        Returns:
            emb: FloatTensor, shape (B, L, embed_dim)
        """
        assert index.dim() == 2, f'index 应为 2D (B, L)，实际 shape: {index.shape}'
        L = index.shape[1]
        assert L <= self.seq_len, (
            f'序列长度 {L} 超过初始化时指定的 seq_len={self.seq_len}')

        # 将负数 id（padding）置为 0，避免越界
        index = index.clone()
        index[index < 0] = 0

        # token embedding
        token_emb = self.emb(index)                                   # (B, L, D)

        # 1D 位置编码
        positions = torch.arange(L, device=index.device).unsqueeze(0)  # (1, L)
        pos_emb   = self.pos_emb(positions)                            # (1, L, D)

        return token_emb + pos_emb                                     # (B, L, D)
