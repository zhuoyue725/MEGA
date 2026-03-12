import torch
import numpy as np
import math
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from .masking import (
    PatchShuffle,
    PatchShuffleScheduled,
    PatchShuffleInference,
)
import torch.nn.functional as F
def analyze_prediction_confidence(full_predicted_img):
    """
    衡量预测置信度、排序并打印详细分析报告
    参数: full_predicted_img [1, 54, 512]
    """
    # 1. 计算概率和置信度
    probs = F.softmax(full_predicted_img, dim=-1) # [1, 54, 512]
    confidence, pred_indices = torch.max(probs[0], dim=-1) # [54]
    
    # 2. 计算信息熵 (越分布均匀，熵越高)
    entropy = -torch.sum(probs[0] * torch.log(probs[0] + 1e-9), dim=-1)
    
    # 3. 排序：按置信度从小到大 (最不确定的排在最前面)
    sorted_conf, sorted_token_indices = torch.sort(confidence, descending=False)

    # 4. 打印报告
    print("\n" + "="*80)
    print(f"{'RANK':<6} | {'TOKEN_IDX':<10} | {'CONFIDENCE':<12} | {'ENTROPY':<10} | {'VISUAL'}")
    print("-" * 80)

    analysis_results = []
    for i in range(len(sorted_token_indices)):
        token_idx = sorted_token_indices[i].item()
        conf_val = sorted_conf[i].item()
        ent_val = entropy[token_idx].item()
        
        # 简单的可视化进度条 (20格)
        bar = "█" * int(conf_val * 20) + "░" * (20 - int(conf_val * 20))
        
        # 只有置信度特别低或者前5个/后5个才打印，防止54行刷屏（可选）
        # 这里为了让你看全，默认全打印
        print(f"{i+1:<6} | {token_idx:<10} | {conf_val:<12.4f} | {ent_val:<10.4f} | {bar}")

        analysis_results.append({
            'rank': i + 1,
            'token_index': token_idx,
            'confidence': conf_val,
            'entropy': ent_val,
            'codebook_idx': pred_indices[token_idx].item()
        })

    print("-" * 80)
    print(f"平均置信度: {confidence.mean().item():.4f}")
    print(f"最不确定 Token: {sorted_token_indices[0].item()} (Conf: {sorted_conf[0].item():.4f})")
    print(f"最确定 Token: {sorted_token_indices[-1].item()} (Conf: {sorted_conf[-1].item():.4f})")
    print("="*80 + "\n")
        
    return analysis_results, sorted_token_indices

def get_num_mask(t, T, L):
    """
    t: 当前迭代次数 (0, 1, ..., T-1)
    T: 总迭代次数
    L: token 序列长度 (此处为 54)
    """
    # 计算 x = t / T，范围从 0 渐进到 (T-1)/T
    x = t / T
    # 余弦调度函数 γ(x)
    gamma = (1 + math.cos(math.pi * x)) / 2
    # 计算掩码数量：向上取整
    num_mask = math.ceil(gamma * L)
    return num_mask

def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


def size_model(model: torch.nn.Module):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")


def add_gumbel_noise(t, temperature):
    return t + torch.Tensor(temperature * np.random.gumbel(size=t.shape)).to(t)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return x


"""
        MASKED Encoder

"""


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        seq_length=None,
        emb_dim=None,
        num_layer=12,
        num_head=2,
        mask_ratio=0.25,
        num_embeddings=512,
        vqvae_embedding=None,
        masking: str = "random",  # ["random", "horizontal", "vertical", "mosaic", "half", "scheduled"]
        trainable_position: bool = True,
    ) -> None:
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        if trainable_position:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(seq_length, 1, emb_dim))
        else:
            self.pos_embedding = PositionalEncoding(d_model=emb_dim, max_len=seq_length)
        if masking.lower() == "random":
            self.shuffle = PatchShuffle(mask_ratio)
        elif masking.lower() == "scheduled":
            self.shuffle = PatchShuffleScheduled()
        else:
            raise Exception("masking must be random or scheduled")
        self.shuffle_inference = PatchShuffleInference()
        self.proj = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=emb_dim
        )
        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.seq_length = seq_length
        self.mask_ratio = mask_ratio
        self.vqvae_embedding = vqvae_embedding
        self.trainable_position = trainable_position
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        if self.trainable_position:
            trunc_normal_(self.pos_embedding, std=0.02)
        if self.vqvae_embedding is not None:
            self.proj = torch.nn.Embedding.from_pretrained(
                embeddings=self.vqvae_embedding, freeze=True
            )

    def forward(self, patches, mask=None, fixed_ratio=None):
        patches = rearrange(patches, "b t -> t b") # [1, 54] -> [54, 1]
        patches = self.proj(patches).reshape(self.seq_length, -1, self.emb_dim) # [54, 1, 1024]
        if self.trainable_position:
            patches = patches + self.pos_embedding
        else:
            patches = self.pos_embedding(patches)
        if mask is None:
            if fixed_ratio is None:
                patches, forward_indexes, backward_indexes = self.shuffle(patches) # 内部随机0-1的fixed_ratio
            else:
                patches, forward_indexes, backward_indexes = self.shuffle(
                    patches, fixed_ratio=fixed_ratio # 打乱，并根据Mask Ratio进行掩码，1表示全部被mask，剔除后面的
                ) # [54, 1, 1024] -> [0, 1, 1024]
        else:
            # mask [1,1,1,0,0,1,1]，1保留（也会被打乱），0被去掉
            patches, forward_indexes, backward_indexes = self.shuffle_inference(
                patches, masks=mask # 内部对原先的索引也打乱了顺序 backward_indexes [1, 3, 0] 表示原先的patches[0]在打乱后的patches[1]
            )
        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        ) # [1, 1, 1024]
        patches = rearrange(patches, "t b c -> b t c") # [1, 1, 1024]
        features = self.layer_norm(self.transformer(patches)) # [1, 1, 1024]
        features = rearrange(features, "b t c -> t b c") # [1, 1, 1024]
        return features, backward_indexes


"""
        MASKED Decoder

"""


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        seq_length=None,
        emb_dim=None,
        num_layer=4,
        num_head=2,
        dim_tokens=32,
        trainable_position: bool = True,
        cond_length: int = 1,
        cond_dim: int = 1280,
        cond_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        if trainable_position:
            self.pos_embedding = torch.nn.Parameter(
                torch.zeros(seq_length + 1, 1, emb_dim)
            )
        else:
            self.pos_embedding = PositionalEncoding(
                d_model=emb_dim, max_len=seq_length + 1
            )
        self.pos_emb_cond = PositionalEncoding(d_model=emb_dim, max_len=cond_length)
        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        self.head = torch.nn.Linear(emb_dim, dim_tokens)
        self.trainable_position = trainable_position
        self.seq_length = seq_length
        self.init_weight()

        self.rotcam_head = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, emb_dim),
            torch.nn.Dropout(),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.Dropout(),
        )
        self.rot_predictor = torch.nn.Linear(emb_dim, 6)
        self.cam_predictor = torch.nn.Linear(emb_dim, 3)

        self.cond_emb = torch.nn.Linear(cond_dim, emb_dim)
        self.cond_length = cond_length
        self.cond_dropout = cond_dropout
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        if self.trainable_position:
            trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features, cond, backward_indexes, cond_dropout=False, T=5):
        """
        前向传播，根据isTrain参数选择训练或推理模式
        
        Args:   
            features: 来自encoder的特征
            cond: 条件特征
            backward_indexes: 用于恢复原始顺序的索引
            cond_dropout: 是否使用条件dropout
            isTrain: True表示训练模式，False表示推理模式（使用迭代解码）
            T: 推理模式下的迭代次数，默认为5
        """
        # 训练模式：原有的一次性解码逻辑
        T_enc = features.shape[0] # features: [1, 1, 1024] 1表示完全不可见（第一维度是cls_token）
        bs = features.shape[1]
        backward_indexes = torch.cat( # [55, 1]
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        features = torch.cat( # [55, 1, 1024]
            [
                features,
                self.mask_token.expand( # 如果是推理，此处是全部被mask，问题是训练的时候这里应该也是全mask吧？有必要吗为什么不直接只用condition来解码
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )
        features = take_indexes(features, backward_indexes) # 未打乱的原始顺序 [55, 1, 1024]
        if self.trainable_position:
            features = features + self.pos_embedding
        else:
            features = self.pos_embedding(features)
        features = rearrange(features, "t b c -> b t c")

        cond_single = self.avg_pool(cond).view(bs, 1, -1) # [1, 720, 7, 7] -> [1, 1, 720]
        cond = rearrange(cond, "b c w h -> (w h) b c") # [1, 720, 7, 7] -> [49, 1, 720]
        cond_emb = self.cond_emb(cond) # [49, 1, 1024]
        cond_emb = self.pos_emb_cond(cond_emb) # 加上位置编码 [49, 1, 1024]
        cond_emb = rearrange(cond_emb, "t b c -> b t c") # [1, 49, 1024]
        if cond_dropout:
            mask_cond = torch.bernoulli(
                torch.ones(bs, device=cond_emb.device) * self.cond_dropout
            ).view(bs, 1)
            cond_emb = cond_emb * (1.0 - mask_cond)

        features = torch.cat([cond_emb, features], dim=1) # 拼接图像condition [1, 49, 1024] 和 token向量 [1, 55, 1024]

        features = self.transformer(features) # [1, 104, 1024] -> [1, 104, 1024] 这里为什么是1024而不是54
        features = rearrange(features, "b t c -> t b c") # [104, 1, 1024]
        pose_features = features[ # 取后面的Pose特征 [54, 1, 1024]
            self.cond_length + 1 :
        ]  # remove image and global feature

        rotcam_feature = self.rotcam_head(cond_single) # [1, 1, 720] ->[1, 1, 1024]
        pred_rot = self.rot_predictor(rotcam_feature).view(-1, 6) # [1, 6]
        pred_cam = self.cam_predictor(rotcam_feature).view(-1, 3) # [1, 3]

        patches = pose_features
        mask = torch.zeros_like(patches) # [54, 1, 1024] 为什么要1024维度
        mask[T_enc:] = 1 # T之前是掩码设置为0，后面的可见的设置为1？
        mask = take_indexes(mask, backward_indexes[1:] - 1) # Encoder中打乱了，这里还原到原本的位置
        mask = rearrange(mask, "t b c -> b t c") # [1, 54, 1024]
        patches = rearrange(patches, "t b c -> b t c") # [1, 54, 1024]
        patches = self.head(patches) # [1, 54, 1024] -> [1, 54, 512] 最后会取最大值
        return patches, pred_rot, pred_cam, mask[:, :, 0]



"""
        MASKED AUTOENCODER

"""


class CVQMAE(torch.nn.Module):
    def __init__(
        self,
        seq_length=54,
        emb_dim=1280,
        encoder_layer=12,
        encoder_head=4,
        decoder_layer=4,
        decoder_head=4,
        mask_ratio=0.75,
        num_embeddings=32,
        dim_tokens=32,
        vqvae_embedding=None,
        masking: str = "scheduled",
        trainable_position: bool = True,
        cond_length: int = 1,
        cond_dim: int = 1280,
        cond_dropout: float = 0.1,
        backbone=None,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        self.encoder = MAE_Encoder(
            seq_length=seq_length,
            emb_dim=emb_dim,
            num_layer=encoder_layer,
            num_head=encoder_head,
            mask_ratio=mask_ratio,
            num_embeddings=num_embeddings,
            vqvae_embedding=vqvae_embedding,
            masking=masking,
            trainable_position=trainable_position,
        )

        self.decoder = MAE_Decoder(
            seq_length=seq_length,
            emb_dim=emb_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
            dim_tokens=dim_tokens,
            trainable_position=trainable_position,
            cond_length=cond_length,
            cond_dim=cond_dim,
            cond_dropout=cond_dropout,
        )
        
        self.segmentation = dict()
        self.segmentation["head"] = [0, 1, 2, 3, 12, 29, 30, 31, 48]
        self.segmentation["left_arm"] = [8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 23]
        self.segmentation["right_arm"] = [39, 40, 41, 42, 43, 44, 45, 46, 47, 49]
        self.segmentation["left_leg"] = [6, 7, 10, 26, 27, 28]
        self.segmentation["right_leg"] = [34, 36, 50, 51, 52, 53]
        self.segmentation["buste"] = [4, 5, 9, 21, 22, 24, 25, 32, 33, 35, 37, 38]

    def forward(self, img, cond, fixed_ratio=None, cond_dropout=False):
        cond = self.backbone(cond) # [1, 3, 224, 224] -> [1, 720, 7, 7]
        
        if fixed_ratio is None:
            features, backward_indexes = self.encoder(img) # 默认0-1的随机fixed_ratio
        else: # 推理阶段，全0的img:[1, 54]被完全mask了，其实输入只有cls token
            features, backward_indexes = self.encoder(img, fixed_ratio=fixed_ratio) # features: [1, 1, 1024] , [54, 1]和img是token: [54, 1]
        predicted_img, predicted_rot, predicted_cam, mask = self.decoder(
            features, cond, backward_indexes, cond_dropout=cond_dropout
        )# [1, 54, 512] [1, 6] [1, 3] [1, 54]
        return predicted_img, predicted_rot.cpu(), predicted_cam.cpu(), mask

    def generate(
        self, cond, nb_steps=5, gen_temp=4.5, device="cuda", return_list=False
    ):
        cond = self.backbone(cond)
        batch_size = cond.shape[0]
        mask = torch.zeros(batch_size, self.encoder.seq_length).to(device)
        patches = torch.zeros(batch_size, self.encoder.seq_length, dtype=torch.int).to(
            device
        )

        if return_list:
            list_indices = []

        for step in range(nb_steps):
            ratio = 1.0 * (step + 1) / nb_steps
            annealed_temp = gen_temp * (1 - ratio)

            is_mask = mask == 0

            features, backward_indexes = self.encoder(patches, mask)
            logits, predicted_rot, predicted_cam, mask = self.decoder(
                features, cond, backward_indexes
            )

            # sampling & scoring
            sampled_ids = add_gumbel_noise(logits, annealed_temp).argmax(dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1
            )
            sampled_ids = torch.where(is_mask, sampled_ids, patches)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_ratio = np.cos(ratio * math.pi * 0.5)
            mask_len = torch.Tensor(
                [np.floor(self.encoder.seq_length * mask_ratio)]
            ).to(device)
            mask_len = torch.maximum(
                torch.Tensor([1]).to(device),
                torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len),
            )[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp)
            sorted_confidence, _ = torch.sort(confidence, axis=-1) # 置信度排序
            cut_off = sorted_confidence[:, mask_len.long() - 1 : mask_len.long()]
            masking = confidence <= cut_off
            patches = torch.where(masking, 0, sampled_ids)
            mask = torch.where(masking, 0, 1)
            if return_list:
                list_indices.append(patches)

        if return_list:
            return patches, predicted_rot, predicted_cam, list_indices
        return patches, predicted_rot, predicted_cam

    def load(self, path_model: str):
        checkpoint = torch.load(path_model)
        state_dict = checkpoint["model"]
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module" in k:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        self.load_state_dict(new_state_dict, strict=False)
        loss = checkpoint["loss"]
        print(f"\t [Model robustSMAE is loaded successfully with loss = {loss}]")


if __name__ == "__main__":
    spect = torch.ones(1, 54, dtype=torch.long)
    spect = spect.to("cuda")

    encoder = MAE_Encoder(seq_length=54, emb_dim=1280, num_layer=6)
    encoder = encoder.to("cuda")
    size_model(encoder)

    decoder = MAE_Decoder(seq_length=54, emb_dim=1280, dim_tokens=512)
    decoder = decoder.to("cuda")
    size_model(decoder)

    cond = torch.ones(1, 1, 1280).to("cuda")

    features_, backward_indexes_ = encoder(spect, cond)
    predicted_spec, predicted_rot, predicted_cam, mask = decoder(
        features_, backward_indexes_
    )

    print(predicted_spec.shape, predicted_rot.shape, predicted_cam.shape)
