# MaskDiff-HMR: Masked Generative Diffusion for Human Mesh Recovery

本仓库实现 **MaskDiff-HMR**（基于 CVQDiffusion 的人体网格重建），是在 [MEGA](https://github.com/g-fiche/MEGA) 框架基础上发展的条件式 VQ-Diffusion 模型。该方法将人体网格 token 化后，利用离散扩散模型以图像为条件从噪声中逐步重建人体姿态与形状。

> 相关论文：[MEGA: Masked Generative Autoencoder for Human Mesh Recovery](https://g-fiche.github.io/research-pages/mega/) (CVPR 2025)

---

## 目录

- [安装](#安装)
- [依赖的子模块](#依赖的子模块)
- [权重下载](#权重下载)
- [数据准备](#数据准备)
- [运行 Demo](#运行-demo)
- [训练](#训练)
- [测试](#测试)
- [文件结构](#文件结构)
- [引用](#引用)

---

## 安装

### 创建环境

```bash
conda create -n mega python=3.9
conda activate mega
```

### 安装 PyTorch

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 安装 PyTorch3D

PyTorch3D 用于网格可视化和旋转变换。手动下载 conda 包然后本地安装：

```bash
conda install --use-local pytorch3d-0.7.8-py39_cu121_pyt240.tar.bz2
```

也可以参照 [官方文档](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) 安装。

若无法安装 PyTorch3D，可使用 [RoMa](https://naver.github.io/roma/) 库替代旋转变换，并跳过 3D 可视化渲染。

### 安装其他依赖

```bash
python -m pip install -r requirements.txt
```

---

## 依赖的子模块

### Mesh-VQ-VAE —— 人体网格离散 Tokenizer

MaskDiff-HMR 依赖 **Mesh-VQ-VAE** 将人体网格编码为离散 token 序列（长度 54，codebook 大小 512）。

1. 克隆 [Mesh-VQ-VAE](https://github.com/g-fiche/Mesh-VQ-VAE) 到本仓库同级目录：

```bash
git clone https://github.com/g-fiche/Mesh-VQ-VAE
```

2. 将 `mesh_vq_vae` 文件夹复制到 MEGA 目录下。
3. 按照 Mesh-VQ-VAE 的说明下载全卷积网格自编码器的相关文件，放置到 `body_models` 目录。

### SMPL 人体模型

SMPL-H 模型用于获取人体网格。

1. 在 [MANO 官网](https://mano.is.tue.mpg.de/index.html) 注册账号。
2. 在 Downloads 页面下载 "Extended SMPL+H model (used in AMASS)"。
3. 将 `smplh.tar.xz` 放到 `body_models` 目录并解压。
4. 将 `male` 子文件夹重命名为 `m`，`female` 重命名为 `f`。

### 关节点回归矩阵

MEGA 是非参数化的，需要关节点回归矩阵来计算人体关节点位置：

- **SMPL 24 关节回归器**：从 SMPL 模型中提取 `J_regressor.npy`：
  ```bash
  cd body_models && unzip smplh/m/model.npz J_regressor.npy
  ```
  重命名为 `J_regressor_24.npy`。

- **Human3.6M 关节回归器**：从 [mmhuman3d](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/getting_started.md#body-model-preparation) 下载，重命名为 `J_regressor_h36m.npy`。

两个文件都放在 `body_models` 目录下。

### 连接矩阵（Mesh Convolution）

在 `body_models/ConnectionMatrices/` 目录下放置 Mesh-VQ-VAE 所需的连接矩阵文件。

### 预训练 Backbone

- **HRNet-W48**：下载 [HRNet 官方仓库](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) 的预训练模型，放到 `body_models` 目录并重命名为 `pose_hrnet_w48.pth`。
- **ViT**（可选）：从 [HMR2.0](https://github.com/shubham-goel/4D-Humans/tree/main) 下载 `vitpose_backbone.pth`。

### YOLOv8 人体检测器（Demo 用）

运行 demo 时，代码会自动下载 YOLOv8x 模型到 `body_models/yolov8x.pt`。也可手动下载。

---

## 权重下载

预训练权重下载地址：https://zenodo.org/records/14974141

- **Mesh-VQ-VAE**：`checkpoint/MESH_VQVAE/mesh_vqvae_54`
- **RotCam 权重**：`checkpoint/CVQMAE/rotcam_weights.pth`
- **CVQDiffusion 条件扩散模型**：待补充

下载后放置结构如下：

```
${checkpoint}
|-- MESH_VQVAE
|   |-- mesh_vqvae_54
|-- CVQMAE
|   |-- rotcam_weights.pth
|-- CVQDIFFUSION
|   |-- <experiment>/
|       |-- model_best_loss
```

---

## 数据准备

### 训练数据集

使用 [BEDLAM](https://bedlam.is.tue.mpg.de/index.html) 提供的 SMPL 标注。参照 [BEDLAM 训练说明](https://github.com/pixelite1201/BEDLAM/blob/master/docs/training.md) 中 `Training CLIFF model with real images` 部分的指引下载训练图像和标注。下载完成后，数据集目录结构如下：

下载完成后，数据集目录结构如下：

```
${dataset}
|-- coco
|   |-- train2014
|   |-- coco.npz
|-- mpii
|   |-- images
|   |-- mpii.npz
|-- h36m_train
|   |-- Images
|   |   |-- S1
|   |   |-- S2 ..
|   |-- h36m_train.npz
|-- mpi-inf-3dhp
|   |-- S1
|   |-- S2 ..
|   |-- mpi_inf_3dhp_train.npz
```

### 测试数据集

使用 [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) 和 [EMDB](https://eth-ait.github.io/emdb/)：
- 从官方网站下载数据集。
- 使用 [VQ-HPS](https://g-fiche.github.io/research-pages/vqhps/) 的 `preprocess_data` 中的脚本准备 `.npz` 标注文件。

### 数据集配置文件

数据集路径通过 `.txt` 文件指定（位于 `configs/config_cvqdiffusion/` 目录下），每行一个 `.npz` 文件的路径。

---

## 运行 Demo

使用训练好的 CVQDiffusion 模型对单张图像或多张图像进行人体网格重建：

```bash
python demo_vqdiffusion.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_path` | `demo_data/` | 输入图像目录或单张图像路径 |
| `--output_path` | `demo_out_vqdiffusion` | 输出目录（保存渲染覆盖图和 `.obj` 文件） |
| `--checkpoint` | 读取配置 | CVQDiffusion 权重路径 |

### 使用示例

```bash
# 处理 demo_data/ 目录下的所有图片
python demo_vqdiffusion.py

# 处理单张图片
python demo_vqdiffusion.py --input_path /path/to/image.jpg

# 指定输出目录和模型权重
python demo_vqdiffusion.py \
    --input_path demo_data/ \
    --output_path my_output \
    --checkpoint checkpoint/CVQDIFFUSION/xxx/model_best_loss
```

### Demo 流程

1. YOLOv8 自动检测图像中的人体。
2. HRNet-W48 backbone 提取图像特征。
3. CVQDiffusion 进行条件扩散采样（从完全掩码开始逐步去噪），得到人体网格 token 序列。
4. Mesh-VQ-VAE 解码 token 为人体网格。
5. 渲染网格覆盖在原始图像上并保存。

---
## 训练

### 基础训练

```bash
python train_vq_diffusion.py
```

使用 Hydra 管理配置，默认读取 `configs/config_cvqdiffusion/config_hrnet_large.yaml`。

### 指定配置

```bash
python train_vq_diffusion.py --config-name config_hrnet_large
```

### 覆盖配置项

```bash
python train_vq_diffusion.py train.batch=16 train.lr=1e-4 \
    training_data.file=configs/config_cvqdiffusion/coco_train_100.txt \
    validation_data.file=configs/config_cvqdiffusion/mpii_train_100.txt
```

### 从 checkpoint 恢复训练

从配置文件加载：

```bash
python train_vq_diffusion.py resume.path=checkpoint/CVQDIFFUSION/xxx/model_checkpoint
```

从命令行恢复时，需要同时设置 `resume.path` 并重新加载 rotcam 参数（训练脚本会自动加载）：

```bash
python train_vq_diffusion.py resume.path=checkpoint/CVQDIFFUSION/xxx/model_checkpoint resume.optimizer=true
```

### 训练配置说明

主要配置项（`configs/config_cvqdiffusion/config_hrnet_large.yaml`）：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `train.batch` | 16 | 批次大小 |
| `train.lr` | 1e-4 | 学习率 |
| `train.total_epoch` | 119 | 总训练轮数 |
| `train.warmup_epoch` | 10 | warmup 轮数 |
| `train.weight_decay` | 1e-4 | 权重衰减 |
| `model.n_layer` | 12 | Transformer 层数 |
| `model.n_emb` | 1024 | Transformer 隐层维度 |
| `model.n_head` | 8 | 注意力头数 |
| `model.diff_step` | 10 | 扩散步数 |
| `save_every_n_steps` | 5000 | 每 N 步保存 checkpoint |

---

## 测试

### 确定性评估（单次采样）

对验证集进行确定性评估（扩散采样时 `filter_ratio=0`，无随机性）：

```bash
python test_vq_diffusion.py -p datasets resume.path=checkpoint/CVQDIFFUSION/xxx/model_best_loss
```

参数说明：
- `-p / --path`：数据集根目录路径（H5 文件所在位置）
- `resume.path`：checkpoint 路径

### 覆盖测试配置

```bash
python test_vq_diffusion.py -p datasets \
    resume.path=checkpoint/CVQDIFFUSION/xxx/model_best_loss \
    train.batch=4
```

测试脚本会输出：
- **V2V 误差**（mm）
- **平均推理时间**（s）
- 模型参数量统计
- 评估结果保存路径

---

## 模型架构概述

MaskDiff-HMR 的总体流程：

```
输入图像
    ↓
HRNet-W48 Backbone → [B, 720, 7, 7] 特征图
    ↓
Condition Embedding → [B, 49, 1024] 条件序列 (+ 位置编码)
    ↓
DiffusionTransformer (12层, 8头, 1024维)
  - 以掩码扩散方式从噪声 token 逐步重建 [B, 54] 离散 token
  - 使用 cross-attention 注入图像条件
    ↓
Mesh-VQ-VAE Decoder → [B, 6890, 3] 人体网格顶点
    ↓
RotNet → 全局旋转 |  CamHead → 相机参数
```

关键组件：
- **Visual Tokenizer** (Mesh-VQ-VAE)：将人体网格压缩为 54 个离散 token，codebook 大小 512
- **Backbone**：HRNet-W48 提取图像特征
- **Condition Encoder**：将图像特征投影为条件序列
- **Diffusion Transformer**：masked discrete diffusion 模型，学习条件分布 $p(x_{1:T} | \text{image})$
- **RotNet & CamHead**：回归全局旋转和相机参数，用于渲染

---

## 文件结构

```
MEGA/
├── train_vq_diffusion.py              # 条件扩散模型训练入口
├── demo_vqdiffusion.py                # Demo 推理
├── test_vq_diffusion.py               # 确定性测试评估
├── configs/
│   └── config_cvqdiffusion/           # 条件扩散模型配置
│       ├── config_hrnet_large.yaml     # 大规模训练配置
│       ├── config_hrnet.yaml           # 小规模训练配置
│       ├── 3dpw_test.txt               # 数据集路径文件
│       ├── emdb_train.txt
│       └── ...
├── mega/
│   ├── model/
│   │   ├── cvqdiffusion/
│   │   │   ├── cvqdiffusion.py          # CVQDiffusion 模型定义
│   │   │   └── __init__.py
│   ├── train/
│   │   ├── train_cvqdiffusion/
│   │   │   ├── train_class.py           # 训练/评估逻辑
│   │   │   └── train.py                 # 辅助函数
│   ├── data/
│   │   └── dataset_demo.py
│   ├── utils/
│   │   ├── demo_renderer.py             # Demo 渲染器
│   │   └── img_renderer.py              # 图像渲染器
│   └── __init__.py
├── scripts/
│   └── extract_rotcam_weights.py        # 提取 RotCam 权重
├── body_models/                         # 人体模型和预训练权重
│   ├── smplh/
│   ├── pose_hrnet_w48.pth
│   ├── J_regressor_24.npy
│   ├── J_regressor_h36m.npy
│   └── ConnectionMatrices/
├── checkpoint/                          # 权重下载目录
│   ├── MESH_VQVAE/
│   ├── CVQMAE/
│   └── CVQDIFFUSION/
├── Mesh-VQ-VAE/                         # Mesh-VQ-VAE 子模块
├── datasets/                            # 数据集
└── requirements.txt
```

---

## 引用

```bibtex
@inproceedings{fiche2024vq,
    title={MEGA: Masked Generative Autoencoder for Human Mesh Recovery},
    author={Fiche, Gu{\'e}nol{\'e} and Leglaive, Simon and Alameda-Pineda, Xavier and Moreno-Noguer, Francesc},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition ({CVPR})},
    year={2025}
}
```
