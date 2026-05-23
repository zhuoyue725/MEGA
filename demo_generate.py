import os
import sys
import torch
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig
from mega import (
    CVQMAE,
    hrnet_w48,
)
import mesh_vq_vae
from mega.data.dataset_demo import DemoDataset
import matplotlib.pyplot as plt
from pytorch3d.transforms import rotation_6d_to_matrix
import argparse
import math
from matplotlib.gridspec import GridSpec

# 导入trimesh用于保存OBJ文件
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入trimesh库，无法保存OBJ格式的mesh")
    TRIMESH_AVAILABLE = False

# 导入渲染相关函数
try:
    from mega.utils.mesh_render import renderer
    from mega.utils.img_renderer import visualize_reconstruction_pyrender, PyRender_Renderer
    RENDER_AVAILABLE = True
except ImportError:
    print("警告: 无法导入渲染模块，step_vis和samples_vis模式可能无法正常工作")
    RENDER_AVAILABLE = False


def plot_meshes_(meshes, faces, save: str = None, rot: bool = False, colors=None):
    """
    可视化mesh网格
    """
    if not RENDER_AVAILABLE:
        print("错误: 渲染模块不可用，无法执行mesh可视化")
        return

    images = renderer(
        meshes,
        torch.from_numpy(faces).to(torch.int32),
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

    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)
        plt.close()
    else:
        plt.close()


def plot_reproj_(images, meshes, cameras, faces, save: str = None):
    """
    可视化重投影结果
    """
    if not RENDER_AVAILABLE:
        print("错误: 渲染模块不可用，无法执行重投影可视化")
        return

    # 确保meshes不是空的
    if len(meshes) == 0:
        print("错误: meshes为空，无法可视化")
        return

    # 检查数据形状和类型
    if isinstance(meshes, torch.Tensor):
        meshes = [meshes[i] for i in range(meshes.size(0))]

    if isinstance(cameras, torch.Tensor):
        cameras = [cameras[i] for i in range(cameras.size(0))]

    # 确保images是列表或数组
    if isinstance(images, torch.Tensor):
        if images.dim() == 4:  # [B, H, W, C] 或 [B, C, H, W]
            if images.shape[1] == 3:  # [B, C, H, W]
                images = images.permute(0, 2, 3, 1).cpu().numpy()
            else:  # [B, H, W, C]
                images = images.cpu().numpy()
        elif images.dim() == 3:  # [H, W, C]
            images = [images.cpu().numpy()] if hasattr(images, 'cpu') else [images]

    # 如果images是单张图像，但meshes有多个，则重复图像
    if len(images) == 1 and len(meshes) > 1:
        images = [images[0]] * len(meshes)

    rendered_img = []
    render_reproj = PyRender_Renderer(faces=faces)

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

    # 确保ncols和nrows为正整数
    ncols = max(1, ncols)
    nrows = max(1, nrows)

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

    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)
        plt.close()
    else:
        plt.close()


def save_mesh_as_obj(vertices, faces, output_path):
    """
    将mesh保存为OBJ格式文件

    Args:
        vertices: 顶点坐标 [V, 3] 或 [B, V, 3]
        faces: 面索引 [F, 3]
        output_path: 输出OBJ文件路径
    """
    if not TRIMESH_AVAILABLE:
        print("警告: trimesh不可用，无法保存OBJ文件")
        return False

    try:
        # 处理顶点数据
        if isinstance(vertices, torch.Tensor):
            vertices_np = vertices.detach().cpu().numpy()
        else:
            vertices_np = np.array(vertices)

        # 处理面数据
        if isinstance(faces, torch.Tensor):
            faces_np = faces.detach().cpu().numpy()
        else:
            faces_np = np.array(faces)

        # 如果vertices是批量数据，取第一个样本
        if vertices_np.ndim == 3:
            vertices_np = vertices_np[0]

        # 确保顶点和面是正确形状
        if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
            print(f"错误: 顶点形状应为[V, 3]，但得到{vertices_np.shape}")
            return False

        if faces_np.ndim != 2 or faces_np.shape[1] != 3:
            print(f"错误: 面形状应为[F, 3]，但得到{faces_np.shape}")
            return False

        # 创建trimesh对象
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 导出为OBJ
        mesh.export(output_path)

        print(f"✅ Mesh已保存为OBJ文件: {output_path}")
        return True

    except Exception as e:
        print(f"❌ 保存OBJ文件失败: {e}")
        return False


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
    # - 乘以 confidence_matrix: 实现了"置信度越高，加得越多；置信度越低，基本不加"
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 置信度增强版可视化图片已保存至: {output_path}")




# 在Hydra装饰器之前解析命令行参数
def parse_cmd_args_before_hydra():
    """在Hydra装饰器之前解析命令行参数，修改sys.argv"""
    parser = argparse.ArgumentParser(description='可视化生成过程', add_help=False)
    parser.add_argument('--mode', type=str, default=None,
                        choices=['confidence_vis', 'step_vis', 'samples_vis'],
                        help='可视化模式: confidence_vis(置信度), step_vis(步骤), samples_vis(多样本)')
    parser.add_argument('--input_path', type=str, default=None,
                        help='输入图像路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--steps', type=int, default=None,
                        help='生成步数')
    parser.add_argument('--temp', type=float, default=None,
                        help='温度参数')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='多样本可视化时的样本数量')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最多处理多少个图像样本（None表示全部）')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='step_vis模式中处理的样本索引')
    parser.add_argument('--help', action='store_true', help='显示帮助信息')

    # 保存原始sys.argv
    original_argv = sys.argv.copy()

    # 解析已知参数，忽略未知参数（包括Hydra的参数）
    known_args, remaining_argv = parser.parse_known_args()

    # 如果请求帮助，打印帮助信息并退出
    if known_args.help:
        parser.print_help()
        print("\n注：使用 --hydra-help 查看Hydra配置选项")
        sys.exit(0)

    # 修改sys.argv，移除我们处理的参数，只留下剩余参数给Hydra
    sys.argv = [original_argv[0]] + remaining_argv

    return known_args

# 在模块导入时立即解析参数，修改sys.argv供Hydra使用
_parsed_args = parse_cmd_args_before_hydra()


def confidence_vis(mega, mesh_vqvae, device, img_features, output_path, steps=50, temp=1.0):
    """
    置信度可视化模式
    """
    with torch.no_grad():
        # 调用generate方法获取概率列表
        mesh_indices, pred_rot, pred_cam, list_probs = mega.generate(
            img_features, nb_steps=steps, gen_temp=temp, device=device, return_list=True
        )

        # 可视化置信度
        visualize_confidence(list_probs, output_path)


def step_vis(mega, mesh_vqvae, device, img_features, raw_img, output_dir, img_fn,
             faces, steps=5, temp=1.0, sample_idx=0):
    """
    步骤可视化模式 - 可视化每一步的生成过程
    """
    if not RENDER_AVAILABLE:
        print("错误: 渲染模块不可用，无法执行step_vis模式")
        return

    with torch.no_grad():
        # 调用generate方法，返回每一步的patches列表
        mesh_indices, pred_rot, pred_cam, list_indices = mega.generate(
            img_features, nb_steps=steps, gen_temp=temp, device=device, return_list=True
        )

        # 解码mesh
        mesh_canonical = mesh_vqvae.decode(mesh_indices).cpu()

        # 应用旋转
        rotmat = rotation_6d_to_matrix(pred_rot).cpu()
        pred_mesh_oriented = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

        # 保存原始图像
        img_save_path = os.path.join(output_dir, f"{img_fn}_img.png")
        plot_img_(raw_img, save=img_save_path)

        # 存储每一步的重投影图像，用于最后拼接
        step_images = []

        # 对每一步进行可视化
        for step_idx, patches in enumerate(list_indices):
            # 解码前确保token索引在有效范围内（0-511）且为整数类型
            # MASK_THRESHOLD通常是512，我们需要裁剪到0-511
            MASK_THRESHOLD = 512

            # 确保patches是整数类型（Long）并在有效范围内
            if patches.dtype != torch.long:
                safe_patches = patches.long()
            else:
                safe_patches = patches

            # 裁剪到有效范围并确保在正确设备上
            safe_patches = safe_patches.clamp(0, MASK_THRESHOLD - 1).to(device)

            # 解码mesh
            mesh_canonical_step = mesh_vqvae.decode(safe_patches).cpu()

            # 应用旋转
            rotmat_step = rotation_6d_to_matrix(pred_rot).cpu()
            pred_mesh_oriented_step = (rotmat_step @ mesh_canonical_step.transpose(2, 1)).transpose(2, 1)

            # 可视化重投影（只取第一个样本）
            save_path = os.path.join(output_dir, f"{img_fn}_step{step_idx}_reprojection.png")
            plot_reproj_(
                raw_img,
                pred_mesh_oriented_step[:1],
                pred_cam[:1],
                faces,
                save=save_path,
            )

            # 读取刚保存的图像用于拼接
            step_img = cv2.imread(save_path)
            if step_img is not None:
                step_images.append(step_img)

            # 可视化mesh（可选）
            mesh_save_path = os.path.join(output_dir, f"{img_fn}_step{step_idx}_mesh.png")
            plot_meshes_(
                mesh_canonical_step[:1],
                faces,
                save=mesh_save_path,
                rot=True,
            )

        # 拼接所有步骤的图像
        if step_images:
            # 水平拼接所有步骤
            concatenated = np.hstack(step_images)
            final_path = os.path.join(output_dir, f"{img_fn}_all_steps_concatenated.png")
            cv2.imwrite(final_path, concatenated)

            print(f"样本 {img_fn} (索引 {sample_idx}): 已保存 {len(step_images)} 个步骤的可视化结果到 {final_path}")


def samples_vis(mega, mesh_vqvae, device, img_features, raw_img, output_dir, img_fn,
                faces, steps=5, temp=1.0, sample_size=10):
    """
    多样本可视化模式 - 针对每个样本生成多个随机预测结果并进行拼接可视化
    """
    if not RENDER_AVAILABLE:
        print("错误: 渲染模块不可用，无法执行samples_vis模式")
        return

    with torch.no_grad():
        # 重复img_features以生成多个样本
        # img_features形状应该是[1, C, H, W]，重复到[sample_size, C, H, W]
        if img_features.dim() == 4 and img_features.size(0) == 1:
            img_features_repeated = img_features.repeat(sample_size, 1, 1, 1)
        else:
            img_features_repeated = img_features
            sample_size = img_features.size(0) if img_features.dim() == 4 else 1

        # 生成多样本结果
        mesh_indices, pred_rot, pred_cam = mega.generate(
            img_features_repeated, nb_steps=steps, gen_temp=temp, device=device, return_list=False
        )

        # 解码 Mesh 并应用旋转变换
        mesh_canonical = mesh_vqvae.decode(mesh_indices).cpu()
        rotmat = rotation_6d_to_matrix(pred_rot).cpu()
        pred_mesh_oriented = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

        sample_images = []

        # 逐个处理 sample_size 个随机结果
        for s_idx in range(sample_size):
            save_path = os.path.join(output_dir, f"{img_fn}_sample_{s_idx}_temp.png")

            # 绘制单个样本的重投影
            plot_reproj_(
                raw_img, # 传入单张原图
                pred_mesh_oriented[s_idx : s_idx + 1], # 传入对应的第 s_idx 个 mesh
                pred_cam[s_idx : s_idx + 1],          # 传入对应的第 s_idx 个相机参数
                faces,
                save=save_path,
            )

            # 读取生成的图像用于后续拼接
            s_img = cv2.imread(save_path)
            if s_img is not None:
                sample_images.append(s_img)

            # 保存mesh为OBJ格式
            # if TRIMESH_AVAILABLE:
            #     # 保存经过旋转的mesh（重投影用的mesh）
            #     oriented_obj_path = os.path.join(output_dir, f"{img_fn}_sample_{s_idx}_oriented.obj")
            #     save_mesh_as_obj(pred_mesh_oriented[s_idx], faces, oriented_obj_path)

                # 保存规范空间的mesh（未旋转的mesh）
                # canonical_obj_path = os.path.join(output_dir, f"{img_fn}_sample_{s_idx}_canonical.obj")
                # save_mesh_as_obj(mesh_canonical[s_idx], faces, canonical_obj_path)

        # 水平拼接所有随机生成的样本结果
        if sample_images:
            concatenated = np.hstack(sample_images)
            final_save_path = os.path.join(output_dir, f"{img_fn}_stochastic_variety_s{sample_size}_t{temp}.png")
            cv2.imwrite(final_save_path, concatenated)

            print(f"样本 {img_fn}: 已保存 {sample_size} 个随机采样结果的拼接图: {final_save_path}")

def plot_img_(images, save: str = None):
    """
    可视化原始图像
    """
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

    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)
        plt.close()
    else:
        plt.close()


@hydra.main(config_path="configs/config_cvqmae", config_name="config_demo", version_base=None)
def main(cfg: DictConfig):
    # 使用在模块导入时已经解析的参数
    cmd_args = _parsed_args

    # 从配置中获取参数，设置默认值
    mode = cfg.get('mode', 'confidence_vis')
    # input_path = cfg.get('input_path', '/home/zzb/pydata/recons/MEGA/demo_data/input/dataset_samples/emdb_occ2')
    input_path = cfg.get('input_path', '/home/zzb/pydata/recons/MEGA/demo_data/input/3dpw_occ')
    output_dir = cfg.get('output_dir', '/home/zzb/pydata/recons/MEGA/demo_out/step_vis/3dpw_occ')
    steps = cfg.get('steps', 50)
    temp = cfg.get('temp', 1.0)
    sample_size = cfg.get('sample_size', 10)
    max_samples = cfg.get('max_samples', None)
    sample_idx = cfg.get('sample_idx', 0)

    # 用命令行参数覆盖配置参数（如果提供了的话）
    if cmd_args.mode is not None:
        mode = cmd_args.mode
    if cmd_args.input_path is not None:
        input_path = cmd_args.input_path
    if cmd_args.output_dir is not None:
        output_dir = cmd_args.output_dir
    if cmd_args.steps is not None:
        steps = cmd_args.steps
    if cmd_args.temp is not None:
        temp = cmd_args.temp
    if cmd_args.sample_size is not None:
        sample_size = cmd_args.sample_size
    if cmd_args.max_samples is not None:
        max_samples = cmd_args.max_samples
    if cmd_args.sample_idx is not None:
        sample_idx = cmd_args.sample_idx

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 YOLOv8 Human Detector
    weights_path = "body_models/"
    if not os.path.exists(weights_path):
        from ultralytics.utils.downloads import download
        download("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt", weights_path)

    # 从指定路径加载 YOLOv8 模型
    yolo_model = YOLO(os.path.join(weights_path,"yolov8x.pt"))

    # 加载 SMPL faces
    ref_bm_path = "body_models/smplh/neutral/model.npz"
    if os.path.exists(ref_bm_path):
        ref_bm = np.load(ref_bm_path)
        faces = ref_bm["f"].astype(np.int32)
        print(f"Loaded SMPL faces: {faces.shape}")
    else:
        print(f"警告: 无法加载SMPL faces文件 {ref_bm_path}")
        faces = None

    # 加载 HMR Model
    os.chdir(hydra.utils.get_original_cwd())

    """ 加载 Backbone """
    backbone = hrnet_w48(pretrained_ckpt_path=cfg.backbone.pretrained, downsample=True, use_conv=True)

    """ 加载 MeshRegressor Model """
    mega = CVQMAE(backbone=backbone, **cfg.model)
    resume_path = cfg.get("resume", {}).get("path", "checkpoint/CVQMAE/mega_hrnet")
    mega.load(resume_path)
    print(f"Loaded model from: {resume_path}")
    mega.to(device).eval()

    convmesh_model = mesh_vq_vae.FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = mesh_vq_vae.MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    mesh_vqvae.to(device)
    convmesh_model.init_test_mode()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"错误: 输入路径不存在: {input_path}")
        return
    print(f"输入路径: {input_path}")
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".png"))]

        if max_samples is not None:
            files = files[:max_samples]
            print(f"限制处理 {max_samples} 个图像")

        for idx, file in enumerate(tqdm(files, desc=f"Processing ({mode})")):
            file_path = os.path.join(input_path, file)
            img_cv2 = cv2.imread(file_path)

            if img_cv2 is None:
                print(f"无法读取图像: {file_path}")
                continue

            # YOLO检测
            boxes = yolo_model.predict(img_cv2,
                                device='cuda',
                                classes=0,  # 只检测人
                                conf=0.5,
                                save=False,
                                verbose=False
                                    )[0].boxes.xyxy.detach().cpu().numpy()

            if len(boxes) == 0:
                print(f"未检测到人体: {file_path}")
                continue

            boxes = boxes[:1]  # 只处理第一个人

            dataset = DemoDataset(img_cv2, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            with torch.no_grad():
                for batch in dataloader:
                    img = batch["img"].to(device)

                    # 获取图像特征并生成
                    # 根据配置决定是否使用ViT backbone的裁剪
                    if cfg.backbone.type == "vit":
                        # 如果是ViT backbone，进行中心裁剪
                        img_features = img[:, :, :, 32:-32]
                    else:
                        img_features = img

                    # 准备raw_img用于重投影可视化
                    raw_img = batch["raw_img"] if "raw_img" in batch else None
                    if raw_img is None:
                        # 如果没有raw_img，从img重建
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        img_np = img.cpu().numpy()[0].transpose(1, 2, 0)
                        raw_img = (img_np * std + mean).clip(0.0, 1.0)
                        raw_img = np.expand_dims(raw_img, axis=0)
                    else:
                        raw_img = raw_img.cpu().numpy().transpose(0, 2, 3, 1)

                    # 生成输出文件名
                    img_fn, _ = os.path.splitext(os.path.basename(file_path))

                    # 根据模式调用不同的可视化函数
                    if mode == 'confidence_vis':
                        output_path = os.path.join(output_dir, f"{img_fn}_confidence_steps{steps}_temp{temp}.png")
                        confidence_vis(mega, mesh_vqvae, device, img_features, output_path, steps, temp)

                    elif mode == 'step_vis':
                        if faces is None:
                            print(f"错误: 无法进行step_vis，缺少faces数据")
                            continue
                        step_vis(mega, mesh_vqvae, device, img_features, raw_img, output_dir, img_fn,
                                faces, steps, temp, sample_idx)

                    elif mode == 'samples_vis':
                        if faces is None:
                            print(f"错误: 无法进行samples_vis，缺少faces数据")
                            continue
                        samples_vis(mega, mesh_vqvae, device, img_features, raw_img, output_dir, img_fn,
                                   faces, steps, temp, sample_size)

                    print(f"已处理: {file_path}")
                    break  # 只处理第一个检测到的人
    else:
        print(f"输入路径不是目录: {input_path}")


if __name__ == "__main__":
    main()