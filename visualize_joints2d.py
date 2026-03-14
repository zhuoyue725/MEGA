"""
可视化2D关节点的脚本
用于调试和检查重投影损失中的关节点对齐情况
"""
import torch
import numpy as np
import os
from mega.utils.loss import visualize_joints_2d, reprojection_loss, orthographic_projection

def test_visualization():
    """测试可视化功能"""
    # 创建一些示例数据
    num_joints = 17  # Human3.6M 17个关节点
    batch_size = 4
    
    # 生成随机的2D关节点 (范围在 [-0.5, 0.5])
    gt_2d = torch.randn(batch_size, num_joints, 2) * 0.3
    pred_2d = gt_2d + torch.randn(batch_size, num_joints, 2) * 0.05  # 添加一些噪声
    
    # 创建输出目录
    output_dir = "./demo_out/joints2d"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化每个batch的样本
    for i in range(batch_size):
        save_path = os.path.join(output_dir, f"test_sample_{i}.png")
        visualize_joints_2d(pred_2d[i], gt_2d[i], save_path)
        print(f"Saved visualization {i+1}/{batch_size}")
    
    print(f"\n所有可视化已保存到: {output_dir}")
    print("绿色: 真实关节点 (GT)")
    print("红色: 预测关节点 (Pred)")
    print("每个关节点旁边标注了序号")


def visualize_from_training_data(gt_2d, pred_v, pred_cam, joints_reg, output_dir="./demo_out/joints2d", prefix="train"):
    """
    从训练数据中可视化关节点
    
    Args:
        gt_2d: 真实2D关节点 [B, N, 2]
        pred_v: 预测的顶点 [B, V, 3]
        pred_cam: 预测的相机参数 [B, 3]
        joints_reg: 关节回归器 [N, V]
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算预测的2D关节点
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    
    # 可视化第一个样本
    save_path = os.path.join(output_dir, f"{prefix}_joints2d_comparison.png")
    visualize_joints_2d(pred_2d, gt_2d, save_path)
    print(f"Saved joints visualization to {save_path}")
    
    # 计算并打印每个关节的误差
    if torch.is_tensor(pred_2d):
        pred_2d_np = pred_2d[0].detach().cpu().numpy()
        gt_2d_np = gt_2d[0].detach().cpu().numpy()
    else:
        pred_2d_np = pred_2d[0]
        gt_2d_np = gt_2d[0]
    
    errors = np.linalg.norm(pred_2d_np - gt_2d_np, axis=1)
    print("\n每个关节的误差 (归一化坐标):")
    for i, error in enumerate(errors):
        print(f"  关节 {i:2d}: {error:.4f}")
    print(f"  平均误差: {errors.mean():.4f}")
    print(f"  最大误差: {errors.max():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("2D关节点可视化测试")
    print("=" * 60)
    test_visualization()
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("在训练代码中，可以这样使用可视化功能:")
    print()
    print("方法1: 直接在loss计算时可视化")
    print("-------")
    print("from mega.utils.loss import reprojection_loss")
    print()
    print("loss = reprojection_loss(")
    print("    gt_2d, pred_v, pred_cam, joints_reg,")
    print("    visualize=True,")
    print("    vis_path='./demo_out/joints2d',")
    print("    vis_counter=step")
    print(")")
    print()
    print("方法2: 单独调用可视化函数")
    print("-------")
    print("from mega.utils.loss import visualize_joints_2d, orthographic_projection")
    print()
    print("# 计算pred_2d")
    print("J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)")
    print("pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)")
    print("pred_2d = orthographic_projection(pred_3dkpt, pred_cam)")
    print()
    print("# 可视化")
    print("visualize_joints_2d(pred_2d, gt_2d, './demo_out/joints2d/comparison.png')")
    print()
    print("=" * 60)
