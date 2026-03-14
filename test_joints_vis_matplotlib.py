"""
使用matplotlib的2D关节点可视化测试
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_joints_2d_matplotlib(pred_2d, gt_2d, save_path):
    """
    可视化预测的2D关节点和真实的2D关节点
    Args:
        pred_2d: 预测的2D关节点 [N, 2]，范围在 [-0.5, 0.5]
        gt_2d: 真实的2D关节点 [N, 2]，范围在 [-0.5, 0.5]
        save_path: 保存路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 定义关节连接关系 (Human3.6M 17关节点骨架)
    skeleton = [
        [0, 1], [1, 2], [2, 3],  # 右腿
        [0, 4], [4, 5], [5, 6],  # 左腿
        [0, 7], [7, 8], [8, 9], [9, 10],  # 脊柱到头部
        [8, 11], [11, 12], [12, 13],  # 右臂
        [8, 14], [14, 15], [15, 16],  # 左臂
    ]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制真实关节点的骨架 (绿色)
    for connection in skeleton:
        pt1 = gt_2d[connection[0]]
        pt2 = gt_2d[connection[1]]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=2, alpha=0.7)
    
    # 绘制预测关节点的骨架 (红色)
    for connection in skeleton:
        pt1 = pred_2d[connection[0]]
        pt2 = pred_2d[connection[1]]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=2, alpha=0.7)
    
    # 绘制真实关节点 (绿色圆圈)
    ax.scatter(gt_2d[:, 0], gt_2d[:, 1], c='green', s=100, zorder=5, label='GT')
    for i, pt in enumerate(gt_2d):
        ax.annotate(str(i), (pt[0] + 0.02, pt[1] + 0.02), 
                   fontsize=8, color='darkgreen', weight='bold')
    
    # 绘制预测关节点 (红色圆圈)
    ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c='red', s=100, zorder=5, label='Pred')
    for i, pt in enumerate(pred_2d):
        ax.annotate(str(i), (pt[0] - 0.05, pt[1] - 0.02), 
                   fontsize=8, color='darkred', weight='bold')
    
    # 设置坐标轴
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title('2D Joint Comparison: Green=GT, Red=Pred', fontsize=14, weight='bold')
    ax.set_xlabel('X coordinate (normalized)', fontsize=12)
    ax.set_ylabel('Y coordinate (normalized)', fontsize=12)
    
    # 反转Y轴，使其与图像坐标系一致
    ax.invert_yaxis()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved joints visualization to {save_path}")


def test_visualization():
    """测试可视化功能"""
    # 创建一些示例数据
    num_joints = 17  # Human3.6M 17个关节点
    num_samples = 4
    
    # 创建输出目录
    output_dir = "./demo_out/joints2d"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成随机的2D关节点 (范围在 [-0.5, 0.5])
    np.random.seed(42)
    
    print("\n生成测试样本...")
    for i in range(num_samples):
        gt_2d = np.random.randn(num_joints, 2) * 0.3
        pred_2d = gt_2d + np.random.randn(num_joints, 2) * 0.05  # 添加一些噪声
        
        save_path = os.path.join(output_dir, f"test_sample_{i}.png")
        visualize_joints_2d_matplotlib(pred_2d, gt_2d, save_path)
        
        # 计算并打印每个关节的误差
        errors = np.linalg.norm(pred_2d - gt_2d, axis=1)
        print(f"\n样本 {i} 的关节误差 (归一化坐标):")
        for j in range(min(5, num_joints)):  # 只打印前5个关节
            print(f"  关节 {j:2d}: {errors[j]:.4f}")
        print(f"  ...")
        print(f"  平均误差: {errors.mean():.4f}")
        print(f"  最大误差: {errors.max():.4f}")
    
    print(f"\n{'='*60}")
    print(f"所有可视化已保存到: {output_dir}")
    print("绿色: 真实关节点 (GT)")
    print("红色: 预测关节点 (Pred)")
    print("每个关节点旁边标注了序号 (0-16)")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("2D关节点可视化测试 (matplotlib版)")
    print("=" * 60)
    test_visualization()
