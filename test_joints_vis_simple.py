"""
简单的2D关节点可视化测试（不依赖torch）
"""
import cv2
import numpy as np
import os


def visualize_joints_2d_simple(pred_2d, gt_2d, save_path, img_size=512):
    """
    可视化预测的2D关节点和真实的2D关节点
    Args:
        pred_2d: 预测的2D关节点 [N, 2]，范围在 [-0.5, 0.5]
        gt_2d: 真实的2D关节点 [N, 2]，范围在 [-0.5, 0.5]
        save_path: 保存路径
        img_size: 图像大小
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将坐标从 [-0.5, 0.5] 转换到图像坐标 [0, img_size]
    pred_2d_img = (pred_2d + 0.5) * img_size
    gt_2d_img = (gt_2d + 0.5) * img_size
    
    # 创建白色背景图像
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 定义关节连接关系 (Human3.6M 17关节点骨架)
    skeleton = [
        [0, 1], [1, 2], [2, 3],  # 右腿
        [0, 4], [4, 5], [5, 6],  # 左腿
        [0, 7], [7, 8], [8, 9], [9, 10],  # 脊柱到头部
        [8, 11], [11, 12], [12, 13],  # 右臂
        [8, 14], [14, 15], [15, 16],  # 左臂
    ]
    
    # 绘制真实关节点的骨架 (绿色)
    for connection in skeleton:
        pt1 = tuple(gt_2d_img[connection[0]].astype(int))
        pt2 = tuple(gt_2d_img[connection[1]].astype(int))
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    # 绘制预测关节点的骨架 (红色)
    for connection in skeleton:
        pt1 = tuple(pred_2d_img[connection[0]].astype(int))
        pt2 = tuple(pred_2d_img[connection[1]].astype(int))
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    
    # 绘制真实关节点 (绿色圆圈)
    for i, pt in enumerate(gt_2d_img):
        pt = tuple(pt.astype(int))
        cv2.circle(img, pt, 5, (0, 255, 0), -1)
        cv2.putText(img, str(i), (pt[0] + 8, pt[1] + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    
    # 绘制预测关节点 (红色圆圈)
    for i, pt in enumerate(pred_2d_img):
        pt = tuple(pt.astype(int))
        cv2.circle(img, pt, 5, (255, 0, 0), -1)
        cv2.putText(img, str(i), (pt[0] - 15, pt[1] - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
    
    # 添加图例
    cv2.putText(img, "Green: GT", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, "Red: Pred", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 保存图像
    cv2.imwrite(save_path, img)
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
    
    for i in range(num_samples):
        gt_2d = np.random.randn(num_joints, 2) * 0.3
        pred_2d = gt_2d + np.random.randn(num_joints, 2) * 0.05  # 添加一些噪声
        
        save_path = os.path.join(output_dir, f"test_sample_{i}.png")
        visualize_joints_2d_simple(pred_2d, gt_2d, save_path)
        
        # 计算并打印每个关节的误差
        errors = np.linalg.norm(pred_2d - gt_2d, axis=1)
        print(f"\n样本 {i} 的关节误差 (归一化坐标):")
        print(f"  平均误差: {errors.mean():.4f}")
        print(f"  最大误差: {errors.max():.4f}")
    
    print(f"\n{'='*60}")
    print(f"所有可视化已保存到: {output_dir}")
    print("绿色: 真实关节点 (GT)")
    print("红色: 预测关节点 (Pred)")
    print("每个关节点旁边标注了序号")


if __name__ == "__main__":
    print("=" * 60)
    print("2D关节点可视化测试 (简化版)")
    print("=" * 60)
    test_visualization()
