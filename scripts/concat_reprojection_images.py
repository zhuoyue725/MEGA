"""
横向拼接不同迭代次数的重投影图像
用于比较1次、5次、10次和20次迭代的结果
"""
import os
from pathlib import Path
from PIL import Image
import numpy as np


def concat_images_horizontal(image_paths, output_path):
    """
    横向拼接多张图像
    
    Args:
        image_paths: 图像路径列表，按顺序拼接
        output_path: 输出路径
    """
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            images.append(Image.open(img_path))
        else:
            print(f"警告: 图像不存在 {img_path}")
            return False
    
    if not images:
        return False
    
    # 获取所有图像的高度，使用最大高度
    heights = [img.height for img in images]
    max_height = max(heights)
    
    # 计算总宽度
    total_width = sum(img.width for img in images)
    
    # 创建新图像
    concatenated = Image.new('RGB', (total_width, max_height))
    
    # 拼接图像
    x_offset = 0
    for img in images:
        # 如果高度不一致，居中对齐
        y_offset = (max_height - img.height) // 2
        concatenated.paste(img, (x_offset, y_offset))
        x_offset += img.width
    
    # 保存结果
    concatenated.save(output_path)
    return True


def main():
    # 基础路径
    base_dir = Path("/home/zzb/pydata/recons/MEGA/checkpoint/CVQMAE/3dpw_step_rand_mask")
    
    # 迭代次数列表
    steps = [1, 5, 10, 20]
    
    # 输出目录
    output_dir = base_dir / "concate_v0"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取step1目录下所有的reprojection图像
    step1_samples = base_dir / "step1" / "samples"
    reprojection_files = sorted([f for f in step1_samples.glob("*_reprojection.png")])
    
    print(f"找到 {len(reprojection_files)} 个重投影图像")
    
    # 对每个图像进行拼接
    success_count = 0
    fail_count = 0
    
    for reprojection_file in reprojection_files:
        filename = reprojection_file.name
        
        # 构建各个step的图像路径
        image_paths = []
        for step in steps:
            img_path = base_dir / f"step{step}" / "samples" / filename
            image_paths.append(img_path)
        
        # 输出路径
        output_path = output_dir / filename
        
        # 拼接图像
        if concat_images_horizontal(image_paths, output_path):
            success_count += 1
            print(f"✓ 已拼接: {filename}")
        else:
            fail_count += 1
            print(f"✗ 拼接失败: {filename}")
    
    print(f"\n完成! 成功: {success_count}, 失败: {fail_count}")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

