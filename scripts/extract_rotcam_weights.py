"""
提取 CVQMAE 模型中用于预测旋转和相机参数的网络权重
包括: rotcam_head, rot_predictor, cam_predictor
"""

import torch
import argparse
from collections import OrderedDict


def extract_rotcam_weights(checkpoint_path, output_path):
    """
    从完整的 CVQMAE checkpoint 中提取旋转和相机预测相关的权重
    
    Args:
        checkpoint_path: 预训练模型的路径
        output_path: 输出权重文件的路径
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取模型的 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 提取相关的权重
    rotcam_weights = OrderedDict()
    
    # 需要提取的模块前缀
    target_modules = [
        'decoder.rotcam_head',
        'decoder.rot_predictor', 
        'decoder.cam_predictor',
        'rotcam_head',
        'rot_predictor',
        'cam_predictor',
    ]
    
    extracted_count = 0
    for key, value in state_dict.items():
        # 移除 'module.' 前缀（如果存在）
        clean_key = key.replace('module.', '')
        
        # 检查是否是我们需要的权重
        for module_name in target_modules:
            if module_name in clean_key:
                rotcam_weights[clean_key] = value
                extracted_count += 1
                print(f"  ✓ Extracted: {clean_key} | Shape: {value.shape}")
                break
    
    if extracted_count == 0:
        print("\n⚠ Warning: No weights extracted! Please check the checkpoint structure.")
        print("\nAvailable keys in checkpoint:")
        for key in list(state_dict.keys())[:20]:
            print(f"  - {key}")
        return
    
    # 保存提取的权重
    save_dict = {
        'rotcam_weights': rotcam_weights,
        'metadata': {
            'source_checkpoint': checkpoint_path,
            'num_parameters': extracted_count,
            'modules': list(set([k.split('.')[0] + '.' + k.split('.')[1] if '.' in k else k for k in rotcam_weights.keys()]))
        }
    }
    
    # 如果原始 checkpoint 包含其他信息，也保存下来
    if 'loss' in checkpoint:
        save_dict['metadata']['source_loss'] = checkpoint['loss']
    if 'epoch' in checkpoint:
        save_dict['metadata']['source_epoch'] = checkpoint['epoch']
    
    torch.save(save_dict, output_path)
    print(f"\n✓ Successfully saved {extracted_count} weight tensors to: {output_path}")
    print(f"  Total parameters: {sum(p.numel() for p in rotcam_weights.values()):,}")
    

def load_rotcam_weights(weights_path, model, strict=False):
    """
    加载提取的旋转和相机预测权重到模型中
    
    Args:
        weights_path: 权重文件路径
        model: 目标模型
        strict: 是否严格匹配所有键
    
    Returns:
        加载信息
    """
    print(f"Loading rotcam weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    rotcam_weights = checkpoint['rotcam_weights']
    metadata = checkpoint.get('metadata', {})
    
    print(f"Metadata: {metadata}")
    
    # 加载权重到模型
    missing_keys, unexpected_keys = model.load_state_dict(rotcam_weights, strict=strict)
    
    if missing_keys:
        print(f"\n⚠ Missing keys (not loaded): {missing_keys}")
    if unexpected_keys:
        print(f"\n⚠ Unexpected keys (not in model): {unexpected_keys}")
    
    print(f"\n✓ Successfully loaded rotcam weights")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract rotation and camera predictor weights from CVQMAE")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="checkpoint/CVQMAE/mega_hrnet",
        help="Path to the pretrained CVQMAE checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoint/CVQMAE/rotcam_weights.pth",
        help="Output path for extracted weights"
    )
    
    args = parser.parse_args()
    
    extract_rotcam_weights(args.checkpoint, args.output)
    
    print("\n" + "="*60)
    print("Usage example to load these weights:")
    print("="*60)
    print("""
from extract_rotcam_weights import load_rotcam_weights

# 在你的模型中
model = YourModel(...)
load_rotcam_weights('checkpoint/CVQMAE/rotcam_weights.pth', model, strict=False)
    """)
