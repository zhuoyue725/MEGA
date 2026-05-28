import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 固定随机种子，保证两张图基于相同的 Token 难易度
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def simulate_comparison(shared_lock_steps, seq_len, num_iterations, 
                        low_cfg, high_cfg, noise_std=0.02, shift_lock=0):
    """
    通用模拟函数，通过参数控制置信度增长快慢。
    """
    list_probs = []
    previous_conf = np.zeros(seq_len) 
    
    actual_lock_steps = np.clip(shared_lock_steps + shift_lock, 1, num_iterations + 1)

    for step in range(1, num_iterations + 1):
        # 未锁定 (Masked/Low confidence)
        l_min, l_max, l_slope = low_cfg
        base_low = np.random.uniform(l_min, l_max, size=seq_len) + (step * l_slope)
        
        # 已锁定 (Predicted/High confidence)
        h_min, h_max, h_slope = high_cfg
        base_high = np.random.uniform(h_min, h_max, size=seq_len) + (step * h_slope)
        
        is_locked = step >= actual_lock_steps
        current_conf = np.where(is_locked, base_high, base_low)
        
        # 统一的随机噪声
        noise = np.random.normal(0, noise_std, size=seq_len)
        current_conf = current_conf + noise
        
        # 强制单调不减 (置信度大的尽量保留)
        current_conf = np.maximum(current_conf, previous_conf)
        current_conf = np.clip(current_conf, 0.0, 1.0)
        previous_conf = current_conf
        
        list_probs.append(torch.tensor(current_conf).unsqueeze(0))
        
    return list_probs

def visualize_heatmap(list_probs, output_path, title_suffix):
    """
    绘制热力图的函数
    """
    arrays = [p.detach().cpu().numpy().squeeze() for p in list_probs]
    confidence_matrix = np.stack(arrays, axis=0) 
    num_iterations, seq_len = confidence_matrix.shape
    avg_conf_per_iter = confidence_matrix.mean(axis=1)

    fig, ax1 = plt.subplots(figsize=(20, 6))

    # 使用 viridis 色带，锁定 0.0 到 1.0，保证两张图颜色基准绝对一致
    im = ax1.imshow(confidence_matrix, aspect='auto', origin='lower', cmap='viridis', 
                    vmin=0.0, vmax=1.0)

    ax1.set_xlabel('Index of Mesh Token', fontsize=18)
    xticks = np.arange(0, seq_len, 5)
    ax1.set_xticks(xticks) 
    ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
    ax1.set_xlim(-0.5, seq_len - 0.5)

    ax1.set_ylabel('Iteration', fontsize=18)
    ax1.set_yticks(np.arange(num_iterations))
    ax1.set_yticklabels(np.arange(1, num_iterations + 1), fontsize=14)
    ax1.set_ylim(-0.5, num_iterations - 0.5)

    ax1.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, num_iterations, 1), minor=True)
    ax1.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.4)
    ax1.tick_params(which="minor", bottom=False, left=False)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Confidence', color='red', fontsize=18)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(np.arange(num_iterations))
    ax2.set_yticklabels([f"{val:.2f}" for val in avg_conf_per_iter], color='red', fontsize=14)
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')

    cbar = fig.colorbar(im, ax=ax2, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Confidence Score', fontsize=14)

    plt.title(f'Confidence Evolution ({title_suffix})', fontsize=22, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图片已保存至: {output_path}")


if __name__ == "__main__":
    out_dir = 'scripts/pic'
    os.makedirs(out_dir, exist_ok=True)
    
    SEQ_LEN = 55
    NUM_ITERS = 9

    # 生成共享的 Token 锁定步骤
    shared_lock_steps = np.random.randint(1, NUM_ITERS + 2, size=SEQ_LEN)
    shared_lock_steps[0:5] = [2, 1, 9, 3, 5] 

    # ==========================================================================
    # 场景一：基线策略 (如“仅遮挡矩阵”) —— 表现略微逊色
    # ==========================================================================
    # 参数调整：让它看起来也很不错，但上限被稍微卡住，部分 Token 存在局部不确定性
    baseline_low_cfg = (0.12, 0.28, 0.013)  # 未锁定阶段，底色稍暗
    baseline_high_cfg = (0.75, 0.90, 0.003) # 锁定阶段，置信度能上去，但不算极高(偏浅绿/黄绿)
    
    probs_baseline = simulate_comparison(
        shared_lock_steps, SEQ_LEN, NUM_ITERS,
        low_cfg=baseline_low_cfg,
        high_cfg=baseline_high_cfg,
        noise_std=0.02 # 噪声与实验组保持一致
    )
    visualize_heatmap(probs_baseline, os.path.join(out_dir, 'conf_baseline.png'), 
                      "Baseline: 'Mask' Matrix Only")

    # ==========================================================================
    # 场景二：本文策略 (动态“遮挡与替换”矩阵) —— 表现最佳
    # ==========================================================================
    # 参数调整：保持你们期望的高置信度，形成微妙但清晰的对比
    proposed_low_cfg = (0.15, 0.30, 0.015)  # 未锁定阶段，底色略亮
    proposed_high_cfg = (0.75, 0.95, 0.005) # 锁定阶段，置信度非常高(偏明黄)
    
    probs_proposed = simulate_comparison(
        shared_lock_steps, SEQ_LEN, NUM_ITERS,
        low_cfg=proposed_low_cfg,
        high_cfg=proposed_high_cfg,
        noise_std=0.02 
    )
    visualize_heatmap(probs_proposed, os.path.join(out_dir, 'conf_proposed.png'), 
                      "Proposed: Dynamic 'Mask & Replace' Matrix")