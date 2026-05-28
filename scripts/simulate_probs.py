import torch
import numpy as np
import matplotlib.pyplot as plt

def simulate_maskgit_probs_monotonic(seq_len=54, num_iterations=9):
    """
    模拟 MaskGIT 置信度，严格保证单调不减（后一步 >= 前一步）。
    """
    list_probs = []
    
    lock_steps = np.random.randint(1, num_iterations + 2, size=seq_len)
    lock_steps[0:5] = [2, 1, 9, 3, 5] 
    
    # ================= 新增：用于记录上一步置信度的变量 =================
    # 初始状态设为 0，保证第一步骤肯定会大于它
    previous_conf = np.zeros(seq_len) 
    # ====================================================================

    for step in range(1, num_iterations + 1):
        # 1. 生成带有随机性的基础置信度
        base_low = np.random.uniform(0.15, 0.30, size=seq_len) + (step * 0.015)
        base_high = np.random.uniform(0.75, 0.95, size=seq_len) + (step * 0.005)
        is_locked = step >= lock_steps
        current_conf = np.where(is_locked, base_high, base_low)
        noise = np.random.normal(0, 0.02, size=seq_len)
        current_conf = current_conf + noise
        
        # ================= 核心修改：强制单调不减 =================
        # 逐个元素对比：如果当前生成的置信度因为随机噪声掉下去了，
        # 就强行拉回到上一步的高度，保底不跌！
        current_conf = np.maximum(current_conf, previous_conf)
        # ==========================================================
        
        # 2. 截断限制范围
        current_conf = np.clip(current_conf, 0.0, 1.0)
        
        # ================= 新增：更新 previous_conf =================
        # 将当前步定格的置信度保存下来，留给下一步对比
        previous_conf = current_conf
        # ============================================================
        
        list_probs.append(torch.tensor(current_conf).unsqueeze(0))
        
    return list_probs

def visualize_simulated_confidence(list_probs, output_path):
    """
    绘制热力图的函数 (复用并微调了坐标轴逻辑)
    """
    arrays = [p.detach().cpu().numpy().squeeze() for p in list_probs]
    confidence_matrix = np.stack(arrays, axis=0) 
    num_iterations, seq_len = confidence_matrix.shape
    avg_conf_per_iter = confidence_matrix.mean(axis=1)

    fig, ax1 = plt.subplots(figsize=(24, 6))

    # 使用 viridis 色带，锁定 0.1 到 1.0，让颜色对比更强烈
    im = ax1.imshow(confidence_matrix, aspect='auto', origin='lower', cmap='viridis', 
                    vmin=0.1, vmax=1.0)

    ax1.set_xlabel('Index of Mesh Token', fontsize=20)
    ax1.set_xticks(np.arange(0, seq_len, 8)) 
    ax1.set_xticklabels(np.arange(0, seq_len, 8), rotation=90, fontsize=18)
    ax1.set_xlim(-0.5, seq_len - 0.5)

    ax1.set_ylabel('Iteration', fontsize=20)
    ax1.set_yticks(np.arange(num_iterations))
    ax1.set_yticklabels(np.arange(1, num_iterations + 1), fontsize=18)
    ax1.set_ylim(-0.5, num_iterations - 0.5)

    ax1.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, num_iterations, 1), minor=True)
    ax1.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.6)
    ax1.tick_params(which="minor", bottom=False, left=False)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Confidence per Iteration', color='red', fontsize=20)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(np.arange(num_iterations))
    ax2.set_yticklabels([f"{val:.2f}" for val in avg_conf_per_iter], color='red', fontsize=18)
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')

    cbar = fig.colorbar(im, ax=ax2, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 模拟突变效果的图片已保存至: {output_path}")

# ================= 运行测试 =================
# 1. 生成模拟的、带有阶梯突变的数据
mock_probs = simulate_maskgit_probs_monotonic(seq_len=55, num_iterations=9)

# 2. 绘制并保存图片
visualize_simulated_confidence(mock_probs, 'scripts/pic/simulated_maskgit_heatmap.png')