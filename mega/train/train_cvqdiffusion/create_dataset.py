"""
创建 CVQDiffusion 训练用的 npz 数据集。
每个样本：图像路径（字符串） + 54 个随机整数索引（0~511）
"""
import numpy as np
import os

IMG_DIR   = 'demo_data/input/3dpw'
SAVE_PATH = 'mega/train/train_cvqdiffusion/data/train_cvqdiffusion.npz'
NUM_TOK   = 512
SEQ_LEN   = 54
N_SAMPLES = 100

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

img_paths = [os.path.join(IMG_DIR, f'{i}.jpg') for i in range(1, N_SAMPLES + 1)]
tokens    = np.random.randint(0, NUM_TOK, size=(N_SAMPLES, SEQ_LEN), dtype=np.int32)

np.savez(SAVE_PATH, img_paths=np.array(img_paths), tokens=tokens)
print(f'Dataset saved to {SAVE_PATH}')
print(f'  img_paths : {len(img_paths)} entries')
print(f'  tokens    : {tokens.shape}  range [{tokens.min()}, {tokens.max()}]')
