from parflow.tools.io import write_pfb
import numpy as np, pathlib
import os

work_dir = '/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press/2025-11-18-11-19_PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep/saved'
path_inputs = os.path.join(work_dir,'inputs.npy')
path_metrics = os.path.join(work_dir,'metrics.npy')
path_preds = os.path.join(work_dir,'preds.npy')
path_trues = os.path.join(work_dir,'trues.npy')

out_dir = pathlib.Path('restored_preds')
out_dir.mkdir(exist_ok=True)
mean = np.load('stats.npz')['mean'].reshape(1,1,-1,1,1)
std  = np.load('stats.npz')['std' ].reshape(1,1,-1,1,1)
preds = np.load('saved/preds.npy')
preds = preds * (std + 1e-6) + mean  # 反归一化

# 将第 i 个样本、第 t 个时间步写出
for i in range(preds.shape[0]):
    for t in range(preds.shape[1]):
        arr = preds[i, t]  # 形状 (C,H,W)
        write_pfb(out_dir / f'pred_{i:03d}_{t:02d}.pfb', arr)
        print(f'Wrote: pred_{i:03d}_{t:02d}.pfb')