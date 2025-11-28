import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 无图形界面环境
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors

# ------------------- 路径和数据加载 -------------------
work_dir = '/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press/2025-11-20-12-42_PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep/saved'
work_time = '2025-11-20-12-42'
save_dir = os.path.join('/home/huanghui/data/ParFlow-transformer/pred_result', work_time)
os.makedirs(save_dir, exist_ok=True)

path_preds = os.path.join(work_dir, 'preds.npy')
path_trues = os.path.join(work_dir, 'trues.npy')
data_preds = np.load(path_preds)
data_trues = np.load(path_trues)

batch_idx = 22 # 选择要查看的批次索引
preds_batch = data_preds[batch_idx]  # shape: [6, 10, 144, 248]
trues_batch = data_trues[batch_idx]  # shape: [6, 10, 144, 248]

# 计算差值：真实值 - 预测值
diffs_batch = trues_batch - preds_batch  # shape: [6, 10, 144, 248]

# 预先计算整个批次的数值范围，确保一致性
# 预测值和真实值使用相同的范围
all_preds = preds_batch.flatten()
all_trues = trues_batch.flatten()
global_min = min(np.min(all_preds), np.min(all_trues))
global_max = max(np.max(all_preds), np.max(all_trues))


# 差值使用对称范围
all_diffs = diffs_batch.flatten()
max_abs_diff = np.max(np.abs(all_diffs))


# ------------------- 修正后的绘图函数 -------------------
def plot_layers(data_hour, hour_idx, mode='preds'):
    """
    绘制指定模式的数据图层
    
    Args:
        data_hour: shape [10, 144, 248] 要绘制的数据
        hour_idx: 小时索引（0~5）
        mode: 'preds', 'trues' 或 'diffs'
    """
    num_layers = data_hour.shape[0]  # 10
    rows = 2
    cols = 5
    fig = plt.figure(figsize=(32, 8))

    # 根据模式设置参数
    if mode == 'preds':
        cmap = 'viridis'
        title_prefix = 'Pred'
        colorbar_label = 'Pressure Value'
        vmin = global_min  # 使用全局最小值
        vmax = global_max  # 使用全局最大值
    elif mode == 'trues':
        cmap = 'viridis'
        title_prefix = 'True'
        colorbar_label = 'Pressure Value'
        vmin = global_min  # 使用全局最小值
        vmax = global_max  # 使用全局最大值
    else:  # diffs
        cmap = 'RdBu_r'
        title_prefix = 'Diff'
        colorbar_label = 'Error (True - Pred)'
        vmin = -max_abs_diff  # 对称最小值
        vmax = max_abs_diff   # 对称最大值

    # 使用 GridSpec 划分布局
    gs = GridSpec(rows, cols + 1, figure=fig, width_ratios=[1]*cols + [0.03])

    ims = []
    for layer_idx in range(num_layers):
        ax = fig.add_subplot(gs[layer_idx // cols, layer_idx % cols])
        layer_data = data_hour[layer_idx]
        
        # 使用统一的数值范围显示数据
        im = ax.imshow(layer_data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{title_prefix} Layer {layer_idx}')
        ax.axis('off')
        ims.append(im)

    # 颜色条设置
    if ims:
        im_ref = ims[0]
        cbar_left = 1 - 0.05
        cbar_bottom = 0.1
        cbar_width = 0.01
        cbar_height = 0.8
        cbar_pos = (cbar_left, cbar_bottom, cbar_width, cbar_height)

        cax = fig.add_axes(cbar_pos)
        cbar = fig.colorbar(im_ref, cax=cax, orientation='vertical')
        cbar.set_label(colorbar_label, rotation=270, labelpad=20)
        
        # 设置颜色条的刻度
        if mode in ['preds', 'trues']:
            # 预测值和真实值：显示合理的刻度
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{tick:.3f}' for tick in ticks])
        else:
            # 差值：显示对称的刻度
            ticks = np.linspace(vmin, vmax, 7)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{tick:.3f}' for tick in ticks])

    # 主标题
    fig.suptitle(f'{mode.upper()} - Batch {batch_idx}, Hour {hour_idx}', fontsize=16, y=0.99)

    # 保存图片
    plt.savefig(
        os.path.join(save_dir, f'{mode}_batch{batch_idx}_hour{hour_idx}.png'),
        dpi=450,
        bbox_inches='tight'
    )
    plt.close(fig)


# ------------------- 为每个小时生成三种图片 -------------------
print("开始生成图像...")

for hour_idx in range(data_preds.shape[1]):  # 0~5小时
    print(f"正在处理 Hour {hour_idx}...")
    
    # 获取当前小时的数据
    preds_hour = preds_batch[hour_idx]   # 预测值
    trues_hour = trues_batch[hour_idx]   # 真实值
    diffs_hour = diffs_batch[hour_idx]   # 差值
    
    # 绘制三种图片
    plot_layers(preds_hour, hour_idx, mode='preds')   # 预测值图
    plot_layers(trues_hour, hour_idx, mode='trues')   # 真实值图
    plot_layers(diffs_hour, hour_idx, mode='diffs')   # 差值图

print(f"所有图像已保存到: {save_dir}")
print("\n生成的文件包括:")
print(f"- preds_batch{batch_idx}_hour0.png ~ hour5.png (6张预测值图)")
print(f"- trues_batch{batch_idx}_hour0.png ~ hour5.png (6张真实值图)")  
print(f"- diffs_batch{batch_idx}_hour0.png ~ hour5.png (6张差值图)")
print(f"总共18张图片")

# 打印数值范围信息
print(f"\n数值范围信息:")
print(f"预测值和真实值范围: [{global_min:.6f}, {global_max:.6f}]")
print(f"差值范围: [{-max_abs_diff:.6f}, {max_abs_diff:.6f}]")