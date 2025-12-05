import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 无图形界面环境
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#评估数据
# ------------------- 路径和数据加载 -------------------
work_dir = '/data/huanghui-data/ParFlow-transformer/work_dirs/ParFlow_press/'
dir_name = '2025-12-04-13-05_PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep/val_saved'
work_dir = os.path.join(work_dir, dir_name)
work_time = '2025-12-04-13-05'
epoch = '097'
save_dir = os.path.join('/home/huanghui/data/ParFlow-transformer/pred_result', work_time)
save_dir = os.path.join(save_dir, f'epoch_{epoch}')
os.makedirs(save_dir, exist_ok=True)
stats_path = '/data/huanghui-data/ParFlow-transformer/stats.npz'

path_preds = os.path.join(work_dir, f'preds_epoch_{epoch}.npy')
path_trues = os.path.join(work_dir, f'trues_epoch_{epoch}.npy')
data_preds = np.load(path_preds)
data_trues = np.load(path_trues)
'''
#测试数据
# ------------------- 路径和数据加载 -------------------
work_dir = '/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press/2025-11-20-12-42_PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep/saved'
work_time = '2025-11-20-12-42'
save_dir = os.path.join('/home/huanghui/data/ParFlow-transformer/pred_result', work_time)
os.makedirs(save_dir, exist_ok=True)
stats_path = '/data/huanghui-data/ParFlow-transformer/stats.npz'

path_preds = os.path.join(work_dir, 'preds.npy')
path_trues = os.path.join(work_dir, 'trues.npy')
data_preds = np.load(path_preds)
data_trues = np.load(path_trues)'''
# ------------------- 反标准化 -------------------
stats = np.load(stats_path)
means = stats['mean']  # shape: (10,) 10个channel的均值
stds = stats['std']    # shape: (10,) 10个channel的标准差

batch_size, time_steps, channels, height, width = data_preds.shape
print(f"数据形状: batch_size={batch_size}, time_steps={time_steps}, channels={channels}, height={height}, width={width}")

means_5d = means.reshape(1, 1, channels, 1, 1)  # (1, 1, 10, 1, 1)
stds_5d = stds.reshape(1, 1, channels, 1, 1)    # (1, 1, 10, 1, 1)

data_preds = data_preds * stds_5d + means_5d
data_trues = data_trues * stds_5d + means_5d

print(f"反标准化后数据范围: [{data_preds.min():.4f}, {data_preds.max():.4f}]")

# ------------------- 选择一个批次进行查看 -------------------
batch_idx = 1
preds_batch = data_preds[batch_idx]  # ✅ shape: [6, 10, 144, 248]
trues_batch = data_trues[batch_idx]  # ✅ shape: [6, 10, 144, 248]

'''
print(f"选定批次 {batch_idx} 形状:")
print(f"preds_batch: {preds_batch.shape}")  # [6, 10, 144, 248]
print(f"trues_batch: {trues_batch.shape}")  # [6, 10, 144, 248])

# 检查一个时间步的数值范围

sample_timestep_preds = preds_batch[0]  # 第一个时间步的所有通道
for ch in range(sample_timestep_preds.shape[0]):
    ch_data = sample_timestep_preds[ch]
    print(f"通道 {ch} 数值范围: [{ch_data.min():.4f}, {ch_data.max():.4f}]")
    print(f"通道{ch}均值：{ch_data.mean():.4f}，标准差：{ch_data.std():.4f}")

sample_timestep_trues = trues_batch[0]
for ch in range(sample_timestep_trues.shape[0]):
    ch_data = sample_timestep_trues[ch]
    print(f"通道 {ch} 数值范围: [{ch_data.min():.4f}, {ch_data.max():.4f}]")
    print(f"通道{ch}均值：{ch_data.mean():.4f}，标准差：{ch_data.std():.4f}")
'''   


# ------------------- 手动指定差值颜色范围 -------------------
GLOBAL_DIFF_MIN = -8  # 手动指定的差值最小值
GLOBAL_DIFF_MAX = 8   # 手动指定的差值最大值
print(f"\n手动指定差值颜色范围: [{GLOBAL_DIFF_MIN}, {GLOBAL_DIFF_MAX}]")

# ------------------- 函数：将范围调整为整数 -------------------
def adjust_to_integers(min_val, max_val, num_ticks=5):
    """将数值范围调整为整数，确保包含足够的整数刻度"""
    # 计算需要的整数范围
    int_min = int(np.floor(min_val))
    int_max = int(np.ceil(max_val))
    
    # 确保范围至少包含num_ticks个不同的整数
    while (int_max - int_min + 1) < num_ticks:
        if int_min > 0:
            int_min -= 1
        elif int_max < 0:
            int_max += 1
        else:
            # 跨越0的情况，向两边扩展
            int_min -= 1
            int_max += 1
    
    # 生成均匀分布的刻度
    if int_max == int_min:
        # 如果只有一个整数，创建一个小范围
        if int_min == 0:
            int_min, int_max = -1, 1
        else:
            int_min -= 1
            int_max += 1
    
    return int_min, int_max

# ------------------- 计算每个channel的preds/trues的整数范围 -------------------
channel_ranges = []
for ch in range(channels):
    channel_preds = preds_batch[:, ch, :, :]
    channel_trues = trues_batch[:, ch, :, :]
    ch_min = min(np.min(channel_preds), np.min(channel_trues))
    ch_max = max(np.max(channel_preds), np.max(channel_trues))
    int_min, int_max = adjust_to_integers(ch_min, ch_max)
    channel_ranges.append((int_min, int_max))
    print(f"Channel {ch} preds/trues整数范围: [{int_min}, {int_max}]")

# ------------------- 修改后的绘图函数 -------------------
def plot_channel_all_timesteps(channel_idx):
    """
    为一个channel绘制包含所有时间步的大图
    
    Args:
        channel_idx: channel索引（0~9）
    """
    # 提取该channel在当前批次所有时间步的数据
    channel_preds = preds_batch[:, channel_idx, :, :]  # [time_steps, height, width]
    channel_trues = trues_batch[:, channel_idx, :, :]  # [time_steps, height, width]
    channel_diffs = channel_trues - channel_preds       # [time_steps, height, width]
    
    # 使用该channel的整数范围
    channel_min, channel_max = channel_ranges[channel_idx]
    
    print(f"Channel {channel_idx} 使用整数范围: [{channel_min}, {channel_max}]")
    
    # 创建图形和布局 - 增加宽度减少空隙
    fig = plt.figure(figsize=(6*time_steps, 10))  # 增加宽度到6*time_steps，高度调整为10
    
    # 使用GridSpec创建布局，调整间距
    gs = GridSpec(3, time_steps + 1, figure=fig, 
                  width_ratios=[1]*time_steps + [0.06],  # 减小颜色条宽度比例
                  height_ratios=[1, 1, 1],
                  hspace=0.1, wspace=0.05)  # 减小子图间距
    
    # 第一行：所有时间步的predictions（使用该channel的整数范围）
    ims_preds = []
    for hour_idx in range(time_steps):
        ax = fig.add_subplot(gs[0, hour_idx])
        layer_data = channel_preds[hour_idx]
        im = ax.imshow(layer_data, cmap='viridis', vmin=channel_min, vmax=channel_max)
        ax.set_title(f'Pred T{hour_idx}', fontsize=10)  # 减小字体大小
        ax.axis('off')
        ims_preds.append(im)
    
    # 第二行：所有时间步的ground truth（使用该channel的整数范围）
    ims_trues = []
    for hour_idx in range(time_steps):
        ax = fig.add_subplot(gs[1, hour_idx])
        layer_data = channel_trues[hour_idx]
        im = ax.imshow(layer_data, cmap='viridis', vmin=channel_min, vmax=channel_max)
        ax.set_title(f'True T{hour_idx}', fontsize=10)  # 减小字体大小
        ax.axis('off')
        ims_trues.append(im)
    
    # 第三行：所有时间步的difference（使用手动指定的固定范围 -8 到 8）
    ims_diffs = []
    for hour_idx in range(time_steps):
        ax = fig.add_subplot(gs[2, hour_idx])
        layer_data = channel_diffs[hour_idx]
        im = ax.imshow(layer_data, cmap='RdBu_r', vmin=GLOBAL_DIFF_MIN, vmax=GLOBAL_DIFF_MAX)
        ax.set_title(f'Diff T{hour_idx}', fontsize=10)  # 减小字体大小
        ax.axis('off')
        ims_diffs.append(im)
    
    # 颜色条1: preds和trues共用（使用该channel的整数范围）
    cax1 = fig.add_subplot(gs[0:2, time_steps])  # 占据前两行
    cbar1 = fig.colorbar(ims_preds[0], cax=cax1, orientation='vertical')
    cbar1.set_label('Pressure Value', rotation=270, labelpad=15, fontsize=10)
    # 设置整数刻度
    pred_ticks = np.arange(channel_min, channel_max + 1, max(1, (channel_max - channel_min) // 4))
    cbar1.set_ticks(pred_ticks)
    cbar1.set_ticklabels([f'{int(tick)}' for tick in pred_ticks], fontsize=8)
    
    # 颜色条2: 差值图（使用手动指定的固定整数范围 -8 到 8）
    cax2 = fig.add_subplot(gs[2, time_steps])  # 只在第三行
    cbar2 = fig.colorbar(ims_diffs[0], cax=cax2, orientation='vertical')
    cbar2.set_label('Error (True-Pred)', rotation=270, labelpad=15, fontsize=10)
    # 使用手动指定的固定整数差值范围
    diff_ticks = np.arange(GLOBAL_DIFF_MIN, GLOBAL_DIFF_MAX + 1, 2)  # 步长为2，显示-8,-6,-4,-2,0,2,4,6,8
    cbar2.set_ticks(diff_ticks)
    cbar2.set_ticklabels([f'{int(tick)}' for tick in diff_ticks], fontsize=8)
    
    # 主标题 - 去掉Denormalized
    fig.suptitle(f'Channel {channel_idx} - All Time Steps Comparison - Batch {batch_idx}', 
                 fontsize=14, y=0.95)
    
    # 保存图片
    plt.savefig(
        os.path.join(save_dir, f'channel_{channel_idx}_all_timesteps_batch{batch_idx}.png'),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.2  # 减小边距
    )
    plt.close(fig)

# ------------------- 遍历所有channel并绘图 -------------------
print(f"\n开始绘制 {channels} 个channels，每个channel生成一张包含所有时间步的对比图...")

for channel_idx in range(channels):  # 0~9 channels
    print(f"绘制 Channel {channel_idx}...")
    plot_channel_all_timesteps(channel_idx)

print("✅ 所有图像处理完成！")
print(f"总共生成 {channels} 张图片（每个channel一张大图）")
print(f"文件命名格式: channel_{channel_idx}_all_timesteps_batch{batch_idx}.png")
print(f"差值图固定使用手动指定范围: [{GLOBAL_DIFF_MIN}, {GLOBAL_DIFF_MAX}]")
print(f"差值color bar刻度: -8, -6, -4, -2, 0, 2, 4, 6, 8")
print("所有color bar都使用整数刻度")

#python  /data/huanghui-data/ParFlow-transformer/tools/draw_press_reslut_val.py