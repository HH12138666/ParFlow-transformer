import os
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

data_root = 'data' 
CROP_H = 146  
CROP_W = 252  
target_h = CROP_H
target_w = CROP_W
save_abnormal_figure = "check_data_figure/lessthan-1000"
save_all_channels_figure = "check_data_figure/raw_figures"
data_threshold = -10000
target_hour_index = 3 

os.makedirs(save_abnormal_figure, exist_ok=True)
os.makedirs(save_all_channels_figure, exist_ok=True)


def _natural_key(p):
    b = os.path.basename(p)
    s = re.split(r'(\d+)', b)
    return [int(t) if t.isdigit() else t for t in s]


def _list_pfb_files(root) :
    files = sorted(glob.glob(os.path.join(root, '*.pfb')), key=_natural_key)
    if not files:
        raise FileNotFoundError(f'No .pfb files found under: {root}')
    return files


def _center_crop_h(arr, target_h=CROP_H,target_w=CROP_W) :
    c, h, w = arr.shape
    if target_h is not None and h != target_h:
        dh = h - target_h
        if dh < 0:
            raise ValueError(f"Need crop/pad to H={target_h}, but input H={h} < target.")
        top = dh // 2 
        arr = arr[:, top:top + target_h, :]
    if target_w is not None and w != target_w:
        dw = w - target_w
        if dw < 0:
            raise ValueError(f"Need crop/pad to W={target_w}, but input W={w} < target.")
        left = dw // 2
        arr = arr[:, :, left:left + target_w]
    return arr



def _read_press_frame(path, target_h=CROP_H, target_w=CROP_W) :

    arr = read_pfb(get_absolute_path(path)).astype(np.float32)  # (C,H,W)
 
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D array per .pfb, got shape {arr.shape} for {path}')

    arr = _center_crop_h(arr, target_h=target_h, target_w = target_w)  
    return arr

# 异常值判断标准
def has_negative_outliers(channel_data, threshold=data_threshold):
    return np.any(channel_data < threshold)


def check_abnormal_values_and_save_images():

    pfb_files = _list_pfb_files(data_root)
    n_files = len(pfb_files)
    abnormal_records = []

    print(f"开始检查 {n_files} 个 .pfb 文件...")

    for hour_idx, pfb_path in enumerate(pfb_files):
        try:
            data = _read_press_frame(pfb_path, target_h, target_w)
            C, H, W = data.shape
            for c in range(C):
                channel_data = data[c, :, :]  # shape: (H, W)
                # 找出该通道中所有 < data_threshold 的元素的坐标
                abnormal_pixels = np.where(channel_data < data_threshold)
                if len(abnormal_pixels[0]) > 0:  # 如果有异常值
                    file_name = os.path.basename(pfb_path)  # 获取文件名（如 "pressure_0001.pfb"）
                    print(f"\n[异常] 文件: {file_name} | 小时索引: {hour_idx} | 通道: {c}")
                    for h_idx, w_idx in zip(*abnormal_pixels):
                        value = channel_data[h_idx, w_idx]
                        print(f"  → 异常位置: (H={h_idx}, W={w_idx}), 值 = {value}")
                    abnormal_records.append((file_name, hour_idx, c, abnormal_pixels, channel_data))
        except Exception as e:
            print(f"[错误] 处理文件 {os.path.basename(pfb_path)} 时出错: {e}")

    n_abnormal = len(abnormal_records)
    print(f"检查完成。共发现 {n_abnormal} 个通道存在异常（值 < {data_threshold}）。")

    '''
    # 可视化异常通道
    for idx, (hour_idx, channel_idx, channel_data) in enumerate(abnormal_records):
        try:
            plt.figure(figsize=(8, 6))
            plt.imshow(channel_data, cmap='jet', origin='lower')
            plt.colorbar(label=f'value (channel {channel_idx})')
            plt.title(f"异常数据\n小时 {hour_idx} | 通道 {channel_idx} | 值 < {data_threshold}")
            hour_str = f"{hour_idx:05d}"
            channel_str = f"{channel_idx:02d}"
            filename = f"abnormal_hour_{hour_str}_channel_{channel_str}.png"
            filepath = os.path.join(save_abnormal_figure, filename)
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"✅ 已保存异常图: {filename}")
        except Exception as e:
            print(f"❌ 保存异常图（小时 {hour_idx} 通道 {channel_idx}）时出错: {e}")
        '''    
def visualize_selected_hour_all_channels():
    global target_hour_index  

    pfb_files = _list_pfb_files(data_root)

    # 检查 hour_index 是否有效
    if target_hour_index < 0 or target_hour_index >= len(pfb_files):
        print(f"❌ 错误：target_hour_index = {target_hour_index} 超出范围。请检查，有效范围为 0 ~ {len(pfb_files) - 1}")
        return

    selected_pfb_path = pfb_files[target_hour_index]
    hour_idx = target_hour_index
    print(f"🔍 正在处理第 {hour_idx} 个小时，对应文件: {os.path.basename(selected_pfb_path)}")

    try:
        data = _read_press_frame(selected_pfb_path, target_h, target_w)
        C, H, W = data.shape

        # 🆕 为当前小时创建一个独立的保存目录，例如：check_data_figure/raw_figures/hour_00003/
        hour_str = f"{hour_idx:05d}"  # 比如 "00003"
        hour_folder_name = f"hour_{hour_str}"  # 比如 "hour_00003"
        hour_save_dir = os.path.join(save_all_channels_figure, hour_folder_name)

        os.makedirs(hour_save_dir, exist_ok=True)  # 创建该目录（如果不存在）


        for c in range(C):
            channel_data = data[c, :, :]
            plt.figure(figsize=(6, 5))
            plt.imshow(channel_data, cmap='jet', origin='lower')
            plt.colorbar(label=f'Value (Channel {c})')
            plt.title(f'Hour {hour_idx} | Channel {c}')

            # 🆕 图片命名与保存路径都基于该小时
            hour_str = f"{hour_idx:05d}"
            channel_str = f"{c:02d}"
            filename = f"hour_{hour_str}_channel_{channel_str}.png"
            filepath = os.path.join(hour_save_dir, filename)  # 👈 重点：保存到 hour_00003/ 目录下

            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"✅ 已生成: Hour {hour_idx} | Channel {c} -> {filename} （保存在 {hour_save_dir}）")

        print(f"🎉 已为小时 {hour_idx} 的所有 {C} 个通道生成图片，保存在: {hour_save_dir}")

    except Exception as e:
        print(f"❌ 处理小时 {hour_idx} 时发生错误: {e}")
        
def main():

    check_abnormal_values_and_save_images()
    
    #visualize_selected_hour_all_channels()  
    


if __name__ == "__main__":
    main()