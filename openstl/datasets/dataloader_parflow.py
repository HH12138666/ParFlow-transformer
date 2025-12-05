import os
import re
import glob
import logging
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

from .utils import create_loader

logger = logging.getLogger(__name__)


EPS = 1e-6

#计算均值和方差相关设置
NORMALIZE = True
NORMALIZE_TARGET = True
STATS_PATH = './stats.npz'                  # 存放均值和方差的路径，如设 None 则不加载 './stats.npz'
STATS_COMPUTE_SAMPLES = 0          # 计算均值和方差时使用的样本数量，如设 0 则不计算
STATS_TIME_STRIDE = 1
STATS_SPATIAL_STRIDE = 1
MAX_FILES = None                    # 设为 None 表示使用全部文件；也可以设为 100 只用前100个
CHANNELS = None    

OUTLIER_THRESHOLD = -10000.0    # 异常值阈值，低于该值的样本将被视为异常并排除在均值和方差计算之外

#数据分割相关设置
time_stride = 5    # 时间步长，用于数据分割



def _natural_key(p):
    b = os.path.basename(p)
    s = re.split(r'(\d+)', b)
    return [int(t) if t.isdigit() else t for t in s]


def _list_pfb_files(root) :
    files = sorted(glob.glob(os.path.join(root, '*.pfb')), key=_natural_key)
    if not files:
        raise FileNotFoundError(f'No .pfb files found under: {root}')
    return files


def _interpolate_outliers(arr, threshold = OUTLIER_THRESHOLD) :
    """Replace outliers (< threshold) along the channel axis using interpolation."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for interpolation, got shape {arr.shape}.")

    repaired = arr.astype(np.float32, copy=True)
    c, h, w = repaired.shape
    flat = repaired.reshape(c, -1)
    mask = flat < threshold
    if not mask.any():
        return repaired

    for idx in range(flat.shape[1]):
        col = flat[:, idx]
        col_mask = mask[:, idx]
        if not col_mask.any():
            continue

        valid_idx = np.where(~col_mask)[0]
        if valid_idx.size == 0:
            # 无有效值时退化为填充 0
            col[:] = 0.0
        elif valid_idx.size == 1:
            col[col_mask] = col[valid_idx[0]]
        else:
            invalid_idx = np.where(col_mask)[0]
            col[col_mask] = np.interp(invalid_idx, valid_idx, col[valid_idx])

    return flat.reshape(c, h, w)
#新增
def _build_space_coords(height, width, space_h, space_w, space_stride_h=None, space_stride_w=None):
    if space_h is None or space_w is None:
        return [(0, 0)]

    stride_h = space_stride_h or space_h
    stride_w = space_stride_w or space_w

    if space_h > height or space_w > width:
        raise ValueError(
            f"Space size {(space_h, space_w)} exceeds frame size {(height, width)}."
        )

    coords_h = list(range(0, height - space_h + 1, stride_h))
    if not coords_h or coords_h[-1] != height - space_h:
        coords_h.append(height - space_h)

    coords_w = list(range(0, width - space_w + 1, stride_w))
    if not coords_w or coords_w[-1] != width - space_w:
        coords_w.append(width - space_w)
    return [(top, left) for top in coords_h for left in coords_w]


def _read_press_frame(path) :

    arr = read_pfb(get_absolute_path(path)).astype(np.float32)  # (C,H,W)

    arr = _interpolate_outliers(arr, threshold=OUTLIER_THRESHOLD)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D array per .pfb, got shape {arr.shape} for {path}')

    return arr

#计算均值和方差
def _welford_update(count, mean, M2, batch_mean, batch_M2, batch_n):
    total_n = count + batch_n
    delta = batch_mean - mean
    mean += delta * (batch_n / np.maximum(total_n, 1))
    M2 += batch_M2 + (delta * delta) * (count * batch_n / np.maximum(total_n, 1))
    count[:] = total_n
    return count, mean, M2

def compute_mean_std(files,
                    spatial_stride=1,
                    time_stride=1,
                    max_files=None,
                    channels=None):
    sel_files = files[::max(1, int(time_stride))]
    if max_files is not None:
        sel_files = sel_files[:int(max_files)]
    a0 = _read_press_frame(sel_files[0])
    if channels is not None:
        a0 = a0[channels, ...]
    if spatial_stride > 1:
        a0 = a0[:, ::spatial_stride, ::spatial_stride]
    C = a0.shape[0]
    count = np.zeros(C, dtype=np.float64)
    mean  = np.zeros(C, dtype=np.float64)
    M2    = np.zeros(C, dtype=np.float64)
    for f in sel_files:
        a = _read_press_frame(f)
        if channels is not None:
            a = a[channels, ...]
        x = a.reshape(C, -1).astype(np.float64, copy=False)
        b_mean = x.mean(axis=1)
        diff   = x - b_mean[:, None]
        b_M2   = (diff * diff).sum(axis=1)
        b_n    = x.shape[1]
        _welford_update(count, mean, M2, b_mean, b_M2, b_n)
    var = M2 / np.maximum(count, 1.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)

#增强数据
def augment_pair(X, Y,
                 p_flip_h=0.5,
                 p_flip_w=0.5,
                 p_noise=0.2,
                 noise_sigma=0.001):
    
    if torch.rand(1).item() < p_flip_h:
        X = X.flip(-2)
        Y = Y.flip(-2)
    if torch.rand(1).item() < p_flip_w:
        X = X.flip(-1)
        Y = Y.flip(-1)
    if torch.rand(1).item() < p_noise:
        X = X + torch.randn_like(X) * float(noise_sigma)
    return X, Y


class ParFlowDataset(Dataset):
    # def __init__(self, data_root, split, pre_seq_length=9, aft_seq_length=1 ,in_shape = None,stride=1,use_augment=False,
    def __init__(self, data_root, split, pre_seq_length=9, aft_seq_length=1 ,in_shape = None,use_augment=False,
                 space_h = None,
                 space_w = None,
                 space_stride_h = None,
                 space_stride_w = None,):
        super().__init__()
        split = str(split).lower()
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid split: {split}. Expected 'train' | 'val' | 'test'.")
        self.split = split

        self.root = data_root
        self.pre = pre_seq_length
        self.aft = aft_seq_length
        self.total = self.pre + self.aft
        self.use_augment = use_augment
        self.space_h = space_h
        self.space_w = space_w
        self.use_space = self.space_h is not None and self.space_w is not None
        
        if self.use_space:
            # Always honor the configured stride so overlapping crops can be
            # used consistently across train/val/test.
            self.space_stride_h = space_stride_h or self.space_h
            self.space_stride_w = space_stride_w or self.space_w
        else:
            self.space_stride_h = None
            self.space_stride_w = None


        self.files = _list_pfb_files(self.root)
        self.num_frames = len(self.files)
        sample = _read_press_frame(self.files[0])
        C, H, W = sample.shape
        self.C, self.H, self.W = C, H, W   
        
        if self.use_space:
            self.space_coords = _build_space_coords(
                self.H, self.W, self.space_h, self.space_w, self.space_stride_h, self.space_stride_w
            )
        else:
            self.space_coords = [(0, 0)]

        self.time_indices = self._build_time_indices()
        
        if self.use_space:
            self.sample_indices = [
                (t, p) for t in self.time_indices for p in range(len(self.space_coords))
            ]
        else:
            self.sample_indices = self.time_indices

        self.mean = None
        self.std  = None

        if NORMALIZE:
            if STATS_PATH and os.path.exists(STATS_PATH):
                data = np.load(STATS_PATH)
                self.mean = np.asarray(data['mean'], dtype=np.float32).reshape(-1)
                self.std  = np.asarray(data['std'],  dtype=np.float32).reshape(-1)
                if self.mean.shape[0] != self.C or self.std.shape[0] != self.C:
                    raise ValueError(f"Stats mismatch: stats C={self.mean.shape[0]} vs dataset C={self.C}.")
            elif STATS_COMPUTE_SAMPLES and STATS_COMPUTE_SAMPLES > 0:
                self.mean, self.std = compute_mean_std(
                    self.files,
                    spatial_stride=STATS_SPATIAL_STRIDE,
                    time_stride=STATS_TIME_STRIDE,
                    max_files=STATS_COMPUTE_SAMPLES,
                    channels=None
                )
            else:
                self.mean = np.zeros((self.C,), dtype=np.float32)
                self.std  = np.ones((self.C,), dtype=np.float32)
        self.mean_t = torch.from_numpy(self.mean).view(1, self.C, 1, 1).float() if self.mean is not None else None
        self.std_t  = torch.from_numpy(self.std ).view(1, self.C, 1, 1).float() if self.std  is not None else None

    def _build_time_indices(self, stride=time_stride):
        n_train = int(self.num_frames * 0.75)
        n_val   = int(self.num_frames * 0.1)
        def build_range(s, e):
            e = e - 1
            max_start = e - self.total + 1
            if max_start < s:
                return []
            return list(range(s, max_start + 1, stride))
        if self.split == 'train':
            return build_range(0, n_train)
        elif self.split == 'val':
            return build_range(n_train, n_train + n_val)
        else:
            return build_range(n_train + n_val, self.num_frames)

    def __len__(self):      
        return len(self.sample_indices)
    # 按时间窗口读取数据确保时间连续   
    def _read_window(self, t0):
        T = self.total
        out = torch.empty((T, self.C, self.H, self.W), dtype=torch.float32)
        for i in range(T):
            path = self.files[t0 + i]
            arr = _read_press_frame(path)
            out[i] = torch.from_numpy(arr)
        return out

    def __getitem__(self, idx):

        if self.use_space:
            t0, p_idx = self.sample_indices[idx]
            top, left = self.space_coords[p_idx]
        else:
            t0 = self.sample_indices[idx]
            top, left = 0, 0
        win = self._read_window(t0)
          
        x = win[: self.pre]
        y = win[self.pre : self.pre + self.aft]
        
        if self.use_space:
            x = x[..., top : top + self.space_h, left : left + self.space_w]
            y = y[..., top : top + self.space_h, left : left + self.space_w]

        if self.use_augment and self.split == 'train':
            x, y = augment_pair(x, y)

        if NORMALIZE and self.mean_t is not None and self.std_t is not None:
            x = (x - self.mean_t) / (self.std_t + EPS)
            if NORMALIZE_TARGET:
                y = (y - self.mean_t) / (self.std_t + EPS)

        return x, y


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers,
              pre_seq_length = 6,
              aft_seq_length = 6,
              in_shape = None,
              distributed = False,
              use_augment = False,
              use_prefetcher = False,
              drop_last = False,
              space_h = None,
              space_w = None,
              space_stride_h= None,
              space_stride_w= None,         
              ):

    train_ds = ParFlowDataset(
        data_root,
        'train',
        pre_seq_length,
        aft_seq_length,
        in_shape=in_shape,
        use_augment=use_augment,
        space_h=space_h,
        space_w=space_w,
        space_stride_h=space_stride_h,
        space_stride_w=space_stride_w,
    )
    val_ds = ParFlowDataset(
        data_root,
        'val',
        pre_seq_length,
        aft_seq_length,
        in_shape=in_shape,
        use_augment=False,
        space_h=space_h,
        space_w=space_w,
        space_stride_h=space_stride_h,
        space_stride_w=space_stride_w,
    )
    test_ds = ParFlowDataset(
        data_root,
        'test',
        pre_seq_length,
        aft_seq_length,
        in_shape=in_shape,
        use_augment=False,
        space_h=space_h,
        space_w=space_w,
        space_stride_h=space_stride_h,
        space_stride_w=space_stride_w,
    )

    input_channels = train_ds.C

    train_loader = create_loader(
        train_ds,
        batch_size=batch_size,
        is_training=True,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
        input_channels=input_channels,
        drop_last=drop_last,
        persistent_workers=True,
    )
    vali_loader = create_loader(
        val_ds,
        batch_size=val_batch_size,
        is_training=False,
        shuffle=False,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
        input_channels=input_channels,
        drop_last=False,
        persistent_workers=True,
    )
    test_loader = create_loader(
        test_ds,
        batch_size=val_batch_size,
        is_training=False,
        shuffle=False,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
        input_channels=input_channels,
        drop_last=False,
        persistent_workers=True,
    )

    return train_loader, vali_loader, test_loader


if __name__ == '__main__':
    
    # 检查数据加载器和mean和std计算是否正确
    dataloader_train, dataloader_vali, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=16,
                  data_root='data/parflow_press',
                  num_workers=4,
                  pre_seq_length=6,
                  aft_seq_length=6,
                  space_h=64,
                  space_w=128,
                  space_stride_h=32,
                  space_stride_w=64,
                  )
    # print(dataloader_train.dataset.mean,dataloader_test.dataset.std)
    

    print(len(dataloader_train),len(dataloader_vali),len(dataloader_test))

    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break

    
    
    
    '''
    mean, std = compute_mean_std(
        files = _list_pfb_files('data/'),
    )
    print(f"✅ 计算完成！")
    print(f"   - 通道数 C = {mean.shape[0]}")
    print(f"   - Mean 示例: {mean[:3]} ...")
    print(f"   - Std  示例: {std[:3]} ...")

    # 保存为 .npz 文件
    np.savez(STATS_PATH, mean=mean, std=std)
    print(f"📦 统计量已保存至: {STATS_PATH}")    
    '''

    '''
    检查各个通道的均值和方差计算是否正确
    import pandas as  pd

    batch = next(iter(dataloader_train))
    x,y = batch
    # print(x[0,0,0,:,:])
    
    pdf = pd.DataFrame(x[0,0,0,:,:].numpy())
    pdf.to_csv("press_pfb_channel_0_augmented.csv",index=False,header=False)

    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            n = x[0,i,j,:,:].numpy().flatten()
            print(f"===== 第 {i} 个时间步，第 {j} 个通道 =====")
            print(f"最小值: {n.min():.2f}, 最大值: {n.max():.2f}, 均值: {n.mean():.2f}")
    '''

    '''
    # 计算均值和方差
    mean, std = compute_mean_std(
        files = _list_pfb_files('data/parflow_press/'),
    )
    print("=== 每个通道的 Mean 和 Std ===")
    C = mean.shape[0]
    for c in range(C):
        print(f"Channel {c}: Mean = {mean[c]:.6f}, Std = {std[c]:.6f}")
    np.savez(STATS_PATH, mean=mean, std=std)  # 保存为 stats.npz 文件
    print(f"✅ 均值和方差已保存到：{STATS_PATH}")
    '''

    '''
    # 统计 Channel 的最大值、最小值、均值、标准差
    for c in range(10):  # 假设有10个通道
        all_channel_data = []
        files = _list_pfb_files('data/')
        for file in files:
            data = _read_press_frame(file)
            all_channel_data.append(data[c, :, :].flatten())
            
        all_channel_data = np.array(all_channel_data)
        max_per_sample = [arr.max() for arr in all_channel_data]
        min_per_sample = [arr.min() for arr in all_channel_data]
        print(f"Channel {c} 的最大值为：", max(max_per_sample))
        print(f"Channel {c} 的最小值为：", min(min_per_sample))
        print(f"Channel {c} 异常样本数（比如最大值 > 1000）：", sum(v > 1000 for v in max_per_sample))
        print(f"Channel {c} 异常样本数（比如最小值 < -1000）：", sum(v < -1000 for v in min_per_sample))
        mean_c = np.mean(np.concatenate(all_channel_data))
        std_c = np.std(np.concatenate(all_channel_data))
        print(f"Channel {c} 的均值为：", mean_c)    
        print(f"Channel {c} 的标准差为：", std_c)
    '''


    '''
    files = _list_pfb_files('data/')
    print(f"🔍 总共读取了 {len(files)} 个 .pfb 文件")
    '''
# python /home/huanghui/data/ParFlow-transformer/openstl/datasets/dataloader_parflow.py
