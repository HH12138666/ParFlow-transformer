import os
import re
import glob
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

from .utils import create_loader

logger = logging.getLogger(__name__)


CROP_H = 128
CROP_W = 240
EPS = 1e-6


NORMALIZE = True
NORMALIZE_TARGET = True
STATS_PATH = None                  # 例如: "/data/stats_128x240.npz"
STATS_COMPUTE_SAMPLES = 0          # 如设 200 将在线抽样估计
STATS_TIME_STRIDE = 2
STATS_SPATIAL_STRIDE = 2



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

#计算均值和方差
def _welford_update(count, mean, M2, batch_mean, batch_M2, batch_n):
    total_n = count + batch_n
    delta = batch_mean - mean
    mean += delta * (batch_n / np.maximum(total_n, 1))
    M2 += batch_M2 + (delta * delta) * (count * batch_n / np.maximum(total_n, 1))
    count[:] = total_n
    return count, mean, M2

def compute_mean_std(files,
                    target_h=CROP_H,
                    target_w=CROP_W,
                    spatial_stride=1,
                    time_stride=1,
                    max_files=None,
                    channels=None):
    sel_files = files[::max(1, int(time_stride))]
    if max_files is not None:
        sel_files = sel_files[:int(max_files)]
    a0 = _read_press_frame(sel_files[0], target_h, target_w)
    if channels is not None:
        a0 = a0[channels, ...]
    if spatial_stride > 1:
        a0 = a0[:, ::spatial_stride, ::spatial_stride]
    C = a0.shape[0]
    count = np.zeros(C, dtype=np.float64)
    mean  = np.zeros(C, dtype=np.float64)
    M2    = np.zeros(C, dtype=np.float64)
    for f in sel_files:
        a = _read_press_frame(f, target_h, target_w)
        if channels is not None:
            a = a[channels, ...]
        if spatial_stride > 1:
            a = a[:, ::spatial_stride, ::spatial_stride]
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
                 noise_sigma=0.01):
    
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

    def __init__(self, data_root, split, pre_seq_length=9, aft_seq_length=1 ,in_shape: Optional[List[int]] = None,stride=1,use_augment=False):
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


        self.files = _list_pfb_files(self.root)
        self.num_frames = len(self.files)

        
         
        if in_shape is not None and len(in_shape) == 4:
            _, C, H, W = in_shape
            # 强制对齐为 10×144×252
            C = 10
            H = 128
            W = 240
        else:
            sample = _read_press_frame(self.files[0], target_h=144)
            C, H, W = sample.shape
        self.C, self.H, self.W = C, H, W   


        self.start_indices = self._build_time_indices(stride=max(1, int(stride)))


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
                    target_h=128, target_w=240,
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

    def _build_time_indices(self, stride=1):
        n_train = int(self.num_frames * 0.70)
        n_val   = int(self.num_frames * 0.15)
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
        return len(self.start_indices)

    def _read_window(self, t0):
        T = self.total
        out = torch.empty((T, self.C, self.H, self.W), dtype=torch.float32)
        for i in range(T):
            path = self.files[t0 + i]
            arr = _read_press_frame(path, target_h=CROP_H, target_w=CROP_W)
            out[i] = torch.from_numpy(arr)
        return out

    def __getitem__(self, idx):
        t0 = self.start_indices[idx]
        win = self._read_window(t0)  # [T, C, 128, 240]
        x = win[: self.pre]
        y = win[self.pre : self.pre + self.aft]

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
              pre_seq_length = 9,
              aft_seq_length = 1,
              in_shape: Optional[List[int]] = None,
              distributed = False,
              use_augment = True,
              use_prefetcher = False,
              drop_last = False,
              stride=1):
    train_ds = ParFlowDataset(data_root, 'train', pre_seq_length, aft_seq_length,in_shape=in_shape,stride=stride, use_augment=use_augment)
    try:
        val_ds = ParFlowDataset(data_root, 'val', pre_seq_length, aft_seq_length, in_shape=in_shape,stride=stride,use_augment=False)
    except Exception:
        val_ds = ParFlowDataset(data_root, 'test', pre_seq_length, aft_seq_length, in_shape=in_shape,stride=stride,use_augment=False)
    test_ds = ParFlowDataset(data_root, 'test', pre_seq_length, aft_seq_length, in_shape=in_shape,stride=stride,use_augment=False)

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
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=64,
                  val_batch_size=64,
                  data_root='data/',
                  num_workers=4,
                  pre_seq_length=9,
                  aft_seq_length=1)

    print(len(dataloader_train), len(dataloader_test))

    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break

    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break


