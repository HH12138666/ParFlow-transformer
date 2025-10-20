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


def _natural_key(p: str):
    b = os.path.basename(p)
    s = re.split(r'(\d+)', b)
    return [int(t) if t.isdigit() else t for t in s]


def _list_pfb_files(root: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, '*.pfb')), key=_natural_key)
    if not files:
        raise FileNotFoundError(f'No .pfb files found under: {root}')
    return files


def _center_crop_h(arr: np.ndarray, target_h: int) -> np.ndarray:
    """arr: (C,H,W) -> crop along H to target_h (center-crop)."""
    C, H, W = arr.shape
    if H == target_h:
        return arr
    if H < target_h:
        raise ValueError(f'Height {H} < target {target_h}, cannot crop')
    off = (H - target_h) // 2
    return arr[:, off:off + target_h, :]

"""
Read a single .pfb press file.
Expected original: [C=10, H=146, W=252]; we center-crop H->144.
"""

def _read_press_frame(path: str, target_h: int = 144) -> np.ndarray:

    arr = read_pfb(get_absolute_path(path)).astype(np.float32)  # (C,H,W)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D array per .pfb, got shape {arr.shape} for {path}')
    
    arr = _center_crop_h(arr, target_h=target_h)
    return arr


class ParFlowDataset(Dataset):
    """
    ParFlow press .pfb sequence dataset.
    - Each file is one timestep: (C,H,W) with C=10, H=146, W=252 (we crop to H=144)
    - Return sliding windows:
        X: [pre, C=10, H=144, W=252]
        Y: [aft, C=10, H=144, W=252]
    """
    def __init__(
        self,
        data_root: str,
        split: str,
        pre_seq_length: int,
        aft_seq_length: int,
        in_shape: Optional[List[int]] = None,
        cfg=None,                   # optional explicit time ranges/strides
        compute_stats_samples: int = 100,
    ):
        super().__init__()
        assert split in ('train', 'val', 'validation', 'valid', 'test')
        self.split = 'val' if split in ('val', 'validation', 'valid') else split
        self.root = data_root
        self.pre = pre_seq_length
        self.aft = aft_seq_length
        self.total = self.pre + self.aft
        self.cfg = cfg

        # enumerate frames
        self.files = _list_pfb_files(self.root)
        self.num_frames = len(self.files)

        # infer shape (after crop)
        if in_shape is not None and len(in_shape) == 4:
            _, C, H, W = in_shape
            # 强制对齐为 10×144×252
            C = 10
            H = 144
            W = 252
        else:
            sample = _read_press_frame(self.files[0], target_h=144)
            C, H, W = sample.shape
        self.C, self.H, self.W = C, H, W   # 期望 10,144,252

        # build split indices
        self.start_indices = self._build_time_indices()

        # estimate per-channel mean/std (用于标准化)
        self.mean, self.std = self._estimate_mean_std(self.files, samples=compute_stats_samples)

        # metadata for OpenSTL
        self.data_name = 'parflow_press'
        self.mean_t = torch.from_numpy(self.mean).view(1, self.C, 1, 1).float()
        self.std_t  = torch.from_numpy(self.std ).view(1, self.C, 1, 1).float()
    """
    Priority 1: cfg.training_* / test_* / (val_*) + stride
    Priority 2: ratio split 70/15/15 with stride=1
    """
    def _build_time_indices(self) -> List[int]:

        if self.cfg is not None:
            try:
                if self.split == 'train':
                    start = int(self.cfg.training_start_step)
                    end   = int(self.cfg.training_end_step)
                    stride = int(getattr(self.cfg, 'st_stride_train', 1))
                else:
                    start = int(self.cfg.test_start_step) if self.split == 'test' else int(getattr(self.cfg, 'val_start_step', 0))
                    if self.split == 'val' and not hasattr(self.cfg, 'val_end_step'):
                        raise AttributeError
                    end   = int(self.cfg.test_end_step) if self.split == 'test' else int(self.cfg.val_end_step)
                    stride = int(getattr(self.cfg, 'st_stride_test', 1))
                start = max(0, start)
                end = min(self.num_frames - 1, end)
                max_start = end - self.total + 1
                if max_start < start:
                    raise ValueError(f'Not enough frames: start={start}, end={end}, total={self.total}')
                return list(range(start, max_start + 1, stride))
            except Exception:
                pass

        n_train = int(self.num_frames * 0.70)
        n_val   = int(self.num_frames * 0.15)

        def build_range(s, e):
            e = e - 1
            max_start = e - self.total + 1
            if max_start < s:
                return []
            return list(range(s, max_start + 1, 1))

        if self.split == 'train':
            return build_range(0, n_train)
        elif self.split == 'val':
            return build_range(n_train, n_train + n_val)
        else:
            return build_range(n_train + n_val, self.num_frames)


    def _estimate_mean_std(self, paths: List[str], samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        n = min(samples, len(paths))
        if n <= 0:
            return np.zeros((self.C,), dtype=np.float32), np.ones((self.C,), dtype=np.float32)
        idxs = np.linspace(0, len(paths) - 1, n, dtype=int)

        sum_m = np.zeros((self.C,), dtype=np.float64)
        sum_ex2 = np.zeros((self.C,), dtype=np.float64)
        for i in idxs:
            arr = _read_press_frame(paths[i], target_h=144)  # (C,144,252)
            x = arr.reshape(self.C, -1)
            m = x.mean(axis=1)
            ex2 = (x**2).mean(axis=1)
            sum_m += m
            sum_ex2 += ex2
        mean = (sum_m / n).astype(np.float32)
        ex2  = (sum_ex2 / n).astype(np.float32)
        std = np.sqrt(np.maximum(ex2 - mean**2, 1e-12)).astype(np.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    def __len__(self) -> int:
        return len(self.start_indices)

    def _read_window(self, t0: int) -> torch.Tensor:
        T = self.total
        out = torch.empty((T, self.C, self.H, self.W), dtype=torch.float32)
        mean = torch.from_numpy(self.mean).view(self.C, 1, 1)
        std  = torch.from_numpy(self.std ).view(self.C, 1, 1)
        for i in range(T):
            path = self.files[t0 + i]
            arr = _read_press_frame(path, target_h=144)   # (10,144,252)
            ten = torch.from_numpy(arr)
            ten = (ten - mean) / std
            out[i] = ten
        return out

    def __getitem__(self, idx: int):
        t0 = self.start_indices[idx]
        win = self._read_window(t0)  # [T, 10, 144, 252]
        x = win[: self.pre]
        y = win[self.pre : self.pre + self.aft]
        return x, y

    """
    Create train/val/test loaders for ParFlow press .pfb sequences (C=10, H->144, W=252).
    data_root: directory containing 00000.pfb ... 08760.pfb

    """
def load_data(batch_size: int,
              val_batch_size: int,
              data_root: str,
              num_workers: int,
              pre_seq_length: int = 12,
              aft_seq_length: int = 12,
              in_shape: Optional[List[int]] = None,
              distributed: bool = False,
              use_augment: bool = False,
              use_prefetcher: bool = False,
              drop_last: bool = False,
              **kwargs):

    cfg = kwargs.get('configs', kwargs.get('cfg', None))

    train_ds = ParFlowDataset(
        data_root=data_root,
        split='train',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        in_shape=in_shape,
        cfg=cfg,
    )
    try:
        val_ds = ParFlowDataset(
            data_root=data_root,
            split='val',
            pre_seq_length=pre_seq_length,
            aft_seq_length=aft_seq_length,
            in_shape=in_shape,
            cfg=cfg,
        )
    except Exception:
        val_ds = ParFlowDataset(
            data_root=data_root,
            split='test',
            pre_seq_length=pre_seq_length,
            aft_seq_length=aft_seq_length,
            in_shape=in_shape,
            cfg=cfg,
        )
    test_ds = ParFlowDataset(
        data_root=data_root,
        split='test',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        in_shape=in_shape,
        cfg=cfg,
    )

    input_channels = train_ds.C  # 10

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
        load_data(batch_size=16,
                  val_batch_size=16,
                  data_root='data/',
                  num_workers=4,
                  pre_seq_length=12,
                  aft_seq_length=12)

    print(len(dataloader_train), len(dataloader_test))

    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break

    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break