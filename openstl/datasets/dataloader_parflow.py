import os
from pathlib import Path
import re
import glob
import logging
import random
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

from openstl.datasets.utils import create_loader

logger = logging.getLogger(__name__)

_STATIC_STACK_CACHE = {}
_STATS_CACHE = {}

#数值稳定常量
EPS = 1e-8

#计算均值和方差相关设置
NORMALIZE = True
NORMALIZE_TARGET = True
STATS_PATH = './stats'   
STATS_NAME = 'stats_wtd.npz'    
STATS_PATH = os.path.join(STATS_PATH, STATS_NAME)         
#是否拼接 static 数据
USE_STATIC = False
# evaptrans 固定使用 6-9 层，路径固定为 data_root/evaptrans
EVAP_CHANNELS = [6, 7, 8, 9]

#数据分割相关设置
time_stride = 5    # 时间步长，用于数据分割


def _compute_patch_padding(height, width, patch_size):
    if patch_size is None or patch_size <= 0:
        return 0, 0
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    return pad_h, pad_w

def _natural_key(p):
    b = os.path.basename(p)
    s = re.split(r'(\d+)', b)
    return [int(t) if t.isdigit() else t for t in s]

def _extract_hour_id(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)(?!.*\d)", name)
    if not m:
        return None
    return int(m.group(1))


def _extract_year(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)(?!.*\d)", name)
    if m:
        token = m.group(1)
        if len(token) >= 8:
            return int(token[:4])
        if len(token) == 4:
            year = int(token)
            if 1900 <= year <= 2100:
                return year

    parent = os.path.basename(os.path.dirname(path))
    if re.fullmatch(r"(19|20)\d{2}", parent):
        return int(parent)
    return None


def _normalize_years(years):
    if years is None:
        return None
    if isinstance(years, int):
        return [years]
    return [int(y) for y in years]


def _build_year_ranges(files):
    year_ranges = {}
    for idx, f in enumerate(files):
        year = _extract_year(f)
        if year is None:
            raise ValueError(f"Cannot parse year from filename/folder: {f}")
        if year not in year_ranges:
            year_ranges[year] = [idx, idx + 1]
        else:
            year_ranges[year][1] = idx + 1
    return {y: (rng[0], rng[1]) for y, rng in year_ranges.items()}

def _build_id_map(files, label):
    items = {}
    for f in files:
        fid = _extract_hour_id(f)
        if fid is None:
            raise ValueError(f"{label} file has no hour id: {f}")
        if fid in items:
            raise ValueError(f"Duplicate {label} hour id {fid}: {items[fid]} and {f}")
        items[fid] = f
    if not items:
        raise ValueError(f"No {label} files with valid hour ids found.")
    return items


def _prepare_press_evap_files(press_root, evap_root=None, align_by_hour_id=False):
    press_files = _list_pfb_files(press_root)
    evap_files = _list_pfb_files(evap_root) if evap_root is not None else None

    if align_by_hour_id:
        press_map = _build_id_map(press_files, "press")
        press_ids = sorted(press_map)
        press_files = [press_map[i] for i in press_ids]
        if evap_files is not None:
            evap_map = _build_id_map(evap_files, "evap")
            missing = [i for i in press_ids if i not in evap_map]
            if missing:
                raise ValueError(f"Missing evap hours for press ids (first 5): {missing[:5]}")
            extra = [i for i in evap_map.keys() if i not in press_map]
            if extra:
                raise ValueError(f"Evap hours not in press ids (first 5): {extra[:5]}")
            evap_files = [evap_map[i] for i in press_ids]
    elif evap_files is not None and len(evap_files) != len(press_files):
        raise ValueError(
            f"press/evap file counts do not match: {len(press_files)} vs {len(evap_files)}"
        )

    return press_files, evap_files


def _list_pfb_files(root) :
    root_path = Path(root)
    files = sorted((str(p) for p in root_path.rglob('*.pfb')), key=_natural_key)
    if not files:
        raise FileNotFoundError(f'No .pfb files found under: {root}')
    return files


def _resolve_parflow_roots(data_root, use_static=USE_STATIC, var_name='press', use_evap=True):
    """
    Allow passing a base directory that contains subfolders:
    - press/wtd/...
    - evaptrans
    - static
    """
    base = Path(data_root)
    press_root = str(base / var_name)
    evap_root = None
    if use_evap:
        evap_root_path = base / "evaptrans"
        if not evap_root_path.exists():
            alt = base / "evapotrans"
            if alt.exists():
                evap_root_path = alt
        evap_root = str(evap_root_path)
    static_root = str(base / "static") if use_static else None
    return press_root, evap_root, static_root


def _parse_static_data(static_data):
    if static_data is None:
        return None
    if isinstance(static_data, (list, tuple)):
        patterns = [str(x).strip() for x in static_data if str(x).strip()]
    else:
        patterns = [p.strip() for p in str(static_data).split(',') if p.strip()]
    return patterns or None


def _filter_static_files(files, static_data):
    patterns = _parse_static_data(static_data)
    if not patterns:
        return files
    matched = []
    for f in files:
        name = os.path.basename(f)
        if any(re.search(pat, name, re.IGNORECASE) for pat in patterns):
            matched.append(f)
    if not matched:
        raise FileNotFoundError(f'No static .pfb files matched patterns: {patterns}')
    return matched

# 根据空间尺寸和步长构建空间裁剪坐标列表
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

# 读取单个动态场文件（press/wtd 等）
def _read_press_frame(press_path) :

    arr = read_pfb(get_absolute_path(press_path)).astype(np.float32)  # (C,H,W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f'Expected 2D/3D array per .pfb, got shape {arr.shape} for {press_path}')
    return arr


# 读取单个蒸发传输场文件
def _read_evap_frame(evap_path):
    if evap_path is None:
        raise ValueError("evap_path is required when reading evaptrans data")

    arr = read_pfb(get_absolute_path(str(evap_path))).astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f'Expected 3D evaptrans array, got shape {arr.shape} for {evap_path}')

    return arr[EVAP_CHANNELS, ...]

# 读取静态数据堆栈
def _read_static_stack(static_root, static_data=None):
    if static_root is None:
        return None
    files = _list_pfb_files(static_root)
    files = _filter_static_files(files, static_data)
    arrays = []
    for f in files:
        arr = read_pfb(get_absolute_path(str(f))).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            raise ValueError(f'Expected 2D/3D static array, got shape {arr.shape} for {f}')
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def _get_static_stack_cached(static_root, static_data=None):
    patterns = _parse_static_data(static_data)
    cache_key = (str(static_root), tuple(patterns) if patterns is not None else None)
    cached = _STATIC_STACK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    arr = _read_static_stack(static_root, static_data)
    _STATIC_STACK_CACHE[cache_key] = arr
    return arr


def _get_stats_cached(stats_path):
    cached = _STATS_CACHE.get(stats_path)
    if cached is not None:
        return cached
    data = np.load(stats_path)
    mean = np.asarray(data['mean'], dtype=np.float32).reshape(-1)
    std = np.asarray(data['std'], dtype=np.float32).reshape(-1)
    _STATS_CACHE[stats_path] = (mean, std)
    return mean, std


def _read_combined_frame(press_path,evap_path = None,static_arr = None):
    """读取压力场，并可选拼接指定层的 evaptrans 和 static 通道"""
    press = _read_press_frame(press_path)
    if evap_path is None:
        combined = press
    else:
        evap = _read_evap_frame(evap_path)
        combined = np.concatenate([press, evap], axis=0)
    if static_arr is not None:
        if static_arr.shape[1:] != combined.shape[1:]:
            raise ValueError(
                f"Static shape {static_arr.shape} does not match frame shape {combined.shape} for {press_path}"
            )
        combined = np.concatenate([combined, static_arr], axis=0)
    return combined

#增强数据,暂时用不到
def augment_pair(X, Y,p_flip_h=0.5,p_flip_w=0.5,p_noise=0.2,noise_sigma=0.001):
    
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
    def __init__(self, press_root, split, pre_seq_length=12, aft_seq_length=12 ,in_shape = None,use_augment=False,
                space_h = None,
                space_w = None,
                space_stride_h = None,
                space_stride_w = None,
                evap_root = None,
                static_root = None,
                out_channels = None,
                static_data = None,
                align_by_hour_id = False,
                press_files = None,
                evap_files = None,
                patch_size = None,
                pad_to_patch = False,
                split_mode = 'ratio',
                train_years = None,
                holdout_years = None,
                val_ratio_in_holdout = 0.25,
                ):
        super().__init__()
        split = str(split).lower()
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid split: {split}. Expected 'train' | 'val' | 'test'.")
        self.split = split

        self.press_root = press_root
        self.pre = pre_seq_length
        self.aft = aft_seq_length
        self.total = self.pre + self.aft
        self.use_augment = use_augment
        self.space_h = space_h
        self.space_w = space_w
        self.use_space = self.space_h is not None and self.space_w is not None
        self.evap_root = evap_root
        self.static_root = static_root
        self.static_data = static_data
        self.static_arr = (
            _get_static_stack_cached(self.static_root, self.static_data)
            if self.static_root is not None else None
        )
        self.out_channels = out_channels  # 仅用于标签 y，输入 x 仍保留全部通道
        self.align_by_hour_id = align_by_hour_id
        self.patch_size = patch_size
        self.pad_to_patch = bool(pad_to_patch)
        self.split_mode = str(split_mode).lower()
        self.train_years = _normalize_years(train_years)
        self.holdout_years = _normalize_years(holdout_years)
        self.val_ratio_in_holdout = float(val_ratio_in_holdout)
        
        if self.use_space:
            self.space_stride_h = space_stride_h or self.space_h
            self.space_stride_w = space_stride_w or self.space_w
        else:
            self.space_stride_h = None
            self.space_stride_w = None


        if press_files is None:
            self.files, self.evap_files = _prepare_press_evap_files(
                self.press_root,
                self.evap_root,
                align_by_hour_id=self.align_by_hour_id,
            )
        else:
            self.files = list(press_files)
            self.evap_files = list(evap_files) if evap_files is not None else None
            if self.evap_files is not None and len(self.evap_files) != len(self.files):
                raise ValueError(
                    f"press/evap file counts do not match: {len(self.files)} vs {len(self.evap_files)}"
                )

        self.num_frames = len(self.files)
        self.year_ranges = _build_year_ranges(self.files)
        sample = _read_combined_frame(
            self.files[0],
            evap_path=self.evap_files[0] if self.evap_files is not None else None,
            static_arr=self.static_arr,
        )
        C, H, W = sample.shape
        self.C, self.H, self.W = C, H, W   
        
        if self.use_space:
            self.space_coords = _build_space_coords(
                self.H, self.W, self.space_h, self.space_w, self.space_stride_h, self.space_stride_w
            )
        else:
            self.space_coords = [(0, 0)]

        self.valid_h = self.space_h if self.use_space else self.H
        self.valid_w = self.space_w if self.use_space else self.W
        self.pad_h, self.pad_w = (0, 0)
        if self.pad_to_patch:
            self.pad_h, self.pad_w = _compute_patch_padding(self.valid_h, self.valid_w, self.patch_size)
        self.padded_h = self.valid_h + self.pad_h
        self.padded_w = self.valid_w + self.pad_w

        self.time_indices = self._build_time_indices()
        
        if self.use_space:
            self.sample_indices = [
                (t, p) for t in self.time_indices for p in range(len(self.space_coords))
            ]
        else:
            self.sample_indices = self.time_indices

        self.mean = None
        self.std = None

        if NORMALIZE:
            if STATS_PATH and os.path.exists(STATS_PATH):
                self.mean, self.std = _get_stats_cached(STATS_PATH)
                if self.mean.shape[0] != self.C or self.std.shape[0] != self.C:
                    raise ValueError(f"Stats mismatch: stats C={self.mean.shape[0]} vs dataset C={self.C}.")
            else:
                raise FileNotFoundError(
                    f"Stats file not found at {STATS_PATH}. "
                    "Please compute mean/std offline and save the npz first."
                )
        self.mean_t = torch.from_numpy(self.mean).view(1, self.C, 1, 1).float() if self.mean is not None else None
        self.std_t = torch.from_numpy(self.std).view(1, self.C, 1, 1).float() if self.std is not None else None
        self.std_eps_t = (self.std_t + EPS) if self.std_t is not None else None
        if self.out_channels is not None and self.mean_t is not None and self.std_t is not None:
            self.mean_y_t = self.mean_t[:, :self.out_channels, :, :]
            self.std_y_eps_t = self.std_t[:, :self.out_channels, :, :] + EPS
        else:
            self.mean_y_t = None
            self.std_y_eps_t = None

    def _build_time_indices(self, stride=time_stride):
        def build_range(s, e):
            end = e - 1
            max_start = end - self.total + 1
            if max_start < s:
                return []
            return list(range(s, max_start + 1, stride))

        if self.split_mode == 'ratio':
            n_train = int(self.num_frames * 0.7)
            n_val = int(self.num_frames * 0.15)
            if self.split == 'train':
                return build_range(0, n_train)
            if self.split == 'val':
                return build_range(n_train, n_train + n_val)
            return build_range(n_train + n_val, self.num_frames)

        if self.split_mode != 'year':
            raise ValueError(
                f"Invalid split_mode={self.split_mode}. Expected 'ratio' or 'year'."
            )

        available_years = sorted(self.year_ranges.keys())
        if not available_years:
            raise ValueError("No available years parsed from files.")

        train_years = self.train_years or [available_years[0]]
        holdout_years = self.holdout_years
        if holdout_years is None:
            holdout_years = [y for y in available_years if y not in train_years]
            if not holdout_years and len(available_years) >= 2:
                holdout_years = [available_years[-1]]

        if not holdout_years:
            raise ValueError(
                f"holdout_years is empty. available_years={available_years}, train_years={train_years}"
            )
        if not (0.0 < self.val_ratio_in_holdout < 1.0):
            raise ValueError(f"val_ratio_in_holdout must be in (0,1), got {self.val_ratio_in_holdout}")

        def year_val_test_range(year):
            if year not in self.year_ranges:
                raise ValueError(f"Year {year} not in dataset. available_years={available_years}")
            s, e = self.year_ranges[year]
            n = e - s
            n_val = int(round(n * self.val_ratio_in_holdout))
            n_val = max(1, min(n - 1, n_val))
            return (s, s + n_val), (s + n_val, e)

        indices = []
        if self.split == 'train':
            for year in train_years:
                if year not in self.year_ranges:
                    raise ValueError(f"Year {year} not in dataset. available_years={available_years}")
                s, e = self.year_ranges[year]
                indices.extend(build_range(s, e))
            return indices

        if self.split == 'val':
            for year in holdout_years:
                (s, e), _ = year_val_test_range(year)
                indices.extend(build_range(s, e))
            return indices

        for year in holdout_years:
            _, (s, e) = year_val_test_range(year)
            indices.extend(build_range(s, e))
        return indices

    def __len__(self):      
        return len(self.sample_indices)
    
    
     
    def _read_window(self, t0, top=0, left=0):
        T = self.total
        if self.use_space:
            h, w = self.valid_h, self.valid_w
        else:
            h, w = self.valid_h, self.valid_w
        out = torch.empty((T, self.C, h, w), dtype=torch.float32)
        for i in range(T):
            path = self.files[t0 + i]
            arr = _read_combined_frame(
                path,
                evap_path=self.evap_files[t0 + i] if self.evap_files is not None else None,
                static_arr=self.static_arr,
            )
            if self.use_space:
                arr = arr[:, top:top + h, left:left + w]
            out[i] = torch.from_numpy(arr)
        return out

    def _pad_tensor(self, tensor):
        if self.pad_h == 0 and self.pad_w == 0:
            return tensor
        pad = (0, self.pad_w, 0, self.pad_h)
        return F.pad(tensor, pad, mode='replicate')

    def __getitem__(self, idx):

        if self.use_space:
            t0, p_idx = self.sample_indices[idx]
            top, left = self.space_coords[p_idx]
        else:
            t0 = self.sample_indices[idx]
            top, left = 0, 0
        win = self._read_window(t0, top=top, left=left)
          
        x = win[: self.pre]
        y = win[self.pre : self.pre + self.aft]
        if self.out_channels is not None:
            y = y[:, :self.out_channels, :, :]  # 只保留预测通道

        if self.use_augment and self.split == 'train':
            x, y = augment_pair(x, y)

        if NORMALIZE and self.mean_t is not None and self.std_t is not None:
            x = x.sub(self.mean_t).div(self.std_eps_t)
            if NORMALIZE_TARGET:
                if self.out_channels is not None:
                    y = y.sub(self.mean_y_t).div(self.std_y_eps_t)
                else:
                    y = y.sub(self.mean_t).div(self.std_eps_t)

        if self.pad_to_patch:
            x = self._pad_tensor(x)
            y = self._pad_tensor(y)

        return x, y


def load_data(batch_size,val_batch_size,data_root,num_workers,pre_seq_length = 6,aft_seq_length = 6,
              in_shape = None,distributed = False,use_augment = False,use_prefetcher = False,drop_last = False,
              space_h = None,space_w = None,space_stride_h= None,space_stride_w= None,out_channels = None,
              static_data = None,
              align_by_hour_id = False,
              patch_size = None,
              pad_to_patch = False,
              split_mode = 'ratio',
              train_years = None,
              holdout_years = None,
              val_ratio_in_holdout = 0.25,
              var_name = 'press',
              use_evap = True,
              use_static_input = USE_STATIC,
              ):

    use_static = use_static_input or static_data is not None
    press_root, evap_root, static_root = _resolve_parflow_roots(
        data_root, use_static=use_static, var_name=var_name, use_evap=use_evap
    )
    all_press_files, all_evap_files = _prepare_press_evap_files(
        press_root,
        evap_root,
        align_by_hour_id=align_by_hour_id,
    )

    train_ds = ParFlowDataset(press_root,'train',pre_seq_length,aft_seq_length,
        in_shape=in_shape,use_augment=use_augment,
        space_h=space_h,space_w=space_w,space_stride_h=space_stride_h,space_stride_w=space_stride_w,
        evap_root=evap_root,static_root=static_root,out_channels=out_channels,static_data=static_data,
        align_by_hour_id=align_by_hour_id,press_files=all_press_files,evap_files=all_evap_files,
        patch_size=patch_size,pad_to_patch=pad_to_patch,
        split_mode=split_mode,train_years=train_years,holdout_years=holdout_years,
        val_ratio_in_holdout=val_ratio_in_holdout,)
    
    val_ds = ParFlowDataset(press_root, 'val',pre_seq_length,aft_seq_length,
        in_shape=in_shape,use_augment=False,
        space_h=space_h,space_w=space_w,space_stride_h=space_stride_h,space_stride_w=space_stride_w,
        evap_root=evap_root,static_root=static_root,out_channels=out_channels,static_data=static_data,
        align_by_hour_id=align_by_hour_id,press_files=all_press_files,evap_files=all_evap_files,
        patch_size=patch_size,pad_to_patch=pad_to_patch,
        split_mode=split_mode,train_years=train_years,holdout_years=holdout_years,
        val_ratio_in_holdout=val_ratio_in_holdout,)
    
    test_ds = ParFlowDataset(press_root,'test',pre_seq_length,aft_seq_length,
        in_shape=in_shape,use_augment=False,
        space_h=space_h,space_w=space_w,space_stride_h=space_stride_h,space_stride_w=space_stride_w,
        evap_root=evap_root,static_root=static_root,out_channels=out_channels,static_data=static_data,
        align_by_hour_id=align_by_hour_id,press_files=all_press_files,evap_files=all_evap_files,
        patch_size=patch_size,pad_to_patch=pad_to_patch,
        split_mode=split_mode,train_years=train_years,holdout_years=holdout_years,
        val_ratio_in_holdout=val_ratio_in_holdout,)

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

if __name__ == "__main__":
    # 简单测试数据集和数据加载器
    data_root = '/home/huanghui/data/ParFlow-transformer/data/parflow'
    batch_size = 28
    val_batch_size = 28
    num_workers = 4
    pre_seq_length = 12
    aft_seq_length = 12
    in_shape = [24, 36, 146, 252]  # 10 个压力层 + 4 个 evaptrans 层 + static 数据
    space_h = 60
    space_w = 84
    space_stride_h = 30
    space_stride_w = 42
    out_channels = 14  # 预测压力 + evaptrans 通道

    train_loader, vali_loader, test_loader = load_data(
        batch_size,
        val_batch_size,
        data_root,
        num_workers,
        pre_seq_length,
        aft_seq_length,
        in_shape,
        distributed=False,
        use_augment=True,
        use_prefetcher=False,
        drop_last=False,
        space_h=space_h,
        space_w=space_w,
        space_stride_h=space_stride_h,
        space_stride_w=space_stride_w,
        out_channels=out_channels,
        static_data='perm_x,alpha_z6-9,n_z6-9,porosity_z6-9',
    )

    for x, y in train_loader:
        print("Train batch - x shape:", x.shape, "y shape:", y.shape)
        break

    for x, y in vali_loader:
        print("Val batch - x shape:", x.shape, "y shape:", y.shape)
        break

    for x, y in test_loader:
        print("Test batch - x shape:", x.shape, "y shape:", y.shape)
        break
#export PYTHONPATH=/home/huanghui/data/ParFlow-transformer:$PYTHONPATH
#python /home/huanghui/data/ParFlow-transformer/openstl/datasets/dataloader_parflow.py
