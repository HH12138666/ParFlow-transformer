import os
from pathlib import Path
import re
import glob
import logging
import random
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

from openstl.datasets.utils import create_loader

logger = logging.getLogger(__name__)

#数值稳定常量
EPS = 1e-6

#计算均值和方差相关设置
NORMALIZE = True
NORMALIZE_TARGET = True
STATS_PATH = './stats'   
STATS_NAME = 'stats_press_evaptrans_perm_x_alpha_n_porosity.npz'    
STATS_PATH = os.path.join(STATS_PATH, STATS_NAME)         
#是否拼接 static 数据
USE_STATIC = True
# evaptrans 固定使用 6-9 层，路径固定为 data_root/evaptrans
EVAP_CHANNELS = [6, 7, 8, 9]

#数据分割相关设置
time_stride = 5    # 时间步长，用于数据分割

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


def _list_pfb_files(root) :
    files = sorted(glob.glob(os.path.join(root, '*.pfb')), key=_natural_key)
    if not files:
        raise FileNotFoundError(f'No .pfb files found under: {root}')
    return files


def _resolve_parflow_roots(data_root, use_static=USE_STATIC):
    """
    Allow passing a base directory that contains subfolders:
    - press
    - evaptrans
    - static
    """
    base = Path(data_root)
    press_root = str(base / "press")
    evap_root = str(base / "evaptrans")
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

# 读取单个压力场文件
def _read_press_frame(press_path) :

    arr = read_pfb(get_absolute_path(press_path)).astype(np.float32)  # (C,H,W)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D array per .pfb, got shape {arr.shape} for {press_path}')
    return arr


# 读取单个蒸发传输场文件
def _read_evap_frame(evap_path):
    if evap_path is None:
        raise ValueError("evap_path is required when reading evaptrans data")
    if not Path(evap_path).exists():
        raise FileNotFoundError(f"Evaptrans file not found: {evap_path}")

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
                align_by_hour_id = False,):
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
        self.static_arr = _read_static_stack(self.static_root, self.static_data) if self.static_root is not None else None
        self.out_channels = out_channels  # 仅用于标签 y，输入 x 仍保留全部通道
        self.align_by_hour_id = align_by_hour_id
        
        if self.use_space:
            self.space_stride_h = space_stride_h or self.space_h
            self.space_stride_w = space_stride_w or self.space_w
        else:
            self.space_stride_h = None
            self.space_stride_w = None


        self.files = _list_pfb_files(self.press_root)
        if self.evap_root is not None:
            self.evap_files = _list_pfb_files(self.evap_root)
        else:
            self.evap_files = None
        if self.align_by_hour_id:
            press_map = _build_id_map(self.files, "press")
            press_ids = sorted(press_map)
            self.files = [press_map[i] for i in press_ids]
            if self.evap_files is not None:
                evap_map = _build_id_map(self.evap_files, "evap")
                missing = [i for i in press_ids if i not in evap_map]
                if missing:
                    raise ValueError(f"Missing evap hours for press ids (first 5): {missing[:5]}")
                extra = [i for i in evap_map.keys() if i not in press_map]
                if extra:
                    raise ValueError(f"Evap hours not in press ids (first 5): {extra[:5]}")
                self.evap_files = [evap_map[i] for i in press_ids]
        elif self.evap_files is not None and len(self.evap_files) != len(self.files):
            raise ValueError(
                f"press/evap file counts do not match: {len(self.files)} vs {len(self.evap_files)}"
            )

        self.num_frames = len(self.files)
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
            else:
                raise FileNotFoundError(
                    f"Stats file not found at {STATS_PATH}. "
                    "Please compute mean/std offline and save the npz first."
                )
        self.mean_t = torch.from_numpy(self.mean).view(1, self.C, 1, 1).float() if self.mean is not None else None
        self.std_t  = torch.from_numpy(self.std ).view(1, self.C, 1, 1).float() if self.std  is not None else None

    def _build_time_indices(self, stride=time_stride):
        n_train = int(self.num_frames * 0.7)
        n_val   = int(self.num_frames * 0.15)
        def build_range(s, e):
            end = e - 1
            max_start = end - self.total + 1
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
    
    
     
    def _read_window(self, t0):
        T = self.total
        out = torch.empty((T, self.C, self.H, self.W), dtype=torch.float32)
        for i in range(T):
            path = self.files[t0 + i]
            arr = _read_combined_frame(
                path,
                evap_path=self.evap_files[t0 + i] if self.evap_files is not None else None,
                static_arr=self.static_arr,
            )
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
        if self.out_channels is not None:
            y = y[:, :self.out_channels, :, :]  # 只保留预测通道
        
        if self.use_space:
            x = x[..., top : top + self.space_h, left : left + self.space_w]
            y = y[..., top : top + self.space_h, left : left + self.space_w]

        if self.use_augment and self.split == 'train':
            x, y = augment_pair(x, y)

        if NORMALIZE and self.mean_t is not None and self.std_t is not None:
            x = (x - self.mean_t) / (self.std_t + EPS)
            if NORMALIZE_TARGET:
                if self.out_channels is not None:
                    mean_y = self.mean_t[:, :self.out_channels, :, :]
                    std_y = self.std_t[:, :self.out_channels, :, :]
                    y = (y - mean_y) / (std_y + EPS)
                else:
                    y = (y - self.mean_t) / (self.std_t + EPS)

        return x, y


def load_data(batch_size,val_batch_size,data_root,num_workers,pre_seq_length = 6,aft_seq_length = 6,
              in_shape = None,distributed = False,use_augment = False,use_prefetcher = False,drop_last = False,
              space_h = None,space_w = None,space_stride_h= None,space_stride_w= None,out_channels = None,
              static_data = None,
              align_by_hour_id = False,
              ):

    use_static = USE_STATIC or static_data is not None
    press_root, evap_root, static_root = _resolve_parflow_roots(data_root, use_static=use_static)

    train_ds = ParFlowDataset(press_root,'train',pre_seq_length,aft_seq_length,
        in_shape=in_shape,use_augment=use_augment,
        space_h=space_h,space_w=space_w,space_stride_h=space_stride_h,space_stride_w=space_stride_w,
        evap_root=evap_root,static_root=static_root,out_channels=out_channels,static_data=static_data,
        align_by_hour_id=align_by_hour_id,)
    
    val_ds = ParFlowDataset(press_root, 'val',pre_seq_length,aft_seq_length,
        in_shape=in_shape,use_augment=False,
        space_h=space_h,space_w=space_w,space_stride_h=space_stride_h,space_stride_w=space_stride_w,
        evap_root=evap_root,static_root=static_root,out_channels=out_channels,static_data=static_data,
        align_by_hour_id=align_by_hour_id,)
    
    test_ds = ParFlowDataset(press_root,'test',pre_seq_length,aft_seq_length,
        in_shape=in_shape,use_augment=False,
        space_h=space_h,space_w=space_w,space_stride_h=space_stride_h,space_stride_w=space_stride_w,
        evap_root=evap_root,static_root=static_root,out_channels=out_channels,static_data=static_data,
        align_by_hour_id=align_by_hour_id,)

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
