import os
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

from openstl.datasets.utils import create_loader

_STATIC_STACK_CACHE = {}
_STATS_CACHE = {}

# =========================
# 全局配置
# =========================

# 数值稳定常量
EPS = 1e-8

# 数据分割相关设置
time_stride = 6    # 时间步长，用于数据分割


# =========================
# 文件名 / 年份 / 小时 ID 解析
# =========================

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
    """把按时间排序的文件列表转换成 {year: (start_idx, end_idx)}。"""
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
    """把文件列表转换成 {hour_id: filepath}，并检查重复 ID。"""
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


def _filter_files_by_years(files, allowed_years):
    """按年份过滤文件列表；None 表示保留全部年份。"""
    if allowed_years is None:
        return files
    year_set = {int(year) for year in allowed_years}
    filtered = [path for path in files if _extract_year(path) in year_set]
    if not filtered:
        raise ValueError(f"No files found for allowed_years={sorted(year_set)}.")
    return filtered


 # =========================
 # 文件收集与路径解析
 # =========================

def _prepare_press_evap_apcp_files(press_root, evap_root=None, apcp_root=None, allowed_years=None):
    """准备主变量与辅助动态文件列表，并按小时 ID 与主变量严格对齐。"""
    press_files = _list_pfb_files(press_root)
    evap_files = _list_pfb_files(evap_root) if evap_root is not None else None
    apcp_files = _list_pfb_files(apcp_root) if apcp_root is not None else None

    press_files = _filter_files_by_years(press_files, allowed_years)
    evap_files = _filter_files_by_years(evap_files, allowed_years) if evap_files is not None else None
    apcp_files = _filter_files_by_years(apcp_files, allowed_years) if apcp_files is not None else None

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

    if apcp_files is not None:
        apcp_map = _build_id_map(apcp_files, "APCP")
        missing = [i for i in press_ids if i not in apcp_map]
        if missing:
            raise ValueError(f"Missing APCP hours for press ids (first 5): {missing[:5]}")
        extra = [i for i in apcp_map.keys() if i not in press_map]
        if extra:
            raise ValueError(f"APCP hours not in press ids (first 5): {extra[:5]}")
        apcp_files = [apcp_map[i] for i in press_ids]

    return press_files, evap_files, apcp_files


def _list_pfb_files(root):
    """递归列出目录下所有 .pfb 文件，并按自然顺序排序。"""
    root_path = Path(root)
    files = sorted((str(p) for p in root_path.rglob('*.pfb')), key=_natural_key)
    if not files:
        raise FileNotFoundError(f'No .pfb files found under: {root}')
    return files


def _resolve_parflow_roots(data_root, use_static=True, var_name='press', use_evap=True, use_apcp=False):
    """根据 data_root 推导主变量、evap、APCP 和 static 的根目录。"""
    base = Path(data_root)
    press_root = str(base / var_name)
    evap_root = None
    apcp_root = None
    if use_evap:
        evap_root_path = base / "evaptrans"
        if not evap_root_path.exists():
            alt = base / "evapotrans"
            if alt.exists():
                evap_root_path = alt
        evap_root = str(evap_root_path)
    if use_apcp:
        apcp_root_path = base / "APCP"
        if not apcp_root_path.exists():
            alt = base / "apcp"
            if alt.exists():
                apcp_root_path = alt
        if apcp_root_path.exists():
            apcp_root = str(apcp_root_path)
    static_root = str(base / "static") if use_static else None
    return press_root, evap_root, apcp_root, static_root


# =========================
# 空间裁剪坐标
# =========================


def _build_space_coords(height, width, space_h, space_w, space_stride_h=None, space_stride_w=None):
    """根据窗口大小和步长，构建所有滑窗左上角坐标。"""
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


# =========================
# PFB 读取与通道拼接
# =========================

def _read_press_frame(press_path):
    """读取主变量单帧，返回形状 (C, H, W)。"""
    arr = read_pfb(get_absolute_path(press_path)).astype(np.float32)  # (C,H,W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f'Expected 2D/3D array per .pfb, got shape {arr.shape} for {press_path}')
    return arr


def _read_evap_frame(evap_path):
    """读取 evaptrans 单帧（保留全部层）。"""
    if evap_path is None:
        raise ValueError("evap_path is required when reading evaptrans data")

    arr = read_pfb(get_absolute_path(str(evap_path))).astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f'Expected 3D evaptrans array, got shape {arr.shape} for {evap_path}')

    return arr


def _read_apcp_frame(apcp_path):
    """读取 APCP 单帧，返回形状 (C, H, W)。"""
    if apcp_path is None:
        raise ValueError("apcp_path is required when reading APCP data")
    arr = read_pfb(get_absolute_path(str(apcp_path))).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f'Expected 2D/3D APCP array, got shape {arr.shape} for {apcp_path}')
    return arr


def _read_static_stack(static_root, static_data=None):
    """仅读取融合后的 static.pfb，返回形状 (C, H, W)。"""
    if static_root is None:
        return None

    merged_static = Path(static_root) / "static.pfb"
    if not merged_static.exists():
        raise FileNotFoundError(
            f"Static file not found: {merged_static}. "
            "Please provide a merged static/static.pfb."
        )

    arr = read_pfb(get_absolute_path(str(merged_static))).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f'Expected 2D/3D static array, got shape {arr.shape} for {merged_static}')
    return arr


def _get_static_stack_cached(static_root, static_data=None):
    cache_key = str(static_root)
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


def _read_combined_frame(press_path, evap_path=None, apcp_path=None, static_arr=None):
    """读取压力场，并可选拼接 evaptrans/APCP/static 通道。"""
    press = _read_press_frame(press_path)
    combined = press
    if evap_path is not None:
        evap = _read_evap_frame(evap_path)
        combined = np.concatenate([combined, evap], axis=0)
    if apcp_path is not None:
        apcp = _read_apcp_frame(apcp_path)
        combined = np.concatenate([combined, apcp], axis=0)
    if static_arr is not None:
        if static_arr.shape[1:] != combined.shape[1:]:
            raise ValueError(
                f"Static shape {static_arr.shape} does not match frame shape {combined.shape} for {press_path}"
            )
        combined = np.concatenate([combined, static_arr], axis=0)
    return combined


# =========================
# 数据增强
# =========================

def augment_pair(X, Y, p_flip_h=0.5, p_flip_w=0.5, p_noise=0.2, noise_sigma=0.001):
    """对输入输出同步做简单空间翻转与噪声增强。"""
    if torch.rand(1).item() < p_flip_h:
        X = X.flip(-2)
        Y = Y.flip(-2)
    if torch.rand(1).item() < p_flip_w:
        X = X.flip(-1)
        Y = Y.flip(-1)
    if torch.rand(1).item() < p_noise:
        X = X + torch.randn_like(X) * float(noise_sigma)
    return X, Y


# =========================
# Dataset 定义
# =========================

class ParFlowDataset(Dataset):
    def __init__(self, press_root, split, pre_seq_length=12, aft_seq_length=12, in_shape=None, use_augment=False,
                 space_h=None,
                 space_w=None,
                 space_stride_h=None,
                 space_stride_w=None,
                 evap_root=None,
                 apcp_root=None,
                 static_root=None,
                 out_channels=None,
                 static_data=None,
                 press_files=None,
                 evap_files=None,
                 apcp_files=None,
                 split_mode='ratio',
                 train_years=None,
                 holdout_years=None,
                 val_ratio_in_holdout=0.25,
                 stats_path=None):
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
        self.apcp_root = apcp_root
        self.static_root = static_root
        self.static_data = static_data
        self.static_arr = (
            _get_static_stack_cached(self.static_root, self.static_data)
            if self.static_root is not None else None
        )
        self.out_channels = out_channels  # 仅用于标签 y，输入 x 仍保留全部通道
        self.split_mode = str(split_mode).lower()
        self.train_years = _normalize_years(train_years)
        self.holdout_years = _normalize_years(holdout_years)
        self.val_ratio_in_holdout = float(val_ratio_in_holdout)
        self.stats_path = stats_path

        if self.use_space:
            self.space_stride_h = space_stride_h or self.space_h
            self.space_stride_w = space_stride_w or self.space_w
        else:
            self.space_stride_h = None
            self.space_stride_w = None

        if press_files is None:
            self.files, self.evap_files, self.apcp_files = _prepare_press_evap_apcp_files(
                self.press_root,
                self.evap_root,
                self.apcp_root,
            )
        else:
            self.files = list(press_files)
            self.evap_files = list(evap_files) if evap_files is not None else None
            self.apcp_files = list(apcp_files) if apcp_files is not None else None
            if self.evap_files is not None and len(self.evap_files) != len(self.files):
                raise ValueError(
                    f"press/evap file counts do not match: {len(self.files)} vs {len(self.evap_files)}"
                )
            if self.apcp_files is not None and len(self.apcp_files) != len(self.files):
                raise ValueError(
                    f"press/APCP file counts do not match: {len(self.files)} vs {len(self.apcp_files)}"
                )

        # 通过首个样本推断通道数和空间尺寸
        self.num_frames = len(self.files)
        self.year_ranges = _build_year_ranges(self.files)
        sample = _read_combined_frame(
            self.files[0],
            evap_path=self.evap_files[0] if self.evap_files is not None else None,
            apcp_path=self.apcp_files[0] if self.apcp_files is not None else None,
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

        # 先按时间切，再按空间滑窗展开样本索引
        self.time_indices = self._build_time_indices()
        if self.use_space:
            self.sample_indices = [
                (t, p) for t in self.time_indices for p in range(len(self.space_coords))
            ]
        else:
            self.sample_indices = self.time_indices

        self.num_sequences = len(self.time_indices)
        self.num_patches_per_frame = len(self.space_coords)
        self.merge_slots = None
        self.merge_tops = None
        self.merge_lefts = None
        if self.use_space:
            self._build_merge_plan()

        self.mean = None
        self.std = None

        if self.stats_path and os.path.exists(self.stats_path):
            self.mean, self.std = _get_stats_cached(self.stats_path)
            if self.mean.shape[0] != self.C or self.std.shape[0] != self.C:
                raise ValueError(f"Stats mismatch: stats C={self.mean.shape[0]} vs dataset C={self.C}.")
        else:
            raise FileNotFoundError(
                f"Stats file not found at {self.stats_path}. "
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

    def _build_merge_plan(self):
        """预计算 patch -> (slot, top, left) 映射，供验证/测试快速拼图。"""
        slot_map = {t: i for i, t in enumerate(self.time_indices)}
        n = len(self.sample_indices)
        self.merge_slots = np.empty(n, dtype=np.int32)
        self.merge_tops = np.empty(n, dtype=np.int32)
        self.merge_lefts = np.empty(n, dtype=np.int32)
        for idx, (t_idx, p_idx) in enumerate(self.sample_indices):
            slot = slot_map.get(t_idx)
            if slot is None:
                raise ValueError(f"Missing merge slot for time index {t_idx}")
            top, left = self.space_coords[p_idx]
            self.merge_slots[idx] = slot
            self.merge_tops[idx] = top
            self.merge_lefts[idx] = left

    def _build_range(self, start, end, stride):
        """在 [start, end) 范围内构建合法时间窗起点。"""
        end = end - 1
        max_start = end - self.total + 1
        if max_start < start:
            return []
        return list(range(start, max_start + 1, stride))

    def _build_time_indices_ratio(self, stride):
        """按固定比例切分 train/val/test。"""
        n_train = int(self.num_frames * 0.75)
        n_val = int(self.num_frames * 0.10)
        if self.split == 'train':
            return self._build_range(0, n_train, stride)
        if self.split == 'val':
            return self._build_range(n_train, n_train + n_val, stride)
        return self._build_range(n_train + n_val, self.num_frames, stride)

    def _year_val_test_range(self, year, available_years):
        if year not in self.year_ranges:
            raise ValueError(f"Year {year} not in dataset. available_years={available_years}")
        s, e = self.year_ranges[year]
        n = e - s
        n_val = int(round(n * self.val_ratio_in_holdout))
        n_val = max(1, min(n - 1, n_val))
        return (s, s + n_val), (s + n_val, e)

    def _build_time_indices_year(self, stride):
        """按年份切分：train_years 训练，holdout_years 再分 val/test。"""
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

        indices = []
        if self.split == 'train':
            for year in train_years:
                if year not in self.year_ranges:
                    raise ValueError(f"Year {year} not in dataset. available_years={available_years}")
                s, e = self.year_ranges[year]
                indices.extend(self._build_range(s, e, stride))
            return indices

        if self.split == 'val':
            for year in holdout_years:
                (s, e), _ = self._year_val_test_range(year, available_years)
                indices.extend(self._build_range(s, e, stride))
            return indices

        for year in holdout_years:
            _, (s, e) = self._year_val_test_range(year, available_years)
            indices.extend(self._build_range(s, e, stride))
        return indices

    def _build_time_indices(self, stride=time_stride):
        """根据当前 split 策略，构造时间窗起点索引列表。"""
        if self.split_mode == 'ratio':
            return self._build_time_indices_ratio(stride)

        if self.split_mode != 'year':
            raise ValueError(
                f"Invalid split_mode={self.split_mode}. Expected 'ratio' or 'year'."
            )
        return self._build_time_indices_year(stride)

    def __len__(self):
        return len(self.sample_indices)

    def _read_window(self, t0, top=0, left=0):
        """读取一个时间窗；若启用空间滑窗，则再裁出对应 patch。"""
        T = self.total
        h, w = self.valid_h, self.valid_w
        out = torch.empty((T, self.C, h, w), dtype=torch.float32)
        for i in range(T):
            path = self.files[t0 + i]
            arr = _read_combined_frame(
                path,
                evap_path=self.evap_files[t0 + i] if self.evap_files is not None else None,
                apcp_path=self.apcp_files[t0 + i] if self.apcp_files is not None else None,
                static_arr=self.static_arr,
            )
            if self.use_space:
                arr = arr[:, top:top + h, left:left + w]
            out[i] = torch.from_numpy(arr)
        return out

    def __getitem__(self, idx):
        """返回单个样本 (x, y)。"""
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

        if self.mean_t is not None and self.std_t is not None:
            x = x.sub(self.mean_t).div(self.std_eps_t)
            if self.out_channels is not None:
                y = y.sub(self.mean_y_t).div(self.std_y_eps_t)
            else:
                y = y.sub(self.mean_t).div(self.std_eps_t)

        return x, y


# =========================
# DataLoader 构建入口
# =========================

def load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length=6, aft_seq_length=6,
              in_shape=None, distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              space_h=None, space_w=None, space_stride_h=None, space_stride_w=None, out_channels=None,
              static_data=None,
              split_mode='ratio',
              train_years=None,
              holdout_years=None,
              val_ratio_in_holdout=0.25,
              var_name='press',
              use_evap=True,
              use_apcp=False,
              use_static_input=True,
              stats_path=None):
    """构建 ParFlow 的 train / val / test 三个 DataLoader。"""

    use_static = use_static_input or static_data is not None
    press_root, evap_root, apcp_root, static_root = _resolve_parflow_roots(
        data_root, use_static=use_static, var_name=var_name, use_evap=use_evap, use_apcp=use_apcp
    )
    allowed_years = None
    if str(split_mode).lower() == 'year':
        train_year_list = _normalize_years(train_years) or []
        holdout_year_list = _normalize_years(holdout_years) or []
        allowed_years = sorted(set(train_year_list + holdout_year_list))
    all_press_files, all_evap_files, all_apcp_files = _prepare_press_evap_apcp_files(
        press_root,
        evap_root,
        apcp_root,
        allowed_years=allowed_years,
    )
    common_ds_kwargs = dict(
        in_shape=in_shape,
        space_h=space_h,
        space_w=space_w,
        space_stride_h=space_stride_h,
        space_stride_w=space_stride_w,
        evap_root=evap_root,
        apcp_root=apcp_root,
        static_root=static_root,
        out_channels=out_channels,
        static_data=static_data,
        press_files=all_press_files,
        evap_files=all_evap_files,
        apcp_files=all_apcp_files,
        split_mode=split_mode,
        train_years=train_years,
        holdout_years=holdout_years,
        val_ratio_in_holdout=val_ratio_in_holdout,
        stats_path=stats_path,
    )

    train_ds = ParFlowDataset(
        press_root, 'train', pre_seq_length, aft_seq_length,
        use_augment=use_augment, **common_ds_kwargs
    )
    val_ds = ParFlowDataset(
        press_root, 'val', pre_seq_length, aft_seq_length,
        use_augment=False, **common_ds_kwargs
    )
    test_ds = ParFlowDataset(
        press_root, 'test', pre_seq_length, aft_seq_length,
        use_augment=False, **common_ds_kwargs
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
