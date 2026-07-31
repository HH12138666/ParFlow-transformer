"""ParFlow Dataset with temporal windows and optional spatial tiling."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import EPS, TIME_STRIDE
from .paths import build_year_ranges, normalize_years, prepare_press_evap_files
from .readers import get_static_stack_cached, get_stats_cached, read_combined_frame


def build_space_coords(height, width, space_h, space_w, space_stride_h=None, space_stride_w=None):
    if space_h is None or space_w is None:
        return [(0, 0)]
    stride_h = space_stride_h or space_h
    stride_w = space_stride_w or space_w
    if space_h > height or space_w > width:
        raise ValueError(f"Space size {(space_h, space_w)} exceeds frame size {(height, width)}.")
    coords_h = _axis_coords(height, space_h, stride_h)
    coords_w = _axis_coords(width, space_w, stride_w)
    return [(top, left) for top in coords_h for left in coords_w]


def _axis_coords(size, window, stride):
    coords = list(range(0, size - window + 1, stride))
    if not coords or coords[-1] != size - window:
        coords.append(size - window)
    return coords


def augment_sample(x, y, future_aux, p_flip_h=0.5, p_flip_w=0.5,
                   p_noise=0.2, noise_sigma=0.001):
    if torch.rand(1).item() < p_flip_h:
        x = x.flip(-2)
        y = y.flip(-2)
        future_aux = future_aux.flip(-2)
    if torch.rand(1).item() < p_flip_w:
        x = x.flip(-1)
        y = y.flip(-1)
        future_aux = future_aux.flip(-1)
    if torch.rand(1).item() < p_noise:
        x = x + torch.randn_like(x) * float(noise_sigma)
    return x, y, future_aux


class ParFlowDataset(Dataset):
    def __init__(self, press_root, split, pre_seq_length=12, aft_seq_length=12, **kwargs):
        super().__init__()
        self._init_basic(press_root, split, pre_seq_length, aft_seq_length, kwargs)
        self._init_files(kwargs)
        self._init_shape_and_indices(kwargs)
        self._init_stats(kwargs.get("stats_path"))

    def _init_basic(self, press_root, split, pre_seq_length, aft_seq_length, kwargs):
        split = str(split).lower()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}. Expected 'train' | 'val' | 'test'.")
        self.press_root = press_root
        self.split = split
        self.pre = pre_seq_length
        self.aft = aft_seq_length
        self.total = self.pre + self.aft
        self.use_augment = bool(kwargs.get("use_augment", False))
        self.out_channels = kwargs.get("out_channels")
        self.split_mode = str(kwargs.get("split_mode", "ratio")).lower()
        self.explicit_time_indices = kwargs.get("explicit_time_indices")
        self.train_years = normalize_years(kwargs.get("train_years"))
        self.holdout_years = normalize_years(kwargs.get("holdout_years"))
        self.val_ratio_in_holdout = float(kwargs.get("val_ratio_in_holdout", 0.5))
        self.use_val = bool(kwargs.get("use_val", False))

    def _init_files(self, kwargs):
        self.evap_root = kwargs.get("evap_root")
        self.static_root = kwargs.get("static_root")
        self.static_arr = get_static_stack_cached(self.static_root) if self.static_root else None
        press_files = kwargs.get("press_files")
        if press_files is None:
            self.files, self.evap_files = prepare_press_evap_files(self.press_root, self.evap_root)
            return
        self.files = list(press_files)
        evap_files = kwargs.get("evap_files")
        self.evap_files = list(evap_files) if evap_files is not None else None
        if self.evap_files is not None and len(self.evap_files) != len(self.files):
            raise ValueError(f"press/evap file counts do not match: {len(self.files)} vs {len(self.evap_files)}")

    def _init_shape_and_indices(self, kwargs):
        self.num_frames = len(self.files)
        self.year_ranges = build_year_ranges(self.files)
        sample = read_combined_frame(self.files[0], self._evap_at(0), self.static_arr)
        self.C, self.H, self.W = sample.shape
        self.space_h = kwargs.get("space_h")
        self.space_w = kwargs.get("space_w")
        self.use_space = self.space_h is not None and self.space_w is not None
        self._init_space(kwargs)
        self.time_indices = self._build_time_indices()
        self.sample_indices = self._build_sample_indices()
        self.num_sequences = len(self.time_indices)
        self.num_patches_per_frame = len(self.space_coords)
        self._init_merge_plan()

    def _init_space(self, kwargs):
        if self.use_space:
            self.space_stride_h = kwargs.get("space_stride_h") or self.space_h
            self.space_stride_w = kwargs.get("space_stride_w") or self.space_w
        else:
            self.space_stride_h = None
            self.space_stride_w = None
        self.space_coords = build_space_coords(self.H, self.W, self.space_h, self.space_w,
                                               self.space_stride_h, self.space_stride_w)
        self.valid_h = self.space_h if self.use_space else self.H
        self.valid_w = self.space_w if self.use_space else self.W

    def _init_stats(self, stats_path):
        if not stats_path or not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found at {stats_path}. Please compute mean/std npz first.")
        self.mean, self.std = get_stats_cached(stats_path)
        if self.mean.shape[0] != self.C or self.std.shape[0] != self.C:
            raise ValueError(f"Stats mismatch: stats C={self.mean.shape[0]} vs dataset C={self.C}.")
        self.mean_t = torch.from_numpy(self.mean).view(1, self.C, 1, 1).float()
        self.std_t = torch.from_numpy(self.std).view(1, self.C, 1, 1).float()
        self.std_eps_t = self.std_t + EPS
        if self.out_channels is None:
            self.mean_y_t = None
            self.std_y_eps_t = None
            return
        self.mean_y_t = self.mean_t[:, :self.out_channels, :, :]
        self.std_y_eps_t = self.std_t[:, :self.out_channels, :, :] + EPS

    def _evap_at(self, idx):
        if self.evap_files is None:
            return None
        return self.evap_files[idx]

    def _build_sample_indices(self):
        if not self.use_space:
            return list(self.time_indices)
        return [(time_idx, patch_idx) for time_idx in self.time_indices for patch_idx in range(len(self.space_coords))]

    def _init_merge_plan(self):
        self.merge_slots = None
        self.merge_tops = None
        self.merge_lefts = None
        if not self.use_space:
            return
        self._build_merge_plan()

    def _build_merge_plan(self):
        slot_map = {time_idx: idx for idx, time_idx in enumerate(self.time_indices)}
        count = len(self.sample_indices)
        self.merge_slots = np.empty(count, dtype=np.int32)
        self.merge_tops = np.empty(count, dtype=np.int32)
        self.merge_lefts = np.empty(count, dtype=np.int32)
        for idx, (time_idx, patch_idx) in enumerate(self.sample_indices):
            top, left = self.space_coords[patch_idx]
            self.merge_slots[idx] = slot_map[time_idx]
            self.merge_tops[idx] = top
            self.merge_lefts[idx] = left

    def _build_range(self, start, end, stride):
        max_start = (end - 1) - self.total + 1
        if max_start < start:
            return []
        return list(range(start, max_start + 1, stride))

    def _build_time_indices_ratio(self, stride):
        n_train = int(self.num_frames * 0.75)
        n_val = int(self.num_frames * 0.10)
        if self.split == "train":
            return self._build_range(0, n_train, stride)
        if not self.use_val:
            return [] if self.split == "val" else self._build_range(n_train, self.num_frames, stride)
        if self.split == "val":
            return self._build_range(n_train, n_train + n_val, stride)
        return self._build_range(n_train + n_val, self.num_frames, stride)

    def _year_val_test_range(self, year, available_years):
        if year not in self.year_ranges:
            raise ValueError(f"Year {year} not in dataset. available_years={available_years}")
        start, end = self.year_ranges[year]
        n_val = int(round((end - start) * self.val_ratio_in_holdout))
        n_val = max(1, min((end - start) - 1, n_val))
        return (start, start + n_val), (start + n_val, end)

    def _build_time_indices_year(self, stride):
        available_years = sorted(self.year_ranges.keys())
        if not available_years:
            raise ValueError("No available years parsed from files.")
        train_years = self.train_years or [available_years[0]]
        holdout_years = self.holdout_years or self._default_holdout_years(available_years, train_years)
        self._validate_year_split(available_years, train_years, holdout_years)
        if self.split == "train":
            return self._collect_year_ranges(train_years, stride)
        if not self.use_val:
            return [] if self.split == "val" else self._collect_year_ranges(holdout_years, stride)
        return self._collect_holdout_ranges(holdout_years, available_years, stride)

    def _default_holdout_years(self, available_years, train_years):
        holdout_years = [year for year in available_years if year not in train_years]
        if not holdout_years and len(available_years) >= 2:
            return [available_years[-1]]
        return holdout_years

    def _validate_year_split(self, available_years, train_years, holdout_years):
        if not holdout_years:
            raise ValueError(f"holdout_years is empty. available_years={available_years}, train_years={train_years}")
        if self.use_val and not (0.0 < self.val_ratio_in_holdout < 1.0):
            raise ValueError(f"val_ratio_in_holdout must be in (0,1), got {self.val_ratio_in_holdout}")

    def _collect_year_ranges(self, years, stride):
        indices = []
        available_years = sorted(self.year_ranges.keys())
        for year in years:
            if year not in self.year_ranges:
                raise ValueError(f"Year {year} not in dataset. available_years={available_years}")
            start, end = self.year_ranges[year]
            indices.extend(self._build_range(start, end, stride))
        return indices

    def _collect_holdout_ranges(self, years, available_years, stride):
        indices = []
        for year in years:
            val_range, test_range = self._year_val_test_range(year, available_years)
            start, end = val_range if self.split == "val" else test_range
            indices.extend(self._build_range(start, end, stride))
        return indices

    def _build_time_indices(self, stride=TIME_STRIDE):
        if self.explicit_time_indices is not None:
            return list(self.explicit_time_indices)
        if self.split_mode == "ratio":
            return self._build_time_indices_ratio(stride)
        if self.split_mode != "year":
            raise ValueError(f"Invalid split_mode={self.split_mode}. Expected 'ratio' or 'year'.")
        return self._build_time_indices_year(stride)

    def __len__(self):
        return len(self.sample_indices)

    def _read_window(self, time_idx, top=0, left=0):
        out = torch.empty((self.total, self.C, self.valid_h, self.valid_w), dtype=torch.float32)
        for offset in range(self.total):
            source_idx = time_idx + offset
            arr = read_combined_frame(self.files[source_idx], self._evap_at(source_idx), self.static_arr)
            if self.use_space:
                arr = arr[:, top:top + self.valid_h, left:left + self.valid_w]
            out[offset] = torch.from_numpy(arr)
        return out

    def __getitem__(self, idx):
        time_idx, top, left = self._sample_location(idx)
        window = self._read_window(time_idx, top=top, left=left)
        x = window[: self.pre]
        target_window = window[self.pre:self.pre + self.aft]
        y = target_window
        if self.out_channels is not None:
            y = y[:, :self.out_channels, :, :]
        future_aux = target_window[:, self.out_channels:, :, :]
        if self.aft <= self.pre:
            future_aux = future_aux[:0]
        if self.use_augment and self.split == "train":
            x, y, future_aux = augment_sample(x, y, future_aux)
        return self._normalize_x(x), self._normalize_y(y), self._normalize_aux(future_aux)

    def _sample_location(self, idx):
        if not self.use_space:
            return self.sample_indices[idx], 0, 0
        time_idx, patch_idx = self.sample_indices[idx]
        top, left = self.space_coords[patch_idx]
        return time_idx, top, left

    def _normalize_x(self, x):
        return x.sub(self.mean_t).div(self.std_eps_t)

    def _normalize_y(self, y):
        if self.out_channels is None:
            return y.sub(self.mean_t).div(self.std_eps_t)
        return y.sub(self.mean_y_t).div(self.std_y_eps_t)

    def _normalize_aux(self, future_aux):
        if future_aux.shape[0] == 0:
            return future_aux
        mean = self.mean_t[:, self.out_channels:, :, :]
        std = self.std_eps_t[:, self.out_channels:, :, :]
        return future_aux.sub(mean).div(std)
