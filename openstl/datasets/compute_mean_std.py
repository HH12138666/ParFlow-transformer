import os
import sys
import re
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb
from openstl.datasets.dataloader_parflow import (
    _list_pfb_files,
    _read_evap_frame,
    _read_static_stack,
    _resolve_parflow_roots,
    _extract_year,
)
#sbatch /home/huanghui/data/slurm_job/run_compute_mean_std.sh
DATA_ROOT = "/home/huanghui/data/ParFlow-transformer/data/parflow"
STATS_OUT = "/home/huanghui/data/ParFlow-transformer/stats"  # 输出目录
# perm_x_alpha_n_porosity
STATS_NAME = "stats_test_0.75_press_evap"
STATS_OUT = os.path.join(STATS_OUT, f"{STATS_NAME}.npz")
os.makedirs(os.path.dirname(STATS_OUT), exist_ok=True)
SPATIAL_STRIDE = 1
TIME_STRIDE = 1
MAX_FILES = 0
# 按比例使用样本（按时间排序后的前比例），None 表示不按比例截取
TRAIN_RATIO = 0.75
PRESS_ROOT = None
EVAP_ROOT = None
MAIN_VAR = "press"
USE_EVAP = True
USE_STATIC_INPUT = False

STATIC_ROOT = None
# perm_x,alpha,n,porosity
STATIC_DATA = "perm_x,alpha,n,porosity"  # 逗号分隔关键词（大小写不敏感），为 None 时使用全部静态数据
# 只统计指定年份；None 表示全部年份，例如 [2019]、[2019, 2020]
ONLY_YEARS =  [2019, 2020]
# 只统计指定 ID 范围（基于文件名末尾数字），None 表示不限制
START_ID = None
END_ID = None

def _read_press_frame_for_stats(press_path):
    arr = read_pfb(get_absolute_path(press_path)).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array per .pfb, got shape {arr.shape} for {press_path}")
    return arr


def _welford_update(count, mean, M2, batch_mean, batch_M2, batch_n):
    total_n = count + batch_n
    delta = batch_mean - mean
    mean += delta * (batch_n / np.maximum(total_n, 1))
    M2 += batch_M2 + (delta * delta) * (count * batch_n / np.maximum(total_n, 1))
    count[:] = total_n
    return count, mean, M2


def _select_files(files, time_stride, max_files):
    sel = files[::max(1, int(time_stride))]
    if max_files is not None:
        sel = sel[:int(max_files)]
    return sel


def _compute_mean_std_from_files(files, read_fn, spatial_stride=1):
    if not files:
        raise ValueError("files is empty")
    a0 = read_fn(files[0])
    if spatial_stride > 1:
        a0 = a0[:, ::spatial_stride, ::spatial_stride]
    C = a0.shape[0]
    count = np.zeros(C, dtype=np.float64)
    mean = np.zeros(C, dtype=np.float64)
    M2 = np.zeros(C, dtype=np.float64)
    for f in files:
        a = read_fn(f)
        if spatial_stride > 1:
            a = a[:, ::spatial_stride, ::spatial_stride]
        x = a.reshape(C, -1).astype(np.float64, copy=False)
        b_mean = x.mean(axis=1)
        diff = x - b_mean[:, None]
        b_M2 = (diff * diff).sum(axis=1)
        b_n = x.shape[1]
        _welford_update(count, mean, M2, b_mean, b_M2, b_n)
    var = M2 / np.maximum(count, 1.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _compute_mean_std_from_array(arr, spatial_stride=1):
    if arr is None:
        return None, None
    if spatial_stride > 1:
        arr = arr[:, ::spatial_stride, ::spatial_stride]
    x = arr.reshape(arr.shape[0], -1).astype(np.float64, copy=False)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return mean.astype(np.float32), std.astype(np.float32)


def _concat_stats(parts):
    keep = [p for p in parts if p is not None and p.size > 0]
    if not keep:
        raise ValueError("No stats to concatenate")
    if len(keep) == 1:
        return keep[0]
    return np.concatenate(keep, axis=0)


def _normalize_years(years):
    if years is None:
        return None
    if isinstance(years, int):
        return {int(years)}
    return {int(y) for y in years}


def _extract_suffix_id(path):
    name = os.path.basename(str(path))
    m = re.search(r"(\d+)(?!.*\d)", name)
    if not m:
        return None
    return int(m.group(1))


def _filter_files_by_year(files, years):
    years_set = _normalize_years(years)
    if years_set is None:
        return files

    out = []
    for f in files:
        y = _extract_year(f)
        if y is None:
            raise ValueError(f"Cannot parse year from filename/folder: {f}")
        if y in years_set:
            out.append(f)
    return out


def _filter_files_by_id_range(files, start_id=None, end_id=None):
    if start_id is None and end_id is None:
        return files

    s = None if start_id is None else int(start_id)
    e = None if end_id is None else int(end_id)
    out = []
    for f in files:
        fid = _extract_suffix_id(f)
        if fid is None:
            continue
        if s is not None and fid < s:
            continue
        if e is not None and fid > e:
            continue
        out.append(f)
    return out


def _resolve_max_files(total_files, max_files, train_ratio):
    if total_files <= 0:
        return 0

    ratio_cap = None
    if train_ratio is not None:
        ratio = float(train_ratio)
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"TRAIN_RATIO must be in (0, 1], got {train_ratio}")
        ratio_cap = max(1, int(total_files * ratio))

    fixed_cap = None
    if max_files is not None:
        fixed_cap = max(1, int(max_files))

    if ratio_cap is None and fixed_cap is None:
        return None
    if ratio_cap is None:
        return min(total_files, fixed_cap)
    if fixed_cap is None:
        return min(total_files, ratio_cap)
    return min(total_files, ratio_cap, fixed_cap)


def compute_mean_std(files,
                     spatial_stride=1,
                     time_stride=1,
                     max_files=None,
                     channels=None,
                     evap_files=None,
                     static_arr=None):
    press_sel = _select_files(files, time_stride, max_files)
    if not press_sel:
        raise ValueError("press_files is empty")
    evap_sel = None
    if evap_files is not None:
        evap_sel = _select_files(evap_files, time_stride, max_files)
        if len(evap_sel) != len(press_sel):
            raise ValueError("press/evap file counts do not match after stride")

    press_mean, press_std = _compute_mean_std_from_files(
        press_sel, _read_press_frame_for_stats, spatial_stride
    )
    evap_mean, evap_std = (None, None)
    if evap_sel is not None:
        evap_mean, evap_std = _compute_mean_std_from_files(
            evap_sel, _read_evap_frame, spatial_stride
        )
    static_mean, static_std = _compute_mean_std_from_array(
        static_arr, spatial_stride
    )

    mean = _concat_stats([press_mean, evap_mean, static_mean])
    std = _concat_stats([press_std, evap_std, static_std])

    if channels is not None:
        mean = mean[channels]
        std = std[channels]
    return mean, std


def main():
    press_root, evap_root, static_root = _resolve_parflow_roots(
        DATA_ROOT,
        var_name=MAIN_VAR,
        use_evap=USE_EVAP,
        use_static=USE_STATIC_INPUT,
    )
    if PRESS_ROOT is not None:
        press_root = PRESS_ROOT
    if EVAP_ROOT is not None:
        evap_root = EVAP_ROOT
    elif not USE_EVAP:
        evap_root = None
    if STATIC_ROOT is not None:
        static_root = STATIC_ROOT
    elif not USE_STATIC_INPUT:
        static_root = None
    if USE_STATIC_INPUT and STATIC_DATA is not None and static_root is None:
        raise ValueError("STATIC_DATA is set but static_root is None")

    press_files = _list_pfb_files(press_root)
    press_files = _filter_files_by_year(press_files, ONLY_YEARS)
    press_files = _filter_files_by_id_range(press_files, START_ID, END_ID)
    if not press_files:
        raise ValueError(
            f"No press files selected after filters: years={ONLY_YEARS}, "
            f"id_range=[{START_ID}, {END_ID}]"
        )

    evap_files = _list_pfb_files(evap_root) if evap_root is not None else None
    if evap_files is not None:
        evap_files = _filter_files_by_year(evap_files, ONLY_YEARS)
        evap_files = _filter_files_by_id_range(evap_files, START_ID, END_ID)
        if not evap_files:
            raise ValueError(
                f"No evap files selected after filters: years={ONLY_YEARS}, "
                f"id_range=[{START_ID}, {END_ID}]"
            )
    static_arr = (
        _read_static_stack(static_root, static_data=STATIC_DATA)
        if (USE_STATIC_INPUT and static_root is not None)
        else None
    )
    max_files_cfg = MAX_FILES if MAX_FILES > 0 else None
    max_files = _resolve_max_files(
        total_files=len(press_files),
        max_files=max_files_cfg,
        train_ratio=TRAIN_RATIO,
    )

    mean, std = compute_mean_std(
        files=press_files,
        spatial_stride=SPATIAL_STRIDE,
        time_stride=TIME_STRIDE,
        max_files=max_files,
        evap_files=evap_files,
        static_arr=static_arr,
    )
    os.makedirs(os.path.dirname(STATS_OUT), exist_ok=True)
    np.savez(STATS_OUT, mean=mean, std=std)
    used_files = len(_select_files(press_files, TIME_STRIDE, max_files))
    used_hours = used_files  # 当前数据是每个文件对应 1 小时
    print(f"Used {used_files} files")
    print(f"Used total hours: {used_hours} h")
    print(f"Main variable: {MAIN_VAR}")
    print(f"Year filter: {ONLY_YEARS}")
    print(f"ID filter: [{START_ID}, {END_ID}]")
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Stats saved to {STATS_OUT}; mean shape={mean.shape}, std shape={std.shape}")


if __name__ == "__main__":
    main()
