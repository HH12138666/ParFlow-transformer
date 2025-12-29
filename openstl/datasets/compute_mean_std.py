import numpy as np

from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

from openstl.datasets.dataloader_parflow import (
    _interpolate_outliers,
    _list_pfb_files,
    _read_evap_frame,
    _read_static_stack,
    _resolve_parflow_roots,
)

DATA_ROOT = "/home/huanghui/data/ParFlow-transformer/data/parflow"
STATS_OUT = "/home/huanghui/data/ParFlow-transformer/stats.npz"
SPATIAL_STRIDE = 1
TIME_STRIDE = 1
MAX_FILES = 0
PRESS_ROOT = None
EVAP_ROOT = None

STATIC_ROOT = None
# Outlier handling parameters for pressure data
APPLY_PRESS_OUTLIER_FIX = True
PRESS_ABS_OUTLIER_THRESHOLD = -10000.0
PRESS_OUTLIER_STD_MULT = 5.0


def _read_press_frame_for_stats(press_path):
    arr = read_pfb(get_absolute_path(press_path)).astype(np.float32)
    if APPLY_PRESS_OUTLIER_FIX:
        arr = _interpolate_outliers(
            arr,
            abs_threshold=PRESS_ABS_OUTLIER_THRESHOLD,
            std_mult=PRESS_OUTLIER_STD_MULT,
        )
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


def compute_press_evap_mean_std(press_files,
                                evap_root,
                                spatial_stride=1,
                                time_stride=1,
                                max_files=None):
    sel_files = press_files[::max(1, int(time_stride))]
    if max_files is not None:
        sel_files = sel_files[:int(max_files)]
    if not sel_files:
        raise ValueError("press_files is empty")
    evap_files = _list_pfb_files(evap_root)
    evap_sel = evap_files[::max(1, int(time_stride))]
    if max_files is not None:
        evap_sel = evap_sel[:int(max_files)]
    if len(evap_sel) != len(sel_files):
        raise ValueError("press/evap file counts do not match after stride")

    p0 = _read_press_frame_for_stats(sel_files[0])
    e0 = _read_evap_frame(evap_sel[0])

    def _init_stats(C):
        return (
            np.zeros(C, dtype=np.float64),
            np.zeros(C, dtype=np.float64),
            np.zeros(C, dtype=np.float64),
        )

    p_sum, p_sumsq, p_cnt = _init_stats(p0.shape[0])
    e_sum, e_sumsq, e_cnt = _init_stats(e0.shape[0])

    def _update(arr, sum_, sumsq_, cnt_):
        if spatial_stride > 1:
            arr = arr[:, ::spatial_stride, ::spatial_stride]
        sum_ += arr.sum(axis=(1, 2))
        sumsq_ += np.square(arr).sum(axis=(1, 2))
        cnt_ += arr.shape[1] * arr.shape[2]
        return sum_, sumsq_, cnt_

    for i, f in enumerate(sel_files):
        p = _read_press_frame_for_stats(f)
        e = _read_evap_frame(evap_sel[i])
        if p.shape[1:] != e.shape[1:]:
            raise ValueError(
                f"Spatial shape mismatch between press and evap: {p.shape} vs {e.shape} for {f}"
            )
        p_sum, p_sumsq, p_cnt = _update(p, p_sum, p_sumsq, p_cnt)
        e_sum, e_sumsq, e_cnt = _update(e, e_sum, e_sumsq, e_cnt)

    def _finish(sum_, sumsq_, cnt_):
        mean = sum_ / np.maximum(cnt_, 1.0)
        var = sumsq_ / np.maximum(cnt_, 1.0) - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean.astype(np.float32), std.astype(np.float32)

    press_mean, press_std = _finish(p_sum, p_sumsq, p_cnt)
    evap_mean, evap_std = _finish(e_sum, e_sumsq, e_cnt)
    return press_mean, press_std, evap_mean, evap_std


def main():
    press_root, evap_root, static_root = _resolve_parflow_roots(DATA_ROOT)
    if PRESS_ROOT is not None:
        press_root = PRESS_ROOT
    if EVAP_ROOT is not None:
        evap_root = EVAP_ROOT
    if STATIC_ROOT is not None:
        static_root = STATIC_ROOT

    press_files = _list_pfb_files(press_root)
    evap_files = _list_pfb_files(evap_root) if evap_root is not None else None
    static_arr = _read_static_stack(static_root) if static_root is not None else None
    max_files = MAX_FILES if MAX_FILES > 0 else None

    mean, std = compute_mean_std(
        files=press_files,
        spatial_stride=SPATIAL_STRIDE,
        time_stride=TIME_STRIDE,
        max_files=max_files,
        evap_files=evap_files,
        static_arr=static_arr,
    )
    np.savez(STATS_OUT, mean=mean, std=std)
    used = len(press_files) if max_files is None else min(len(press_files), max_files)
    print(f"Used {used} files")
    print(f"Stats saved to {STATS_OUT}; mean shape={mean.shape}, std shape={std.shape}")


if __name__ == "__main__":
    main()
