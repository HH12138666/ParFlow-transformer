"""PFB readers and per-worker caches for ParFlow data loading."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

from .constants import DEFAULT_PFB_CACHE_SIZE

_STATIC_STACK_CACHE = {}
_STATS_CACHE = {}


@lru_cache(maxsize=DEFAULT_PFB_CACHE_SIZE)
def _cached_read_pfb(path):
    return read_pfb(get_absolute_path(str(path))).astype(np.float32)


def _as_channel_first(arr, path, label, allow_2d=True):
    if arr.ndim == 2 and allow_2d:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    expected = "2D/3D" if allow_2d else "3D"
    raise ValueError(f"Expected {expected} {label} array, got shape {arr.shape} for {path}")


def read_press_frame(press_path):
    return _as_channel_first(_cached_read_pfb(str(press_path)), press_path, "press", allow_2d=True)


def read_evap_frame(evap_path):
    if evap_path is None:
        raise ValueError("evap_path is required when reading evaptrans data")
    return _as_channel_first(_cached_read_pfb(str(evap_path)), evap_path, "evaptrans", allow_2d=False)


def read_static_stack(static_root):
    if static_root is None:
        return None
    merged_static = Path(static_root) / "static.pfb"
    if not merged_static.exists():
        raise FileNotFoundError(f"Static file not found: {merged_static}. Please provide static/static.pfb.")
    return _as_channel_first(_cached_read_pfb(str(merged_static)), merged_static, "static", allow_2d=True)


def get_static_stack_cached(static_root):
    cache_key = str(static_root)
    cached = _STATIC_STACK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    arr = read_static_stack(static_root)
    _STATIC_STACK_CACHE[cache_key] = arr
    return arr


def get_stats_cached(stats_path):
    cached = _STATS_CACHE.get(stats_path)
    if cached is not None:
        return cached
    data = np.load(stats_path)
    mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(data["std"], dtype=np.float32).reshape(-1)
    _STATS_CACHE[stats_path] = (mean, std)
    return mean, std


def read_combined_frame(press_path, evap_path=None, static_arr=None):
    combined = read_press_frame(press_path)
    if evap_path is not None:
        combined = np.concatenate([combined, read_evap_frame(evap_path)], axis=0)
    if static_arr is not None:
        if static_arr.shape[1:] != combined.shape[1:]:
            raise ValueError(f"Static shape {static_arr.shape} does not match frame shape {combined.shape}")
        combined = np.concatenate([combined, static_arr], axis=0)
    return combined
