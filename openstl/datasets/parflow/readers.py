"""PFB readers and per-worker caches for ParFlow data loading."""

from functools import lru_cache
from pathlib import Path
import re

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
        return arr[None]
    if arr.ndim == 3:
        return arr
    expected = "2D/3D" if allow_2d else "3D"
    raise ValueError(f"Expected {expected} {label}, got {arr.shape} for {path}")


def read_press_frame(press_path):
    return _as_channel_first(
        _cached_read_pfb(str(press_path)), press_path, "pressure"
    )


def read_evap_frame(evap_path):
    if evap_path is None:
        raise ValueError("evap_path is required")
    return _as_channel_first(
        _cached_read_pfb(str(evap_path)),
        evap_path,
        "evaptrans",
        allow_2d=False,
    )


def read_static_stack(static_root, static_data=None):
    if static_root is None:
        return None
    root = Path(static_root)
    merged = root / "static.pfb"
    if merged.exists():
        return _as_channel_first(_cached_read_pfb(str(merged)), merged, "static")
    files = _select_static_files(root, static_data)
    arrays = [
        _as_channel_first(_cached_read_pfb(str(path)), path, "static")
        for path in files
    ]
    return np.concatenate(arrays, axis=0)


def _select_static_files(root, static_data):
    files = sorted(root.glob("*.pfb"))
    patterns = _static_patterns(static_data)
    if patterns:
        files = [
            path
            for path in files
            if any(re.search(pattern, path.name, re.I) for pattern in patterns)
        ]
    if not files:
        raise FileNotFoundError(f"No static PFB files found under {root}")
    return files


def _static_patterns(static_data):
    if static_data is None:
        return []
    if isinstance(static_data, (list, tuple)):
        return [str(item).strip() for item in static_data if str(item).strip()]
    return [item.strip() for item in str(static_data).split(",") if item.strip()]


def get_static_stack_cached(static_root):
    cache_key = str(static_root)
    if cache_key not in _STATIC_STACK_CACHE:
        _STATIC_STACK_CACHE[cache_key] = read_static_stack(static_root)
    return _STATIC_STACK_CACHE[cache_key]


def get_stats_cached(stats_path):
    if stats_path not in _STATS_CACHE:
        data = np.load(stats_path)
        mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(data["std"], dtype=np.float32).reshape(-1)
        _STATS_CACHE[stats_path] = (mean, std)
    return _STATS_CACHE[stats_path]


def read_combined_frame(press_path, evap_path=None, static_arr=None):
    parts = [read_press_frame(press_path)]
    if evap_path is not None:
        parts.append(read_evap_frame(evap_path))
    if static_arr is not None:
        if static_arr.shape[1:] != parts[0].shape[1:]:
            raise ValueError(
                f"Static shape {static_arr.shape} does not match {parts[0].shape}"
            )
        parts.append(static_arr)
    return np.concatenate(parts, axis=0)
