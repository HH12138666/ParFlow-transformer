#!/usr/bin/env python3
"""
Fix outliers in ParFlow press .pfb files and save corrected data.

Examples:
  # Write corrected files to a new directory
  python tools/fix_press_outliers.py \
    --input_dir /path/to/press \
    --output_dir /path/to/press_fixed

  # In-place overwrite (write temp then replace)
  python tools/fix_press_outliers.py \
    --input_dir /path/to/press \
    --in_place
"""
import shutil
from pathlib import Path

import numpy as np
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb, write_pfb

from openstl.datasets.dataloader_parflow import _list_pfb_files


# 输入/输出路径
INPUT_DIR = "/home/huanghui/data/ParFlow-transformer/data/parflow/press"
OUTPUT_DIR = None  # 复制到新目录时填写路径；为 None 时表示原地覆盖
IN_PLACE = True    # True 表示覆盖 INPUT_DIR 中的原始文件

# 异常值修复参数
ABS_THRESHOLD = -10000.0  # 绝对阈值
STD_MULT = 5.0            # 标准差倍数阈值

# 处理数量限制（0 表示全部）
LIMIT = 0

# numerical stability constant
EPS = 1e-6


def _interpolate_outliers(arr, abs_threshold=-10000.0, std_mult=5.0):
    """
    Simple + dynamic outlier repair:
    - values below abs_threshold are outliers;
    - values deviating from mean by > std_mult * std are outliers (both sides);
    - replace outliers with channel mean.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for interpolation, got shape {arr.shape}.")

    repaired = arr.astype(np.float64, copy=True)
    c, h, w = repaired.shape
    flat = repaired.reshape(c, -1)

    abs_mask = flat < abs_threshold if abs_threshold is not None else np.zeros_like(flat, dtype=bool)
    valid = np.ma.array(flat, mask=abs_mask)
    channel_mean = valid.mean(axis=1, keepdims=True).filled(0.0)
    channel_std = valid.std(axis=1, keepdims=True).filled(0.0)
    channel_std = np.maximum(channel_std, EPS)

    low = channel_mean - std_mult * channel_std
    high = channel_mean + std_mult * channel_std

    mask_dyn = (flat < low) | (flat > high)
    mask = abs_mask | mask_dyn
    if not mask.any():
        return repaired

    flat[mask] = np.broadcast_to(channel_mean, flat.shape)[mask]
    return flat.reshape(c, h, w)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fix_file(src: Path,
              dst: Path,
              abs_threshold: float,
              std_mult: float) -> None:
    arr = read_pfb(get_absolute_path(str(src))).astype(np.float64)
    fixed = _interpolate_outliers(
        arr,
        abs_threshold=abs_threshold,
        std_mult=std_mult,
    )
    write_pfb(str(dst), fixed.astype(np.float64), dist=False)


def main() -> None:
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    if IN_PLACE and OUTPUT_DIR:
        raise ValueError("Use either --in_place or --output_dir, not both.")
    if not IN_PLACE and not OUTPUT_DIR:
        raise ValueError("Provide --output_dir or use --in_place.")

    output_dir = input_dir if IN_PLACE else Path(OUTPUT_DIR)
    _ensure_dir(output_dir)

    files = _list_pfb_files(str(input_dir))
    if LIMIT and LIMIT > 0:
        files = files[:LIMIT]

    for f in files:
        src = Path(f)
        dst = output_dir / src.name

        if IN_PLACE:
            tmp = output_dir / (src.name + ".tmp")
            _fix_file(src, tmp, ABS_THRESHOLD, STD_MULT)
            shutil.move(str(tmp), str(dst))
        else:
            _fix_file(src, dst, ABS_THRESHOLD, STD_MULT)

    print(f"Processed {len(files)} files")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
