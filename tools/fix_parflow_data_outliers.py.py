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


# =========================
# Config: edit here only

# =========================
# 输入/输出路径
INPUT_DIR = "/home/huanghui/data/ParFlow_train_data/evaptrans_raw/2019"
OUTPUT_DIR = "/home/huanghui/data/ParFlow_train_data/evaptrans/2019"  # 复制到新目录时填写路径；为 None 时表示原地覆盖
IN_PLACE = False    # True 表示覆盖 INPUT_DIR 中的原始文件

# 修复模式:
# - "manual_points": 仅修复 MANUAL_POINTS
# - "auto_or_manual_union": 自动异常 + MANUAL_POINTS 并集
REPAIR_MODE = "manual_points"

# 自动异常修复参数（REPAIR_MODE="auto_or_manual_union" 时生效）
ABS_THRESHOLD = -7.0     # 绝对阈值
STD_MULT = 10000.0        # 标准差倍数阈值

# 手动点位修复配置
# 每个点格式: (layer, row, col)
MANUAL_POINTS = [
(7,115,245),(8,115,245), (9,115,245)
]
#  2019  (7,115,245),(8,115,245), (9,115,245)
#  2020  (9,36,96),(9,68,27),(8,68,127),(8,68,245),(9,115,245),(7,68,127)
# 空间插值最大迭代次数（邻域扩散轮数）
INTERP_MAX_ITERS = 1

# 处理数量限制（0 表示全部）
LIMIT = 0

# 只排查不修复（True 时不写任何输出文件）
CHECK_ONLY = False

# numerical stability constant
EPS = 1e-8


def _spatial_interpolate_2d(layer, bad_mask, max_iters=512):
    """
    Spatial interpolation on a 2D layer:
    - keep valid values unchanged;
    - iteratively fill bad cells using 8-neighbor mean;
    - if isolated cells remain after iterations, fallback to layer mean.
    """
    if layer.ndim != 2 or bad_mask.ndim != 2:
        raise ValueError("layer and bad_mask must be 2D.")
    if layer.shape != bad_mask.shape:
        raise ValueError("layer and bad_mask must have the same shape.")
    if not bad_mask.any():
        return layer

    filled = layer.astype(np.float64, copy=True)
    filled[bad_mask] = np.nan
    h, w = filled.shape

    for _ in range(max_iters):
        missing = np.isnan(filled)
        if not missing.any():
            break

        padded = np.pad(filled, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)
        nei_sum = np.zeros((h, w), dtype=np.float64)
        nei_cnt = np.zeros((h, w), dtype=np.float64)

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nei = padded[1 + di:1 + di + h, 1 + dj:1 + dj + w]
                ok = ~np.isnan(nei)
                nei_sum += np.where(ok, nei, 0.0)
                nei_cnt += ok

        fillable = missing & (nei_cnt > 0)
        if not fillable.any():
            break
        filled[fillable] = nei_sum[fillable] / nei_cnt[fillable]

    remain = np.isnan(filled)
    if remain.any():
        valid_vals = filled[~remain]
        fallback = float(valid_vals.mean()) if valid_vals.size else 0.0
        filled[remain] = fallback

    return filled


def _build_manual_mask(arr_shape, manual_points):
    c, h, w = arr_shape
    mask = np.zeros(arr_shape, dtype=bool)
    for item in manual_points:
        if len(item) != 3:
            raise ValueError(f"Manual point must be (layer,row,col), got: {item}")
        layer, row, col = int(item[0]), int(item[1]), int(item[2])
        if layer < 0 or layer >= c:
            raise ValueError(f"Manual point layer out of range: {item}, valid [0,{c-1}]")
        if row < 0 or row >= h:
            raise ValueError(f"Manual point row out of range: {item}, valid [0,{h-1}]")
        if col < 0 or col >= w:
            raise ValueError(f"Manual point col out of range: {item}, valid [0,{w-1}]")
        mask[layer, row, col] = True
    return mask


def _detect_outlier_mask_1d(flat, abs_threshold, std_mult, manual_flat_mask, only_manual):
    if only_manual:
        return manual_flat_mask

    abs_mask = flat < abs_threshold if abs_threshold is not None else np.zeros_like(flat, dtype=bool)
    valid_vals = flat[~abs_mask]
    if valid_vals.size == 0:
        layer_mean = 0.0
        layer_std = EPS
    else:
        layer_mean = float(np.mean(valid_vals))
        layer_std = max(float(np.std(valid_vals)), EPS)

    low = layer_mean - std_mult * layer_std
    high = layer_mean + std_mult * layer_std
    mask_dyn = (flat < low) | (flat > high)
    return abs_mask | mask_dyn | manual_flat_mask


def _interpolate_outliers(arr, abs_threshold=-10000.0, std_mult=5.0,
                          manual_points=None, use_manual=False, only_manual=False):
    """
    Simple + dynamic outlier repair:
    - values below abs_threshold are outliers;
    - values deviating from mean by > std_mult * std are outliers (both sides);
    - repair outliers with spatial interpolation (8-neighbor iterative fill).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for interpolation, got shape {arr.shape}.")

    repaired = arr.astype(np.float64, copy=True)
    c = repaired.shape[0]
    layer_fixed_counts = np.zeros(c, dtype=np.int64)
    manual_mask_3d = np.zeros_like(repaired, dtype=bool)
    if use_manual and manual_points:
        manual_mask_3d = _build_manual_mask(repaired.shape, manual_points)

    # Process each layer independently: outlier detection and interpolation
    # both only use values from the same layer.
    for ci in range(c):
        layer = repaired[ci]
        manual_layer_mask = manual_mask_3d[ci]
        flat = layer.reshape(-1)
        mask = _detect_outlier_mask_1d(
            flat,
            abs_threshold=abs_threshold,
            std_mult=std_mult,
            manual_flat_mask=manual_layer_mask.reshape(-1),
            only_manual=only_manual,
        )

        fixed_count = int(np.count_nonzero(mask))
        layer_fixed_counts[ci] = fixed_count
        if not mask.any():
            continue

        repaired[ci] = _spatial_interpolate_2d(
            layer,
            mask.reshape(layer.shape),
            max_iters=INTERP_MAX_ITERS,
        )
    return repaired, layer_fixed_counts


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fix_file(src: Path,
              dst: Path,
              abs_threshold: float,
              std_mult: float,
              manual_points=None,
              use_manual=False,
              only_manual=False) -> None:
    arr = read_pfb(get_absolute_path(str(src))).astype(np.float64)
    fixed, layer_fixed_counts = _interpolate_outliers(
        arr,
        abs_threshold=abs_threshold,
        std_mult=std_mult,
        manual_points=manual_points,
        use_manual=use_manual,
        only_manual=only_manual,
    )
    write_pfb(str(dst), fixed.astype(np.float64), dist=False)
    return layer_fixed_counts


def _scan_file(src: Path,
               abs_threshold: float,
               std_mult: float,
               manual_points=None,
               use_manual=False,
               only_manual=False):
    arr = read_pfb(get_absolute_path(str(src))).astype(np.float64)
    c = arr.shape[0]
    layer_fixed_counts = np.zeros(c, dtype=np.int64)
    layer_positions = {}

    manual_mask_3d = np.zeros_like(arr, dtype=bool)
    if use_manual and manual_points:
        manual_mask_3d = _build_manual_mask(arr.shape, manual_points)

    for ci in range(c):
        layer = arr[ci]
        manual_layer_mask = manual_mask_3d[ci]
        flat = layer.reshape(-1)
        mask = _detect_outlier_mask_1d(
            flat,
            abs_threshold=abs_threshold,
            std_mult=std_mult,
            manual_flat_mask=manual_layer_mask.reshape(-1),
            only_manual=only_manual,
        )
        layer_fixed_counts[ci] = int(np.count_nonzero(mask))

        if layer_fixed_counts[ci] > 0:
            mask2d = mask.reshape(layer.shape)
            rc_idx = np.argwhere(mask2d)
            points = []
            for row, col in rc_idx:
                points.append((int(row), int(col), float(layer[row, col])))
            layer_positions[ci] = points

    return layer_fixed_counts, layer_positions


def main() -> None:
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    if not CHECK_ONLY:
        if IN_PLACE and OUTPUT_DIR:
            raise ValueError("Use either --in_place or --output_dir, not both.")
        if not IN_PLACE and not OUTPUT_DIR:
            raise ValueError("Provide --output_dir or use --in_place.")
        output_dir = input_dir if IN_PLACE else Path(OUTPUT_DIR)
        _ensure_dir(output_dir)
    else:
        output_dir = None

    files = _list_pfb_files(str(input_dir))
    if LIMIT and LIMIT > 0:
        files = files[:LIMIT]

    if REPAIR_MODE not in {"manual_points", "auto_or_manual_union"}:
        raise ValueError(
            f"Invalid REPAIR_MODE={REPAIR_MODE}. "
            "Use 'manual_points' or 'auto_or_manual_union'."
        )

    use_manual_points = len(MANUAL_POINTS) > 0
    if REPAIR_MODE == "manual_points":
        if not use_manual_points:
            raise ValueError("REPAIR_MODE='manual_points' requires non-empty MANUAL_POINTS.")
        use_manual = True
        only_manual = True
    else:
        use_manual = use_manual_points
        only_manual = False

    global_total_fixed = 0
    global_layer_fixed = None
    print(f"CHECK_ONLY={CHECK_ONLY}")
    print(f"REPAIR_MODE={REPAIR_MODE}")
    print(f"use_manual={use_manual}, only_manual={only_manual}")
    print(f"MANUAL_POINTS={MANUAL_POINTS}")

    for f in files:
        src = Path(f)
        pos_detail = {}
        if CHECK_ONLY:
            layer_fixed, pos_detail = _scan_file(
                src, ABS_THRESHOLD, STD_MULT,
                manual_points=MANUAL_POINTS,
                use_manual=use_manual,
                only_manual=only_manual,
            )
        else:
            dst = output_dir / src.name
            if IN_PLACE:
                tmp = output_dir / (src.name + ".tmp")
                layer_fixed = _fix_file(
                    src, tmp, ABS_THRESHOLD, STD_MULT,
                    manual_points=MANUAL_POINTS,
                    use_manual=use_manual,
                    only_manual=only_manual
                )
                shutil.move(str(tmp), str(dst))
            else:
                layer_fixed = _fix_file(
                    src, dst, ABS_THRESHOLD, STD_MULT,
                    manual_points=MANUAL_POINTS,
                    use_manual=use_manual,
                    only_manual=only_manual
                )

        if global_layer_fixed is None:
            global_layer_fixed = np.zeros_like(layer_fixed, dtype=np.int64)
        global_layer_fixed += layer_fixed

        total_fixed = int(np.sum(layer_fixed))
        global_total_fixed += total_fixed
        if total_fixed > 0:
            nz = np.where(layer_fixed > 0)[0]
            detail = ", ".join([f"L{int(i)}:{int(layer_fixed[i])}" for i in nz])
            print(f"{src.name}: fixed={total_fixed} ({detail})")
            if CHECK_ONLY and pos_detail:
                for li in sorted(pos_detail.keys()):
                    pts = pos_detail[li]
                    pts_txt = ", ".join([f"({r},{c},v={v:.6g})" for r, c, v in pts])
                    print(f"  L{li} positions: {pts_txt}")
        else:
            print(f"{src.name}: fixed=0")

    print(f"Processed {len(files)} files")
    if CHECK_ONLY:
        print("Output dir: (not used, check-only mode)")
    else:
        print(f"Output dir: {output_dir}")
    print(f"Total fixed points: {global_total_fixed}")
    if global_layer_fixed is not None:
        layer_detail = ", ".join(
            [f"L{i}:{int(v)}" for i, v in enumerate(global_layer_fixed)]
        )
        print(f"Global layer fixed counts: {layer_detail}")


if __name__ == "__main__":
    main()


# python /home/huanghui/data/ParFlow-transformer/tools/fix_parflow_data_outliers.py
