#!/usr/bin/env python3
# sbatch /home/huanghui/data/slurm_job/compute_apcp14_extra_stats.sh
"""根据指定小时计算普通训练集 + 额外数据的 mean/std。

思路和旧版 compute_mean_std.py 保持一致：
1. 动态变量 press、evaptrans 分别按小时读取并累计。
2. 额外数据 CSV 只提供起始时间 t0，脚本展开成 t0 ~ t0+23。
3. 静态变量不随额外数据变化，只从普通数据目录读取一次。
4. 最后把动态变量 stats 和静态变量 stats 按通道拼起来保存。
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from parflow_extra_data_common import (
    ChannelAccumulator,
    hour_to_index,
    prepare_frame_files,
    read_evap_frame,
    read_manifest_rows,
    read_press_frame,
    read_static_stack,
)

# ===================== 用户配置区 =====================
# NORMAL_DATA_ROOT: 普通 ParFlow 数据根目录。
NORMAL_DATA_ROOT = "/home/huanghui/data/ParFlow-transformer/data/parflow"

# NORMAL_YEARS: 普通训练数据年份，可以写成 "2020,2021"。
NORMAL_YEARS = "2020,2021"

# EXTRA_DATA_ROOT: 额外数据 CSV 不写 data_root 时使用的默认额外数据根目录。
EXTRA_DATA_ROOT = "/home/huanghui/data/ParFlow_train_data/apcp1.4"

# MANIFEST_DIR: 额外数据 CSV 所在目录。
MANIFEST_DIR = "/home/huanghui/data/ParFlow-transformer/extra_data/extra_apcp14_training_design"

# STATS_OUT_DIR: stats npz 输出目录，和旧 compute_mean_std.py 保持一致。
STATS_OUT_DIR = "/home/huanghui/data/ParFlow-transformer/stats"

# VAR_NAME: 主变量名。当前是 pressure。
VAR_NAME = "press"
USE_STATIC = True
USE_EVAP = True

# WINDOW_HOURS: 一个 t0 需要展开统计的小时数，输入 12h + 输出 12h = 24h。
WINDOW_HOURS = 24

# EXTRA_MANIFEST: 这次要加入的额外数据 CSV 文件名。空字符串表示只统计普通训练集。
EXTRA_MANIFEST = "extra_apcp14_regime_moderate_heavy_2020_2021.csv"

# OUT_NPZ: 本次输出的 stats 文件名。
OUT_NPZ = "stats1_1.4_press_evap_static_2020_2021_moderate_heavy.npz"
# =====================================================


def parse_years(text: str) -> list[int]:
    """把配置区的年份字符串转成整数列表。"""
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def accumulate_aligned_hours(acc: ChannelAccumulator, press_files: list[str], evap_files: list[str] | None,
                             hours: list[int] | None) -> None:
    """累计指定小时的 press/evap；hours=None 表示统计全部对齐后的小时。"""
    indices = selected_indices(press_files, hours)
    for pos, idx in enumerate(indices, start=1):
        frame = dynamic_frame(press_files[idx], None if evap_files is None else evap_files[idx])
        acc.update(frame)
        if pos % 500 == 0 or pos == len(indices):
            print(f"[dynamic {pos}/{len(indices)}]")


def selected_indices(press_files: list[str], hours: list[int] | None) -> list[int]:
    """把小时列表转换成 press_files 中的下标。"""
    if hours is None:
        return list(range(len(press_files)))
    mapping = hour_to_index(press_files)
    missing = [hour for hour in hours if hour not in mapping]
    if missing:
        raise ValueError(f"Missing dynamic hours in aligned files: {missing[:5]} count={len(missing)}")
    return [mapping[hour] for hour in sorted(set(hours))]


def dynamic_frame(press_path: str, evap_path: str | None) -> np.ndarray:
    """读取一个小时的动态输入通道。"""
    press = read_press_frame(press_path)
    if evap_path is None:
        return press
    return np.concatenate([press, read_evap_frame(evap_path)], axis=0)


def extra_hours_by_root() -> dict[str, list[int]]:
    """从额外 CSV 中读取 t0，并按 data_root 展开为真实需要统计的小时。"""
    if not EXTRA_MANIFEST:
        return {}
    grouped = defaultdict(set)
    manifest_path = Path(MANIFEST_DIR) / EXTRA_MANIFEST
    for row in read_manifest_rows(manifest_path, default_root=EXTRA_DATA_ROOT):
        grouped[str(row["data_root"])].update(expand_window_hours(int(row["t0"])))
    return {root: sorted(hours) for root, hours in grouped.items()}


def expand_window_hours(t0: int) -> list[int]:
    """把一个样本起始时间 t0 展开成 t0 ~ t0+23。"""
    return [t0 + offset for offset in range(WINDOW_HOURS)]


def add_normal_dynamic(acc: ChannelAccumulator) -> None:
    """累计普通训练年份的全部动态小时。"""
    years = parse_years(NORMAL_YEARS)
    press_files, evap_files, _ = prepare_frame_files(NORMAL_DATA_ROOT, years, USE_STATIC, VAR_NAME, USE_EVAP)
    accumulate_aligned_hours(acc, press_files, evap_files, None)


def add_extra_dynamic(acc: ChannelAccumulator) -> None:
    """累计额外 CSV 指定的动态小时。"""
    for data_root, hours in extra_hours_by_root().items():
        years = sorted({int(str(hour)[:4]) for hour in hours})
        press_files, evap_files, _ = prepare_frame_files(data_root, years, USE_STATIC, VAR_NAME, USE_EVAP)
        accumulate_aligned_hours(acc, press_files, evap_files, hours)


def static_stats() -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """静态变量只从普通数据目录读取一次。"""
    if not USE_STATIC:
        return None, None
    _, _, static_root = prepare_frame_files(NORMAL_DATA_ROOT, parse_years(NORMAL_YEARS), True, VAR_NAME, USE_EVAP)
    acc = ChannelAccumulator()
    acc.update(read_static_stack(static_root))
    return acc.mean_std()


def concat_stats(dynamic_stats, static_stats_pair):
    """把动态变量和静态变量的 stats 拼成完整输入通道 stats。"""
    dyn_mean, dyn_std = dynamic_stats
    static_mean, static_std = static_stats_pair
    if static_mean is None:
        return dyn_mean, dyn_std
    return np.concatenate([dyn_mean, static_mean]), np.concatenate([dyn_std, static_std])


def save_stats(dynamic_acc: ChannelAccumulator) -> None:
    """保存 mean/std/count 到 stats 目录。"""
    mean, std = concat_stats(dynamic_acc.mean_std(), static_stats())
    out = Path(STATS_OUT_DIR) / OUT_NPZ
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mean=mean.astype(np.float32), std=std.astype(np.float32), count=np.array(dynamic_acc.count))
    print(f"saved stats: {out} channels={mean.shape[0]} dynamic_pixels={dynamic_acc.count}")


def main() -> None:
    """脚本入口：普通动态 + 指定额外动态 + 静态，最后保存一份 stats。"""
    print(f"[stats] normal_years={NORMAL_YEARS} extra_manifest={EXTRA_MANIFEST or 'None'}")
    dynamic_acc = ChannelAccumulator()
    add_normal_dynamic(dynamic_acc)
    add_extra_dynamic(dynamic_acc)
    save_stats(dynamic_acc)


if __name__ == "__main__":
    main()
