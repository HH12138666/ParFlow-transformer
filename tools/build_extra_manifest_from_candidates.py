#!/usr/bin/env python3
# sbatch /home/huanghui/data/slurm_job/build_apcp14_extra_manifests.sh

"""根据降雨 pfb 直接生成额外训练数据 CSV。

这个脚本是第一个功能入口：
1. 读取指定 APCP 降雨目录。
2. 按模型样本方式生成 24h、stride=6h 的候选窗口。
3. 按配置区定义的策略，输出后续训练可读取的 extra manifest CSV。
"""
import csv
from pathlib import Path

import numpy as np

from parflow_extra_data_common import (
    RAIN_KEYS,
    build_stride_candidates,
    hour_to_index,
    prepare_frame_files,
    read_apcp_hourly,
)
# ===================== 用户配置区 =====================
# APCP_DIRS: 额外数据对应的降雨 forcing 目录，可以一次写多个年份。
# 工具会先分别在每个目录内生成 24h/stride=6h 候选窗口，再合并后统一筛选。
APCP_DIRS = [
    "/home/huanghui/data/ParFlow_train_data/parflow_forcing/APCP1.4/2020",
    "/home/huanghui/data/ParFlow_train_data/parflow_forcing/APCP1.4/2021",
]

# APCP_DIR: 单目录兼容入口；当 APCP_DIRS 为空时才会使用它。
APCP_DIR = "/home/huanghui/data/ParFlow_train_data/parflow_forcing/APCP1.4/2021"

# EXTRA_DATA_ROOT: 额外 ParFlow 动态数据根目录，只用于检查 t0 是否能组成完整样本。
# 注意：CSV 只写 split 和 t0，不写 data_root；训练时仍通过 --extra_data_root 指定这个目录。
EXTRA_DATA_ROOT = "/home/huanghui/data/ParFlow_train_data/apcp1.4"

# WINDOW_HOURS: 一个训练样本需要的总小时数，输入 12h + 输出 12h = 24h。
WINDOW_HOURS = 24

# OUT_DIR: 输出 CSV 的目录。
OUT_DIR = "/home/huanghui/data/ParFlow-transformer/extra_data/extra_apcp14_training_design"

# SELECT_MODE: 选择你这次要生成哪一种额外训练数据 CSV。
# 可选值：
# "all"       = 加入全部候选样本。
# "top_hours" = 按降雨指标排序，只加入降雨最强的一部分。
# "regime"    = 只加入指定降雨情景。
SELECT_MODE = "regime"

# OUT_CSV: 本次输出的训练 manifest 文件名。
OUT_CSV = "extra_apcp14_regime_moderate_heavy_2020_2021.csv"

# TOP_HOURS / TOP_SCORE: 仅当 SELECT_MODE = "top_hours" 时生效。
# TOP_HOURS=2400 表示约选 2400 小时；由于样本 stride=6h，实际选 ceil(2400/6) 个窗口。
TOP_HOURS = 4800
TOP_SCORE = "rain_total"

# SELECT_REGIMES: 仅当 SELECT_MODE = "regime" 时生效。
# 多个情景用英文逗号分隔，只要某个 24h 窗口命中任一情景，就会被加入 CSV。
# 可选情景：dry, light, moderate, heavy, strong_6h, persistent_wet, dry_to_wet, wet_to_dry。
SELECT_REGIMES = "moderate, heavy"
# =====================================================


def selected_indices(data: dict[str, np.ndarray], job: dict[str, object]) -> np.ndarray:
    """根据单个 job 的筛选模式，返回应该写入 CSV 的候选窗口下标。"""
    mode = str(job["mode"])
    total = len(data["t0"])
    if mode == "all":
        return np.arange(total, dtype=np.int64)
    if mode == "top_hours":
        return top_hour_indices(data, job, total)
    if mode == "regime":
        return np.flatnonzero(build_regime_mask(data, str(job["regimes"])))
    raise ValueError(f"Invalid mode={mode}; expected all | top_hours | regime")


def top_hour_indices(data: dict[str, np.ndarray], job: dict[str, object], total: int) -> np.ndarray:
    """选出降雨指标最大的若干个 24h 窗口。"""
    count = int(np.ceil(int(job["top_hours"]) / 6.0))
    order = np.argsort(np.asarray(data[str(job["score"])], dtype=np.float64))[::-1]
    return np.sort(order[:min(count, total)])


def build_regime_mask(data: dict[str, np.ndarray], regimes_text: str) -> np.ndarray:
    """把多个降雨情景合成一个布尔掩膜，只要命中任一情景就会被选中。"""
    regimes = [item.strip() for item in regimes_text.split(",") if item.strip()]
    unknown = sorted(set(regimes) - set(RAIN_KEYS))
    if unknown:
        raise ValueError(f"Unknown regimes: {unknown}; valid={RAIN_KEYS}")
    mask = np.zeros(len(data["t0"]), dtype=bool)
    for regime in regimes:
        mask |= np.asarray(data[regime]).astype(bool)
    return mask


def manifest_path(job: dict[str, object]) -> Path:
    """得到当前 job 对应的 CSV 输出路径。"""
    return Path(OUT_DIR) / str(job["out_csv"])


def write_manifest(job: dict[str, object], data: dict[str, np.ndarray], indices: np.ndarray) -> None:
    """把筛选后的候选窗口写成训练可用的最简 CSV。"""
    fields = ["split", "t0"]
    path = manifest_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for idx in indices:
            writer.writerow(build_row(data, int(idx)))


def build_row(data: dict[str, np.ndarray], idx: int) -> dict[str, object]:
    """组装 CSV 的一行；训练端会用 extra_data_root 补充数据根目录。"""
    return {"split": "train", "t0": int(data["t0"][idx])}


def run_job(data: dict[str, np.ndarray], job: dict[str, object]) -> None:
    """执行一个额外数据方案，并打印输出行数。"""
    indices = selected_indices(data, job)
    if indices.size == 0:
        raise ValueError(f"No rows selected for job={job}")
    write_manifest(job, data, indices)
    print(f"manifest_rows={indices.size} out={manifest_path(job)}")


def build_job() -> dict[str, object]:
    """把配置区的单选参数组装成内部使用的 job。"""
    job = {"mode": SELECT_MODE, "out_csv": OUT_CSV}
    if SELECT_MODE == "top_hours":
        job.update({"top_hours": TOP_HOURS, "score": TOP_SCORE})
    if SELECT_MODE == "regime":
        job.update({"regimes": SELECT_REGIMES})
    return job


def active_apcp_dirs() -> list[Path]:
    """返回本次要读取的 APCP 目录；优先使用多目录配置。"""
    dirs = [Path(item) for item in APCP_DIRS if str(item).strip()]
    if dirs:
        return dirs
    return [Path(APCP_DIR)]


def build_all_candidates() -> dict[str, np.ndarray]:
    """分别读取多个 APCP 目录，并把候选窗口合并成一个候选池。"""
    parts = []
    for apcp_dir in active_apcp_dirs():
        hour_ids, apcp = read_apcp_hourly(apcp_dir)
        part = build_stride_candidates(hour_ids, apcp)
        print(f"candidate_windows_raw={len(part['t0'])} apcp_dir={apcp_dir}")
        parts.append(part)
    return concat_candidates(parts)


def concat_candidates(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """合并多个年份候选窗口，并按 t0 排序。"""
    if not parts:
        raise ValueError("No APCP candidate parts generated")
    keys = parts[0].keys()
    merged = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    order = np.argsort(merged["t0"].astype(np.int64))
    return {key: value[order] for key, value in merged.items()}


def filter_complete_windows(candidates: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """只保留在额外动态数据中可以组成完整 24h 样本的 t0。"""
    years = sorted({int(str(hour)[:4]) for hour in candidates["t0"]})
    press_files, _, _ = prepare_frame_files(EXTRA_DATA_ROOT, years, False, "press", True)
    mapping = hour_to_index(press_files)
    keep_mask = np.asarray([is_complete_t0(mapping, int(t0)) for t0 in candidates["t0"]])
    dropped = int(np.count_nonzero(~keep_mask))
    if dropped:
        dropped_t0 = candidates["t0"][~keep_mask]
        print(f"dropped_incomplete_windows={dropped} first={int(dropped_t0[0])} last={int(dropped_t0[-1])}")
    return {key: value[keep_mask] for key, value in candidates.items()}


def is_complete_t0(mapping: dict[int, int], t0: int) -> bool:
    """判断某个 t0 是否能在 press/evap 对齐后形成完整 WINDOW_HOURS 窗口。"""
    return all((t0 + offset) in mapping for offset in range(WINDOW_HOURS))


def main() -> None:
    """脚本入口：读降雨、建候选窗口、过滤无效 t0，再按当前选择写一个 CSV。"""
    candidates = build_all_candidates()
    candidates = filter_complete_windows(candidates)
    print(f"candidate_windows={len(candidates['t0'])} apcp_dirs={len(active_apcp_dirs())}")
    run_job(candidates, build_job())


if __name__ == "__main__":
    main()
