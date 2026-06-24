#!/usr/bin/env python3
"""ParFlow 额外训练数据工具的公共函数。

这个文件不是单独运行入口，只服务于两个功能脚本：
1. build_extra_manifest_from_candidates.py
2. compute_mean_std_with_extra_manifest.py
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from parflow.tools.io import read_pfb

# 让 tools 脚本可以直接导入 ParFlow-transformer 项目里的 openstl 模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openstl.datasets.parflow.paths import extract_hour_id, prepare_press_evap_files, resolve_parflow_roots

# APCP pfb 通常是 m/s 或等价通量，这里乘 3600 转成每小时降雨深度。
SECONDS_PER_HOUR = 3600.0

# 和训练样本保持一致：一个候选样本看 24h，起点每 6h 滑动一次。
SAMPLE_HOURS = 24
SAMPLE_STRIDE = 6
WET_HOUR_THRESHOLD = 0.1

# 固定降雨情景字段，和论文分析中的 24h 样本划分保持一致。
RAIN_KEYS = (
    "dry", "light", "moderate", "heavy",
    "strong_6h", "persistent_wet", "dry_to_wet", "wet_to_dry",
)


def natural_key(path: Path) -> list[object]:
    """按文件名中的数字自然排序，避免 10 排在 2 前面。"""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def list_pfb(root: Path) -> list[Path]:
    """列出目录下所有 pfb 文件，如果没有文件则直接报错。"""
    files = sorted(root.rglob("*.pfb"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No .pfb files found under {root}")
    return files


def read_apcp_hourly(apcp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 APCP 小时序列，返回 hour_id 和空间平均降雨量。"""
    hour_ids, values = [], []
    for path in list_pfb(apcp_dir):
        arr = read_pfb(str(path)).astype(np.float64)
        arr = squeeze_apcp_field(arr, path)
        hour_ids.append(extract_hour_id(str(path)))
        values.append(float(np.mean(arr) * SECONDS_PER_HOUR))
    return sort_hourly_values(hour_ids, values)


def squeeze_apcp_field(arr: np.ndarray, path: Path) -> np.ndarray:
    """把 APCP pfb 统一成 2D 空间场。"""
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D/3D APCP field, got {arr.shape} for {path}")
    return arr


def sort_hourly_values(hour_ids: list[int], values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """按 hour_id 排序，保证后续滑动窗口是时间顺序。"""
    ids = np.asarray(hour_ids, dtype=np.int64)
    order = np.argsort(ids)
    return ids[order], np.asarray(values, dtype=np.float64)[order]


def max_window(values: np.ndarray, width: int) -> float:
    """计算连续 width 小时累计降雨的最大值。"""
    if values.size < width:
        return float(np.sum(values))
    return float(np.max(np.convolve(values, np.ones(width), mode="valid")))


def rainfall_metrics(values: np.ndarray) -> dict[str, float]:
    """计算一个 24h 候选窗口的降雨统计指标。"""
    input_values = values[:12]
    output_values = values[12:24]
    return {
        "rain_total": float(np.sum(values)),
        "rain_mean": float(np.mean(values)),
        "rain_std": float(np.std(values)),
        "rain_max1h": float(np.max(values)),
        "rain_max6h": max_window(values, 6),
        "rain_hours": int(np.count_nonzero(values >= WET_HOUR_THRESHOLD)),
        "wet_frac": float(np.mean(values >= WET_HOUR_THRESHOLD)),
        "rain_in12": float(np.sum(input_values)),
        "rain_out12": float(np.sum(output_values)),
        "rain_jump": float(np.sum(output_values) - np.sum(input_values)),
    }


def category_flags(metrics: dict[str, float]) -> dict[str, int]:
    """把连续降雨指标转成多个降雨情景标签。"""
    total = metrics["rain_total"]
    max6h = metrics["rain_max6h"]
    return {
        "dry": int(total < 1.0),
        "light": int(1.0 <= total < 10.0),
        "moderate": int(10.0 <= total < 25.0),
        "heavy": int(total >= 25.0),
        "strong_6h": int(max6h >= 10.0),
        "persistent_wet": int(metrics["rain_hours"] >= 12),
        "dry_to_wet": int(metrics["rain_in12"] < 1.0 and metrics["rain_out12"] >= 5.0),
        "wet_to_dry": int(metrics["rain_in12"] >= 5.0 and metrics["rain_out12"] < 1.0),
    }


def build_stride_candidates(hour_ids: np.ndarray, apcp: np.ndarray) -> dict[str, np.ndarray]:
    """按 24h 窗口、6h 步长生成所有候选训练样本。"""
    rows = []
    for pos in range(0, len(hour_ids) - SAMPLE_HOURS + 1, SAMPLE_STRIDE):
        rows.append(build_candidate_row(hour_ids, apcp, pos))
    return rows_to_arrays(rows)


def build_candidate_row(hour_ids: np.ndarray, apcp: np.ndarray, pos: int) -> dict[str, object]:
    """生成单个候选窗口的一行统计结果。"""
    window_hours = hour_ids[pos:pos + SAMPLE_HOURS]
    if not np.all(np.diff(window_hours) == 1):
        raise ValueError(f"Non-contiguous APCP window starting at {window_hours[0]}")
    metrics = rainfall_metrics(apcp[pos:pos + SAMPLE_HOURS])
    return {"t0": int(window_hours[0]), "end_hour": int(window_hours[-1]), **metrics, **category_flags(metrics)}


def rows_to_arrays(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """把候选窗口列表转成按字段组织的 numpy 数组。"""
    if not rows:
        raise ValueError("No candidate rows generated")
    keys = rows[0].keys()
    return {key: np.asarray([row[key] for row in rows]) for key in keys}


def read_manifest_rows(path: Path, default_root: str | None = None) -> list[dict[str, object]]:
    """读取额外数据 CSV，只保留 stats 计算真正需要的 data_root 和 t0。"""
    rows = []
    with path.open(newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            data_root = row.get("data_root") or default_root
            if not data_root:
                raise ValueError(f"Manifest row missing data_root in {path}")
            rows.append({"data_root": data_root, "t0": int(row.get("t0") or row.get("start_hour"))})
    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    return rows


@dataclass
class ChannelAccumulator:
    """逐通道累计 sum/sum_of_square，用于流式计算 mean/std。"""
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None
    count: int = 0

    def update(self, arr: np.ndarray) -> None:
        """加入一个 channel-first 的 3D 数据块。"""
        data = np.asarray(arr, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError(f"Expected channel-first 3D array, got {data.shape}")
        sums = data.sum(axis=(1, 2))
        sums_sq = np.square(data).sum(axis=(1, 2))
        self._init_if_needed(data, sums, sums_sq)
        self.total += sums
        self.total_sq += sums_sq
        self.count += data.shape[1] * data.shape[2]

    def _init_if_needed(self, data: np.ndarray, sums: np.ndarray, sums_sq: np.ndarray) -> None:
        """首次更新时初始化累计数组，并检查通道数一致。"""
        if self.total is None:
            self.total = np.zeros_like(sums, dtype=np.float64)
            self.total_sq = np.zeros_like(sums_sq, dtype=np.float64)
        if self.total.shape[0] != data.shape[0]:
            raise ValueError(f"Channel mismatch: accumulator={self.total.shape[0]} arr={data.shape[0]}")

    def mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        """根据累计结果计算逐通道均值和标准差。"""
        if self.total is None or self.total_sq is None or self.count == 0:
            raise ValueError("Cannot finalize empty accumulator")
        mean = self.total / self.count
        var = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def prepare_frame_files(data_root: str, years: list[int] | None, use_static: bool, var_name: str, use_evap: bool):
    """根据数据根目录和年份，准备 press、evaptrans、static 的文件列表。"""
    press_root, evap_root, static_root = resolve_parflow_roots(data_root, use_static, var_name, use_evap)
    return (*prepare_press_evap_files(press_root, evap_root, allowed_years=years), static_root)


def hour_to_index(files: list[str]) -> dict[int, int]:
    """建立 hour_id 到文件列表下标的映射，方便从 t0 找窗口。"""
    out = {}
    for idx, path in enumerate(files):
        hour_id = extract_hour_id(path)
        if hour_id in out:
            raise ValueError(f"Duplicate hour_id={hour_id}: {path}")
        out[hour_id] = idx
    return out
