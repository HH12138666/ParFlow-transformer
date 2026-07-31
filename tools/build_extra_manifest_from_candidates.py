#!/usr/bin/env python3
# sbatch /home/huanghui/data/slurm_job/build_extra_manifest_from_candidates.sh
"""根据 APCP 降雨强度生成额外训练样本 manifest。

这个脚本是额外训练样本的唯一入口：
1. 支持一个或多个增强数据源，例如 APCP×1.4、APCP×1.8。
2. 只保留 dry/light/moderate/heavy 四种 24h 降雨强度情景。
3. 输出训练代码可直接读取的 CSV，包含 data_root 和 press 起始时刻 t0。
"""
from __future__ import annotations

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
# 输出目录：生成的 extra manifest CSV 都会放在这里。
OUT_DIR = Path("/home/huanghui/data/ParFlow-transformer/data/parflow/extra_data_index")

# 只用训练年份生成额外训练数据；2019 是测试集，不放进这里。
YEARS = (2020, 2021)

# 一个训练样本总长度：输入 12h + 输出 12h = 24h。
WINDOW_HOURS = 24

# APCP 文件从 h+1 开始，press 样本从 h 开始，所以 APCP 起点转 press t0 要减 1。
APCP_TO_PRESS_T0_OFFSET = -1

# 额外数据源；可以保留一个，也可以同时保留多个。
SOURCES = [
    {
        "name": "apcp14",
        "apcp_root": Path("/home/huanghui/data/ParFlow_train_data/parflow_forcing/APCP1.4"),
        "data_root": "/home/huanghui/data/ParFlow-transformer/data/parflow/extra_data_apcp14",
    },
    {
        "name": "apcp18",
        "apcp_root": Path("/home/huanghui/data/ParFlow_train_data/parflow_forcing/APCP1.8"),
        "data_root": "/home/huanghui/data/ParFlow-transformer/data/parflow/extra_data_apcp18",
    },
]

# 本次只生成一份 manifest；你需要什么额外数据，就改这里后重新运行。
# SELECT_MODE 可选：
# "all"       = 保留全部候选样本。
# "top_hours" = 按降雨指标从大到小选择一部分样本。
# "regime"    = 按 dry/light/moderate/heavy 情景选择样本。
JOB_NAME = "heavy_all"
SELECT_MODE = "regime"
OUT_CSV = "extra_apcp_heavy_all_2020_2021.csv"

# SELECT_REGIMES 只在 SELECT_MODE="regime" 时生效。
# 可选情景：dry, light, moderate, heavy。
SELECT_REGIMES = ("heavy",)

# SAMPLE_COUNT=0 表示保留全部命中样本；>0 表示只取指定数量。
SAMPLE_COUNT = 0

# SAMPLE_SELECT 控制取样顺序：
# "natural"         = 按时间顺序取前 N 个。
# "rain_total_desc" = 按 24h 累计降雨从大到小取 N 个。
# "rain_total_asc"  = 按 24h 累计降雨从小到大取 N 个。
# "random"          = 固定随机种子随机抽 N 个。
SAMPLE_SELECT = "rain_total_desc"

# top_hours 模式配置：按 TOP_SCORE 从大到小选 TOP_HOURS/6 个样本。
TOP_HOURS = 4800
TOP_SCORE = "rain_total"

# 随机选样时使用的种子。
RANDOM_SEED = 2026
# =====================================================


def main() -> None:
    """生成候选池、写汇总表，并输出本次配置指定的一份 manifest。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_candidate_pool()
    write_summary(rows)
    job = build_job()
    selected = select_rows(rows, job)
    write_manifest(job, selected)
    print_job_summary(job, selected)


def build_job() -> dict[str, object]:
    """把顶部配置区组装成本次 manifest 生成任务。"""
    return {
        "name": JOB_NAME,
        "mode": SELECT_MODE,
        "out_csv": OUT_CSV,
        "regimes": SELECT_REGIMES,
        "sample_count": SAMPLE_COUNT,
        "sample_select": SAMPLE_SELECT,
        "top_hours": TOP_HOURS,
        "score": TOP_SCORE,
    }


def build_candidate_pool() -> list[dict[str, object]]:
    """合并所有数据源和年份的候选样本。"""
    rows: list[dict[str, object]] = []
    for source in SOURCES:
        source_rows = build_source_rows(source)
        rows.extend(source_rows)
    rows.sort(key=lambda row: (int(row["t0"]), str(row["source"])))
    print(f"candidate_pool_rows={len(rows)}")
    return rows


def build_source_rows(source: dict[str, object]) -> list[dict[str, object]]:
    """读取一个增强数据源，并过滤掉无法组成完整 24h 样本的窗口。"""
    raw: list[dict[str, object]] = []
    for year in YEARS:
        raw.extend(build_source_year_rows(source, int(year)))
    filtered = filter_complete_rows(raw, str(source["data_root"]))
    print(f"source={source['name']} raw={len(raw)} usable={len(filtered)}")
    return filtered


def build_source_year_rows(source: dict[str, object], year: int) -> list[dict[str, object]]:
    """按单个年份生成 stride=6 的 24h 候选窗口。"""
    apcp_dir = Path(source["apcp_root"]) / str(year)
    hour_ids, apcp = read_apcp_hourly(apcp_dir)
    candidates = build_stride_candidates(hour_ids, apcp)
    rows = candidate_arrays_to_rows(candidates, source, year)
    print(f"source={source['name']} year={year} raw_windows={len(rows)}")
    return rows


def candidate_arrays_to_rows(data: dict[str, np.ndarray], source: dict[str, object], year: int) -> list[dict[str, object]]:
    """把 numpy 候选数组转为带数据源信息的行。"""
    return [candidate_row(data, source, year, idx) for idx in range(len(data["t0"]))]


def candidate_row(data: dict[str, np.ndarray], source: dict[str, object], year: int, idx: int) -> dict[str, object]:
    """构造单个候选样本行。"""
    row = {key: scalar(data[key][idx]) for key in data}
    row["t0"] = int(row["t0"]) + APCP_TO_PRESS_T0_OFFSET
    row["source"] = str(source["name"])
    row["data_root"] = str(source["data_root"])
    row["year"] = int(year)
    return row


def scalar(value: object) -> object:
    """把 numpy 标量转成 Python 标量，方便 csv 写出。"""
    return value.item() if hasattr(value, "item") else value


def filter_complete_rows(rows: list[dict[str, object]], data_root: str) -> list[dict[str, object]]:
    """只保留 press/evap 对齐后能组成完整窗口的 t0。"""
    years = sorted({int(str(row["t0"])[:4]) for row in rows})
    press_files, _, _ = prepare_frame_files(data_root, years, False, "press", True)
    mapping = hour_to_index(press_files)
    return [row for row in rows if is_complete_window(mapping, int(row["t0"]))]


def is_complete_window(mapping: dict[int, int], t0: int) -> bool:
    """检查 t0 到 t0+23 是否都存在。"""
    return all((t0 + offset) in mapping for offset in range(WINDOW_HOURS))


def select_rows(rows: list[dict[str, object]], job: dict[str, object]) -> list[dict[str, object]]:
    """按 job 配置筛选并限制样本数。"""
    candidates = rows_for_mode(rows, job)
    ordered = order_rows(candidates, str(job.get("sample_select", "natural")))
    return limit_rows(ordered, int(job.get("sample_count", 0)), str(job["name"]))


def rows_for_mode(rows: list[dict[str, object]], job: dict[str, object]) -> list[dict[str, object]]:
    """根据 all/top_hours/regime 三种模式返回候选行。"""
    mode = str(job["mode"])
    if mode == "all":
        return list(rows)
    if mode == "top_hours":
        return top_hour_rows(rows, job)
    if mode == "regime":
        return regime_rows(rows, tuple(job["regimes"]))
    raise ValueError(f"Invalid job mode={mode}")


def top_hour_rows(rows: list[dict[str, object]], job: dict[str, object]) -> list[dict[str, object]]:
    """选出降雨指标最高的若干个窗口。"""
    score = str(job.get("score", TOP_SCORE))
    top_hours = int(job.get("top_hours", TOP_HOURS))
    count = int(np.ceil(top_hours / 6.0))
    ordered = sorted(rows, key=lambda row: (-float(row[score]), int(row["t0"])))
    return ordered[: min(count, len(ordered))]


def regime_rows(rows: list[dict[str, object]], regimes: tuple[str, ...]) -> list[dict[str, object]]:
    """选出命中任一指定降雨情景的窗口。"""
    validate_regimes(regimes)
    return [row for row in rows if any(int(row[regime]) == 1 for regime in regimes)]


def validate_regimes(regimes: tuple[str, ...]) -> None:
    """确保只使用 dry/light/moderate/heavy 四种情景。"""
    unknown = sorted(set(regimes) - set(RAIN_KEYS))
    if unknown:
        raise ValueError(f"Unknown regimes: {unknown}; valid={RAIN_KEYS}")


def order_rows(rows: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    """对候选样本排序。"""
    if mode == "rain_total_desc":
        return sorted(rows, key=lambda row: (-float(row["rain_total"]), int(row["t0"])))
    if mode == "rain_total_asc":
        return sorted(rows, key=lambda row: (float(row["rain_total"]), int(row["t0"])))
    if mode == "random":
        return random_rows(rows)
    if mode == "natural":
        return sorted(rows, key=lambda row: (int(row["t0"]), str(row["source"])))
    raise ValueError(f"Invalid sample_select={mode}")


def random_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """固定随机种子打乱候选样本。"""
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(rows))
    return [rows[int(idx)] for idx in order]


def limit_rows(rows: list[dict[str, object]], count: int, name: str) -> list[dict[str, object]]:
    """限制最终写出的样本数；count=0 表示全保留。"""
    if count <= 0:
        return rows
    if count > len(rows):
        raise ValueError(f"{name} needs {count} rows, only {len(rows)} available")
    return rows[:count]


def write_manifest(job: dict[str, object], rows: list[dict[str, object]]) -> None:
    """写出训练代码可读取的 manifest CSV。"""
    path = OUT_DIR / str(job["out_csv"])
    fields = ["split", "data_root", "t0", "source", "year", "rain_total", *RAIN_KEYS]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(manifest_row(row))
    print(f"saved_manifest={path} rows={len(rows)}")


def manifest_row(row: dict[str, object]) -> dict[str, object]:
    """只保留训练和复核需要的字段。"""
    out = {"split": "train", "data_root": row["data_root"], "t0": int(row["t0"])}
    out.update({"source": row["source"], "year": int(row["year"]), "rain_total": float(row["rain_total"])})
    out.update({key: int(row[key]) for key in RAIN_KEYS})
    return out


def write_summary(rows: list[dict[str, object]]) -> None:
    """写出候选池中各来源和各情景的样本数。"""
    path = OUT_DIR / "extra_apcp_2020_2021_summary.csv"
    groups = ["all", *sorted({str(row["source"]) for row in rows})]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["group", "rows", *RAIN_KEYS])
        writer.writeheader()
        for group in groups:
            selected = rows if group == "all" else [row for row in rows if row["source"] == group]
            writer.writerow(summary_row(group, selected))
    print(f"saved_summary={path}")


def summary_row(group: str, rows: list[dict[str, object]]) -> dict[str, object]:
    """汇总一个来源组的四情景数量。"""
    out = {"group": group, "rows": len(rows)}
    out.update({key: sum(int(row[key]) for row in rows) for key in RAIN_KEYS})
    return out


def print_job_summary(job: dict[str, object], rows: list[dict[str, object]]) -> None:
    """打印每个输出文件的来源和情景数量。"""
    sources = {source: sum(1 for row in rows if row["source"] == source) for source in sorted({row["source"] for row in rows})}
    regimes = {key: sum(int(row[key]) for row in rows) for key in RAIN_KEYS}
    print(f"job={job['name']} rows={len(rows)} sources={sources} regimes={regimes}")


if __name__ == "__main__":
    main()
