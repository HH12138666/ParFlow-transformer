"""Manifest-selected extra ParFlow training samples."""

import csv

from .dataset import ParFlowDataset
from .paths import extract_hour_id, prepare_press_evap_files, resolve_parflow_roots


def read_extra_manifest(manifest_path, default_root=None):
    if not manifest_path:
        raise ValueError("use_extra_data=True requires extra_manifest_path")
    rows = []
    with open(manifest_path, newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for line_no, row in enumerate(reader, start=2):
            parsed = _parse_manifest_row(row, line_no, default_root)
            if parsed is not None:
                rows.append(parsed)
    if not rows:
        raise ValueError(f"Extra manifest has no train rows: {manifest_path}")
    return rows


def _parse_manifest_row(row, line_no, default_root):
    split = str(row.get("split", "train")).strip().lower()
    if split and split != "train":
        return None
    data_root = (row.get("data_root") or default_root or "").strip()
    if not data_root:
        raise ValueError(f"Extra manifest line {line_no} missing data_root")
    return {"data_root": data_root, "t0": _manifest_t0(row, line_no)}


def _manifest_t0(row, line_no):
    for key in ("t0", "hour_id", "start_hour"):
        value = row.get(key)
        if value not in (None, ""):
            return int(value)
    raise ValueError(f"Extra manifest line {line_no} missing t0/hour_id/start_hour")


def group_manifest_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["data_root"], []).append(row["t0"])
    return grouped


def indices_from_manifest_hours(files, hour_ids, total):
    hour_to_index = {extract_hour_id(path): idx for idx, path in enumerate(files)}
    indices = []
    for hour_id in hour_ids:
        if hour_id not in hour_to_index:
            raise ValueError(f"Extra manifest t0={hour_id} not found in data files")
        idx = hour_to_index[hour_id]
        if idx + total > len(files):
            raise ValueError(f"Extra manifest t0={hour_id} cannot form full window length={total}")
        indices.append(idx)
    return indices


def build_extra_train_datasets(base_kwargs, manifest_path, default_root, use_static, var_name, use_evap):
    datasets = []
    rows = read_extra_manifest(manifest_path, default_root=default_root)
    for data_root, hour_ids in group_manifest_rows(rows).items():
        datasets.append(_build_one_extra_dataset(base_kwargs, data_root, hour_ids, use_static, var_name, use_evap))
    return datasets


def _build_one_extra_dataset(base_kwargs, data_root, hour_ids, use_static, var_name, use_evap):
    # 额外数据只替换动态变量 press/evaptrans；静态变量沿用普通训练集。
    press_root, evap_root, _ = resolve_parflow_roots(data_root, False, var_name, use_evap)
    years = sorted({int(str(hour_id)[:4]) for hour_id in hour_ids})
    press_files, evap_files = prepare_press_evap_files(press_root, evap_root, allowed_years=years)
    kwargs = dict(base_kwargs)
    pre = kwargs.pop("pre")
    aft = kwargs.pop("aft")
    kwargs.update({
        "evap_root": evap_root,
        "static_root": base_kwargs.get("static_root") if use_static else None,
        "press_files": press_files,
        "evap_files": evap_files,
        "explicit_time_indices": indices_from_manifest_hours(press_files, hour_ids, pre + aft),
    })
    return ParFlowDataset(press_root, "train", pre, aft, **kwargs)
