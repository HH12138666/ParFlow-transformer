"""Path discovery, year parsing, and dynamic-variable time alignment."""

import os
from pathlib import Path
import re


def natural_key(path):
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]


def extract_hour_id(path):
    name = os.path.basename(path)
    match = re.search(r"(\d+)(?!.*\d)", name)
    if not match:
        return None
    return int(match.group(1))


def extract_year(path):
    name = os.path.basename(path)
    match = re.search(r"(\d+)(?!.*\d)", name)
    if match:
        year = _year_from_token(match.group(1))
        if year is not None:
            return year
    parent = os.path.basename(os.path.dirname(path))
    if re.fullmatch(r"(19|20)\d{2}", parent):
        return int(parent)
    return None


def _year_from_token(token):
    if len(token) >= 8:
        return int(token[:4])
    if len(token) == 4:
        year = int(token)
        if 1900 <= year <= 2100:
            return year
    return None


def normalize_years(years):
    if years is None:
        return None
    if isinstance(years, int):
        return [years]
    return [int(year) for year in years]


def list_pfb_files(root):
    root_path = Path(root)
    files = sorted((str(path) for path in root_path.rglob("*.pfb")), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No .pfb files found under: {root}")
    return files


def filter_files_by_years(files, allowed_years):
    if allowed_years is None:
        return files
    year_set = {int(year) for year in allowed_years}
    filtered = [path for path in files if extract_year(path) in year_set]
    if not filtered:
        raise ValueError(f"No files found for allowed_years={sorted(year_set)}.")
    return filtered


def build_year_ranges(files):
    year_ranges = {}
    for idx, path in enumerate(files):
        year = extract_year(path)
        if year is None:
            raise ValueError(f"Cannot parse year from filename/folder: {path}")
        year_ranges.setdefault(year, [idx, idx + 1])
        year_ranges[year][1] = idx + 1
    return {year: (rng[0], rng[1]) for year, rng in year_ranges.items()}


def build_id_map(files, label):
    items = {}
    for path in files:
        hour_id = extract_hour_id(path)
        if hour_id is None:
            raise ValueError(f"{label} file has no hour id: {path}")
        if hour_id in items:
            raise ValueError(f"Duplicate {label} hour id {hour_id}: {items[hour_id]} and {path}")
        items[hour_id] = path
    if not items:
        raise ValueError(f"No {label} files with valid hour ids found.")
    return items


def terminal_ids_by_group(hour_ids):
    groups = {}
    for hour_id in hour_ids:
        token = str(int(hour_id))
        group = int(token[:4]) if len(token) >= 8 else None
        groups.setdefault(group, []).append(hour_id)
    return {max(ids) for ids in groups.values()}


def shift_evap_map_to_press_hours(press_ids, evap_map):
    shifted = {hour_id - 1: path for hour_id, path in evap_map.items()}
    missing = [hour_id for hour_id in press_ids if hour_id not in shifted]
    allowed_missing = terminal_ids_by_group(press_ids)
    unexpected = [hour_id for hour_id in missing if hour_id not in allowed_missing]
    if unexpected:
        raise ValueError(f"Missing shifted evap hours for press ids (first 5): {unexpected[:5]}")
    kept_ids = [hour_id for hour_id in press_ids if hour_id in shifted]
    _print_evap_alignment(missing)
    return kept_ids, shifted


def _print_evap_alignment(missing):
    if missing:
        print(
            "[evap alignment] press(h) uses evaptrans(h+1); "
            f"dropped terminal press ids: {missing[:5]} (count={len(missing)})"
        )
        return
    print("[evap alignment] press(h) uses evaptrans(h+1); no press ids dropped")


def prepare_press_evap_files(press_root, evap_root=None, allowed_years=None):
    press_files = filter_files_by_years(list_pfb_files(press_root), allowed_years)
    evap_files = filter_files_by_years(list_pfb_files(evap_root), allowed_years) if evap_root else None
    press_map = build_id_map(press_files, "press")
    press_ids = sorted(press_map)
    if evap_files is None:
        return [press_map[hour_id] for hour_id in press_ids], None
    raw_evap_map = build_id_map(evap_files, "evap")
    press_ids, evap_map = shift_evap_map_to_press_hours(press_ids, raw_evap_map)
    return [press_map[hour_id] for hour_id in press_ids], [evap_map[hour_id] for hour_id in press_ids]


def resolve_parflow_roots(data_root, use_static=True, var_name="press", use_evap=True):
    base = Path(data_root)
    press_root = str(base / var_name)
    evap_root = None
    if use_evap:
        evap_root_path = base / "evaptrans"
        if not evap_root_path.exists():
            alt = base / "evapotrans"
            if alt.exists():
                evap_root_path = alt
        evap_root = str(evap_root_path)
    static_root = str(base / "static") if use_static else None
    return press_root, evap_root, static_root
