import glob
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb, write_pfb

REPO = "/home/huanghui/data/ParFlow-transformer"
sys.path.insert(0, REPO)
from openstl.models import PredFormer_Model


# sbatch /home/huanghui/data/slurm_job/inference.sh


@dataclass(frozen=True)
class InferenceConfig:
    run_dir: str = "/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press/2026-06-13-17-38_FACTS"
    checkpoint_file: str = "latest.pth"
    explicit_checkpoint_path: str = ""
    data_root_override: str = ""
    output_dir: str = "/home/huanghui/data/ParFlow-transformer/inference_data/press"
    run_param: str = "test1_2019_press_evap_static_train1_1.4_moderate_heavy_20_21_post_p8_latest"
    start_hour: int = 20190000
    end_hour: int = 20198759
    use_rollout: bool = True
    rollout_hours: int = 720
    use_amp: bool = True
    amp_dtype: str = "fp16"
    patch_batch_size: int = 28
    preload_rollout_aux: bool = True
    empty_cache_each_block: bool = False
    eps: float = 1e-8


CONFIG = InferenceConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def natural_key(path):
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(part) if part.isdigit() else part for part in parts]


def extract_id(path):
    match = re.search(r"(\d+)(?!.*\d)", os.path.basename(path))
    return int(match.group(1)) if match else None


def list_pfb_files(root, recursive=True):
    pattern = "**/*.pfb" if recursive else "*.pfb"
    files = sorted(glob.glob(os.path.join(root, pattern), recursive=recursive), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No .pfb files found under: {root}")
    return files


def resolve_checkpoint(run_dir, checkpoint_file, explicit_path=""):
    if explicit_path:
        path = Path(explicit_path)
    else:
        run_path = Path(run_dir)
        file_path = Path(checkpoint_file)
        if file_path.is_absolute():
            path = file_path
        elif (run_path / file_path).exists():
            path = run_path / file_path
        else:
            path = run_path / "checkpoints" / file_path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def load_model_params(checkpoint_path):
    ckpt_path = Path(checkpoint_path).resolve()
    candidates = [ckpt_path.parent.parent / "model_param.json", ckpt_path.parent / "model_param.json"]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as file_obj:
                params = json.load(file_obj)
            if not isinstance(params, dict):
                raise ValueError(f"model_param.json must be a dict, got {type(params)}")
            return params, path
    raise FileNotFoundError(f"model_param.json not found for checkpoint: {checkpoint_path}")


def require_param(params, key):
    if key not in params:
        raise KeyError(f"model_param.json missing required key: {key}")
    return params[key]


def require_model_key(model_config, key):
    if key not in model_config:
        raise KeyError(f"model_config missing required key: {key}")
    return model_config[key]


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def resolve_parflow_roots(data_root, var_name, use_evap, use_static):
    base = Path(data_root)
    var_root = str(base / var_name)
    evap_root = None
    if use_evap:
        evap_path = base / "evaptrans"
        alt_path = base / "evapotrans"
        evap_root = str(evap_path if evap_path.exists() else alt_path)
    static_root = str(base / "static") if use_static else None
    return var_root, evap_root, static_root


def read_var_frame(path):
    arr = read_pfb(get_absolute_path(str(path))).astype(np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D var array, got shape {arr.shape} for {path}")
    return arr


def read_evap_frame(path):
    arr = read_pfb(get_absolute_path(str(path))).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D evap array, got shape {arr.shape} for {path}")
    return arr


def parse_static_patterns(static_data):
    if static_data is None:
        return None
    if isinstance(static_data, (list, tuple)):
        patterns = [str(item).strip() for item in static_data if str(item).strip()]
    else:
        patterns = [item.strip() for item in str(static_data).split(",") if item.strip()]
    return patterns or None


def filter_static_files(files, static_data):
    patterns = parse_static_patterns(static_data)
    if not patterns:
        return files
    matched = [path for path in files if any(re.search(pat, os.path.basename(path), re.I) for pat in patterns)]
    if not matched:
        raise FileNotFoundError(f"No static .pfb files matched patterns: {patterns}")
    return matched


def ensure_3d(arr, source):
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D static array, got shape {arr.shape} for {source}")
    return arr


def read_static_stack(static_root, static_data=None):
    if static_root is None:
        return None
    merged = Path(static_root) / "static.pfb"
    if merged.exists():
        return ensure_3d(read_pfb(get_absolute_path(str(merged))).astype(np.float32), merged)
    arrays = [ensure_3d(read_pfb(get_absolute_path(path)).astype(np.float32), path)
              for path in filter_static_files(list_pfb_files(static_root, recursive=False), static_data)]
    return np.concatenate(arrays, axis=0)


def read_combined_frame(var_path, evap_path=None, static_arr=None):
    parts = [read_var_frame(var_path)]
    if evap_path is not None:
        parts.append(read_evap_frame(evap_path))
    if static_arr is not None:
        if static_arr.shape[1:] != parts[0].shape[1:]:
            raise ValueError(f"Static shape {static_arr.shape} does not match frame shape {parts[0].shape}")
        parts.append(static_arr)
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def build_file_items(files):
    items = [(extract_id(path), path) for path in files]
    items = [(hour_id, path) for hour_id, path in items if hour_id is not None]
    if not items:
        raise ValueError("No files with valid numeric ids found")
    return sorted(items, key=lambda item: item[0])


def hour_group(hour_id):
    token = str(int(hour_id))
    return int(token[:4]) if len(token) >= 8 else None


def terminal_ids_by_group(hour_ids):
    groups = {}
    for hour_id in hour_ids:
        groups.setdefault(hour_group(hour_id), []).append(hour_id)
    return {max(ids) for ids in groups.values()}


def shift_evap_items_to_var_hours(var_ids, evap_items):
    shifted = {hour_id - 1: path for hour_id, path in evap_items}
    missing = [hour_id for hour_id in var_ids if hour_id not in shifted]
    unexpected = [hour_id for hour_id in missing if hour_id not in terminal_ids_by_group(var_ids)]
    if unexpected:
        preview = ", ".join(str(x) for x in unexpected[:5])
        raise ValueError(f"Missing shifted evap files for {len(unexpected)} hour ids, e.g. {preview}")
    kept_ids = [hour_id for hour_id in var_ids if hour_id in shifted]
    if missing:
        preview = ", ".join(str(x) for x in missing[:5])
        print(f"[evap alignment] var(h) uses evaptrans(h+1); dropped terminal var ids: {preview} (count={len(missing)})")
    else:
        print("[evap alignment] var(h) uses evaptrans(h+1); no var ids dropped")
    return kept_ids, shifted


def build_aligned_var_evap_items(var_root, evap_root=None):
    var_items = build_file_items(list_pfb_files(var_root, recursive=True))
    var_ids = [hour_id for hour_id, _ in var_items]
    evap_map = None
    if evap_root is not None:
        var_ids, evap_map = shift_evap_items_to_var_hours(var_ids, build_file_items(list_pfb_files(evap_root, True)))
        var_items = [(hour_id, dict(var_items)[hour_id]) for hour_id in var_ids]
    return [(hour_id, var_path, evap_map[hour_id] if evap_map is not None else None) for hour_id, var_path in var_items]


def find_index_by_hour(items, hour):
    for idx, (found_hour, _) in enumerate(items):
        if found_hour == hour:
            return idx
    raise ValueError(f"Hour {hour} not found in input files")


def build_space_coords(height, width, space_h, space_w, stride_h=None, stride_w=None):
    if space_h is None or space_w is None:
        return [(0, 0)]
    stride_h = stride_h or space_h
    stride_w = stride_w or space_w
    if space_h > height or space_w > width:
        raise ValueError(f"Space size {(space_h, space_w)} exceeds frame size {(height, width)}")
    tops = list(range(0, height - space_h + 1, stride_h))
    lefts = list(range(0, width - space_w + 1, stride_w))
    if not tops or tops[-1] != height - space_h:
        tops.append(height - space_h)
    if not lefts or lefts[-1] != width - space_w:
        lefts.append(width - space_w)
    return [(top, left) for top in tops for left in lefts]


def strip_module_prefix(state_dict):
    if not state_dict or not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}

@dataclass(frozen=True)
class RunSpec:
    checkpoint_path: Path
    model_param_path: Path
    model_config: dict
    var_name: str
    use_evap: bool
    use_static: bool
    static_data: object
    stats_path: str
    data_root: str


@dataclass(frozen=True)
class Sources:
    files: list
    evap_files: list
    items: list
    hours: list
    rel_paths: list
    static_arr: np.ndarray | None
    start_idx: int
    end_idx: int
    c_in: int
    height: int
    width: int


@dataclass(frozen=True)
class NormStats:
    mean_t: torch.Tensor
    std_eps_t: torch.Tensor
    mean_y: np.ndarray
    std_y_eps: np.ndarray


@dataclass
class TimingStats:
    prepare_time: float = 0.0
    read_time: float = 0.0
    forward_time: float = 0.0
    write_time: float = 0.0
    total_time: float = 0.0
    output_count: int = 0


def sync_cuda():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def load_run_spec(config):
    checkpoint_path = resolve_checkpoint(config.run_dir, config.checkpoint_file, config.explicit_checkpoint_path)
    params, model_param_path = load_model_params(checkpoint_path)
    model_config = require_param(params, "model_config")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError("model_param.json key 'model_config' must be a non-empty dict.")
    data_root = config.data_root_override.strip() or require_param(params, "data_root")
    return RunSpec(
        checkpoint_path=checkpoint_path,
        model_param_path=model_param_path,
        model_config=dict(model_config),
        var_name=require_param(params, "var_name"),
        use_evap=to_bool(require_param(params, "use_evap")),
        use_static=to_bool(require_param(params, "use_static_input")),
        static_data=params.get("static_data", None),
        stats_path=require_param(params, "stats_path"),
        data_root=data_root,
    )


def prepare_sources(config, spec):
    var_root, evap_root, static_root = resolve_parflow_roots(
        spec.data_root, spec.var_name, spec.use_evap, spec.use_static
    )
    aligned = build_aligned_var_evap_items(var_root, evap_root if spec.use_evap else None)
    files = [var_path for _, var_path, _ in aligned]
    evap_files = [evap_path for _, _, evap_path in aligned]
    items = [(hour_id, var_path) for hour_id, var_path, _ in aligned]
    hours = [hour_id for hour_id, _, _ in aligned]
    rel_paths = [Path(var_path).relative_to(var_root) for var_path in files]
    start_idx = find_index_by_hour(items, int(config.start_hour))
    end_idx = find_index_by_hour(items, int(config.end_hour))
    if end_idx < start_idx:
        raise ValueError(f"end_hour={config.end_hour} is before start_hour={config.start_hour}")
    check_contiguous_hours(hours, start_idx, end_idx)
    static_arr = read_static_stack(static_root, static_data=spec.static_data) if static_root else None
    sample = read_combined_frame(files[start_idx], evap_files[start_idx], static_arr)
    return Sources(files, evap_files, items, hours, rel_paths, static_arr, start_idx, end_idx, *sample.shape)


def check_contiguous_hours(hours, start_idx, end_idx):
    for idx in range(start_idx, end_idx):
        if hours[idx + 1] != hours[idx] + 1:
            raise ValueError(f"Missing hours between {hours[idx]} and {hours[idx + 1]}")


def validate_config(spec, sources):
    cfg = spec.model_config
    expected_in = int(require_model_key(cfg, "input_channels"))
    expected_h = int(require_model_key(cfg, "height"))
    expected_w = int(require_model_key(cfg, "width"))
    if expected_in != sources.c_in:
        raise ValueError(f"Input C mismatch: model={expected_in}, data={sources.c_in}")
    if expected_h != sources.height or expected_w != sources.width:
        raise ValueError(f"Spatial size mismatch: model=({expected_h},{expected_w}), data=({sources.height},{sources.width})")


def load_stats(spec, sources, config):
    if not spec.stats_path or not Path(spec.stats_path).exists():
        raise FileNotFoundError(f"Stats file not found: {spec.stats_path}")
    stats = np.load(spec.stats_path)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    if mean.shape[0] != sources.c_in or std.shape[0] != sources.c_in:
        raise ValueError(f"Stats C mismatch: stats={mean.shape[0]}, data={sources.c_in}")
    out_channels = int(require_model_key(spec.model_config, "out_channels"))
    return NormStats(
        mean_t=torch.from_numpy(mean).view(1, sources.c_in, 1, 1).float().to(DEVICE),
        std_eps_t=torch.from_numpy(std + config.eps).view(1, sources.c_in, 1, 1).float().to(DEVICE),
        mean_y=mean[:out_channels].reshape(1, -1, 1, 1),
        std_y_eps=(std[:out_channels] + config.eps).reshape(1, -1, 1, 1),
    )


def load_model(spec):
    model = PredFormer_Model(spec.model_config).to(DEVICE)
    ckpt = torch.load(spec.checkpoint_path, map_location=DEVICE)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    incompatible = model.load_state_dict(strip_module_prefix(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model


def autocast_ctx(config):
    if DEVICE != "cuda" or not config.use_amp:
        return nullcontext()
    dtype = torch.float16 if config.amp_dtype.lower() == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def predict_sequence(model, x, pre_seq, aft_seq, c_in, out_channels):
    if aft_seq == pre_seq:
        return model(x)
    if aft_seq < pre_seq:
        return model(x)[:, :aft_seq]
    pred_blocks, cur_seq = [], x.clone()
    for _ in range(aft_seq // pre_seq):
        pred = model(cur_seq)
        pred_blocks.append(pred)
        cur_seq = merge_pred_with_aux(pred, cur_seq, c_in, out_channels)
    if aft_seq % pre_seq:
        pred_blocks.append(model(cur_seq)[:, :aft_seq % pre_seq])
    return torch.cat(pred_blocks, dim=1)


def merge_pred_with_aux(pred, prev_seq, c_in, out_channels):
    if c_in == out_channels:
        return pred
    return torch.cat([pred, prev_seq[:, :, out_channels:c_in]], dim=2)




class SimpleSources:
    def __init__(self, pred_full):
        self.height = pred_full.shape[-2]
        self.width = pred_full.shape[-1]


def space_shape(cfg, sources):
    return int(cfg.get("space_h") or sources.height), int(cfg.get("space_w") or sources.width)

def predict_full_block(model, x, coords, spec, sources, config, timing):
    cfg = spec.model_config
    pre_seq = int(require_model_key(cfg, "pre_seq"))
    aft_seq = int(require_model_key(cfg, "after_seq"))
    out_channels = int(require_model_key(cfg, "out_channels"))
    pred_full = np.zeros((aft_seq, out_channels, sources.height, sources.width), dtype=np.float32)
    counts = np.zeros_like(pred_full, dtype=np.int32)
    for start in range(0, len(coords), config.patch_batch_size):
        chunk = coords[start:start + config.patch_batch_size]
        patch_h, patch_w = space_shape(cfg, sources)
        x_batch = torch.cat([x[..., top:top + patch_h, left:left + patch_w] for top, left in chunk], dim=0)
        sync_cuda()
        forward_start = time.perf_counter()
        with torch.inference_mode(), autocast_ctx(config):
            pred = predict_sequence(model, x_batch, pre_seq, aft_seq, sources.c_in, out_channels)
        sync_cuda()
        timing.forward_time += time.perf_counter() - forward_start
        stitch_predictions(pred.float().cpu().numpy(), pred_full, counts, chunk, cfg)
    mask = counts > 0
    pred_full[mask] = pred_full[mask] / counts[mask]
    return pred_full


def stitch_predictions(pred, pred_full, counts, coords, cfg):
    for idx, (top, left) in enumerate(coords):
        patch_h, patch_w = space_shape(cfg, SimpleSources(pred_full))
        bottom = top + patch_h
        right = left + patch_w
        pred_full[:, :, top:bottom, left:right] += pred[idx]
        counts[:, :, top:bottom, left:right] += 1


def build_aux_map(sources, spec, start_idx, end_idx):
    if not spec.use_evap and sources.static_arr is None:
        return None
    aux_map = {}
    static = sources.static_arr.astype(np.float32, copy=False) if sources.static_arr is not None else None
    for idx in range(start_idx, end_idx):
        parts = []
        if spec.use_evap:
            parts.append(read_evap_frame(sources.evap_files[idx]))
        if static is not None:
            parts.append(static)
        aux_map[idx] = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return aux_map


def build_next_input(pred_frame, hour_idx, sources, spec, aux_map, out_channels, timing):
    parts = [pred_frame[:out_channels].astype(np.float32, copy=False)]
    if aux_map is not None:
        parts.append(aux_map[hour_idx])
    elif spec.use_evap or sources.static_arr is not None:
        read_start = time.perf_counter()
        aux = read_combined_frame(sources.files[hour_idx], sources.evap_files[hour_idx], sources.static_arr)[out_channels:]
        timing.read_time += time.perf_counter() - read_start
        parts.append(aux)
    frame = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    if frame.shape[0] != sources.c_in:
        raise ValueError(f"Rollout input C mismatch at idx={hour_idx}: {frame.shape[0]} vs {sources.c_in}")
    return frame


def write_prediction_block(pred_full, t0, predicted, block, out_dir, sources, out_channels, timing):
    write_start = time.perf_counter()
    for k in range(block):
        idx = t0 + predicted + k
        rel = sources.rel_paths[idx]
        out_path = out_dir / rel.parent / f"pred_{rel.name}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = pred_full[k][:out_channels].astype(np.float64)
        write_pfb(str(out_path), arr[0] if arr.shape[0] == 1 else arr, dist=False)
    timing.write_time += time.perf_counter() - write_start
    timing.output_count += block


def run_inference(config, spec, sources, stats, model, timing):
    cfg = spec.model_config
    pre_seq = int(require_model_key(cfg, "pre_seq"))
    aft_seq = int(require_model_key(cfg, "after_seq"))
    out_channels = int(require_model_key(cfg, "out_channels"))
    total_aft = config.rollout_hours if config.use_rollout else aft_seq
    if sources.end_idx - sources.start_idx + 1 < pre_seq + total_aft:
        raise ValueError("Not enough frames for requested range and sequence lengths")
    coords = build_space_coords(sources.height, sources.width, cfg.get("space_h"), cfg.get("space_w"), cfg.get("space_stride_h"), cfg.get("space_stride_w"))
    out_dir = Path(config.output_dir) / f"{datetime.now():%Y%m%d}_{config.run_param}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = total_aft if config.use_rollout else aft_seq
    stop = sources.end_idx - pre_seq - total_aft + 2
    for t0 in range(sources.start_idx, stop, stride):
        print_window(sources, t0, pre_seq, total_aft)
        read_start = time.perf_counter()
        history = [read_combined_frame(sources.files[t0 + i], sources.evap_files[t0 + i], sources.static_arr) for i in range(pre_seq)]
        aux_map = build_aux_map(sources, spec, t0 + pre_seq, t0 + pre_seq + total_aft) if config.preload_rollout_aux else None
        timing.read_time += time.perf_counter() - read_start
        run_one_window(config, model, history, t0, coords, out_dir, spec, sources, stats, aux_map, timing)


def print_window(sources, t0, pre_seq, total_aft):
    pred_start = sources.hours[t0 + pre_seq]
    pred_end = sources.hours[t0 + pre_seq + total_aft - 1]
    print(f"[window] pred_range={pred_start}..{pred_end}")


def print_timing_summary(timing):
    print("========== Timing Summary ==========")
    print(f"prepare_time : {timing.prepare_time:.2f}s")
    print(f"read_time    : {timing.read_time:.2f}s")
    print(f"forward_time : {timing.forward_time:.2f}s")
    print(f"write_time   : {timing.write_time:.2f}s")
    print(f"total_time   : {timing.total_time:.2f}s")
    print(f"output_pfb   : {timing.output_count}")


def run_one_window(config, model, history, t0, coords, out_dir, spec, sources, stats, aux_map, timing):
    cfg = spec.model_config
    pre_seq = int(require_model_key(cfg, "pre_seq"))
    aft_seq = int(require_model_key(cfg, "after_seq"))
    out_channels = int(require_model_key(cfg, "out_channels"))
    target_hours = config.rollout_hours if config.use_rollout else aft_seq
    predicted = 0
    while predicted < target_hours:
        block = min(aft_seq, target_hours - predicted)
        x = normalize_history(history[-pre_seq:], stats)
        pred_full = predict_full_block(model, x, coords, spec, sources, config, timing)
        pred_full = pred_full * stats.std_y_eps + stats.mean_y
        write_prediction_block(pred_full, t0 + pre_seq, predicted, block, out_dir, sources, out_channels, timing)
        for k in range(block):
            idx = t0 + pre_seq + predicted + k
            history.append(build_next_input(pred_full[k], idx, sources, spec, aux_map, out_channels, timing))
            if len(history) > pre_seq:
                history.pop(0)
        if DEVICE == "cuda" and config.empty_cache_each_block:
            torch.cuda.empty_cache()
        predicted += block


def normalize_history(history, stats):
    x = np.stack(history, axis=0)
    x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
    return (x - stats.mean_t) / stats.std_eps_t


def main():
    timing = TimingStats()
    total_start = time.perf_counter()
    prepare_start = time.perf_counter()
    spec = load_run_spec(CONFIG)
    print(f"[checkpoint] {spec.checkpoint_path}")
    print(f"[model_param] {spec.model_param_path}")
    print(f"[data_root] {spec.data_root}")
    sources = prepare_sources(CONFIG, spec)
    validate_config(spec, sources)
    stats = load_stats(spec, sources, CONFIG)
    model = load_model(spec)
    timing.prepare_time = time.perf_counter() - prepare_start
    run_inference(CONFIG, spec, sources, stats, model, timing)
    timing.total_time = time.perf_counter() - total_start
    print_timing_summary(timing)
    print(f"Inference done. Elapsed time: {timing.total_time:.2f}s")


if __name__ == "__main__":
    main()
