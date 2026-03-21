import os
import re
import sys
import time
import glob
import json
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

repo = "/home/huanghui/data/ParFlow-transformer"
sys.path.insert(0, repo)

from configs.parflow.PredFormer_infer import model_config as _model_cfg
from openstl.models import PredFormer_Model
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb, write_pfb


# ---- User configuration ----
# Training output dir (contains timestamp subdirs with checkpoint.pth)
WORK_DIR = "/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_wtd"
CHECKPOINT_NAME = "2026-03-19-23-50_FACTS"
CHECKPOINT_PATH = os.path.join(WORK_DIR, CHECKPOINT_NAME, "checkpoint.pth")

DATA_ROOT = "/home/huanghui/data/ParFlow-transformer/data/parflow"
OUTPUT_DIR = "/home/huanghui/data/ParFlow-transformer/inference_data/wtd"
RUN_PARAM = "wtd_evap_static_h60_w84_in12_out12_rollout700"

# Must match current training data pipeline
VAR_NAME = "wtd"
USE_EVAP = True
USE_STATIC = True
STATIC_DATA = "perm_x,alpha_z6-9,n_z6-9,porosity_z6-9"

STATS_PATH = "/home/huanghui/data/ParFlow-transformer/stats/stats_wtd_evaptrans_perm_x_alpha_n_porosity.npz"
EPS = 1e-6

# Prediction range (parsed from filename tail digits, e.g., 20190001)
START_HOUR = 20204369
END_HOUR = 20208760

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = False
# options: "bf16" or "fp16"
AMP_DTYPE = "fp16"

# Rollout config
USE_ROLLOUT = True
ROLL_AFT = 700
EVAP_CHANNELS = [6, 7, 8, 9]
PRELOAD_ROLLOUT_AUX = True

# Prefer checkpoint-aligned settings from model_param.json
USE_MODEL_PARAM_JSON = True


def _natural_key(p):
    b = os.path.basename(p)
    s = re.split(r"(\d+)", b)
    return [int(t) if t.isdigit() else t for t in s]


def _extract_id(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)(?!.*\d)", name)
    if not m:
        return None
    return int(m.group(1))


def _list_pfb_files(root, recursive=True):
    pattern = "**/*.pfb" if recursive else "*.pfb"
    files = sorted(glob.glob(os.path.join(root, pattern), recursive=recursive), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No .pfb files found under: {root}")
    return files


def _resolve_parflow_roots(data_root, var_name="wtd", use_evap=False, use_static=False):
    base = Path(data_root)
    var_root = str(base / var_name)
    evap_root = str(base / "evaptrans") if use_evap else None
    static_root = str(base / "static") if use_static else None
    return var_root, evap_root, static_root


def _parse_static_data(static_data):
    if static_data is None:
        return None
    if isinstance(static_data, (list, tuple)):
        patterns = [str(x).strip() for x in static_data if str(x).strip()]
    else:
        patterns = [p.strip() for p in str(static_data).split(",") if p.strip()]
    return patterns or None


def _filter_static_files(files, static_data):
    patterns = _parse_static_data(static_data)
    if not patterns:
        return files
    matched = []
    for f in files:
        name = os.path.basename(f)
        if any(re.search(pat, name, re.IGNORECASE) for pat in patterns):
            matched.append(f)
    if not matched:
        raise FileNotFoundError(f"No static .pfb files matched patterns: {patterns}")
    return matched


def _read_static_stack(static_root, static_data=None):
    if static_root is None:
        return None
    files = _list_pfb_files(static_root, recursive=False)
    files = _filter_static_files(files, static_data)
    arrays = []
    for f in files:
        arr = read_pfb(get_absolute_path(str(f))).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            raise ValueError(f"Expected 2D/3D static array, got shape {arr.shape} for {f}")
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def _read_var_frame(var_path):
    arr = read_pfb(get_absolute_path(str(var_path))).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape {arr.shape} for {var_path}")
    return arr


def _read_evap_frame(evap_path):
    arr = read_pfb(get_absolute_path(str(evap_path))).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D evap array, got shape {arr.shape} for {evap_path}")
    return arr[EVAP_CHANNELS, ...]


def _read_combined_frame(var_path, evap_path=None, static_arr=None):
    arr = _read_var_frame(var_path)
    if evap_path is not None:
        evap = _read_evap_frame(evap_path)
        arr = np.concatenate([arr, evap], axis=0)
    if static_arr is not None:
        if static_arr.shape[1:] != arr.shape[1:]:
            raise ValueError(
                f"Static shape {static_arr.shape} does not match frame shape {arr.shape} for {var_path}"
            )
        arr = np.concatenate([arr, static_arr], axis=0)
    return arr


def _build_rollout_input_frame(
    pred_frame,
    hour_idx,
    c_in,
    out_channels,
    use_evap=False,
    evap_files=None,
    static_arr=None,
    aux_by_index=None,
):
    frame_parts = [pred_frame[:out_channels].astype(np.float32, copy=False)]
    aux = None
    if aux_by_index is not None:
        aux = aux_by_index.get(hour_idx)
    elif use_evap or static_arr is not None:
        aux_parts = []
        if use_evap:
            if evap_files is None:
                raise ValueError("use_evap=True requires evap_files in rollout.")
            evap = _read_evap_frame(evap_files[hour_idx])
            aux_parts.append(evap.astype(np.float32, copy=False))
        if static_arr is not None:
            aux_parts.append(static_arr.astype(np.float32, copy=False))
        if len(aux_parts) == 1:
            aux = aux_parts[0]
        elif len(aux_parts) > 1:
            aux = np.concatenate(aux_parts, axis=0)
    if aux is not None:
        frame_parts.append(aux)
    if len(frame_parts) == 1:
        frame = frame_parts[0]
    else:
        frame = np.concatenate(frame_parts, axis=0)
    if frame.shape[0] != c_in:
        raise ValueError(
            f"Rollout input channels mismatch at idx={hour_idx}: frame C={frame.shape[0]}, expected C={c_in}."
        )
    return frame


def _build_rollout_aux_map(start_idx, end_idx, use_evap=False, evap_files=None, static_arr=None):
    if not use_evap and static_arr is None:
        return None
    aux_map = {}
    static_fp32 = static_arr.astype(np.float32, copy=False) if static_arr is not None else None
    for idx in range(start_idx, end_idx):
        aux_parts = []
        if use_evap:
            if evap_files is None:
                raise ValueError("use_evap=True requires evap_files for aux preload.")
            evap = _read_evap_frame(evap_files[idx])
            aux_parts.append(evap.astype(np.float32, copy=False))
        if static_fp32 is not None:
            aux_parts.append(static_fp32)
        if len(aux_parts) == 1:
            aux_map[idx] = aux_parts[0]
        else:
            aux_map[idx] = np.concatenate(aux_parts, axis=0)
    return aux_map


def _build_space_coords(height, width, space_h, space_w, space_stride_h=None, space_stride_w=None):
    if space_h is None or space_w is None:
        return [(0, 0)]
    stride_h = space_stride_h or space_h
    stride_w = space_stride_w or space_w
    if space_h > height or space_w > width:
        raise ValueError(f"Space size {(space_h, space_w)} exceeds frame size {(height, width)}")

    coords_h = list(range(0, height - space_h + 1, stride_h))
    if not coords_h or coords_h[-1] != height - space_h:
        coords_h.append(height - space_h)

    coords_w = list(range(0, width - space_w + 1, stride_w))
    if not coords_w or coords_w[-1] != width - space_w:
        coords_w.append(width - space_w)

    return [(top, left) for top in coords_h for left in coords_w]


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _merge_pred_with_aux(pred, prev_seq, in_ch, out_ch):
    if in_ch == out_ch:
        return pred
    aux = prev_seq[:, :, out_ch:in_ch, :, :]
    return torch.cat([pred, aux], dim=2)


def _predict_rollout(model, x, pre_seq, aft_seq, in_ch, out_ch):
    if aft_seq == pre_seq:
        return model(x)
    if aft_seq < pre_seq:
        return model(x)[:, :aft_seq]

    pred_blocks = []
    cur_seq = x.clone()
    d = aft_seq // pre_seq
    m = aft_seq % pre_seq

    for _ in range(d):
        pred_block = model(cur_seq)
        pred_blocks.append(pred_block)
        cur_seq = _merge_pred_with_aux(pred_block, cur_seq, in_ch, out_ch)

    if m:
        pred_block = model(cur_seq)
        pred_blocks.append(pred_block[:, :m])

    return torch.cat(pred_blocks, dim=1)


def _autocast_ctx():
    if DEVICE != "cuda" or not USE_AMP:
        return nullcontext()
    if AMP_DTYPE.lower() == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _parse_index(value, default=None):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    value = str(value).strip()
    if value == "":
        return default
    return int(value)


def _build_file_items(files):
    items = []
    for f in files:
        fid = _extract_id(f)
        if fid is None:
            continue
        items.append((fid, f))
    if not items:
        raise ValueError("No files with valid numeric ids found")
    items.sort(key=lambda x: x[0])
    return items


def _build_aligned_var_evap_items(var_root, evap_root=None):
    var_items = _build_file_items(_list_pfb_files(var_root, recursive=True))
    if evap_root is None:
        return [(h, vp, None) for h, vp in var_items]

    evap_items = _build_file_items(_list_pfb_files(evap_root, recursive=True))
    evap_map = {h: p for h, p in evap_items}
    aligned = []
    missing = []
    for h, vp in var_items:
        ep = evap_map.get(h)
        if ep is None:
            missing.append(h)
            continue
        aligned.append((h, vp, ep))
    if missing:
        preview = ", ".join(str(x) for x in missing[:5])
        raise ValueError(f"Missing evap files for {len(missing)} hour ids, e.g. {preview}")
    return aligned


def _find_index_by_hour(items, hour):
    for idx, (h, _) in enumerate(items):
        if h == hour:
            return idx
    raise ValueError(f"Hour {hour} not found in input files")


def _compute_patch_padding(height, width, patch_size):
    if patch_size is None or patch_size <= 0:
        return 0, 0
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    return pad_h, pad_w


def _pad_input_for_model(x, patch_size, pad_to_patch):
    # x: (B, T, C, H, W)
    if not pad_to_patch:
        return x, 0, 0
    h = x.shape[-2]
    w = x.shape[-1]
    pad_h, pad_w = _compute_patch_padding(h, w, patch_size)
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    # F.pad with replicate mode expects 4D for 4-value padding.
    # Flatten (B, T) so we pad spatial dims on 4D tensors.
    b, t, c, _, _ = x.shape
    x = x.reshape(b * t, c, h, w)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    x = x.reshape(b, t, c, h + pad_h, w + pad_w)
    return x, pad_h, pad_w


def _crop_pred_back(pred, pad_h, pad_w):
    if pad_h == 0 and pad_w == 0:
        return pred
    h_end = pred.shape[-2] - pad_h if pad_h > 0 else pred.shape[-2]
    w_end = pred.shape[-1] - pad_w if pad_w > 0 else pred.shape[-1]
    return pred[..., :h_end, :w_end]


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _load_model_params(checkpoint_path):
    model_param_path = Path(checkpoint_path).resolve().parent / "model_param.json"
    if not model_param_path.exists():
        return {}, None
    with model_param_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError(f"model_param.json must be a dict, got {type(params)}")
    return params, model_param_path


def main():
    start_time = time.time()

    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    run_params, model_param_path = ({}, None)
    if USE_MODEL_PARAM_JSON:
        run_params, model_param_path = _load_model_params(CHECKPOINT_PATH)
        if model_param_path is not None:
            print(f"[config] loaded model params: {model_param_path}")

    cfg = dict(_model_cfg)
    cfg_from_run = run_params.get("model_config")
    if isinstance(cfg_from_run, dict) and cfg_from_run:
        cfg = dict(cfg_from_run)

    var_name = run_params.get("var_name", VAR_NAME)
    use_evap = _to_bool(run_params.get("use_evap", USE_EVAP))
    use_static = _to_bool(run_params.get("use_static_input", USE_STATIC))
    static_data = run_params.get("static_data", STATIC_DATA)
    print(
        f"[config] var_name={var_name}, use_evap={use_evap}, "
        f"use_static={use_static}, static_data={static_data}"
    )

    var_root, evap_root, static_root = _resolve_parflow_roots(
        DATA_ROOT,
        var_name=var_name,
        use_evap=use_evap,
        use_static=use_static,
    )

    aligned_items = _build_aligned_var_evap_items(var_root, evap_root if use_evap else None)
    files = [vp for _, vp, _ in aligned_items]
    evap_files = [ep for _, _, ep in aligned_items]
    items = [(h, vp) for h, vp, _ in aligned_items]
    hours = [h for h, _, _ in aligned_items]

    raw_end = _parse_index(END_HOUR, None)
    end_idx = len(files) - 1 if raw_end is None else _find_index_by_hour(items, int(raw_end))
    raw_start = _parse_index(START_HOUR, None)
    start_idx = 0 if raw_start is None else _find_index_by_hour(items, int(raw_start))

    if end_idx < start_idx:
        raise ValueError(f"END_HOUR ({END_HOUR}) is before START_HOUR ({START_HOUR})")

    for i in range(start_idx, end_idx):
        if hours[i + 1] != hours[i] + 1:
            raise ValueError(f"Missing hours between {hours[i]} and {hours[i + 1]}")

    static_arr = (
        _read_static_stack(static_root, static_data=static_data)
        if static_root is not None
        else None
    )

    sample = _read_combined_frame(files[start_idx], evap_path=evap_files[start_idx], static_arr=static_arr)
    c_in, h, w = sample.shape

    pre_seq = int(cfg["pre_seq"])
    aft_seq = int(cfg["after_seq"])
    out_channels = int(cfg["out_channels"])
    patch_size = int(cfg.get("patch_size", 1))
    pad_to_patch = bool(cfg.get("pad_to_patch", False))

    total_aft = ROLL_AFT if USE_ROLLOUT else aft_seq
    if end_idx - start_idx + 1 < pre_seq + total_aft:
        raise ValueError("Not enough frames for the requested range and seq lengths")

    stats = np.load(STATS_PATH)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    if mean.shape[0] != c_in or std.shape[0] != c_in:
        raise ValueError(f"Stats mismatch: stats C={mean.shape[0]} vs data C={c_in}")

    mean_t = torch.from_numpy(mean).view(1, c_in, 1, 1).float().to(DEVICE)
    std_t = torch.from_numpy(std).view(1, c_in, 1, 1).float().to(DEVICE)
    mean_y = mean[:out_channels]
    std_y = std[:out_channels]

    model = PredFormer_Model(cfg).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    incompatible = model.load_state_dict(_strip_module_prefix(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing_preview = ", ".join(incompatible.missing_keys[:5]) if incompatible.missing_keys else "-"
        unexpected_preview = ", ".join(incompatible.unexpected_keys[:5]) if incompatible.unexpected_keys else "-"
        raise RuntimeError(
            "Checkpoint/model mismatch detected. "
            f"missing_keys={len(incompatible.missing_keys)} (e.g., {missing_preview}); "
            f"unexpected_keys={len(incompatible.unexpected_keys)} (e.g., {unexpected_preview}). "
            "Please use the matching model_param.json/model_config for this checkpoint."
        )
    model.eval()

    space_h = cfg.get("space_h", h)
    space_w = cfg.get("space_w", w)
    space_stride_h = cfg.get("space_stride_h", None)
    space_stride_w = cfg.get("space_stride_w", None)
    coords = _build_space_coords(h, w, space_h, space_w, space_stride_h, space_stride_w)

    stride = ROLL_AFT if USE_ROLLOUT else aft_seq

    run_tag = datetime.now().strftime("%Y%m%d")
    if RUN_PARAM:
        run_tag = f"{run_tag}_{RUN_PARAM}"
    out_dir = Path(OUTPUT_DIR) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    total_windows = max(0, (end_idx - start_idx - pre_seq - total_aft + 2 + stride - 1) // stride)
    window_idx = 0

    for t0 in range(start_idx, end_idx - pre_seq - total_aft + 2, stride):
        window_idx += 1
        pred_start = hours[t0 + pre_seq]
        pred_end = hours[t0 + pre_seq + total_aft - 1]
        print(f"[window {window_idx}/{total_windows}] pred_range={pred_start}..{pred_end}")

        history = []
        for i in range(pre_seq):
            history.append(
                _read_combined_frame(
                    files[t0 + i],
                    evap_path=evap_files[t0 + i],
                    static_arr=static_arr,
                )
            )

        if USE_ROLLOUT:
            rollout_aux_map = None
            if PRELOAD_ROLLOUT_AUX and (use_evap or static_arr is not None):
                rollout_aux_map = _build_rollout_aux_map(
                    t0 + pre_seq,
                    t0 + pre_seq + total_aft,
                    use_evap=use_evap,
                    evap_files=evap_files,
                    static_arr=static_arr,
                )
            predicted = 0
            while predicted < ROLL_AFT:
                block = min(aft_seq, ROLL_AFT - predicted)

                x = np.stack(history[-pre_seq:], axis=0)
                x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
                x = (x - mean_t) / (std_t + EPS)

                pred_full = np.zeros((aft_seq, out_channels, h, w), dtype=np.float32)
                counts = np.zeros_like(pred_full, dtype=np.int32)

                for top, left in coords:
                    x_patch = x[..., top: top + space_h, left: left + space_w]
                    x_patch, pad_h, pad_w = _pad_input_for_model(x_patch, patch_size, pad_to_patch)
                    with torch.no_grad():
                        with _autocast_ctx():
                            pred = _predict_rollout(model, x_patch, pre_seq, aft_seq, c_in, out_channels)
                    pred = _crop_pred_back(pred, pad_h, pad_w)
                    pred = pred.squeeze(0).float().cpu().numpy()
                    pred_full[:, :, top: top + space_h, left: left + space_w] += pred
                    counts[:, :, top: top + space_h, left: left + space_w] += 1

                mask = counts > 0
                pred_full[mask] = pred_full[mask] / counts[mask]
                pred_full = pred_full * std_y.reshape(1, -1, 1, 1) + mean_y.reshape(1, -1, 1, 1)

                for k in range(block):
                    idx = t0 + pre_seq + predicted + k
                    src = Path(files[idx])
                    rel = src.relative_to(var_root)
                    out_path = out_dir / rel.parent / f"pred_{rel.name}"
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    arr = pred_full[k][:out_channels].astype(np.float64)
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    write_pfb(str(out_path), arr, dist=False)

                    next_input = _build_rollout_input_frame(
                        pred_full[k],
                        hour_idx=idx,
                        c_in=c_in,
                        out_channels=out_channels,
                        use_evap=use_evap,
                        evap_files=evap_files,
                        static_arr=static_arr,
                        aux_by_index=rollout_aux_map,
                    )
                    history.append(next_input)
                    if len(history) > pre_seq:
                        history.pop(0)

                predicted += block
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
        else:
            x = np.stack(history, axis=0)
            x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
            x = (x - mean_t) / (std_t + EPS)

            pred_full = np.zeros((aft_seq, out_channels, h, w), dtype=np.float32)
            counts = np.zeros_like(pred_full, dtype=np.int32)

            for top, left in coords:
                x_patch = x[..., top: top + space_h, left: left + space_w]
                x_patch, pad_h, pad_w = _pad_input_for_model(x_patch, patch_size, pad_to_patch)
                with torch.no_grad():
                    with _autocast_ctx():
                        pred = _predict_rollout(model, x_patch, pre_seq, aft_seq, c_in, out_channels)
                pred = _crop_pred_back(pred, pad_h, pad_w)
                pred = pred.squeeze(0).float().cpu().numpy()
                pred_full[:, :, top: top + space_h, left: left + space_w] += pred
                counts[:, :, top: top + space_h, left: left + space_w] += 1

            mask = counts > 0
            pred_full[mask] = pred_full[mask] / counts[mask]
            pred_full = pred_full * std_y.reshape(1, -1, 1, 1) + mean_y.reshape(1, -1, 1, 1)

            for k in range(aft_seq):
                idx = t0 + pre_seq + k
                src = Path(files[idx])
                rel = src.relative_to(var_root)
                out_path = out_dir / rel.parent / f"pred_{rel.name}"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                arr = pred_full[k][:out_channels].astype(np.float64)
                if arr.shape[0] == 1:
                    arr = arr[0]
                write_pfb(str(out_path), arr, dist=False)

    elapsed = time.time() - start_time
    print(f"Inference done. Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
