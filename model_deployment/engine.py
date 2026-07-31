import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from parflow.tools.io import write_pfb

from openstl.datasets.parflow.dataset import build_space_coords
from openstl.datasets.parflow.readers import read_combined_frame, read_evap_frame

from .common import DEVICE, autocast_ctx, require_model_key, sync_cuda
class SimpleSources:
    def __init__(self, pred_full):
        self.height = pred_full.shape[-2]
        self.width = pred_full.shape[-1]


def space_shape(cfg, sources):
    return int(cfg.get("space_h") or sources.height), int(cfg.get("space_w") or sources.width)

def predict_full_block(model, x, coords, spec, sources, config, timing, block):
    cfg = spec.model_config
    out_channels = int(require_model_key(cfg, "out_channels"))
    pred_full = np.zeros((block, out_channels, sources.height, sources.width), dtype=np.float32)
    counts = np.zeros_like(pred_full, dtype=np.int32)
    for start in range(0, len(coords), config.patch_batch_size):
        chunk = coords[start:start + config.patch_batch_size]
        patch_h, patch_w = space_shape(cfg, sources)
        x_batch = torch.cat([x[..., top:top + patch_h, left:left + patch_w] for top, left in chunk], dim=0)
        sync_cuda()
        forward_start = time.perf_counter()
        with torch.inference_mode(), autocast_ctx(config):
            pred = model(x_batch)[:, :block]
        sync_cuda()
        timing.forward_time += time.perf_counter() - forward_start
        stitch_predictions(pred.float().cpu().numpy(), pred_full, counts, chunk)
    mask = counts > 0
    pred_full[mask] = pred_full[mask] / counts[mask]
    return pred_full


def stitch_predictions(pred, pred_full, counts, coords):
    for idx, (top, left) in enumerate(coords):
        patch_h, patch_w = pred[idx].shape[-2:]
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
    block_capacity = min(pre_seq, aft_seq)
    target_hours = config.rollout_hours if config.use_rollout else aft_seq
    predicted = 0
    while predicted < target_hours:
        block = min(block_capacity, target_hours - predicted)
        x = normalize_history(history[-pre_seq:], stats)
        pred_full = predict_full_block(
            model, x, coords, spec, sources, config, timing, block
        )
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


