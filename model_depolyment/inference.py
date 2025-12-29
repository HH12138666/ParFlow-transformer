import os
import sys
from pathlib import Path
import numpy as np
import time
from datetime import datetime
import torch
repo = "/home/huanghui/data/ParFlow-transformer"
sys.path.insert(0, repo)

from configs.parflow.PredFormer import model_config as _model_cfg
from openstl.models import PredFormer_Model
from openstl.datasets.dataloader_parflow import (
    _resolve_parflow_roots,
    _list_pfb_files,
    _read_evap_frame,
    _read_static_stack,
    _build_space_coords,
    _interpolate_outliers,
    EPS,
    )
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb, write_pfb

# ---- User configuration ----
# Fill these before running.
PARFLOW_PATH = "/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press"
CHECKPOINT_NAME = "2025-12-22-14-40_PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep"
CHECKPOINT_PATH = os.path.join(PARFLOW_PATH, CHECKPOINT_NAME, "checkpoint.pth")
DATA_ROOT = "/home/huanghui/data/ParFlow-transformer/data/parflow"
OUTPUT_DIR = "/home/huanghui/data/ParFlow-transformer/inference_data"
RUN_PARAM = "press+evapotrans_in10_out10_rollout700"  # 
USE_STATIC = False
ABS_OUTLIER_THRESHOLD = -10000.0
OUTLIER_STD_MULT = 5

# Prediction range (index in sorted press files).
# Use strings like "00000" if you want to keep the visual format.
START_INDEX = "07000"
END_INDEX = "08760"  # empty means use last available index

DEVICE = "cuda"

# Use the same pre/aft as training (can override).
PRE_SEQ = 10
AFT_SEQ = 10
USE_ROLLOUT = True  # set True to roll forward with model outputs
ROLL_AFT = 700  # total hours to roll out when USE_ROLLOUT=True
STRIDE = ROLL_AFT if USE_ROLLOUT else AFT_SEQ  # no overlap between prediction windows

# Spatial tiling (enable for large frames).
SPACE_H = 60
SPACE_W = 84
SPACE_STRIDE_H = 30
SPACE_STRIDE_W = 42

# Stats for normalization.
STATS_PATH = "/home/huanghui/data/ParFlow-transformer/stats.npz"
OUT_CHANNELS = 10  # press channels


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


def _parse_index(value, default=None):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    value = str(value).strip()
    if value == "":
        return default
    return int(value)


def _read_combined_frame_infer(press_path, evap_path=None, static_arr=None):
    # Keep press outlier handling consistent with training (_read_press_frame).
    press = read_pfb(get_absolute_path(press_path)).astype(np.float32)
    press = _interpolate_outliers(
        press,
        abs_threshold=ABS_OUTLIER_THRESHOLD,
        std_mult=OUTLIER_STD_MULT,
    )
    if evap_path is None:
        combined = press
    else:
        evap = _read_evap_frame(evap_path)
        combined = np.concatenate([press, evap], axis=0)
    if static_arr is not None:
        if static_arr.shape[1:] != combined.shape[1:]:
            raise ValueError(
                f"Static shape {static_arr.shape} does not match frame shape {combined.shape} for {press_path}"
            )
        combined = np.concatenate([combined, static_arr], axis=0)
    return combined


def _build_combined_from_pred(press_pred, evap_path=None, static_arr=None):
    combined = press_pred.astype(np.float32, copy=False)
    if evap_path is not None:
        evap = _read_evap_frame(evap_path)
        combined = np.concatenate([combined, evap], axis=0)
    if static_arr is not None:
        if static_arr.shape[1:] != combined.shape[1:]:
            raise ValueError(
                f"Static shape {static_arr.shape} does not match frame shape {combined.shape} for {evap_path}"
            )
        combined = np.concatenate([combined, static_arr], axis=0)
    return combined


def main():
    start_time = time.time()
    if not CHECKPOINT_PATH:
        raise ValueError("CHECKPOINT_PATH is empty")
    if not DATA_ROOT:
        raise ValueError("DATA_ROOT is empty")
    if not OUTPUT_DIR:
        raise ValueError("OUTPUT_DIR is empty")

    press_root, evap_root, static_root = _resolve_parflow_roots(DATA_ROOT)
    if not USE_STATIC:
        static_root = None
    press_files = _list_pfb_files(press_root)
    evap_files = _list_pfb_files(evap_root)
    if len(press_files) != len(evap_files):
        raise ValueError(f"press/evap file counts do not match: {len(press_files)} vs {len(evap_files)}")

    static_arr = _read_static_stack(static_root) if static_root is not None else None

    end_idx = _parse_index(END_INDEX, len(press_files) - 1)
    start_idx = _parse_index(START_INDEX, 0)
    total_aft = ROLL_AFT if USE_ROLLOUT else AFT_SEQ
    if end_idx - start_idx + 1 < PRE_SEQ + total_aft:
        raise ValueError("Not enough frames for the requested range and seq lengths")

    sample = _read_combined_frame_infer(
        press_files[start_idx],
        evap_path=evap_files[start_idx],
        static_arr=static_arr,
    )
    C, H, W = sample.shape

    stats = np.load(STATS_PATH)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    if mean.shape[0] != C or std.shape[0] != C:
        raise ValueError(f"Stats mismatch: stats C={mean.shape[0]} vs data C={C}")

    mean_t = torch.from_numpy(mean).view(1, C, 1, 1).float()
    std_t = torch.from_numpy(std).view(1, C, 1, 1).float()
    mean_y = mean[:OUT_CHANNELS]
    std_y = std[:OUT_CHANNELS]

    cfg = dict(_model_cfg)
    cfg["pre_seq"] = PRE_SEQ
    cfg["after_seq"] = AFT_SEQ
    cfg["space_h"] = SPACE_H
    cfg["space_w"] = SPACE_W
    cfg["in_channels"] = C
    cfg["out_channels"] = OUT_CHANNELS

    model = PredFormer_Model(cfg).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(_strip_module_prefix(state), strict=False)
    model.eval()

    coords = _build_space_coords(H, W, SPACE_H, SPACE_W, SPACE_STRIDE_H, SPACE_STRIDE_W)
    run_tag = datetime.now().strftime("%Y%m%d")
    if RUN_PARAM:
        run_tag = f"{run_tag}_{RUN_PARAM}"
    out_dir = Path(OUTPUT_DIR) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    total_windows = max(0, (end_idx - start_idx - PRE_SEQ - total_aft + 2 + STRIDE - 1) // STRIDE)
    window_idx = 0
    for t0 in range(start_idx, end_idx - PRE_SEQ - total_aft + 2, STRIDE):
        window_idx += 1
        if USE_ROLLOUT:
            print(f"[rollout {window_idx}/{total_windows}] t0={t0}, range={t0 + PRE_SEQ}..{t0 + PRE_SEQ + total_aft - 1}")
        else:
            print(f"[window {window_idx}/{total_windows}] t0={t0}, range={t0 + PRE_SEQ}..{t0 + PRE_SEQ + total_aft - 1}")
        history = []
        for i in range(PRE_SEQ):
            arr = _read_combined_frame_infer(
                press_files[t0 + i],
                evap_path=evap_files[t0 + i],
                static_arr=static_arr,
            )
            history.append(arr)

        if USE_ROLLOUT:
            predicted = 0
            block_idx = 0
            while predicted < ROLL_AFT:
                block_idx += 1
                block = min(AFT_SEQ, ROLL_AFT - predicted)
                print(f"  block {block_idx}: {block} steps (pred={predicted}->{predicted + block})")
                x = np.stack(history[-PRE_SEQ:], axis=0)  # (T, C, H, W)
                x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
                x = (x - mean_t.to(DEVICE)) / (std_t.to(DEVICE) + EPS)

                pred_full = np.zeros((AFT_SEQ, OUT_CHANNELS, H, W), dtype=np.float32)
                counts = np.zeros_like(pred_full, dtype=np.int32)

                for top, left in coords:
                    x_patch = x[..., top: top + SPACE_H, left: left + SPACE_W]
                    with torch.no_grad():
                        pred = _predict_rollout(model, x_patch, PRE_SEQ, AFT_SEQ, C, OUT_CHANNELS)
                    pred = pred.squeeze(0).cpu().numpy()
                    pred_full[:, :, top: top + SPACE_H, left: left + SPACE_W] += pred
                    counts[:, :, top: top + SPACE_H, left: left + SPACE_W] += 1

                mask = counts > 0
                pred_full[mask] = pred_full[mask] / counts[mask]

                pred_full = pred_full * std_y.reshape(1, -1, 1, 1) + mean_y.reshape(1, -1, 1, 1)

                for k in range(block):
                    idx = t0 + PRE_SEQ + predicted + k
                    src_name = Path(press_files[idx]).name
                    out_name = f"pred_{src_name}"
                    out_path = out_dir / out_name
                    write_pfb(str(out_path), pred_full[k].astype(np.float64), dist=False)

                    combined = _build_combined_from_pred(
                        pred_full[k],
                        evap_path=evap_files[idx] if evap_files is not None else None,
                        static_arr=static_arr,
                    )
                    history.append(combined)
                    if len(history) > PRE_SEQ:
                        history.pop(0)

                predicted += block
        else:
            x = np.stack(history, axis=0)  # (T, C, H, W)
            x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
            x = (x - mean_t.to(DEVICE)) / (std_t.to(DEVICE) + EPS)

            pred_full = np.zeros((AFT_SEQ, OUT_CHANNELS, H, W), dtype=np.float32)
            counts = np.zeros_like(pred_full, dtype=np.int32)

            for top, left in coords:
                x_patch = x[..., top: top + SPACE_H, left: left + SPACE_W]
                with torch.no_grad():
                    pred = _predict_rollout(model, x_patch, PRE_SEQ, AFT_SEQ, C, OUT_CHANNELS)
                pred = pred.squeeze(0).cpu().numpy()
                pred_full[:, :, top: top + SPACE_H, left: left + SPACE_W] += pred
                counts[:, :, top: top + SPACE_H, left: left + SPACE_W] += 1

            mask = counts > 0
            pred_full[mask] = pred_full[mask] / counts[mask]

            pred_full = pred_full * std_y.reshape(1, -1, 1, 1) + mean_y.reshape(1, -1, 1, 1)

            for k in range(AFT_SEQ):
                idx = t0 + PRE_SEQ + k
                src_name = Path(press_files[idx]).name
                out_name = f"pred_{src_name}"
                out_path = out_dir / out_name
                write_pfb(str(out_path), pred_full[k].astype(np.float64), dist=False)

    elapsed = time.time() - start_time
    print(f"✅ Inference done. Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
