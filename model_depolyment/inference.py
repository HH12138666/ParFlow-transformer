import os
import sys
from pathlib import Path
import glob
import re
import numpy as np
import time
from datetime import datetime
import torch
repo = "/home/huanghui/data/ParFlow-transformer"
sys.path.insert(0, repo)

from configs.parflow.PredFormer_infer import model_config as _model_cfg
from openstl.models import PredFormer_Model
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb, write_pfb

# ---- User configuration ----
# Fill these before running.
PARFLOW_PATH = "/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press"
# 选择的checkpoint名称
CHECKPOINT_NAME = "2026-03-04-21-24_FACTS"
CHECKPOINT_PATH = os.path.join(PARFLOW_PATH, CHECKPOINT_NAME, "checkpoint.pth")
DATA_ROOT = "/home/huanghui/data/ParFlow-transformer/data/parflow"
OUTPUT_DIR = "/home/huanghui/data/ParFlow-transformer/inference_data/press"
# +perm_x_n_alpha_porosity
RUN_PARAM = "press+evaptrans+perm_x_n_alpha_porosity_time5_train0.75_cnn5_k1_in12_out12_rollout700"  # 
USE_STATIC = True
#perm_x,alpha_z6-9,n_z6-9,porosity_z6-9
STATIC_DATA = "perm_x,alpha_z6-9,n_z6-9,porosity_z6-9"  
USE_PRED_EVAP = False  # use predicted evap instead of reading real evap files

# Keep in sync with training preprocessing.
EPS = 1e-6
EVAP_CHANNELS = [6, 7, 8, 9]
PRESS_CHANNELS = 10

# Prediction range (hour id in file names, 1-based if files start at 00001).
# Use strings like "00001" if you want to keep the visual format.
START_HOUR = "20207000"
END_HOUR = "20207711"  # empty means use last available hour

DEVICE = "cuda"

# Use config defaults from PredFormer_infer.py.
PRE_SEQ = int(_model_cfg["pre_seq"])
AFT_SEQ = int(_model_cfg["after_seq"])

USE_ROLLOUT = True  # set True to roll forward with model outputs
ROLL_AFT = 700  # total hours to roll out when USE_ROLLOUT=True
STRIDE = ROLL_AFT if USE_ROLLOUT else AFT_SEQ  # no overlap between prediction windows

# Spatial tiling (enable for large frames).
SPACE_H = _model_cfg.get("space_h", None)
SPACE_W = _model_cfg.get("space_w", None)
SPACE_STRIDE_H = _model_cfg.get("space_stride_h", None)
SPACE_STRIDE_W = _model_cfg.get("space_stride_w", None)

# Stats for normalization.
STATS_PATH = "/home/huanghui/data/ParFlow-transformer/stats/stats_press_evaptrans_perm_x_alpha_n_porosity.npz"
OUT_CHANNELS = int(_model_cfg["out_channels"])


def _natural_key(p):
    b = os.path.basename(p)
    s = re.split(r'(\d+)', b)
    return [int(t) if t.isdigit() else t for t in s]


def _extract_id(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)(?!.*\d)", name)
    if not m:
        return None
    return m.group(1)


def _list_pfb_files(root):
    files = sorted(glob.glob(os.path.join(root, '*.pfb')), key=_natural_key)
    if not files:
        raise FileNotFoundError(f'No .pfb files found under: {root}')
    return files


def _resolve_parflow_roots(data_root, use_static=True):
    base = Path(data_root)
    press_root = str(base / "press")
    evap_root = str(base / "evaptrans")
    static_root = str(base / "static") if use_static else None
    return press_root, evap_root, static_root


def _parse_static_data(static_data):
    if static_data is None:
        return None
    if isinstance(static_data, (list, tuple)):
        patterns = [str(x).strip() for x in static_data if str(x).strip()]
    else:
        patterns = [p.strip() for p in str(static_data).split(',') if p.strip()]
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
        raise FileNotFoundError(f'No static .pfb files matched patterns: {patterns}')
    return matched


def _read_static_stack(static_root, static_data=None):
    if static_root is None:
        return None
    files = _list_pfb_files(static_root)
    files = _filter_static_files(files, static_data)
    arrays = []
    for f in files:
        arr = read_pfb(get_absolute_path(str(f))).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            raise ValueError(f'Expected 2D/3D static array, got shape {arr.shape} for {f}')
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def _read_evap_frame(evap_path):
    if evap_path is None:
        raise ValueError("evap_path is required when reading evaptrans data")
    if not Path(evap_path).exists():
        raise FileNotFoundError(f"Evaptrans file not found: {evap_path}")
    arr = read_pfb(get_absolute_path(str(evap_path))).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D evaptrans array, got shape {arr.shape} for {evap_path}')
    return arr[EVAP_CHANNELS, ...]



def _build_space_coords(height, width, space_h, space_w, space_stride_h=None, space_stride_w=None):
    if space_h is None or space_w is None:
        return [(0, 0)]
    stride_h = space_stride_h or space_h
    stride_w = space_stride_w or space_w
    if space_h > height or space_w > width:
        raise ValueError(
            f"Space size {(space_h, space_w)} exceeds frame size {(height, width)}."
        )
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
    press = read_pfb(get_absolute_path(press_path)).astype(np.float32)
    if press.shape[0] > PRESS_CHANNELS:
        press = press[:PRESS_CHANNELS]
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
    if evap_path is not None and press_pred.shape[0] > PRESS_CHANNELS:
        press_pred = press_pred[:PRESS_CHANNELS]
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


def _build_evap_map(evap_files):
    evap_map = {}
    for f in evap_files:
        fid = _extract_id(f)
        if fid is None:
            continue
        evap_map[fid] = f
    return evap_map


def _get_evap_path_for_press(press_path, evap_map):
    fid = _extract_id(press_path)
    if fid is None:
        raise ValueError(f"Cannot extract id from press file: {press_path}")
    evap_path = evap_map.get(fid)
    if evap_path is None:
        raise ValueError(f"Evap file not found for id {fid} (press {press_path})")
    return evap_path


def _build_press_list(press_files):
    items = []
    for f in press_files:
        fid = _extract_id(f)
        if fid is None:
            continue
        items.append((int(fid), fid, f))
    if not items:
        raise ValueError("No press files with valid ids found.")
    items.sort(key=lambda x: x[0])
    return items


def _find_index_by_hour(items, hour):
    for idx, (h, _, _) in enumerate(items):
        if h == hour:
            return idx
    raise ValueError(f"Hour {hour} not found in press files.")


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
    press_items = _build_press_list(press_files)
    press_files = [p for _, _, p in press_items]
    press_hours = [h for h, _, _ in press_items]
    evap_files = _list_pfb_files(evap_root)
    evap_map = _build_evap_map(evap_files)
    if not evap_map:
        raise ValueError("No evap files with valid ids found.")

    static_arr = (
        _read_static_stack(static_root, static_data=STATIC_DATA)
        if static_root is not None
        else None
    )
    print("static_arr channels:", None if static_arr is None else static_arr.shape[0])

    raw_end = _parse_index(END_HOUR, None)
    end_idx = len(press_files) - 1 if raw_end is None else _find_index_by_hour(press_items, int(raw_end))
    raw_start = _parse_index(START_HOUR, None)
    start_idx = 0 if raw_start is None else _find_index_by_hour(press_items, int(raw_start))
    if end_idx < start_idx:
        raise ValueError(f"END_HOUR ({END_HOUR}) is before START_HOUR ({START_HOUR}).")
    for i in range(start_idx, end_idx):
        if press_hours[i + 1] != press_hours[i] + 1:
            raise ValueError(
                f"Missing press hours between {press_hours[i]} and {press_hours[i + 1]}."
            )
    total_aft = ROLL_AFT if USE_ROLLOUT else AFT_SEQ
    if end_idx - start_idx + 1 < PRE_SEQ + total_aft:
        raise ValueError("Not enough frames for the requested range and seq lengths")

    press_path = press_files[start_idx]
    evap_path = _get_evap_path_for_press(press_path, evap_map)
    sample = _read_combined_frame_infer(
        press_path,
        evap_path=evap_path,
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
    use_pred_evap = USE_PRED_EVAP and OUT_CHANNELS > PRESS_CHANNELS
    if USE_PRED_EVAP and OUT_CHANNELS <= PRESS_CHANNELS:
        raise ValueError(
            "USE_PRED_EVAP=True requires OUT_CHANNELS to include evap (e.g., 14)."
        )

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
        start_hour = press_hours[t0]
        pred_start_hour = press_hours[t0 + PRE_SEQ]
        pred_end_hour = press_hours[t0 + PRE_SEQ + total_aft - 1]
        if USE_ROLLOUT:
            print(
                f"[rollout {window_idx}/{total_windows}] t0={t0} (hour={start_hour}), "
                f"pred_range={pred_start_hour}..{pred_end_hour}"
            )
        else:
            print(
                f"[window {window_idx}/{total_windows}] t0={t0} (hour={start_hour}), "
                f"pred_range={pred_start_hour}..{pred_end_hour}"
            )
        history = []
        for i in range(PRE_SEQ):
            press_path = press_files[t0 + i]
            evap_path = _get_evap_path_for_press(press_path, evap_map)
            arr = _read_combined_frame_infer(
                press_path,
                evap_path=evap_path,
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
                    press_path = press_files[idx]
                    src_name = Path(press_path).name
                    out_name = f"pred_{src_name}"
                    out_path = out_dir / out_name
                    write_pfb(
                        str(out_path),
                        pred_full[k][:PRESS_CHANNELS].astype(np.float64),
                        dist=False,
                    )

                    combined = _build_combined_from_pred(
                        pred_full[k],
                        evap_path=None if use_pred_evap else _get_evap_path_for_press(press_path, evap_map),
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
                press_path = press_files[idx]
                src_name = Path(press_path).name
                out_name = f"pred_{src_name}"
                out_path = out_dir / out_name
                write_pfb(
                    str(out_path),
                    pred_full[k][:PRESS_CHANNELS].astype(np.float64),
                    dist=False,
                )

    elapsed = time.time() - start_time
    print(f"✅ Inference done. Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
