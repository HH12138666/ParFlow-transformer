import sys
from pathlib import Path
import numpy as np
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb, write_pfb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from openstl.datasets.dataloader_parflow import (  
    _list_pfb_files,
    _interpolate_outliers,
)

# ---- User configuration ----
PRESS_ROOT = "/home/huanghui/data/ParFlow-transformer/data/parflow/press"
OUTPUT_DIR = "/home/huanghui/data/ParFlow-transformer/inference_data/true_press"
START_INDEX = "07009"   # e.g. "00000"
END_INDEX = "07009"     # e.g. "01000"
STRIDE = 1
MAX_FILES = None   # e.g. 100
OVERWRITE = True
ABS_OUTLIER_THRESHOLD = -10000.0
OUTLIER_STD_MULT = 5


def _parse_index(value, default=None):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    value = str(value).strip()
    if value == "":
        return default
    return int(value)


def main():
    press_files = _list_pfb_files(PRESS_ROOT)
    if not press_files:
        raise FileNotFoundError(f"No .pfb files found under: {PRESS_ROOT}")

    start_idx = _parse_index(START_INDEX, 0)
    end_idx = _parse_index(END_INDEX, len(press_files) - 1)
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError(f"Invalid range: {start_idx}..{end_idx}")

    sel = press_files[start_idx:end_idx + 1: max(1, int(STRIDE))]
    if MAX_FILES is not None:
        sel = sel[:int(MAX_FILES)]
    if not sel:
        raise ValueError("No files selected with the given range/stride.")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(sel, 1):
        arr = read_pfb(get_absolute_path(f)).astype(np.float32)
        arr = _interpolate_outliers(
            arr,
            abs_threshold=ABS_OUTLIER_THRESHOLD,
            std_mult=OUTLIER_STD_MULT,
        )
        out_path = out_dir / Path(f).name
        if out_path.exists() and not OVERWRITE:
            continue
        write_pfb(str(out_path), arr.astype(np.float64), dist=False)
        if i % 10 == 0 or i == len(sel):
            print(f"[{i}/{len(sel)}] saved {out_path}")

    print(f"Done. Wrote {len(sel)} files to {out_dir}")


if __name__ == "__main__":
    main()

#python /home/huanghui/data/ParFlow-transformer/tools/handing_outliers.py