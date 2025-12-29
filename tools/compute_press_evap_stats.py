#!/usr/bin/env python3
"""
Compute and save mean/std separately for pressure and evaptrans channels (fixed 6-9).

Example:
python tools/compute_press_evap_stats.py \
    --press_dir /home/huanghui/share/parflow-group/sunaoqi-share-old-server/sunaoqi-share/standard_2018/output_press \
    --evap_root /home/huanghui/share/parflow-group/sunaoqi-share-old-server/sunaoqi-share/standard_2018/output_evapotrans \
    --press_out press_stats.npz \
    --evap_out evap_stats.npz
"""
import argparse
import numpy as np

from openstl.datasets.dataloader_parflow import _list_pfb_files
from openstl.datasets.compute_mean_std import compute_press_evap_mean_std


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean/std for press and evaptrans separately."
    )
    parser.add_argument("--press_dir", required=True, help="Directory of press .pfb files")
    parser.add_argument("--evap_root", required=True, help="Directory of evaptrans .pfb files")
    parser.add_argument("--spatial_stride", type=int, default=1, help="Spatial stride for subsampling")
    parser.add_argument("--time_stride", type=int, default=1, help="Temporal stride for subsampling files")
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Max files to use (0 means all files)",
    )
    parser.add_argument("--press_out", default="press_stats.npz", help="Output npz for press mean/std")
    parser.add_argument("--evap_out", default="evap_stats.npz", help="Output npz for evap mean/std")
    args = parser.parse_args()

    press_files = _list_pfb_files(args.press_dir)
    max_files = args.max_files if args.max_files > 0 else None
    pm, ps, em, es = compute_press_evap_mean_std(
        press_files,
        evap_root=args.evap_root,
        spatial_stride=args.spatial_stride,
        time_stride=args.time_stride,
        max_files=max_files,
    )

    np.savez(args.press_out, mean=pm, std=ps)
    np.savez(args.evap_out, mean=em, std=es)

    print(f"Used {len(press_files) if max_files is None else min(len(press_files), max_files)} files")
    print(f"Press  stats saved to {args.press_out}; shape={pm.shape}")
    print(f"Evap   stats saved to {args.evap_out}; shape={em.shape}")


if __name__ == "__main__":
    main()
