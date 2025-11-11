#!/usr/bin/env bash

set -euo pipefail

# Resolve repository root relative to this script's location.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <work_dir> [checkpoint_path] [output_dir] [device]" >&2
  echo "  work_dir:      Experiment directory created during training (required)." >&2
  echo "  checkpoint:    Optional checkpoint path. Defaults to <work_dir>/checkpoints/checkpoint.pth." >&2
  echo "  output_dir:    Optional directory to store .pfb predictions. Defaults to <work_dir>/pred_result." >&2
  echo "  device:        cuda | cpu (default: cuda)." >&2
  exit 1
fi

WORK_DIR=$1
CHECKPOINT=${2:-}
OUTPUT_DIR=${3:-}
DEVICE=${4:-cuda}

cd "$REPO"
export PYTHONPATH="$REPO:$PYTHONPATH"

# Let callers override CUDA_VISIBLE_DEVICES; otherwise prefer GPU 0.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ARGS=(
  --work-dir "$WORK_DIR"
  --device "$DEVICE"
)

if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

python tools/parflow_inference_demo.py "${ARGS[@]}"
