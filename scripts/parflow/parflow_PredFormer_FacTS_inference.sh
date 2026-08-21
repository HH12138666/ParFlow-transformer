#!/bin/bash
set -euo pipefail

# Example inference script for a trained ParFlow PredFormer checkpoint.
# Edit the configuration block below before running on a new machine.

# ===================== User configuration =====================
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

# GPU selection. Set CUDA_VISIBLE_DEVICES before running if needed.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Trained experiment directory and checkpoint file.
RUN_DIR=${RUN_DIR:-"$PROJECT_ROOT/work_dirs/ParFlow_press/2026-07-11-18-00_FACTS"}
CHECKPOINT_FILE=${CHECKPOINT_FILE:-latest_state_dict.pth}

# Input data and output directory.
DATA_ROOT=${DATA_ROOT:-"$PROJECT_ROOT/data/parflow/normal_data"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/inference_data/example_press"}
RUN_NAME=${RUN_NAME:-example_rollout12h}

# Inference period and rollout settings.
START_HOUR=${START_HOUR:-20190000}
END_HOUR=${END_HOUR:-20190035}
ROLLOUT_HOURS=${ROLLOUT_HOURS:-12}
PATCH_BATCH_SIZE=${PATCH_BATCH_SIZE:-4}
# ==============================================================

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m model_deployment.inference \
    --run-dir "$RUN_DIR" \
    --checkpoint-file "$CHECKPOINT_FILE" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --start-hour "$START_HOUR" \
    --end-hour "$END_HOUR" \
    --rollout-hours "$ROLLOUT_HOURS" \
    --patch-batch-size "$PATCH_BATCH_SIZE" \
    --use-rollout \
    --use-amp
