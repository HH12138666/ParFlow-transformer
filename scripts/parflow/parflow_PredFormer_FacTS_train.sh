#!/bin/bash
set -euo pipefail

# Example training script for the ParFlow PredFormer surrogate.
# Edit the configuration block below before running on a new machine.

# ===================== User configuration =====================
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

# GPU selection. Set CUDA_VISIBLE_DEVICES before running if needed.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Data and normalization statistics.
DATA_ROOT=${DATA_ROOT:-"$PROJECT_ROOT/data/parflow/normal_data"}
STATS_PATH=${STATS_PATH:-"$PROJECT_ROOT/stats/stats1_press_evap_static_2019_2020.npz"}

# Training and holdout years.
TRAIN_YEARS=${TRAIN_YEARS:-'[2019]'}
HOLDOUT_YEARS=${HOLDOUT_YEARS:-'[2019]'}

# Output directory and experiment name.
RES_DIR=${RES_DIR:-"$PROJECT_ROOT/work_dirs"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"ParFlow_press/example_smoke_test"}

# Core training hyperparameters.
BATCH_SIZE=${BATCH_SIZE:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
NUM_WORKERS=${NUM_WORKERS:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
TEST_INTERVAL=${TEST_INTERVAL:-1}

# Optional extra-sample training.
# Set USE_EXTRA_DATA=True and provide EXTRA_MANIFEST_PATH to append selected
# APCP-perturbed ParFlow-CLM samples to the baseline training years.
USE_EXTRA_DATA=${USE_EXTRA_DATA:-False}
EXTRA_MANIFEST_PATH=${EXTRA_MANIFEST_PATH:-"$PROJECT_ROOT/data/parflow/extra_data_index/extra_sample_manifest.csv"}
EXTRA_DATA_ROOT=${EXTRA_DATA_ROOT:-}
# ==============================================================

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXTRA_ARGS=()
if [[ "$USE_EXTRA_DATA" == "True" ]]; then
    EXTRA_ARGS=(
        --use-extra-data
        --extra-manifest-path "$EXTRA_MANIFEST_PATH"
    )
    if [[ -n "$EXTRA_DATA_ROOT" ]]; then
        EXTRA_ARGS+=(--extra-data-root "$EXTRA_DATA_ROOT")
    fi
fi

python tools/train.py \
    --config-file configs/parflow/PredFormer.py \
    --data-root "$DATA_ROOT" \
    --stats-path "$STATS_PATH" \
    --res-dir "$RES_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-epochs 0 \
    --empty-cache \
    --fp16 \
    --no-use-val \
    --save-interval "$SAVE_INTERVAL" \
    --test-interval "$TEST_INTERVAL" \
    --weight-decay 1e-2 \
    --num-workers "$NUM_WORKERS" \
    --var-name press \
    --use-evap \
    --use-static-input \
    --split-mode year \
    --train-years "$TRAIN_YEARS" \
    --holdout-years "$HOLDOUT_YEARS" \
    "${EXTRA_ARGS[@]}"
