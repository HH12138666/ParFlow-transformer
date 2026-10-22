#!/bin/bash
set -euo pipefail

# Change this root to the installed repository. No target-basin training is used.
PROJECT_ROOT=/home/huanghui/data/ParFlow-transformer
# Unzip the Hejiang press, four-layer evaptrans, and static archives under this root.
HEJIANG_DATA_ROOT=${PROJECT_ROOT}/data/parflow/hejiang_data

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# The saved model_param.json selects the unchanged UBRB WY2019-WY2020 statistics.
# PME is aligned to pressure by the existing loader: press(h) with evaptrans(h+1).
# 12 complete cases: outputs 20210012..20218651; no incomplete year-end tail.
# Each case is initialized with its own 12-hour local reference pressure history.
python -m model_deployment.inference \
    --run-dir "${PROJECT_ROOT}/work_dirs/ParFlow_press/2026-07-11-18-00_FACTS" \
    --checkpoint-file latest_state_dict.pth \
    --data-root "${HEJIANG_DATA_ROOT}" \
    --output-dir "${PROJECT_ROOT}/inference_data/press" \
    --run-name hejiang_wy2021_zero_shot_rollout720 \
    --start-hour 20210000 \
    --end-hour 20218651 \
    --rollout-hours 720 \
    --patch-batch-size 12 \
    --use-rollout \
    --use-amp
