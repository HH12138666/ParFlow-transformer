#!/bin/bash
set -e

# Modify PROJECT_ROOT when this repository is placed in a different directory.
PROJECT_ROOT=/home/huanghui/data/ParFlow-transformer
export CUDA_VISIBLE_DEVICES=0

cd ${PROJECT_ROOT}
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

# Main paths to modify for a new inference run:
#   --run-dir: trained experiment directory containing model_param.json and checkpoints/.
#   --checkpoint-file: checkpoint file name under --run-dir/checkpoints/.
#   --data-root: ParFlow-CLM data root used as inference input.
#   --output-dir: directory for predicted pressure-head PFB files.

python -m model_deployment.inference \
    --run-dir ${PROJECT_ROOT}/work_dirs/ParFlow_press/2026-07-11-18-00_FACTS \
    --checkpoint-file latest_state_dict.pth \
    --data-root ${PROJECT_ROOT}/data/parflow/normal_data \
    --output-dir ${PROJECT_ROOT}/inference_data/example_press \
    --run-name example_rollout12h \
    --start-hour 20190000 \
    --end-hour 20190035 \
    --rollout-hours 12 \
    --patch-batch-size 4 \
    --use-rollout \
    --use-amp
