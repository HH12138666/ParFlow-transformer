#!/bin/bash
set -euo pipefail

# sbatch /home/huanghui/data/ParFlow-transformer/slurm/parflow_inference.sh

# ===================== 用户配置区 =====================
REPO=/home/huanghui/data/ParFlow-transformer


# 选择使用的单张 GPU。
export CUDA_VISIBLE_DEVICES=1

# 需要推理的训练结果目录。
RUN_DIR=/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press/2026-06-12-13-59_FACTS
CHECKPOINT_FILE=latest.pth

# 推理数据和输出目录。
DATA_ROOT=/home/huanghui/data/ParFlow-transformer/data/parflow/normal_data
OUTPUT_DIR=/home/huanghui/data/ParFlow-transformer/inference_data/press
RUN_NAME="test"

# 推理时间范围和 rollout 设置。
START_HOUR=20190000
END_HOUR=20198759
ROLLOUT_HOURS=720
PATCH_BATCH_SIZE=28
# =====================================================

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

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
