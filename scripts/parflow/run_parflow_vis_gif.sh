#!/bin/bash

# Script to generate ParFlow GIF visualizations.

REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO" || exit 1

export PYTHONPATH="$REPO:$PYTHONPATH"

# Update these paths if needed
WORK_DIR="/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press/2025-12-11-21-29_PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep"
SAVE_DIR="/home/huanghui/data/ParFlow-transformer/vis_figures"
CHANNEL=0
INDEX=0

python tools/visualizations/vis_video.py \
  --work_dir "$WORK_DIR" \
  --save_dir "$SAVE_DIR" \
  --vis_channel "$CHANNEL" \
  --index "$INDEX"

# bash /home/huanghui/data/ParFlow-transformer/scripts/parflow/run_parflow_vis_gif.sh
