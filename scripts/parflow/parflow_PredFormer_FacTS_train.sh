#!/bin/bash
set -euo pipefail

# sbatch /home/huanghui/data/ParFlow-transformer/slurm/parflow_train.sh

# ===================== 用户配置区 =====================
REPO=/home/huanghui/data/ParFlow-transformer


# 选择使用的单张 GPU。
export CUDA_VISIBLE_DEVICES=0


# 普通 ParFlow 数据和标准化统计量。
DATA_ROOT=/home/huanghui/data/ParFlow-transformer/data/parflow/normal_data
STATS_PATH=/home/huanghui/data/ParFlow-transformer/stats/stats1_press_evap_static_2019_2020.npz

# 年份划分：2020/2021 训练，2019 测试；默认不使用验证集。
TRAIN_YEARS='[2019,2020]'
HOLDOUT_YEARS='[2021]'

# 是否加入额外训练数据。
# True: 训练集 = 普通训练年份 + EXTRA_MANIFEST_PATH 指定的额外 t0 样本。
# False: 只使用普通训练年份。
USE_EXTRA_DATA=False

# manifest 新格式已包含 data_root，可同时混合 APCP×1.4 和 APCP×1.8。
# EXTRA_DATA_ROOT 只作为旧 manifest 缺少 data_root 列时的默认根目录。
EXTRA_DATA_ROOT=
EXTRA_MANIFEST_PATH=/home/huanghui/data/ParFlow-transformer/data/parflow/extra_data_index/extra_apcp_moderate_heavy_1000_2020_2021.csv
# - 不加额外数据时，用普通训练年份 stats。
# - 加额外数据时，用普通训练年份 + 这份额外 manifest 重新计算后的 stats。
# 示例：
# dry-1000: stats1_press_evap_static_2020_2021_extra_apcp_dry_1000.npz
# moderate+heavy-1000: stats1_press_evap_static_2020_2021_extra_apcp_moderate_heavy_1000.npz
# heavy-all: stats1_press_evap_static_2020_2021_extra_apcp_heavy_all.npz
# =====================================================

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_press/${CURRENT_TIME}_FACTS"

EXTRA_ARGS=()
if [[ "$USE_EXTRA_DATA" == "True" ]]; then
    EXTRA_ARGS=(
        --use-extra-data
        --extra-manifest-path "$EXTRA_MANIFEST_PATH"
        --extra-data-root "$EXTRA_DATA_ROOT"
    )
fi

python tools/train.py \
    --config-file configs/parflow/PredFormer.py \
    --data-root "$DATA_ROOT" \
    --stats-path "$STATS_PATH" \
    --res-dir work_dirs \
    --batch-size 28 \
    --val-batch-size 28 \
    --epochs 60 \
    --learning-rate 3e-4 \
    --warmup-epochs 0 \
    --empty-cache \
    --fp16 \
    --no-use-val \
    --save-interval 15 \
    --test-interval 15 \
    --weight-decay 1e-2 \
    --experiment-name "$EX_NAME" \
    --num-workers 16 \
    --var-name press \
    --use-evap \
    --use-static-input \
    --split-mode year \
    --train-years "$TRAIN_YEARS" \
    --holdout-years "$HOLDOUT_YEARS" \
    "${EXTRA_ARGS[@]}"
