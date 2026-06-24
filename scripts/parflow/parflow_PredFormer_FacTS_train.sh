#!/bin/bash
set -euo pipefail

# sbatch /home/huanghui/data/ParFlow-transformer/slurm/parflow_train.sh

# ===================== 用户配置区 =====================
REPO=/home/huanghui/data/ParFlow-transformer


# 选择使用的单张 GPU。
export CUDA_VISIBLE_DEVICES=1

# 普通 ParFlow 数据和标准化统计量。
DATA_ROOT=/home/huanghui/data/ParFlow-transformer/data/parflow/normal_data
STATS_PATH=/home/huanghui/data/ParFlow-transformer/stats/stats1_1.4_press_evap_static_2020_2021_moderate_heavy.npz

# 年份划分：2020/2021 训练，2019 测试；默认不使用验证集。
TRAIN_YEARS='[2020,2021]'
HOLDOUT_YEARS='[2019]'

# 是否加入额外训练数据。
# True: 训练集 = 普通训练年份 + EXTRA_MANIFEST_PATH 指定的额外 t0 样本。
# False: 只使用普通训练年份。
USE_EXTRA_DATA=True
EXTRA_DATA_ROOT=/home/huanghui/data/ParFlow-transformer/data/parflow/extra_data_apcp14
EXTRA_MANIFEST_PATH=/home/huanghui/data/ParFlow-transformer/data/parflow/extra_data_index/extra_apcp14_top_hours_4800h_2020_2021.csv

# stats_path 必须和训练数据组成一致：
# - 不加额外数据时，用普通训练年份 stats。
# - 加额外数据时，用普通训练年份 + 这份额外 manifest 重新计算后的 stats。
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
