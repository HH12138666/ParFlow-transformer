#!/bin/bash

# 切到当前仓库根目录（根据你的路径调整为真实绝对路径）
REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"

# 让当前仓库优先于任何其他已安装/旧仓库
export PYTHONPATH="$REPO:$PYTHONPATH"

# sbatch /home/huanghui/data/slurm_job/ParFlow_transformer.sh

# ===================== 用户配置区 =====================
# 选择使用的单张 GPU
export CUDA_VISIBLE_DEVICES=1

# 是否加入额外训练数据。
# True: 训练集 = 普通训练年份 + EXTRA_MANIFEST_PATH 指定的额外 t0 样本。
# False: 只使用普通训练年份。
USE_EXTRA_DATA=True
EXTRA_MANIFEST_PATH=/home/huanghui/data/ParFlow-transformer/extra_data/extra_apcp14_training_design/extra_apcp14_regime_moderate_heavy_2020_2021.csv
EXTRA_DATA_ROOT=/home/huanghui/data/ParFlow_train_data/apcp1.4

# stats_path 必须和训练数据组成一致：
# - 不加额外数据时，用普通训练年份 stats。
# - 加额外数据时，用普通训练年份 + 这份额外 manifest 重新计算后的 stats。
# =====================================================

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_press/${CURRENT_TIME}_FACTS"

EXTRA_ARGS=""
if [ "$USE_EXTRA_DATA" = "True" ]; then
    EXTRA_ARGS="--use_extra_data True --extra_manifest_path $EXTRA_MANIFEST_PATH --extra_data_root $EXTRA_DATA_ROOT"
fi

python tools/train.py \
    --config_file configs/parflow/PredFormer.py \
    --dataname parflow \
    --data_root /home/huanghui/data/ParFlow-transformer/data/parflow \
    --res_dir work_dirs \
    --batch_size 28 \
    --val_batch_size 28\
    --epoch 60 \
    --lr 3e-4 \
    --warmup_epoch 0 \
    --empty_cache \
    --fp16 \
    --use_val False \
    --save_interval 15 \
    --test_interval 15 \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --num_workers 16 \
    --var_name press \
    --use_evap True  \
    --use_static_input True \
    --stats_path /home/huanghui/data/ParFlow-transformer/stats/stats1_1.4_press_evap_static_2020_2021_moderate_heavy.npz \
    --split_mode year \
    --train_years [2020,2021] \
    --holdout_years [2019] \
    $EXTRA_ARGS
