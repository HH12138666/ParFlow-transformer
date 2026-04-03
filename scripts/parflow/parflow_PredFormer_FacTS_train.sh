#!/bin/bash

# 切到当前仓库根目录（根据你的路径调整为真实绝对路径）
REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"

# 让当前仓库优先于任何其他已安装/旧仓库
export PYTHONPATH="$REPO:$PYTHONPATH"

# sbatch /home/huanghui/data/slurm_job/ParFlow_transformer.sh

# 选择使用的单张 GPU
export CUDA_VISIBLE_DEVICES=1
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_press/${CURRENT_TIME}_FACTS"

python tools/train.py \
    --config_file configs/parflow/PredFormer.py \
    --dataname parflow \
    --data_root /home/huanghui/data/ParFlow-transformer/data/parflow \
    --res_dir work_dirs \
    --batch_size 28 \
    --val_batch_size 28 \
    --epoch 80 \
    --overwrite \
    --lr 3e-4 \
    --sched cosine \
    --warmup_epoch 0 \
    --opt adamw \
    --empty_cache \
    --fp16 \
    --log_step 1 \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --early_stop_epoch 60\
    --num_workers 28 \
    --var_name press \
    --use_evap False \
    --use_apcp True \
    --use_static_input False \
    --stats_path /home/huanghui/data/ParFlow-transformer/stats/stats_press_APCP1.8.npz \
    --split_mode year \
    --train_years [2020,2021] \
    --holdout_years [2019] \
    --val_ratio_in_holdout 0.3
