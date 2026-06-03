#!/bin/bash

# 切到当前仓库根目录（根据你的路径调整为真实绝对路径）
REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"

# 让当前仓库优先于任何其他已安装/旧仓库
export PYTHONPATH="$REPO:$PYTHONPATH"

# sbatch /home/huanghui/data/slurm_job/ParFlow_transformer.sh

# 选择使用的单张 GPU
export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_press/${CURRENT_TIME}_FACTS"

python tools/train.py \
    --config_file configs/parflow/PredFormer.py \
    --dataname parflow \
    --data_root /home/huanghui/data/ParFlow-transformer/data/parflow \
    --res_dir work_dirs \
    --batch_size 28 \
    --val_batch_size 28 \
    --epoch 60 \
    --lr 3e-4 \
    --warmup_epoch 0 \
    --empty_cache \
    --fp16 \
    --use_val False \
    --save_interval 20 \
    --test_interval 20 \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --num_workers 16 \
    --var_name press \
    --use_evap True \
    --use_static_input True \
    --stats_path /home/huanghui/data/ParFlow-transformer/stats/stats1_press_evap_static_2019_2020.npz \
    --split_mode year \
    --train_years [2019,2020] \
    --holdout_years [2021]
