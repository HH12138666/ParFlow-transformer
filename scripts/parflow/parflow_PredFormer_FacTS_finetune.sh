#!/bin/bash

# 切到当前仓库根目录
REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"

# 让当前仓库优先于任何其他已安装/旧仓库
export PYTHONPATH="$REPO:$PYTHONPATH"

# sbatch /home/huanghui/data/slurm_job/parflow_PredFormer_FacTS_finetune.sh

# 选择使用的单张 GPU
export CUDA_VISIBLE_DEVICES=3

# 需要微调的预训练权重
FINETUNE_CKPT="/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press/2026-04-21-22-36_FACTS/checkpoint.pth"

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_press/${CURRENT_TIME}_FACTS_finetune"

python tools/train.py \
    --config_file configs/parflow/PredFormer_finetune.py \
    --dataname parflow \
    --data_root /home/huanghui/data/ParFlow_train_data/apcp1.4 \
    --res_dir work_dirs \
    --batch_size 28 \
    --val_batch_size 28 \
    --epoch 50 \
    --overwrite \
    --lr 1e-4 \
    --sched cosine \
    --warmup_epoch 0 \
    --opt adamw \
    --empty_cache \
    --fp16 \
    --log_step 1 \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --early_stop_epoch 5 \
    --num_workers 28 \
    --var_name press \
    --use_evap True \
    --use_apcp False \
    --use_static_input True \
    --stats_path /home/huanghui/data/ParFlow-transformer/stats/stats1.4_press_evap_static_2020_2021.npz \
    --split_mode year \
    --train_years [2020,2021] \
    --holdout_years [2019] \
    --val_ratio_in_holdout 0.3 \
    --finetune_from "$FINETUNE_CKPT"
