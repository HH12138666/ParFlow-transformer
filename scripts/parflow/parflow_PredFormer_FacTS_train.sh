#切到当前仓库根目录（根据你的路径调整为真实绝对路径）
REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"

#让当前仓库优先于任何其他已安装/旧仓库
export PYTHONPATH="$REPO:$PYTHONPATH"

# 选择使用的GPU
export CUDA_VISIBLE_DEVICES=3
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_wtd/${CURRENT_TIME}_FACTS"

python tools/train.py \
    --config_file configs/parflow/PredFormer.py \
    --dataname parflow \
    --data_root /home/huanghui/data/ParFlow-transformer/data/parflow \
    --res_dir work_dirs \
    --batch_size 28 \
    --val_batch_size 28 \
    --epoch 60 \
    --overwrite \
    --lr 5e-4 \
    --sched cosine \
    --warmup_epoch 0 \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --early_stop_epoch 30 \
    --num_workers 28 \
    --var_name wtd \
    --use_evap "" \
    --use_static_input "" \
    --loss_channels 1 \
    --save_channels 1 \
    --split_mode year \
    --train_years [2019] \
    --holdout_years [2020] \
    --val_ratio_in_holdout 0.5
