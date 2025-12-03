#切到当前仓库根目录（根据你的路径调整为真实绝对路径）
REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"

#让当前仓库优先于任何其他已安装/旧仓库
export PYTHONPATH="$REPO:$PYTHONPATH"

# 选择使用的GPU
export CUDA_VISIBLE_DEVICES=2
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_press/${CURRENT_TIME}_PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep"

python tools/train.py \
    --config_file configs/parflow/PredFormer.py \
    --dataname parflow \
    --data_root data/parflow_press \
    --res_dir work_dirs \
    --batch_size 5 \
    --val_batch_size 5 \
    --epoch 20 \
    --overwrite \
    --lr 5e-4 \
    --sched cosine \
    --warmup_epoch 0 \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --tb_dir logs_tb/11_28 \


