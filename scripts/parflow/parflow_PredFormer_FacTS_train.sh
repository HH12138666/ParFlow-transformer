#切到当前仓库根目录（根据你的路径调整为真实绝对路径）
REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"

#让当前仓库优先于任何其他已安装/旧仓库
export PYTHONPATH="$REPO:$PYTHONPATH"


export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_press/${CURRENT_TIME}_PredFormer_depth4_FacTS_sd0.25_dp0.1_ps4_bs16_256_8_32_5e-4_Adamw_cosine_200ep"

python tools/train.py \
    --config_file configs/parflow/PredFormer.py \
    --dataname parflow \
    --data_root data \
    --res_dir work_dirs \
    --batch_size 4 \
    --epoch 30 \
    --overwrite \
    --lr 5e-4 \
    --sched cosine \
    --warmup_epoch 0 \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --tb_dir logs_tb/03_08

# nohup bash /home/huanghui/data/ParFlow-transformer/scripts/parflow/parflow_PredFormer_FacTS_train.sh > "train_log_2$(date +'%Y%m%d_%H%M%S').log" 2>&1 &