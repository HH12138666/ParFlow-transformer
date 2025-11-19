#切到当前仓库根目录（根据你的路径调整为真实绝对路径）
REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"

#让当前仓库优先于任何其他已安装/旧仓库
export PYTHONPATH="$REPO:$PYTHONPATH"

# 选择使用的GPU
export CUDA_VISIBLE_DEVICES=3
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
#自己写清楚实验名称
EX_NAME="ParFlow_press/PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep"

python tools/inference.py \
    --config_file configs/parflow/PredFormer.py \
    --dataname parflow \
    --data_root data/parflow_press \
    --res_dir work_dirs \
    --batch_size 5 \
    --ex_name "$EX_NAME" \