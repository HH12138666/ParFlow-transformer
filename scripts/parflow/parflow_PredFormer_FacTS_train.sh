export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_press/${CURRENT_TIME}_PredFormer_depth4_FacTS_sd0.25_dp0.1_ps4_bs16_256_8_32_5e-4_Adamw_cosine_200ep"

python tools/train.py \
    --config_file configs/parflow/PredFormer.py \  #改
    --dataname parflow \         #改
    --data_root data \              #改 数据存放目录地址
    --res_dir work_dirs \               #不改
    --batch_size 16 \       #不改或改
    --epoch 30 \            #不改或改
    --overwrite \             # 如果之前已经存在同名的训练结果（比如相同实验名），​直接覆盖它。
    --lr 5e-4 \             #不改或改
    --sched cosine \        #不改
    --warmup_epoch 0 \          #不改
    --opt adamw \               #不改
    --weight_decay 1e-2 \           #不改
    --ex_name "$EX_NAME"  \         #不改
    --tb_dir logs_tb/03_08          #不改  训练日志保存地址