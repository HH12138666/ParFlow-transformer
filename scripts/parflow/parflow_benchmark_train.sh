#!/usr/bin/env bash
set -euo pipefail

REPO=/home/huanghui/data/ParFlow-transformer
cd "$REPO"
export PYTHONPATH="$REPO:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

METHOD=${1:-cnn}
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ParFlow_benchmark/${CURRENT_TIME}_${METHOD}"

case "$METHOD" in
  cnn)
    CFG="configs/parflow/cnn.py"
    BS=${BATCH_SIZE:-16}
    ;;
  rnn)
    CFG="configs/parflow/rnn.py"
    BS=${BATCH_SIZE:-4}
    ;;
  lstm)
    CFG="configs/parflow/lstm.py"
    BS=${BATCH_SIZE:-4}
    ;;
  convlstm)
    CFG="configs/parflow/convlstm.py"
    BS=${BATCH_SIZE:-6}
    ;;
  *)
    echo "Unsupported METHOD=$METHOD. Use one of: cnn | rnn | lstm | convlstm"
    exit 1
    ;;
esac

python tools/train.py \
  --method "$METHOD" \
  --config_file "$CFG" \
  --dataname parflow \
  --data_root /home/huanghui/data/ParFlow-transformer/data/parflow \
  --res_dir work_dirs \
  --batch_size "$BS" \
  --val_batch_size "$BS" \
  --epoch 60 \
  --overwrite \
  --fp16 \
  --lr 5e-4 \
  --sched cosine \
  --warmup_epoch 0 \
  --opt adamw \
  --weight_decay 1e-2 \
  --ex_name "$EX_NAME" \
  --early_stop_epoch 40 \
  --num_workers 16 \
  --var_name wtd \
  --use_evap "true" \
  --use_static_input "" \
  --loss_channels 1 \
  --save_channels 1 \
  --split_mode year \
  --train_years [2019] \
  --holdout_years [2020] \
  --val_ratio_in_holdout 0.5
