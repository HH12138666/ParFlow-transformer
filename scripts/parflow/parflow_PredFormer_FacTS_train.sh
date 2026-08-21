#!/bin/bash
set -e

# Modify PROJECT_ROOT when this repository is placed in a different directory.
PROJECT_ROOT=/home/huanghui/data/ParFlow-transformer
RUN_DATE=$(date +%Y-%m-%d-%H-%M)
export CUDA_VISIBLE_DEVICES=0

cd ${PROJECT_ROOT}
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

# Main paths to modify for a new experiment:
#   --data-root: baseline ParFlow-CLM data root, containing press/, evaptrans/, and static/.
#   --stats-path: normalization stats. Use normal+extra stats when extra data are enabled.
#   --res-dir: output directory for logs and checkpoints.
#
# To train with extra ParFlow-CLM samples, add these arguments to the python command below:
#   --use-extra-data \
#   --extra-manifest-path ${PROJECT_ROOT}/data/parflow/extra_data_index/extra_sample_manifest.csv
#
# The extra manifest should provide split=train and t0/hour_id/start_hour.
# If the manifest does not contain data_root, also add:
#   --extra-data-root ${PROJECT_ROOT}/data/parflow/extra_data_apcp14

python tools/train.py \
    --config-file configs/parflow/PredFormer.py \
    --data-root ${PROJECT_ROOT}/data/parflow/normal_data \
    --stats-path ${PROJECT_ROOT}/stats/stats1_press_evap_static_2019_2020.npz \
    --res-dir ${PROJECT_ROOT}/work_dirs \
    --experiment-name ParFlow_press/${RUN_DATE}_FACTS \
    --batch-size 2 \
    --val-batch-size 2 \
    --epochs 1 \
    --learning-rate 3e-4 \
    --warmup-epochs 0 \
    --empty-cache \
    --fp16 \
    --no-use-val \
    --save-interval 1 \
    --test-interval 1 \
    --weight-decay 1e-2 \
    --num-workers 0 \
    --var-name press \
    --use-evap \
    --use-static-input \
    --split-mode year \
    --train-years '[2019]' \
    --holdout-years '[2019]'
