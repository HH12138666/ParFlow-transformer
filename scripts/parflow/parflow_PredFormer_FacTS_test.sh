cd /home/huanghui/data/ParFlow-transformer
export PYTHONPATH="/home/huanghui/data/ParFlow-transformer:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

python tools/test.py \
  --config_file configs/parflow/PredFormer.py \
  --dataname parflow \
  --data_root /home/huanghui/data/ParFlow-transformer/data/parflow \
  --res_dir work_dirs \
  --ex_name ParFlow_press/2025-12-30-16-38_FACTS \
  --batch_size 28 \
  --val_batch_size 28 \
  --static_data alpha,n_z6-9,porosity \
  --test
