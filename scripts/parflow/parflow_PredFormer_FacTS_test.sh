cd /home/huanghui/data/ParFlow-transformer
export PYTHONPATH="/home/huanghui/data/ParFlow-transformer:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=2

python tools/test.py \
  --config_file configs/parflow/PredFormer.py \
  --dataname parflow \
  --data_root /home/huanghui/data/ParFlow-transformer/data/parflow \
  --res_dir work_dirs \
  --ex_name ParFlow_press/2026-04-19-22-24_FACTS \
  --batch_size 28 \
  --val_batch_size 28 \
  --static_data alpha,n_z6-9,porosity \
  --test
