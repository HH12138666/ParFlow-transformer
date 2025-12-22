method = 'PredFormer'

model_config = {
    # press h w c
    'height': 146,
    'width': 252,
    'in_channels': 74,   # 输入：压力 10 层 + evaptrans 4 层
    'out_channels': 10,  # 输出：仅压力 10 层
    
    # space stride
    'space_h': 60,
    'space_w': 84,
    'space_stride_h': 30,
    'space_stride_w': 42,
    'val_save_stride': 0,
    
    # video length in and out
    'pre_seq': 10,
    'after_seq': 10,
    # patch size
    'patch_size': 4,
    'dim': 256, 
    'heads': 8,
    'dim_head': 32,
    # dropout
    'dropout': 0.1,
    'attn_dropout': 0.1,
    'drop_path': 0.25,
    'scale_dim': 2,
    # depth
    'depth': 4,
    'Ndepth': 6, # For FullAttention-8, for BinaryST, BinaryST, FacST, FacTS-4, for TST,STS-3, for TSST, STTS-2
}

# 默认的空间裁剪/步长配置（让 CLI 不传参时也能从配置文件生效）
space_h = model_config['space_h']
space_w = model_config['space_w']
space_stride_h = model_config['space_stride_h']
space_stride_w = model_config['space_stride_w']
val_save_stride = model_config['val_save_stride']
