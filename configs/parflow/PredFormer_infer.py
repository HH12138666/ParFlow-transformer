method = 'PredFormer'

model_config = {
    # press h w c
    'height': 146,
    'width': 252,
    'input_channels': 36,
    'dynamic_channels': 14,
    'static_in_channels': 22, # 如果不适用cnn处理静态数据，则设为None
    'static_out_channels': 10,    # 如果不适用cnn处理静态数据，则设为None
    'in_channels': 24,
    'out_channels': 14,
    
    # space stride
    'space_h': 60,
    'space_w': 84,
    'space_stride_h': 30,
    'space_stride_w': 42,
    'val_save_stride': 0,
    
    # video length in and out
    'pre_seq': 12,
    'after_seq': 12,
    
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
    'Ndepth': 6,
}

