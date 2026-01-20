method = 'PredFormer'

model_config = {
    # press h w c
    'height': 146,
    'width': 252,
    'input_channels': 36,   # 原始输入：压力 10 层 + evaptrans 4 层 + static 22 通道
    'dynamic_channels': 14, # 压力 10 层 + evaptrans 4 层
    'static_in_channels': 22, # 静态输入通道数
    'static_out_channels': 10, # 静态压缩到 10 层，如果不使用cnn处理静态数据，则设为None
    'in_channels': 24,   # 动态 14 + 静态压缩 10
    'out_channels': 14,  # 输出：压力 10 层 ，为了让batch_y能有真实的evap数据，方便在自回归的时候可以用到真实的evap数据进行预测
    
    # attention type
    'pre_attn_type': 'none',  # none or self or cross
    
    # cnn卷积核大小
    'static_kernel_size':5,
    
    # space stride
    'space_h': 60,
    'space_w': 84,
    'space_stride_h': 30,
    'space_stride_w': 42,
    'val_save_stride': 0,# 验证集保存步长，0表示不保存
    
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
    'Ndepth': 6, # For FullAttention-8, for BinaryST, BinaryST, FacST, FacTS-4, for TST,STS-3, for TSST, STTS-2
}

# 默认的空间裁剪/步长配置（让 CLI 不传参时也能从配置文件生效）
space_h = model_config['space_h']
space_w = model_config['space_w']
space_stride_h = model_config['space_stride_h']
space_stride_w = model_config['space_stride_w']
val_save_stride = model_config['val_save_stride']
