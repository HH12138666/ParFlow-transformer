method = 'PredFormer'

model_config = {
    # wtd h w c
    'height': 146,
    'width': 252,
    'input_channels': 5,   # wtd 1 层 + static 22 通道
    'dynamic_channels': 5,  # wtd 单通道
    'static_in_channels': None, # 静态输入通道数
    'static_out_channels': None, # 静态压缩到 n 层，如果不使用cnn处理静态数据，则设为None
    'in_channels': 5,   # 动态 1 + 静态压缩 5
    'out_channels': 1,  # 输出：wtd 单通道
    
    # attention type
    'attn_type': 'none',  # none or pre_cross or post_cross or film
    
    # cnn卷积核大小
    'static_kernel_size':1,
    
    # space stride
    'space_h': 60,
    'space_w': 84,
    'space_stride_h': 30, # None表示不裁剪，直接用全图，整数表示裁剪成patch的大小
    'space_stride_w': 42, # None表示不裁剪，直接用全图，整数表示裁剪成patch的大小
    'val_save_stride': 0,# 验证集保存步长，0表示不保存
    'pad_to_patch': True, # 整图模式下补到 patch_size 的整数倍，滑窗模式可关掉

    
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

# 默认的空间裁剪/步长配置（让 CLI 不传参时也能从配置文件生效）
space_h = model_config['space_h']
space_w = model_config['space_w']
space_stride_h = model_config['space_stride_h']
space_stride_w = model_config['space_stride_w']
val_save_stride = model_config['val_save_stride']
patch_size = model_config['patch_size']
pad_to_patch = model_config['pad_to_patch']