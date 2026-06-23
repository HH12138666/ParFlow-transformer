method = 'PredFormer'

# 这里主要放“模型结构”和“输入/输出通道结构”。
# 训练数据选择、loss/save 通道、数据划分方式等实验参数，
# 统一放到 scripts/parflow/parflow_PredFormer_FacTS_train.sh 中控制。

model_config = {
    # 当前这套配置对应 press + evap + static 的输入形式
    'height': 146,
    'width': 252,
    'input_channels': 45,   # 原始输入总通道 = press 10 + evap 4 + static 28
    'dynamic_channels': 14,  # 动态输入通道 = press 10 + evap 4
    'static_in_channels': 31, # 静态输入通道数
    'static_out_channels': 8, # 静态通道经卷积压缩后的输出通道数；若不压缩则设为 None
    'in_channels': 22 ,  # 送入 patch embedding 的通道 = 动态 10 + 压缩后静态 0
    'out_channels': 10,  # 模型输出通道 = 只预测 press 的 10 层
    
    # attention type
    'attn_type': 'post_cross',  # none or pre_cross or post_cross
    
    # 静态数据压缩卷积核大小
    'static_kernel_size':3,
    
    # space stride
    'space_h': 64,
    'space_w': 80,
    'space_stride_h': 32, # 滑窗步长；None 表示不做空间滑窗，直接使用全图
    'space_stride_w': 40, # 滑窗步长；None 表示不做空间滑窗，直接使用全图

    
    # video length in and out
    'pre_seq': 12,
    'after_seq': 12,
    # patch size
    'patch_size': 8,
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


