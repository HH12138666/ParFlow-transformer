dataset_parameters = {
    # parflow自动对齐
    'parflow': {
    'in_shape': [24, 14, 146, 252],  # 10 个压力层 + 4 个 evaptrans 层
    'pre_seq_length': 12,
    'aft_seq_length': 12,
    'total_length': 24,
    'data_name': 'parflow',
    'metrics': ['mae', 'mse', 'rmse', 'mape'],
    # 额外拼接的 evaptrans 数据配置（固定 data_root/evapotrans）
    # 只预测压力通道
    'target_channels': 10,
    },
}
