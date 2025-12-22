dataset_parameters = {
    # parflow自动对齐
    'parflow': {
    'in_shape': [20, 74, 146, 252],  # 10 个压力层 + 4 个 evaptrans 层
    'pre_seq_length': 10,
    'aft_seq_length': 10,
    'total_length': 20,
    'data_name': 'parflow',
    'metrics': ['mae', 'mse', 'rmse', 'mape'],
    # 额外拼接的 evaptrans 数据配置（固定 data_root/evapotrans）
    # 只预测压力通道
    'target_channels': 10,
    },
}
