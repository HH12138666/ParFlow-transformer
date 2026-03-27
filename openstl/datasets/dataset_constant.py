dataset_parameters = {
    # parflow自动对齐
    'parflow': {
    'in_shape': [24, 27, 146, 252],  # 原始输入：压力 10 层 + evaptrans 4 层 + static 22 通道
    'pre_seq_length': 12,
    'aft_seq_length': 12,
    'total_length': 24,
    'data_name': 'parflow',
    'metrics': ['mae', 'mse', 'rmse', 'mape'],
    },
}
