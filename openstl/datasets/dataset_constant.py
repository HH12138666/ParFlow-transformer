dataset_parameters = {
    # parflow自动对齐
    'parflow': {
    'in_shape': [12, 10, 144, 248],   # [T, C, H, W]；T用于展示/约束
    'pre_seq_length': 6,
    'aft_seq_length': 6,
    'total_length': 12,
    'data_name': 'parflow',
    'metrics': ['mae', 'mse', 'rmse', 'mape'],
    },
}