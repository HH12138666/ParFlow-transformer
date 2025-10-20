dataset_parameters = {
    # parflow自动对齐
    'parflow': {
    'in_shape': [12, 10, 144, 252],   # [T, C, H, W]；T用于展示/约束
    'pre_seq_length': 12,
    'aft_seq_length': 12,
    'total_length': 24,
    'data_name': 'parflow',
    'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
}