dataset_parameters = {
    # parflow自动对齐
    'parflow': {
    'in_shape': [10, 7, 128, 240],   # [T, C, H, W]；T用于展示/约束
    'pre_seq_length': 9,
    'aft_seq_length': 1,
    'total_length': 10,
    'data_name': 'parflow',
    'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
}