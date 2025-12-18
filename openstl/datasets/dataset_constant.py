dataset_parameters = {
    # parflow自动对齐
    'parflow': {
    'in_shape': [60, 14, 146, 252],  # 10 个压力层 + 4 个 evaptrans 层
    'pre_seq_length': 15,
    'aft_seq_length': 45,
    'total_length': 60,
    'data_name': 'parflow',
    'metrics': ['mae', 'mse', 'rmse', 'mape'],
    # 额外拼接的 evaptrans 数据配置（可按需覆盖）
    'evap_root': '/home/huanghui/share/parflow-group/sunaoqi-share-old-server/sunaoqi-share/standard_2018/output_evapotrans',
    'evap_channels': [6, 7, 8, 9],
    },
}
