method = 'convlstm'

model_config = {
    'height': 146,
    'width': 252,
    'input_channels': 5,
    'out_channels': 1,
    'pre_seq': 12,
    'after_seq': 12,
    'hidden_dim': 32,
    'kernel_size': 3,

    # keep the same tiling setup for fair comparison
    'space_h': 60,
    'space_w': 84,
    'space_stride_h': 30,
    'space_stride_w': 42,
    'val_save_stride': 0,
    'pad_to_patch': False,
}

space_h = model_config['space_h']
space_w = model_config['space_w']
space_stride_h = model_config['space_stride_h']
space_stride_w = model_config['space_stride_w']
val_save_stride = model_config['val_save_stride']
pad_to_patch = model_config['pad_to_patch']
