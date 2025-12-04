method = 'PredFormer'

model_config = {
    # press h w c
    'height': 48,
    'width': 84,
    'num_channels': 10,
    
    # space stride
    'space_h': 48,
    'space_w': 84,
    'space_stride_h': 24,
    'space_stride_w': 42,
    'eval_non_overlap': True,
    
    # video length in and out
    'pre_seq': 6,
    'after_seq': 6,
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
    'Ndepth': 6, # For FullAttention-8, for BinaryST, BinaryST, FacST, FacTS-4, for TST,STS-3, for TSST, STTS-2
}
