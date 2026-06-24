method = "predformer"
dataname = "parflow"

# ParFlow PredFormer architecture.
model_config = {
    "height": 146,
    "width": 252,
    "input_channels": 45,  # pressure 10 + evaptrans 4 + static 31
    "dynamic_channels": 14,
    "static_in_channels": 31,
    "static_out_channels": 8,
    "in_channels": 22,  # dynamic 14 + projected static 8
    "out_channels": 10,
    "attn_type": "post_cross",
    "static_kernel_size": 3,
    "space_h": 64,
    "space_w": 80,
    "space_stride_h": 32,
    "space_stride_w": 40,
    "pre_seq": 12,
    "after_seq": 12,
    "patch_size": 8,
    "dim": 256,
    "heads": 8,
    "dim_head": 32,
    "dropout": 0.1,
    "attn_dropout": 0.1,
    "drop_path": 0.25,
    "scale_dim": 2,
    "depth": 4,
    "Ndepth": 6,
}
