"""Framework-level defaults shared by experiment configurations."""


DEFAULT_CONFIG = {
    # Runtime.
    "device": "cuda",
    "res_dir": "work_dirs",
    "ex_name": "Debug",
    "use_gpu": True,
    "fp16": False,
    "seed": 42,
    "fps": False,
    "empty_cache": True,
    "resume_from": None,
    "finetune_from": None,
    "auto_resume": False,
    "test": False,
    "inference": False,
    "deterministic": False,
    "launcher": "none",
    "local_rank": 0,
    "port": 29500,
    "find_unused_parameters": False,
    "broadcast_buffers": False,
    "no_display_method_info": False,
    # Data loading.
    "batch_size": 16,
    "val_batch_size": 16,
    "num_workers": 4,
    "use_augment": False,
    "drop_last": False,
    "val_ratio_in_holdout": 0.5,
    "use_time_grouped_sampler": True,
    "use_extra_data": False,
    "extra_manifest_path": None,
    "extra_data_root": None,
    # Training.
    "epoch": 50,
    "log_step": 1,
    "opt": "adamw",
    "opt_eps": None,
    "opt_betas": None,
    "weight_decay": 0.0,
    "clip_grad": None,
    "clip_mode": "norm",
    "early_stop_epoch": 30,
    "save_interval": 5,
    "test_interval": 5,
    "sched": "cosine",
    "lr": 1e-3,
    "lr_k_decay": 1.0,
    "warmup_lr": 1e-5,
    "min_lr": 1e-6,
    "warmup_epoch": 0,
    "filter_bias_and_bn": False,
}


def get_default_config():
    """Return an independent copy for one experiment."""
    return DEFAULT_CONFIG.copy()
