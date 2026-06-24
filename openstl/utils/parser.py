import argparse


def parse_list(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    stripped = str(value).strip().strip("[]")
    if not stripped:
        return []
    return [int(item.strip()) for item in stripped.split(",")]


def _add_override(parser, *flags, **kwargs):
    parser.add_argument(*flags, default=argparse.SUPPRESS, **kwargs)


def _add_bool_override(parser, name, help_text):
    dashed_name = name.replace("_", "-")
    flags = [f"--{name}"]
    if dashed_name != name:
        flags.append(f"--{dashed_name}")
    _add_override(
        parser,
        *flags,
        action=argparse.BooleanOptionalAction,
        help=help_text,
    )


def _add_runtime_args(parser):
    _add_override(parser, "--device", type=str)
    _add_override(parser, "--res_dir", "--res-dir", type=str)
    _add_override(parser, "--ex_name", "--experiment-name", "-ex", type=str)
    _add_override(parser, "--seed", type=int)
    _add_override(parser, "--resume_from", "--resume-from", type=str)
    _add_override(parser, "--finetune_from", "--finetune-from", type=str)
    _add_bool_override(parser, "use_gpu", "Use CUDA for training")
    _add_bool_override(parser, "fp16", "Use native automatic mixed precision")
    _add_bool_override(parser, "empty_cache", "Empty the CUDA cache after each epoch")
    _add_bool_override(parser, "auto_resume", "Resume from the latest checkpoint")
    _add_bool_override(parser, "deterministic", "Enable deterministic CUDNN behavior")


def _add_data_args(parser):
    _add_override(parser, "--batch_size", "--batch-size", "-b", type=int)
    _add_override(parser, "--val_batch_size", "--val-batch-size", "-vb", type=int)
    _add_override(parser, "--num_workers", "--num-workers", type=int)
    _add_override(parser, "--data_root", "--data-root", type=str)
    _add_override(parser, "--stats_path", "--stats-path", type=str)
    _add_override(parser, "--train_years", "--train-years", type=parse_list)
    _add_override(parser, "--holdout_years", "--holdout-years", type=parse_list)
    _add_override(parser, "--val_ratio_in_holdout", "--val-ratio-in-holdout", type=float)
    _add_override(parser, "--var_name", "--var-name", type=str)
    _add_override(parser, "--split_mode", "--split-mode", choices=("ratio", "year"))
    _add_bool_override(parser, "use_augment", "Apply spatial augmentation during training")
    _add_bool_override(parser, "drop_last", "Drop the final incomplete training batch")
    _add_bool_override(parser, "use_evap", "Include evaptrans channels")
    _add_bool_override(parser, "use_static_input", "Include static channels")
    _add_bool_override(parser, "use_time_grouped_sampler", "Group spatial patches by time")
    _add_bool_override(parser, "use_val", "Enable validation and early stopping")
    _add_bool_override(parser, "use_extra_data", "Append manifest-selected extra data")
    _add_override(parser, "--extra_manifest_path", "--extra-manifest-path", type=str)
    _add_override(parser, "--extra_data_root", "--extra-data-root", type=str)


def _add_training_args(parser):
    _add_override(parser, "--epoch", "--epochs", "-e", type=int)
    _add_override(parser, "--log_step", "--log-step", type=int)
    _add_override(parser, "--lr", "--learning-rate", type=float)
    _add_override(parser, "--weight_decay", "--weight-decay", type=float)
    _add_override(parser, "--opt_eps", "--optimizer-eps", type=float)
    _add_override(parser, "--opt_betas", "--optimizer-betas", type=float, nargs=2)
    _add_override(parser, "--clip_grad", "--clip-grad", type=float)
    _add_override(parser, "--clip_mode", "--clip-mode", choices=("norm", "value", "agc"))
    _add_override(parser, "--early_stop_epoch", "--early-stop-epoch", type=int)
    _add_override(parser, "--save_interval", "--save-interval", type=int)
    _add_override(parser, "--test_interval", "--test-interval", type=int)
    _add_override(parser, "--warmup_lr", "--warmup-lr", type=float)
    _add_override(parser, "--min_lr", "--min-lr", type=float)
    _add_override(parser, "--warmup_epoch", "--warmup-epochs", type=int)
    _add_override(parser, "--lr_k_decay", "--lr-k-decay", type=float)
    _add_bool_override(parser, "filter_bias_and_bn", "Disable decay for bias and norm parameters")


def _add_distributed_args(parser):
    _add_override(parser, "--launcher", choices=("none", "pytorch"))
    _add_override(parser, "--local_rank", "--local-rank", type=int)
    _add_override(parser, "--port", type=int)
    _add_bool_override(parser, "find_unused_parameters", "Enable DDP unused parameter detection")
    _add_bool_override(parser, "broadcast_buffers", "Broadcast DDP buffers")


def create_parser():
    parser = argparse.ArgumentParser(description="Train the ParFlow PredFormer model")
    parser.add_argument(
        "--config_file",
        "--config-file",
        "-c",
        default="configs/parflow/PredFormer.py",
        help="Experiment configuration file",
    )
    _add_runtime_args(parser)
    _add_data_args(parser)
    _add_training_args(parser)
    _add_distributed_args(parser)
    return parser
