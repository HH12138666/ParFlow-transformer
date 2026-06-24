import argparse
import warnings

warnings.filterwarnings("ignore")

from openstl.api import BaseExperiment
from openstl.utils.default_config import get_default_config
from openstl.utils import (
    create_parser,
    destroy_dist,
    get_dist_info,
    init_dist,
    load_config,
    setup_multi_processes,
)


MODEL_CONFIG_KEYS = (
    "space_h",
    "space_w",
    "space_stride_h",
    "space_stride_w",
)


def resolve_config(parsed_args):
    cli_values = vars(parsed_args).copy()
    config_path = cli_values.pop("config_file")
    config = get_default_config()
    config.update(load_config(config_path))
    config.update(cli_values)
    config["config_file"] = config_path
    _apply_model_defaults(config)
    _validate_config(config)
    return argparse.Namespace(**config)


def _apply_model_defaults(config):
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError("model_config must be a dictionary")
    for key in MODEL_CONFIG_KEYS:
        if config.get(key) is None:
            config[key] = model_config.get(key)
    config["pre_seq_length"] = model_config["pre_seq"]
    config["aft_seq_length"] = model_config["after_seq"]
    config["total_length"] = config["pre_seq_length"] + config["aft_seq_length"]
    config["in_shape"] = [
        config["total_length"],
        model_config["input_channels"],
        model_config["height"],
        model_config["width"],
    ]
    config["data_name"] = config["dataname"]
    config["metrics"] = ["mae", "rmse"]


def _validate_config(config):
    required = (
        "method",
        "dataname",
        "data_root",
        "stats_path",
        "epoch",
        "batch_size",
        "val_batch_size",
        "num_workers",
        "lr",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required configuration keys: {missing}")
    if config["stats_path"] is None:
        raise ValueError("stats_path must be set in the config or CLI")


def configure_distributed(args):
    args.distributed = args.launcher != "none"
    args.rank = 0
    args.world_size = 1
    if not args.distributed:
        return
    init_dist(args.launcher, port=args.port)
    args.rank, args.world_size = get_dist_info()


def main():
    args = resolve_config(create_parser().parse_args())
    setup_multi_processes(args.__dict__)
    configure_distributed(args)

    if not args.distributed or args.rank == 0:
        print(">" * 35 + " training " + "<" * 35)
    experiment = BaseExperiment(args)
    experiment.train()

    if not args.distributed or args.rank == 0:
        print(">" * 35 + " testing  " + "<" * 35)
    experiment.test()

    if args.distributed:
        destroy_dist()


if __name__ == "__main__":
    main()
