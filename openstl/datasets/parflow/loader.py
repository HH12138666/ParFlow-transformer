"""DataLoader factory for ParFlow datasets."""

from torch.utils.data import ConcatDataset

from openstl.datasets.utils import LoaderConfig, create_loader

from .config import ParFlowDataConfig
from .dataset import ParFlowDataset
from .manifest import build_extra_train_datasets
from .paths import normalize_years, prepare_press_evap_files, resolve_parflow_roots
from .sampler import build_train_sampler


def load_data(config: ParFlowDataConfig):
    roots = resolve_parflow_roots(
        config.data_root,
        config.use_static_input,
        config.var_name,
        config.use_evap,
    )
    press_root, evap_root, static_root = roots
    press_files, evap_files = _prepare_all_files(press_root, evap_root, config)
    common = _dataset_kwargs(config, evap_root, static_root, press_files, evap_files)
    train_base = ParFlowDataset(
        press_root,
        "train",
        config.pre_seq_length,
        config.aft_seq_length,
        use_augment=config.use_augment,
        **common,
    )
    train_dataset = _append_extra_data(train_base, common, config)
    val_dataset = _build_val_dataset(press_root, common, config)
    test_dataset = ParFlowDataset(
        press_root,
        "test",
        config.pre_seq_length,
        config.aft_seq_length,
        use_augment=False,
        **common,
    )
    return _create_loaders(train_dataset, val_dataset, test_dataset, config)


def _prepare_all_files(press_root, evap_root, config):
    allowed_years = None
    if config.split_mode.lower() == "year":
        train_years = normalize_years(config.train_years) or []
        holdout_years = normalize_years(config.holdout_years) or []
        allowed_years = sorted(set(train_years + holdout_years))
    return prepare_press_evap_files(press_root, evap_root, allowed_years=allowed_years)


def _dataset_kwargs(config, evap_root, static_root, press_files, evap_files):
    return {
        "space_h": config.space_h,
        "space_w": config.space_w,
        "space_stride_h": config.space_stride_h,
        "space_stride_w": config.space_stride_w,
        "evap_root": evap_root,
        "static_root": static_root,
        "out_channels": config.out_channels,
        "press_files": press_files,
        "evap_files": evap_files,
        "split_mode": config.split_mode,
        "train_years": config.train_years,
        "holdout_years": config.holdout_years,
        "val_ratio_in_holdout": config.val_ratio_in_holdout,
        "use_val": config.use_val,
        "stats_path": config.stats_path,
    }


def _append_extra_data(train_base, common, config):
    if not config.use_extra_data:
        return train_base
    extra_kwargs = dict(common)
    extra_kwargs.update({
        "pre": config.pre_seq_length,
        "aft": config.aft_seq_length,
        "use_augment": config.use_augment,
    })
    extra_datasets = build_extra_train_datasets(
        extra_kwargs,
        config.extra_manifest_path,
        config.extra_data_root,
        config.use_static_input,
        config.var_name,
        config.use_evap,
    )
    _validate_extra_channels(train_base, extra_datasets)
    count = sum(len(dataset) for dataset in extra_datasets)
    print(f"[extra data] appended {count} manifest samples")
    return ConcatDataset([train_base, *extra_datasets])


def _validate_extra_channels(train_base, extra_datasets):
    mismatched = [dataset.C for dataset in extra_datasets if dataset.C != train_base.C]
    if mismatched:
        raise ValueError(f"Extra dataset channels {mismatched} do not match normal C={train_base.C}")


def _build_val_dataset(press_root, common, config):
    if not config.use_val:
        return None
    return ParFlowDataset(
        press_root,
        "val",
        config.pre_seq_length,
        config.aft_seq_length,
        use_augment=False,
        **common,
    )


def _create_loaders(train_dataset, val_dataset, test_dataset, config):
    sampler = build_train_sampler(
        train_dataset,
        config.use_time_grouped_sampler,
        config.distributed,
    )
    train_loader = _make_loader(train_dataset, config.batch_size, True, sampler, config)
    val_loader = None
    if config.use_val:
        val_loader = _make_loader(val_dataset, config.val_batch_size, False, None, config)
    test_loader = _make_loader(test_dataset, config.val_batch_size, False, None, config)
    return train_loader, val_loader, test_loader


def _make_loader(dataset, batch_size, is_training, sampler, config):
    loader_config = LoaderConfig(
        batch_size=batch_size,
        is_training=is_training,
        num_workers=config.num_workers,
        distributed=config.distributed,
        drop_last=config.drop_last if is_training else False,
    )
    return create_loader(dataset, loader_config, sampler=sampler)
