"""DataLoader factory for ParFlow datasets."""

from torch.utils.data import ConcatDataset

from openstl.datasets.utils import create_loader

from .constants import DEFAULT_VAL_RATIO_IN_HOLDOUT
from .dataset import ParFlowDataset
from .manifest import build_extra_train_datasets
from .paths import normalize_years, prepare_press_evap_files, resolve_parflow_roots
from .sampler import build_train_sampler


def load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length=6, aft_seq_length=6,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              space_h=None, space_w=None, space_stride_h=None, space_stride_w=None,
              out_channels=None, split_mode="ratio", train_years=None, holdout_years=None,
              val_ratio_in_holdout=DEFAULT_VAL_RATIO_IN_HOLDOUT, var_name="press",
              use_evap=True, use_static_input=True, stats_path=None, use_val=False,
              use_extra_data=False, extra_manifest_path=None, extra_data_root=None,
              use_time_grouped_sampler=True):
    use_static = bool(use_static_input)
    press_root, evap_root, static_root = resolve_parflow_roots(data_root, use_static, var_name, use_evap)
    press_files, evap_files = _prepare_all_files(press_root, evap_root, split_mode, train_years, holdout_years)
    common_kwargs = _common_dataset_kwargs(
        space_h, space_w, space_stride_h, space_stride_w, evap_root, static_root,
        out_channels, press_files, evap_files, split_mode, train_years, holdout_years,
        val_ratio_in_holdout, use_val, stats_path,
    )
    train_base_ds = ParFlowDataset(
        press_root, "train", pre_seq_length, aft_seq_length, use_augment=use_augment, **common_kwargs
    )
    train_ds = _maybe_append_extra_data(
        train_base_ds, common_kwargs, pre_seq_length, aft_seq_length, use_augment,
        use_extra_data, extra_manifest_path, extra_data_root, use_static, var_name, use_evap,
    )
    val_ds = _build_val_dataset(press_root, pre_seq_length, aft_seq_length, use_val, common_kwargs)
    test_ds = ParFlowDataset(press_root, "test", pre_seq_length, aft_seq_length, use_augment=False, **common_kwargs)
    return _create_loaders(
        train_ds, val_ds, test_ds, train_base_ds.C, batch_size, val_batch_size, num_workers,
        distributed, use_prefetcher, drop_last, use_val, use_time_grouped_sampler,
    )


def _prepare_all_files(press_root, evap_root, split_mode, train_years, holdout_years):
    allowed_years = None
    if str(split_mode).lower() == "year":
        train_year_list = normalize_years(train_years) or []
        holdout_year_list = normalize_years(holdout_years) or []
        allowed_years = sorted(set(train_year_list + holdout_year_list))
    return prepare_press_evap_files(press_root, evap_root, allowed_years=allowed_years)


def _common_dataset_kwargs(space_h, space_w, space_stride_h, space_stride_w, evap_root, static_root,
                           out_channels, press_files, evap_files, split_mode, train_years,
                           holdout_years, val_ratio, use_val, stats_path):
    return dict(
        space_h=space_h,
        space_w=space_w,
        space_stride_h=space_stride_h,
        space_stride_w=space_stride_w,
        evap_root=evap_root,
        static_root=static_root,
        out_channels=out_channels,
        press_files=press_files,
        evap_files=evap_files,
        split_mode=split_mode,
        train_years=train_years,
        holdout_years=holdout_years,
        val_ratio_in_holdout=val_ratio,
        use_val=use_val,
        stats_path=stats_path,
    )


def _maybe_append_extra_data(train_base_ds, common_kwargs, pre, aft, use_augment, use_extra,
                             manifest_path, extra_root, use_static, var_name, use_evap):
    if not use_extra:
        return train_base_ds
    extra_kwargs = dict(common_kwargs)
    extra_kwargs.update({"pre": pre, "aft": aft, "use_augment": use_augment})
    extra_datasets = build_extra_train_datasets(
        extra_kwargs, manifest_path, extra_root, use_static, var_name, use_evap
    )
    _validate_extra_channels(train_base_ds, extra_datasets)
    print(f"[extra data] appended {sum(len(dataset) for dataset in extra_datasets)} manifest samples")
    return ConcatDataset([train_base_ds] + extra_datasets)


def _validate_extra_channels(train_base_ds, extra_datasets):
    for extra_ds in extra_datasets:
        if extra_ds.C != train_base_ds.C:
            raise ValueError(f"Extra dataset C={extra_ds.C} does not match normal C={train_base_ds.C}")


def _build_val_dataset(press_root, pre, aft, use_val, common_kwargs):
    if not use_val:
        return None
    return ParFlowDataset(press_root, "val", pre, aft, use_augment=False, **common_kwargs)


def _create_loaders(train_ds, val_ds, test_ds, input_channels, batch_size, val_batch_size, num_workers,
                    distributed, use_prefetcher, drop_last, use_val, use_time_grouped_sampler):
    train_sampler = build_train_sampler(train_ds, use_time_grouped_sampler, distributed)
    train_loader = _make_loader(
        train_ds, batch_size, True, train_sampler, num_workers, distributed,
        use_prefetcher, input_channels, drop_last,
    )
    val_loader = None
    if use_val:
        val_loader = _make_loader(
            val_ds, val_batch_size, False, None, num_workers, distributed,
            use_prefetcher, input_channels, False,
        )
    test_loader = _make_loader(
        test_ds, val_batch_size, False, None, num_workers, distributed,
        use_prefetcher, input_channels, False,
    )
    return train_loader, val_loader, test_loader


def _make_loader(dataset, batch_size, is_training, sampler, num_workers, distributed,
                 use_prefetcher, input_channels, drop_last):
    return create_loader(
        dataset,
        batch_size=batch_size,
        is_training=is_training,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
        input_channels=input_channels,
        drop_last=drop_last,
        persistent_workers=True,
    )
