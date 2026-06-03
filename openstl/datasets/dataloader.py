def _get_cfg_value(kwargs, model_cfg, key, default=None):
    value = kwargs.get(key, None)
    if value is None and isinstance(model_cfg, dict):
        value = model_cfg.get(key, default)
    if value is None:
        value = default
    return value


def _collect_kwargs(kwargs, defaults):
    """Collect a subset of kwargs with fallback defaults."""
    return {k: kwargs.get(k, v) for k, v in defaults.items()}


def _build_parflow_loader_cfg(kwargs, model_cfg):
    direct_defaults = {
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        'distributed': False,
        'space_h': None,
        'space_w': None,
        'space_stride_h': None,
        'space_stride_w': None,
        'use_extra_data': False,
        'extra_manifest_path': None,
        'extra_data_root': None,
        'use_val': False,
        'use_time_grouped_sampler': True,
    }
    cfg = _collect_kwargs(kwargs, direct_defaults)
    cfg['out_channels'] = _get_cfg_value(kwargs, model_cfg, 'out_channels', None)

    cfg['split_mode'] = _get_cfg_value(kwargs, model_cfg, 'split_mode', 'ratio')
    cfg['train_years'] = _get_cfg_value(kwargs, model_cfg, 'train_years', None)
    cfg['holdout_years'] = _get_cfg_value(kwargs, model_cfg, 'holdout_years', None)
    cfg['val_ratio_in_holdout'] = _get_cfg_value(kwargs, model_cfg, 'val_ratio_in_holdout', 0.5)
    cfg['var_name'] = _get_cfg_value(kwargs, model_cfg, 'var_name', 'press')
    cfg['use_evap'] = _get_cfg_value(kwargs, model_cfg, 'use_evap', True)
    cfg['use_static_input'] = _get_cfg_value(kwargs, model_cfg, 'use_static_input', True)
    cfg['stats_path'] = _get_cfg_value(kwargs, model_cfg, 'stats_path', None)
    cfg['use_extra_data'] = _get_cfg_value(kwargs, model_cfg, 'use_extra_data', False)
    cfg['extra_manifest_path'] = _get_cfg_value(kwargs, model_cfg, 'extra_manifest_path', None)
    cfg['extra_data_root'] = _get_cfg_value(kwargs, model_cfg, 'extra_data_root', None)
    cfg['use_val'] = _get_cfg_value(kwargs, model_cfg, 'use_val', False)
    cfg['use_time_grouped_sampler'] = _get_cfg_value(kwargs, model_cfg, 'use_time_grouped_sampler', True)
    return cfg


def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    model_cfg = kwargs.get('model_config', {})
    cfg_dataloader = _build_parflow_loader_cfg(kwargs, model_cfg)

    if dataname != 'parflow':
        raise ValueError(f'Dataname {dataname} is unsupported')

    from .dataloader_parflow import load_data as load_data_parflow
    return load_data_parflow(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
