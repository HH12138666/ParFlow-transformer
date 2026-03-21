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


def _resolve_pad_to_patch(kwargs, model_cfg):
    """Preserve existing precedence: explicit True in kwargs wins."""
    pad_to_patch = kwargs.get('pad_to_patch', False)
    if not pad_to_patch and isinstance(model_cfg, dict):
        pad_to_patch = model_cfg.get('pad_to_patch', False)
    return pad_to_patch


def _build_parflow_loader_cfg(kwargs, model_cfg, dist):
    direct_defaults = {
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        'space_h': None,
        'space_w': None,
        'space_stride_h': None,
        'space_stride_w': None,
        'static_data': None,
        'align_by_hour_id': True,
    }
    cfg = _collect_kwargs(kwargs, direct_defaults)
    cfg['distributed'] = dist

    cfg['out_channels'] = _get_cfg_value(kwargs, model_cfg, 'out_channels', None)
    cfg['patch_size'] = _get_cfg_value(kwargs, model_cfg, 'patch_size', None)
    cfg['pad_to_patch'] = _resolve_pad_to_patch(kwargs, model_cfg)

    cfg['split_mode'] = _get_cfg_value(kwargs, model_cfg, 'split_mode', 'ratio')
    cfg['train_years'] = _get_cfg_value(kwargs, model_cfg, 'train_years', None)
    cfg['holdout_years'] = _get_cfg_value(kwargs, model_cfg, 'holdout_years', None)
    cfg['val_ratio_in_holdout'] = _get_cfg_value(kwargs, model_cfg, 'val_ratio_in_holdout', 0.25)
    cfg['var_name'] = _get_cfg_value(kwargs, model_cfg, 'var_name', 'press')
    cfg['use_evap'] = _get_cfg_value(kwargs, model_cfg, 'use_evap', True)
    cfg['use_static_input'] = _get_cfg_value(kwargs, model_cfg, 'use_static_input', True)
    return cfg


def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    model_cfg = kwargs.get('model_config', {})
    cfg_dataloader = _build_parflow_loader_cfg(kwargs, model_cfg, dist)

    if dataname != 'parflow':
        raise ValueError(f'Dataname {dataname} is unsupported')

    from .dataloader_parflow import load_data as load_data_parflow
    return load_data_parflow(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
