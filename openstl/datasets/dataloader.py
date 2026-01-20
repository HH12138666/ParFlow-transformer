def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    model_cfg = kwargs.get('model_config', {})
    out_channels = kwargs.get('out_channels', None)
    if out_channels is None and isinstance(model_cfg, dict):
        out_channels = model_cfg.get('out_channels')
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 12),
        aft_seq_length=kwargs.get('aft_seq_length', 12),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
        space_h=kwargs.get('space_h', None),
        space_w=kwargs.get('space_w', None),
        space_stride_h=kwargs.get('space_stride_h', None),
        space_stride_w=kwargs.get('space_stride_w', None),
        out_channels=out_channels,
        static_data=kwargs.get('static_data', None),
        align_by_hour_id=kwargs.get('align_by_hour_id', True),
    )
    if dataname == 'parflow':
        from .dataloader_parflow import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
