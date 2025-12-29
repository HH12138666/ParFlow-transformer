def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 6),
        aft_seq_length=kwargs.get('aft_seq_length', 6),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
        space_h=kwargs.get('space_h', None),
        space_w=kwargs.get('space_w', None),
        space_stride_h=kwargs.get('space_stride_h', None),
        space_stride_w=kwargs.get('space_stride_w', None),
        target_channels=kwargs.get('target_channels', None),
    )
    if dataname == 'parflow':
        from .dataloader_parflow import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
