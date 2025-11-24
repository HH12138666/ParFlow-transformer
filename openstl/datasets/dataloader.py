def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    patch_h = kwargs.get('patch_h', None)
    patch_w = kwargs.get('patch_w', None)
    patching = patch_h is not None and patch_w is not None
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 9),
        aft_seq_length=kwargs.get('aft_seq_length', 1),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
        patch_h=patch_h,
        patch_w=patch_w,
        patch_stride_h=kwargs.get('patch_stride_h', None),
        patch_stride_w=kwargs.get('patch_stride_w', None),
        random_patch=kwargs.get('random_patch', False),
        return_patch_coords=kwargs.get('return_patch_coords', patching),
    )
    if dataname == 'parflow':
        from .dataloader_parflow import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
