import os

import torch
import torch.distributed as dist


def init_dist(launcher='pytorch', backend='nccl', port=29500):
    """Initialize torch distributed for single-node multi-GPU training."""
    if launcher == 'none':
        return False
    if launcher != 'pytorch':
        raise ValueError(f'Unsupported launcher: {launcher}')

    if dist.is_available() and dist.is_initialized():
        return True

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    os.environ.setdefault('MASTER_PORT', str(port))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size,
    )
    return True


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if not is_dist_avail_and_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def is_main_process():
    rank, _ = get_dist_info()
    return rank == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def destroy_dist():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
