"""PyTorch DataLoader helpers used by the ParFlow pipeline."""

from dataclasses import dataclass
from functools import partial
import random
from typing import Callable

import numpy as np
import torch
from timm.data.distributed_sampler import OrderedDistributedSampler


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    is_training: bool
    num_workers: int
    distributed: bool
    drop_last: bool
    pin_memory: bool = False
    persistent_workers: bool = True
    worker_seeding: str = "all"


def worker_init(worker_id, worker_seeding="all"):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None or worker_info.id != worker_id:
        raise RuntimeError(f"Invalid DataLoader worker id: {worker_id}")
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        return
    if worker_seeding not in ("all", "part"):
        raise ValueError(f"Unknown worker_seeding={worker_seeding}")
    if worker_seeding == "all":
        np.random.seed(worker_info.seed % (2**32 - 1))


def create_loader(dataset, config: LoaderConfig, *, sampler=None, collate_fn=None):
    if config.distributed and sampler is not None:
        raise ValueError("A custom sampler cannot be combined with distributed loading.")
    resolved_sampler = _resolve_sampler(
        dataset, config.is_training, config.distributed, sampler
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.is_training and resolved_sampler is None,
        num_workers=config.num_workers,
        sampler=resolved_sampler,
        collate_fn=collate_fn or torch.utils.data.default_collate,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        worker_init_fn=partial(
            worker_init, worker_seeding=config.worker_seeding
        ),
        persistent_workers=config.persistent_workers and config.num_workers > 0,
    )


def _resolve_sampler(dataset, is_training, distributed, sampler):
    if sampler is not None or not distributed:
        return sampler
    if is_training:
        return torch.utils.data.distributed.DistributedSampler(dataset)
    return OrderedDistributedSampler(dataset)
