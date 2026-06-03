"""Samplers that improve ParFlow spatial-window cache locality."""

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler


class ParFlowTimeGroupedSampler(Sampler):
    """Yield all spatial windows for the same time window close together."""

    def __init__(self, dataset, shuffle_groups=True, shuffle_within_group=True, seed=0):
        self.dataset = dataset
        self.shuffle_groups = bool(shuffle_groups)
        self.shuffle_within_group = bool(shuffle_within_group)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = _build_groups(dataset)
        self.group_keys = list(self.groups.keys())
        self.length = sum(len(indices) for indices in self.groups.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self.group_keys)
        if self.shuffle_groups:
            rng.shuffle(keys)
        for key in keys:
            indices = list(self.groups[key])
            if self.shuffle_within_group:
                rng.shuffle(indices)
            yield from indices
        self.epoch += 1

    def __len__(self):
        return self.length


def _build_groups(dataset):
    if isinstance(dataset, ConcatDataset):
        return _build_concat_groups(dataset)
    return _build_dataset_groups(dataset, offset=0, prefix=0)


def _build_concat_groups(concat_dataset):
    groups = {}
    offset = 0
    for dataset_id, dataset in enumerate(concat_dataset.datasets):
        groups.update(_build_dataset_groups(dataset, offset=offset, prefix=dataset_id))
        offset += len(dataset)
    return groups


def _build_dataset_groups(dataset, offset, prefix):
    if not hasattr(dataset, "sample_indices"):
        raise ValueError("ParFlowTimeGroupedSampler requires datasets with sample_indices.")
    groups = {}
    for local_idx, sample_idx in enumerate(dataset.sample_indices):
        time_idx = sample_idx[0] if isinstance(sample_idx, tuple) else sample_idx
        groups.setdefault((prefix, time_idx), []).append(offset + local_idx)
    return groups


def build_train_sampler(dataset, enabled, distributed):
    if not enabled:
        return None
    if distributed:
        raise ValueError("ParFlowTimeGroupedSampler is not implemented for distributed training.")
    return ParFlowTimeGroupedSampler(dataset, seed=int(torch.initial_seed() % (2 ** 32)))
