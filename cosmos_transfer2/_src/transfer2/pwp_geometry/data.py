# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PwP data sampling: a new video order every epoch and an independent mask draw for every (epoch, video).

The original training path repeats itself:
- the trainer never calls ``DistributedSampler.set_epoch``, so every epoch visits the videos in the same order;
- ``MaskPoolSampler`` is seeded with ``mask_pool_sampler_seed + worker_id`` only, so every GPU draws the same mask
  sequence, and because the dataloader workers restart every epoch, every epoch repeats that sequence.

Here the sampler reshuffles on every pass and hands the epoch to the dataset together with each index, and the
dataset seeds the mask draw from ``(mask_pool_sampler_seed, epoch, video index)``. GPUs read different videos, so
they draw different masks; the same (epoch, video) always gets the same mask, whatever the worker or restart.
"""

import random

import torch.distributed as dist
from torch.utils.data import DistributedSampler

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.local_datasets.singleview_dataset_mask import (
    SingleViewTransferDatasetMask,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.anyview_masks import mask_seed


class PwPEpochShuffleSampler(DistributedSampler):
    """``DistributedSampler`` over the data-parallel ranks that moves to the next epoch on every pass.

    With ``yield_epoch`` it yields ``(index, epoch)`` for ``PwPSingleViewTransferDatasetMask``.
    """

    def __init__(self, dataset, seed: int = 0, yield_epoch: bool = False) -> None:
        if dist.is_available() and dist.is_initialized():
            from megatron.core import parallel_state

            num_replicas = parallel_state.get_data_parallel_world_size()
            rank = parallel_state.get_data_parallel_rank()
        else:
            num_replicas, rank = 1, 0
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=int(seed))
        self.yield_epoch = yield_epoch
        self._next_epoch = 0
        log.info(
            f"PwP sampler: reshuffle every epoch, seed={self.seed}, rank {rank}/{num_replicas}, "
            f"{self.num_samples} videos per rank per epoch, mask seed per (epoch, video)={yield_epoch}"
        )

    def __iter__(self):
        # The trainer creates a new dataloader iterator for every pass and never calls set_epoch.
        self.set_epoch(self._next_epoch)
        self._next_epoch += 1
        indices = super().__iter__()
        if not self.yield_epoch:
            return indices
        return ((index, self.epoch) for index in indices)


class PwPSingleViewTransferDatasetMask(SingleViewTransferDatasetMask):
    """Mask-pool dataset whose mask draw depends on (mask_pool_sampler_seed, epoch, video index) only."""

    _pwp_mask_seed: int = 0

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        self._pwp_mask_seed = mask_seed(self.mask_pool_sampler_seed, epoch, index)
        return super().__getitem__(index)

    def _sample_dynamic_mask(self):
        sampler = self._get_dynamic_mask_sampler()
        if not isinstance(getattr(sampler, "rng", None), random.Random):
            raise TypeError(f"{type(sampler).__name__} has no random.Random 'rng' attribute to seed per sample")
        sampler.rng.seed(self._pwp_mask_seed)
        return super()._sample_dynamic_mask()


__all__ = ["PwPEpochShuffleSampler", "PwPSingleViewTransferDatasetMask", "mask_seed"]
