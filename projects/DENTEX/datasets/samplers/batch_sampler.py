# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
from torch.utils.data import BatchSampler, Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class InstanceCountBatchSampler(BatchSampler):

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
        max_instances: int = 50,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last        
        
        self.max_instances = max_instances
        self.instance_counts = np.array([])
        self.buckets = []

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            instance_count = len(data_info['instances'])
            
            fits = (self.instance_counts + instance_count) <= self.max_instances
            if fits.shape[0] == 0 or not np.any(fits):
                bucket_idx = len(self.buckets)
                self.instance_counts = np.concatenate((
                    self.instance_counts, np.array([instance_count]),
                ))
                self.buckets.append([idx])
            else:
                bucket_idx = fits.nonzero()[0][-1]
                self.instance_counts[bucket_idx] += instance_count
                self.buckets[bucket_idx].append(idx)

            idxs = np.argsort(self.instance_counts)
            self.instance_counts = self.instance_counts[idxs]
            self.buckets = [self.buckets[idx] for idx in idxs]
            bucket_idx = (idxs == bucket_idx).argmax()

            # yield a batch of indices in a group with less than maximum instances
            if len(self.buckets[bucket_idx]) == self.batch_size:
                yield self.buckets[bucket_idx]
                self.instance_counts = np.concatenate((
                    self.instance_counts[:bucket_idx],
                    self.instance_counts[bucket_idx + 1:],
                ))
                self.buckets = self.buckets[:bucket_idx] + self.buckets[bucket_idx + 1:]

        # yield the rest data and reset the bucket
        left_data = [idx for idxs in self.buckets for idx in idxs]
        self.instance_counts = np.array([])
        self.buckets = []
        while len(left_data) > 0:
            if not self.drop_last:
                yield left_data[:1]
                left_data = left_data[1:]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
