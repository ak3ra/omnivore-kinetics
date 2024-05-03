import os
from pathlib import Path

import PIL
import scipy.io
import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets.vision import VisionDataset


class OmnivoreKinetics(torchvision.datasets.kinetics.Kinetics):
    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label, video_idx


class ConcatDataLoaderIterator:
    def __init__(self, _obj):
        # Copy necessary data from _obj
        self.data_loaders = _obj.data_loaders
        self.output_keys = _obj.output_keys
        self.max_total_steps = _obj.max_total_steps
        self.epoch = _obj.epoch

        # Construct iterators
        self.step_counter = 0

        self.iterators = [iter(dl) for dl in self.data_loaders]
        self.indices = torch.cat(
            [
                torch.ones(_obj.iterator_lengths[i], dtype=torch.int32) * i
                for i in range(_obj.num_data_loaders)
            ]
        )
        assert self.max_total_steps == len(self.indices)

        if _obj.shuffle:
            g = torch.Generator()
            if self.epoch is not None:
                # Have deterministic behaviour when epoch is set
                g.manual_seed(_obj.seed + self.epoch)
            shuffle_indices = torch.randperm(len(self.indices), generator=g)
            self.indices = self.indices[shuffle_indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step_counter >= self.max_total_steps:
            raise StopIteration

        idx = self.indices[self.step_counter]
        output_key = self.output_keys[idx]
        try:
            batch = next(self.iterators[idx])
        except StopIteration:
            # We cycle over the data_loader to the beginning. This can happen when repeat_factor > 1
            # Take note that in this case we always use same shuffling from same data_loader in an epoch
            self.iterators[idx] = iter(self.data_loaders[idx])
            batch = next(self.iterators[idx])

        self.step_counter += 1
        # Return batch and output_key
        return batch, output_key

    def __len__(self):
        return self.max_total_steps
    

class ConcatDataLoader:
    """
    ConcatDataLoader is used to group data loader objects.
    When user iterate on this object, we will sample random data loader and
    return their batch data with coresponding output_key.
    With repeat_factors, user can do upsampling or downsampling the data loader.

    Args:
        data_loaders: the iterable objects that will be grouped
        output_keys: List of keys that is used to identify the iterable output.
            The list length should be the same as number of data_loaders.
        repeat_factors: List of numbers that represent the upsampling / downsampling factor
            to the coresponding data_loaders. Should have same length as data_loaders.
        shuffle: Boolean that determine whether we should shuffle the ordering of the
            data loaders (default: ``False``)
        seed: the seed for randomness (default: ``42``)
    """

    def __init__(
        self, data_loaders, output_keys, repeat_factors, shuffle=False, seed=42
    ):
        self.data_loaders = data_loaders
        self.output_keys = output_keys
        self.repeat_factors = repeat_factors
        self.shuffle = shuffle
        self.seed = seed
        self.num_data_loaders = len(self.data_loaders)
        assert self.num_data_loaders == len(output_keys)
        assert self.num_data_loaders == len(repeat_factors)

        # The iterator len is adjusted with repeat_factors
        self.iterator_lengths = [
            int(repeat_factors[i] * len(itb)) for i, itb in enumerate(self.data_loaders)
        ]
        self.max_total_steps = sum(self.iterator_lengths)
        self.epoch = None

    def __len__(self):
        return self.max_total_steps

    def __iter__(self):
        return ConcatDataLoaderIterator(self)

    def set_epoch(self, epoch, is_distributed=False):
        # Setting epoch will result in reproducible shuffling
        self.epoch = epoch
        if is_distributed:
            # In distributed mode, we want to call set_epoch for the samplers
            for data_loader in self.data_loaders:
                data_loader.sampler.set_epoch(epoch)