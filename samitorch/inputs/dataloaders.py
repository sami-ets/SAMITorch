# -*- coding: utf-8 -*-
# Copyright 2019 SAMITorch Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import numpy as np

from typing import Union, Tuple
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class DataLoader(torch.utils.data.DataLoader):
    """
    Base class for all data loaders.
    """

    def __init__(self, dataset: torch.utils.data.dataset.Dataset, batch_size: int, shuffle: bool,
                 validation_split: Union[int, float], num_workers: int, collate_fn=default_collate,
                 samplers: Union[tuple, list] = None):
        """
        DataLoader initializer.

        Args:
            dataset (:obj:`torch.utils.data.dataset.Dataset`): A PyTorch Dataset object which contains the training data.
            batch_size (int): The number of elements to load at each batch.
            shuffle (bool): Whether to shuffle the batch or not.
            validation_split (int_or_float): If int, takes this number of elements to produce a validation set.
                If float, computes the proportion of the training set to place in a validation set.
            num_workers (int): Number of parallel workers to execute.
            collate_fn (callable): The collate function that merges a list of samples to form a mini-batch.
            samplers: Optional list or tuple containing both a training and validation sampler.
        """
        self._validation_split = validation_split
        self._shuffle = shuffle
        self._batch_idx = 0
        self._n_samples = len(dataset)

        if samplers is None:
            self._sampler, self._valid_sampler = self._split_sampler(self._validation_split)
        else:
            self._sampler, self._valid_sampler = samplers
            self._shuffle = False

        self._init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self._shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available()
        }
        super().__init__(sampler=self._sampler, **self._init_kwargs)

    def _split_sampler(self, split: Union[int, float]) -> Tuple[Union[
                                                                    torch.utils.data.sampler.Sampler, None], Union[
                                                                    torch.utils.data.sampler.Sampler, None]]:
        """
        Split a sampler for training and validation split.

        Args:
            split (int_or_float): If int, takes this number of elements to produce a validation set.
                If float, computes the proportion of the training set to place in a validation set.

        Returns:
            Tuple_of_:obj:`torch.utils.data.sampler.Sampler`: A tuple of samplers for training and validation sets.
        """
        if split == 0.0:
            return None, None

        idx_full = np.arange(self._n_samples)

        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self._n_samples, "Validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self._n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Turn off shuffle option which is mutually exclusive with sampler
        self._shuffle = False
        self._n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def get_validation_dataloader(self) -> Union[torch.utils.data.dataloader.DataLoader, None]:
        """
        Get the validation dataloader object based on previously created validation sampler.

        Returns:
            :obj:`torch.utils.data.dataloader.Dataloader`: A PyTorch DataLoader with the validation sampler (if exists),
                else returns None.
        """
        if self._valid_sampler is None:
            return None
        else:
            return torch.utils.data.dataloader.DataLoader(sampler=self._valid_sampler, **self._init_kwargs)
