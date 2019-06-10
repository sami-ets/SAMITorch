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

from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class DataLoader(torch.utils.data.DataLoader):
    """
    Base class for all data loaders.
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self._validation_split = validation_split
        self._shuffle = shuffle
        self._batch_idx = 0
        self._n_samples = len(dataset)

        self._sampler, self._valid_sampler = self._split_sampler(self._validation_split)

        self._init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self._shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self._sampler, **self._init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self._n_samples)

        np.random.seed(0)
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

    def split_validation(self):
        if self._valid_sampler is None:
            return None
        else:
            return torch.utils.data.dataloader.DataLoader(sampler=self._valid_sampler, **self._init_kwargs)
