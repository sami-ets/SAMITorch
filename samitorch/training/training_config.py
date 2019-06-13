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

"""Utility class to declare attributes a Trainer object must have for training a model.
"""

import torch
from torch.utils import data
from typing import Union, List
from samitorch.configs.configurations import RunningConfiguration


class TrainingConfig(object):

    def __init__(self, checkpoint_every: int, max_epoch: int, criterion: Union[List[torch.nn.Module], torch.nn.Module],
                 metric,
                 model: Union[List[torch.nn.Module], torch.nn.Module],
                 optimizer: Union[List[torch.nn.Module], torch.nn.Module],
                 dataloader: Union[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader],
                 running_config: RunningConfiguration) -> None:
        super(TrainingConfig, self).__init__()
        self._checkpoint_every = checkpoint_every
        self._max_epoch = max_epoch
        self._criterion = criterion
        self._metric = metric
        self._model = model
        self._optimizer = optimizer
        self._dataloader = dataloader
        self._running_config = running_config

    @property
    def checkpoint_every(self) -> int:
        """
        int: The frequency (in epoch) at which we should save a model checkpoint.
        """
        return self._checkpoint_every

    @property
    def max_epoch(self) -> int:
        """
        int: The max number of epoch.
        """
        return self._max_epoch

    @property
    def criterion(self):
        """
        :obj:`torch.nn.Module`: A PyTorch loss function. Can be a custom made function.
        """
        return self._criterion

    @property
    def metric(self):
        """
        :obj:`torch.nn.Module`: A PyTorch metric. Can be a custom made metric.
        """
        return self._metric

    @property
    def model(self):
        """
        :obj:`torch.nn.Module`: A PyTorch model implementing the `forward` method.
        """
        return self._model

    @property
    def optimizer(self):
        """
        :obj:`torch.nn.Module`: A PyTorch optimizer.
        """
        return self._optimizer

    @property
    def dataloader(self):
        """
        :obj:`torch.utils.data.DataLoader`: A PyTorch dataloader object.
        """
        return self._dataloader

    @property
    def running_config(self):
        """
        The running configuration, which groups multiple execution environment related variables.

        Returns:
            :obj:`samitorch.configs.configurations.RunningConfiguration`: The running configuration.
        """
        return self._running_config