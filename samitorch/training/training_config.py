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

import torch.nn


class TrainingConfig(object):

    def __init__(self, checkpoint_every: int, criterion: torch.nn.Module, metric, model: torch.nn.Module,
                 optimizer: torch.nn.Module):
        super(TrainingConfig).__init__()
        self._checkpoint_every = checkpoint_every
        self._criterion = self._register_criterion(criterion)
        self._metric = self._register_metric(metric)
        self._model = self._register_model(model)
        self._optimizer = self._register_optimizer(optimizer)

    @property
    def checkpoint_every(self):
        """int: The frequency (in epoch) at which we should save a model checkpoint."""
        return self._checkpoint_every

    @property
    def criterion(self):
        """:obj:`torch.nn.Module`: A PyTorch loss function. Can be a custom made function."""
        return self._criterion

    @property
    def metric(self):
        """:obj:`torch.nn.Module`: A PyTorch metric. Can be a custom made metric."""
        return self._metric

    @property
    def model(self):
        """:obj:`torch.nn.Module`: A PyTorch model implementing the `forward` method."""
        return self._model

    @property
    def optimizer(self):
        """:obj:`torch.nn.Module`: A PyTorch optimizer."""
        return self._optimizer

    @staticmethod
    def _register_criterion(criterion: torch.nn.Module) -> torch.nn.Module:
        """Register an optimizer to the current training configuration.

        Args:
            criterion (:obj:`torch.nn.Module`): A PyTorch loss function.

        Returns:
            Registered criterion.
        """
        assertion_str = "Given criterion is not valid; Must be instance of 'torch.nn.Module' " \
                        "and implement function 'forward'"

        assert \
            isinstance(criterion, torch.nn.Module) \
            and (hasattr(criterion, "forward")), assertion_str

        return criterion

    @staticmethod
    def _register_metric(metric: torch.nn.Module) -> torch.nn.Module:
        """Register a metric to the current training configuration.

        Args:
            metric (:obj:`torch.nn.Module`): A PyTorch metric that contains `forward` function.

        Returns:
            Registered metric.
        """
        assertion_str = "Given metric is not valid; Must be instance of 'torch.nn.Module' " \
                        "and implement function 'forward'"

        assert \
            isinstance(metric, torch.nn.Module) \
            and (hasattr(metric, "forward")), assertion_str

        return metric

    @staticmethod
    def _register_model(model: torch.nn.Module) -> torch.nn.Module:
        """Register a model to the current training configuration.

        Args:
            model (:obj:`torch.nn.Module`): A PyTorch model.

        Returns:
            Registered model.
        """
        assertion_str = "Given model is not valid; Must be instance of 'torch.nn.Module' or " \
                        "'torch.nn.DataParallel' and implement function 'forward'"

        assert \
            isinstance(model, torch.nn.Module) \
            or isinstance(model, torch.nn.DataParallel) \
            and (hasattr(model, "forward")), assertion_str

        return model

    @staticmethod
    def _register_optimizer(optimizer: torch.nn.Module) -> torch.nn.Module:
        """Register an optimizer to the current training configuration.

        Args:
            optimizer (:obj:`torch.nn.Module`): A PyTorch optimizer.

        Returns:
            Registered optimizer.
        """
        assertion_str = "Given optimizer is not valid; Must be instance of 'torch.nn.Module' " \
                        "and implement function 'forward'"

        assert \
            isinstance(optimizer, torch.nn.Module) \
            and (hasattr(optimizer, "forward")), assertion_str

        return optimizer
