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

"""A base trainer class.

Declare methods a Trainer object must have.
"""

import abc
import torch

from typing import List, Optional
from samitorch.callbacks.callbacks import Callback
from samitorch.training.trainer_configuration import TrainerConfiguration


class Trainer(object):
    def __init__(self, config: TrainerConfiguration, callbacks: Optional[List[Callback]]):
        """Class initializer.

        Args:
            config (:obj:`samitorch.training.training_config.TrainingConfig`): A TrainingConfig containing training configuration.
            callbacks (:obj:`list` of :obj:`Callback`): A list of Callback objects to register.
        """
        assert config is not None, "Training must at least have a list of one configuration."
        self._config = None

        self._register_config(config)

        if callbacks is not None:
            self._callbacks = list()
            for cbck in callbacks:
                self._register_callback(cbck)
        self._epoch = 0

    @property
    def epoch(self):
        """int: The current epoch count."""
        return self._epoch

    @property
    def config(self):
        """:obj:`list` of :obj:`TrainingConfig`: A list of registered training configuration, one per model."""
        return self._config

    @property
    def callbacks(self):
        """:obj:`list` of :obj:`Callback`: A list of registered callbacks."""
        return self._callbacks

    @abc.abstractmethod
    def train(self):
        """Main training loop.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _train_epoch(self, epoch_num: int, **kwargs):
        """Train a model for one epoch.

        Args:
            epoch_num (int): current epoch number.
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _train_batch(self, data_dict: dict, fold=0, **kwargs):
        """Function which handles prediction from batch, logging, loss calculation and optimizer step.

        This function is needed for the framework to provide a generic trainer function which works with all kind of
        networks and loss functions. The closure function must implement all steps from forwarding, over loss
        calculation, metric calculation, logging, and the actual backpropagation. It is called with an empty
        optimizer-dict to evaluate and should thus work with optional optimizers.

        Args:
            data_dict (dict): dictionary containing the data.
            fold (int): Current Fold in Cross Validation (default: 0).
            **kwargs (dict): additional keyword arguments.

        Returns:
            dict: Metric values (with same keys as input dict metrics).
            dict: Loss values (with same keys as input dict criterions).
            list: Arbitrary number of predictions.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @staticmethod
    def _prepare_batch(batch: dict, input_device: torch.device, output_device: torch.device):
        """Converts a numpy batch of data and labels to suitable datatype and pushes them to correct devices

        Args
            batch (dict): dictionary containing the batch (must have keys 'data' and 'label'
            input_device (:obj:`torch.device`): device for network inputs
            output_device (:obj:`torch.device`): device for network outputs

        Returns:
            dict: dictionary containing all necessary data in right format and type and on the correct device

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_epoch(self, epoch_num: int, **kwargs):
        """Run validation phase.

         Args:
            epoch_num (int): Current epoch number.
            kwargs: keyword arguments

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _finalize(self, *args, **kwargs):
        """Finalize all operations (e.g. save checkpoint, finalize Data Loaders and close logger if required).

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _setup(self, *args, **kwargs):
        """Defines the actual Trainer Setup.

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_training_begin(self, *args, **kwargs):
        """Defines the behaviour at beginnig of the training.

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_training_end(self, *args, **kwargs):
        """Defines the behaviour at the end of the training.

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_epoch_begin(self, *args, **kwargs):
        """Defines the behaviour at beginnig of each epoch.

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_epoch_end(self, *args, **kwargs):
        """
        Defines the behaviour at the end of each epoch.

        Args:
            *args: positional arguments
            **kwargs(dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """

        raise NotImplementedError()

    @staticmethod
    def _is_better_val_scores(old_val_score: float, new_val_score: float, mode='highest'):
        """Check whether the new val score is better than the old one with respect to the optimization goal.

        Args:
            old_val_score (float): old validation score
            new_val_score (float): new validation score
            mode (str): String to specify whether a higher or lower validation score is optimal;
                must be in ['highest', 'lowest']

        Returns:
            bool: True if new score is better, False otherwise.
        """

        assert mode in ['highest', 'lowest'], "Invalid Comparison Mode"

        if mode == 'highest':
            return new_val_score > old_val_score
        elif mode == 'lowest':
            return new_val_score < old_val_score

    def _register_callback(self, callback: Callback):
        """Register Callback to Trainer.

        Args:
            callback (:class:`Callback`): the callback to register

        Raises:
            AssertionError: `callback` is not an instance of :class:`Callback` and has not both methods
            ['at_epoch_begin', 'at_epoch_end'].
        """

        assertion_str = "Given callback is not valid; Must be instance of " \
                        "Callback or provide functions " \
                        "'at_epoch_begin' and 'at_epoch_end'"

        assert \
            isinstance(callback, Callback) \
            or (hasattr(callback, "at_epoch_begin")
                and hasattr(callback, "at_epoch_end")), assertion_str

        self._callbacks.append(callback)

    def _register_config(self, config: TrainerConfiguration):
        """Register a TrainingConfig to Trainer.

        Args:
            config (:obj:`TrainingConfig`): the training configuration to register.

        Raises:
             AssertionError: `config` is not an instance of :class:`TrainingConfig`.

        """
        assertion_str = "Given config is not valid; Must be instance of 'TrainingConfig'"

        assert isinstance(config, TrainerConfiguration), assertion_str

        self._config = config
