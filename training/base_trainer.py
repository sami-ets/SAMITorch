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

Declares methods a Trainer object must have.
"""

import abc
from callbacks.base_callback import BaseCallback


class BaseTrainer(object):
    def __init__(self, config, callbacks=[]):
        """Class initializer.

        Args:
            config: a dictionary containing configuration.
            callbacks: list of callbacks to register.
        """
        self._config = config
        self._callbacks = callbacks
        for cbck in callbacks:
            self.register_callback(cbck)

    @abc.abstractmethod
    def train(self):
        """Main training loop.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train_epoch(self, epoch_num, **kwargs):
        """Train a model for one epoch.

        Args:
            *epoch_num: current epoch number.
            **kwargs: keyword arguments.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train_batch(self, data_dict: dict, optimizers: dict, criterions={}, metrics={}, fold=0, **kwargs):
        """Function which handles prediction from batch, logging, loss calculation and optimizer step.

        This function is needed for the framework to provide a generic trainer function which works with all kind of
        networks and loss functions. The closure function must implement all steps from forwarding, over loss
        calculation, metric calculation, logging, and the actual backpropagation. It is called with an empty
        optimizer-dict to evaluate and should thus work with optional optimizers.

        Args:
            model: (:class:`Model`) model to forward data through.
            data_dict: (dict) dictionary containing the data.
            optimizers: (dict) dictionary containing all optimizers to perform parameter update.
            criterions: (dict) Functions or classes to calculate criterions.
            metrics: (dict) Functions or classes to calculate other metrics.
            fold: (int) Current Fold in Crossvalidation (default: 0).
            kwargs : (dict) additional keyword arguments.

        Returns:
            dict: Metric values (with same keys as input dict metrics).
            dict: Loss values (with same keys as input dict criterions).
            list: Arbitrary number of predictions.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """Converts a numpy batch of data and labels to suitable datatype and pushes them to correct devices

        Args
            batch: (dict) dictionary containing the batch (must have keys 'data' and 'label'
            input_device: device for network inputs
            output_device: device for network outputs

        Returns:
            dict: dictionary containing all necessary data in right format and type and on the correct device

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def validate_epoch(self, epoch_num, **kwargs):
        """Run validation phase.

         Args:
            epoch_num: current epoch number.
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def finalize(self, *args, **kwargs):
        """Finalize all operations (e.g. save checkpoint, finalize Data Loaders and close logger if required).

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _setup(self, *args, **kwargs):
        """Defines the actual Trainer Setup.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_training_begin(self, *args, **kwargs):
        """Defines the behaviour at beginnig of the training.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_training_end(self, *args, **kwargs):
        """Defines the behaviour at the end of the training.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_epoch_begin(self, *args, **kwargs):
        """Defines the behaviour at beginnig of each epoch.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

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
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """

        raise NotImplementedError()

    def register_callback(self, callback: BaseCallback):
        """Register Callback to Trainer.

        Args:
            callback: :class:`BaseCallback` the callback to register

        Raises:
            AssertionError: `callback` is not an instance of :class:`AbstractCallback` and has not both methods
            ['at_epoch_begin', 'at_epoch_end'].
        """

        assertion_str = "Given callback is not valid; Must be instance of " \
                        "BasetCallback or provide functions " \
                        "'at_epoch_begin' and 'at_epoch_end'"

        assert isinstance(callback, BaseCallback) or \
               (hasattr(callback, "at_epoch_begin")
                and hasattr(callback, "at_epoch_end")), assertion_str

        self._callbacks.append(callback)

    @staticmethod
    def _is_better_val_scores(old_val_score, new_val_score, mode='highest'):
        """Check whether the new val score is better than the old one with respect to the optimization goal.

        Args:

            old_val_score : old validation score
            new_val_score : new validation score
            mode: String to specify whether a higher or lower validation score is optimal;
                must be in ['highest', 'lowest']

        Returns:
            bool: True if new score is better, False otherwise.
        """

        assert mode in ['highest', 'lowest'], "Invalid Comparison Mode"

        if mode == 'highest':
            return new_val_score > old_val_score
        elif mode == 'lowest':
            return new_val_score < old_val_score
