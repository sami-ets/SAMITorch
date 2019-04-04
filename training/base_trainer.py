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

from abc import abstractmethod
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

    @abstractmethod
    def train(self, *args, **kwargs):
        """Main training loop.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        """Train a model for one epoch.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def validate(self, *args, **kwargs):
        """Run validation phase.

         Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, *args, **kwargs):
        """Finalize all operations (e.g. save checkpoint, finalize Data Loaders and close logger if required).

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def _setup(self, *args, **kwargs):
        """Defines the actual Trainer Setup.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def _at_training_begin(self, *args, **kwargs):
        """Defines the behaviour at beginnig of the training.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def _at_training_end(self, *args, **kwargs):
        """Defines the behaviour at the end of the training.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def _at_epoch_begin(self, *args, **kwargs):
        """Defines the behaviour at beginnig of each epoch.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
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
