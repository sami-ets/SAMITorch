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

import abc
import datetime

from samitorch.training.trainer import Trainer
from samitorch.utils.model_io import save


class TrainingStrategy(object):

    def __init__(self, trainer):
        self._trainer = self._register_trainer(trainer)
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def _register_trainer(trainer: Trainer) -> Trainer:
        """Register a trainer to the current training strategy.

        Args:
            trainer (:obj:`Trainer`): A Trainer object.

        Returns:
            Registered trainer.
        """
        assertion_str = "Given trainer is not valid; Must be instance of 'Trainer'."

        assert \
            isinstance(trainer, Trainer), assertion_str

        return trainer


class EarlyStoppingStrategy(TrainingStrategy):
    """Define an early stopping strategy.

    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience: int, trainer: Trainer):
        """Class constructor.

        Args:
            patience (int): The number of events (epochs) to count before triggering an action.
            trainer (:obj:`Trainer`): A trainer on which to trigger an action.
        """
        super(EarlyStoppingStrategy).__init__(trainer)
        self._patience = self._register_patience(patience)
        self._counter = 0
        self._best_score = None

    def __call__(self, metric):
        """Verify if given metric is better than older one. If so, increase a counter. When counter equals to
        desired patience, finalise the training process.

        Args:
            metric (float): An evaluation score.
        """
        if self._best_score is None:
            self._best_score = metric
        elif metric <= self._best_score:
            self._counter += 1
            if self._counter >= self._patience:
                self._trainer.finalize()
        else:
            self._best_score = metric
            self._counter = 0

    @staticmethod
    def _register_patience(patience: int) -> int:
        """Register a patience to the current training strategy.

        Args:
            patience (int): The number of events to wait before stopping the training process.

        Returns:
            Registered patience.
        """
        assertion_str = "Given patience is not valid; Must be instance of 'int'."
        assertion_str_greater_or_equal = "Given patience is not valid; Must be greater or equal to 1."

        assert isinstance(patience, int), assertion_str
        assert patience >= 1, assertion_str_greater_or_equal

        return patience


class CheckpointStrategy(TrainingStrategy):
    """Define a checkpoint strategy.

    Checkpoint strategy is declared to periodically save a Trainer's model(s).

    """

    def __init__(self, trainer):
        """Class constructor.

        Args:
            trainer (:obj:`Trainer`): A trainer.
        """
        super(CheckpointStrategy).__init__(trainer)

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Call method.

        Args:
            *args:
            **kwargs:

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()


class CheckpointStrategyScore(CheckpointStrategy):
    """Define a checkpoint strategy based on an evaluation score.

    This strategy checks if the metric in parameter is best seen. If so, save the model. If not, simply pass.
    """

    def __init__(self, trainer):
        super(CheckpointStrategy).__init__(trainer)
        self._best_score = None

    def __call__(self, metric):
        """Verify if the given metric in parameter is better than an older one. If so, save the model.

        Args:
            metric (float): An evalution score.
        """
        if self._best_score is None:
            self._best_score = metric
        elif metric <= self._best_score:
            pass
        else:
            self._best_score = metric
            time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            save("model-{}.pytorch".format(time), self._trainer.configs.model, self._trainer.epoch,
                 self._trainer.configs.optimizer)
