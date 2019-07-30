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

"""A base callback class.

Declare methods a Callback must have.
"""
import abc
from datetime import datetime

from samitorch.utils.model_io import save


class Callback(object):
    """Implements abstract callback interface.

    All callbacks should be derived from this class

    See Also:
        class:`Trainer`
    """

    @abc.abstractmethod
    def on_epoch_begin(self, *args, **kwargs):
        """Function which will be executed at begin of each epoch

        Args:
            **kwargs: additional keyword arguments

        Returns:
            dict: modified trainer attributes, where the name must correspond to the trainer's attribute name
        """
        raise NotImplementedError

    @abc.abstractmethod
    def on_epoch_end(self, *args, **kwargs):
        """Function which will be executed at end of each epoch

        Args:
            **kwargs: additional keyword arguments

        Returns:
            dict: modified trainer attributes, where the name must correspond to the trainer's attribute name
        """
        raise NotImplementedError


class LossCheckpointCallback(Callback):
    """Define a checkpoint strategy based on a loss value.

    This strategy checks if the loss in parameter is best seen. If so, save the model. If not, simply pass.
    """

    def __init__(self, model_name: str):
        self._best_score = None
        self._model_name = model_name

    def on_epoch_begin(self):
        raise NotImplementedError

    def on_epoch_end(self, epoch_num, loss, model, optimizer):
        """ Verify if the given loss in parameter is better than an older one. If so, save the model.

            Args: loss (float): A loss value.
        """
        if self._best_score is None:
            self._best_score = loss
        elif loss <= self._best_score:
            self._best_score = loss
            time = datetime.now().strftime("%Y%m%d-%H%M%S")
            save("{}-{}.tar".format(self._model_name, time), model, epoch_num, optimizer)
        else:
            pass


class MetricCheckpointStrategy(Callback):
    """Define a checkpoint strategy based on an evaluation score.

    This strategy checks if the metric in parameter is best seen. If so, save the model. If not, simply pass.
    """

    def __init__(self, model_name: str):
        self._best_score = None
        self._model_name = model_name

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch_num, metric, model, optimizer):
        """Verify if the given metric in parameter is better than an older one. If so, save the model.

            Args:
                epoch_num: The current epoch number
                metric (float): An evalution score.
                model: The model to save on disk
                optimizer: The oprimizer to save on disk
        """
        if self._best_score is None:
            self._best_score = metric
        elif metric <= self._best_score:
            pass
        else:
            self._best_score = metric
            time = datetime.now().strftime("%Y%m%d-%H%M%S")
            save("{}-{}.tar".format(self._model_name, time), model, epoch_num, optimizer)
