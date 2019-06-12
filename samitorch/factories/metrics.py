#  -*- coding: utf-8 -*-
#  Copyright 2019 SAMITorch Authors. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import abc

from enum import Enum

from typing import Union

from samitorch.metrics.metrics import Dice

from samitorch.factories.enums import Metrics


class AbstractMetricFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_metric(self, function: Union[str, Metrics], *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, function: str, creator):
        raise NotImplementedError


class MetricsFactory(AbstractMetricFactory):
    def __init__(self):
        super(MetricsFactory, self).__init__()

        self._metrics = {
            "Dice": Dice
        }

    def create_metric(self, metric: Union[str, Metrics], *args, **kwargs):
        """
        Instanciate an optimizer based on its name.

        Args:
            metric (str_or_Enum): The optimizer's name.
            *args: Other arguments.

        Returns:
            The metric, a torch or ignite module.

        Raises:
            KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        metric = self._metrics[metric.name if isinstance(metric, Metrics) else metric]
        return metric(*args, **kwargs)

    def register(self, metric: str, creator):
        """
        Add a new activation layer.

        Args:
           metric (str): Metric's name.
           creator: A torch or ignite module object wrapping the new custom metric function.
        """
        self._metrics[metric] = creator
