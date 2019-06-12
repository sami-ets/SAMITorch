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
import torch

from enum import Enum


class AbstractOptimizerFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_optimizer(self, function: str, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, function: str, creator):
        raise NotImplementedError


class OptimizerFactory(AbstractOptimizerFactory):

    def __init__(self):
        super(OptimizerFactory, self).__init__()

        self._optimizers = {
            "Adam": torch.optim.Adam,
            "Adagrad": torch.optim.Adagrad,
            "SGD": torch.optim.SGD
        }

    def create_optimizer(self, function: Enum, *args):
        """
        Instanciate an optimizer based on its name.

        Args:
            function (Enum): The optimizer's name.
            *args: Other arguments.

        Returns:
            :obj:`torch.optim.Optimizer`: The optimizer.

        Raises:
            KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        optimizer = self._optimizers[function.name]
        return optimizer

    def register(self, function: str, creator: torch.optim.Optimizer):
        """
        Add a new activation layer.

        Args:
           function (str): Activation layer name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom optimizer function.
        """
        self._optimizers[function] = creator
