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

from typing import Union

from samitorch.losses.losses import DiceLoss

class AbstractCriterionFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_criterion(self, criterion: Union[str, Enum], *args):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, criterion: str, creator):
        raise NotImplementedError


class CriterionFactory(AbstractCriterionFactory):
    def __init__(self):
        super(CriterionFactory, self).__init__()

        self._criterion = {
            "DiceLoss": DiceLoss,
            "Cross_Entropy": torch.nn.functional.cross_entropy
        }

    def create_criterion(self, criterion: Union[str, Enum], *args):
        """
        Instanciate a loss function based on its name.

        Args:
           criterion (str_or_Enum): The criterion's name.
           *args: Other arguments.

        Returns:
           :obj:`torch.nn.Module`: The criterion.

        Raises:
           KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        optimizer = self._criterion[criterion.name if isinstance(criterion, Enum) else criterion]
        return optimizer

    def register(self, function: str, creator):
        """
        Add a new criterion.

        Args:
           function (str): Criterion's name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom criterion function.
        """
        self._criterion[function] = creator
