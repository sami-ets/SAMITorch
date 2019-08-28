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

import torch


class ActivationLayers(Enum):
    ReLU = "ReLU"
    LeakyReLU = "LeakyReLU"
    PReLU = "PReLU"

    def __str__(self):
        return self.value


class PaddingLayers(Enum):
    ReplicationPad3d = "ReplicationPad3d"

    def __str__(self):
        return self.value


class PoolingLayers(Enum):
    MaxPool3d = "MaxPool3d"
    AvgPool3d = "AvgPool3d"
    Conv3d = "Conv3d"

    def __str__(self):
        return self.value


class NormalizationLayers(Enum):
    GroupNorm = "GroupNorm"
    BatchNorm3d = "BatchNorm3d"

    def __str__(self):
        return self.value


class AbstractLayerFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, function: str, creator: torch.nn.Module):
        raise NotImplementedError


class ActivationLayerFactory(AbstractLayerFactory):
    """
    Object to instantiate an activation layer.
    """

    def __init__(self):
        super(ActivationLayerFactory, self).__init__()

        self._activation_functions = {
            "ReLU": torch.nn.ReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            'PReLU': torch.nn.PReLU
        }

    def create(self, function: Union[str, ActivationLayers], *args, **kwargs):
        """
        Instantiate an activation layer based on its name.

        Args:
            function (str): The activation layer's name.
            *args: Optional arguments for the respective activation function.

        Returns:
            :obj:`torch.nn.Module`: The activation layer.

        Raises:
            KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        activation_function = self._activation_functions[
            str(function) if isinstance(function, ActivationLayers) else function]
        return activation_function(*args, **kwargs)

    def register(self, function: str, creator: torch.nn.Module):
        """
        Add a new activation layer.

        Args:
            function (str): Activation layer name.
            creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom activation function.
        """
        self._activation_functions[function] = creator


class PaddingLayerFactory(AbstractLayerFactory):
    """
    Object to instantiate a padding layer.
    """

    def __init__(self):
        super(PaddingLayerFactory, self).__init__()
        self._padding_strategies = {
            "ReplicationPad3d": torch.nn.ReplicationPad3d
        }

    def create(self, strategy: Union[str, PaddingLayers], dims: tuple, *args, **kwargs):
        """
        Instantiate a new padding layer.

        Args:
            strategy (:obj:samitorch.models.layers.PaddingLayers): The padding strategy.
            dims (tuple): The number of  where to apply padding.
            *args: Optional arguments for the respective activation function.

        Returns:
            :obj:`torch.nn.Module`: The padding layer.

        Raises:
            KeyError: Raises KeyError Exception if Padding Function is not found.
        """
        padding = self._padding_strategies[str(strategy) if isinstance(strategy, PaddingLayers) else strategy]
        return padding(dims, *args, **kwargs)

    def register(self, strategy: str, creator: torch.nn.Module):
        """
        Add a new padding strategy.

        Args:
            strategy (str): The padding strategy name.
            creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom padding layer.
        """
        self._padding_strategies[strategy] = creator


class PoolingLayerFactory(AbstractLayerFactory):
    """
    An object to instantiate pooling layers.
    """

    def __init__(self):
        super(PoolingLayerFactory, self).__init__()
        self._pooling_strategies = {
            "MaxPool3d": torch.nn.MaxPool3d,
            "AvgPool3d": torch.nn.AvgPool3d,
            "Conv3d": torch.nn.Conv3d
        }

    def create(self, strategy: Union[str, PoolingLayers], kernel_size: int, stride: int = None, *args,
               **kwargs):
        """
        Instantiate a new pooling layer with mandatory parameters.

        Args:
            strategy (str): The pooling strategy name.
            kernel_size (int): Pooling kernel's size.
            stride (int): The size of the stride of the window.
            *kwargs: Optional keywords.

        Returns:
            :obj:`torch.nn.Module`: The pooling layer.

        Raises:
            KeyError: Raises KeyError Exception if Pooling Function is not found.
        """
        pooling = self._pooling_strategies[str(strategy) if isinstance(strategy, PoolingLayers) else strategy]

        if not "Conv3d" in str(strategy):
            return pooling(kernel_size, stride, *args, **kwargs)

        else:
            return pooling(*args, kernel_size, stride, **kwargs)

    def register(self, strategy: str, creator: torch.nn.Module):
        """
        Add a new pooling layer.

        Args:
            strategy (str): The pooling strategy name.
            creator (obj:`torch.nn.Module`): A torch module object wrapping the new custom pooling layer.
        """
        self._pooling_strategies[strategy] = creator


class NormalizationLayerFactory(AbstractLayerFactory):
    """
    An object to instantiate normalization layers.
    """

    def __init__(self):
        super(NormalizationLayerFactory, self).__init__()
        self._normalization_strategies = {
            "GroupNorm": torch.nn.GroupNorm,
            "BatchNorm3d": torch.nn.BatchNorm3d
        }

    def create(self, strategy: Union[str, NormalizationLayers], *args, **kwargs):
        """
        Instantiate a new normalization layer.

        Args:
            strategy (str_or_Enum): The normalization strategy layer.
            *args: Optional keywords arguments.

        Returns:
            :obj:`torch.nn.Module`: The normalization layer.

        Raises:
            KeyError: Raises KeyError Exception if Normalization Function is not found.
        """
        norm = self._normalization_strategies[
            strategy.name if isinstance(strategy, NormalizationLayers) else strategy]
        return norm(*args, **kwargs)

    def register(self, strategy: str, creator: torch.nn.Module):
        """
        Add a new normalization layer.

        Args:
            strategy (str): The normalization strategy name.
            creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom normalization layer.
        """
        self._normalization_strategies[strategy] = creator
