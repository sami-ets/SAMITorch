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
import torch


class AbstractLayerFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_layer(self, *args):
        pass

    @abc.abstractmethod
    def register(self, function: str, creator: torch.nn.Module):
        pass


class ActivationFunctionsFactory(AbstractLayerFactory):

    def __init__(self):
        self._activation_functions = {
            "ReLU": torch.nn.ReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            'PReLU': torch.nn.PReLU
        }

    def get_layer(self, function: str, *kwargs):
        activation_function = self._activation_functions.get(function)
        if not activation_function:
            raise ValueError(function)
        return activation_function(*kwargs)

    def register(self, function: str, creator: torch.nn.Module):
        self._activation_functions[function] = creator


class PaddingFactory(AbstractLayerFactory):

    def __init__(self):
        self._padding_strategies = {
            "ReplicationPad3d": torch.nn.ReplicationPad3d
        }

    def get_layer(self, strategy: str, dims: tuple):
        padding = self._padding_strategies.get(strategy)
        if not padding:
            raise ValueError(strategy)
        return padding(dims)

    def register(self, strategy: str, creator: torch.nn.Module):
        self._padding_strategies[strategy] = creator


class PoolingFactory(AbstractLayerFactory):

    def __init__(self):
        self._pooling_strategies = {
            "MaxPool3d": torch.nn.MaxPool3d,
            "AvgPool3d": torch.nn.AvgPool3d,
            "Conv3d": torch.nn.Conv3d
        }

    def get_layer(self, strategy: str, kernel_size: int, stride: int = None, *kwargs):
        pooling = self._pooling_strategies.get(strategy)
        if not pooling:
            raise ValueError(strategy)

        if not "Conv3d" in strategy:
            return pooling(kernel_size, stride)

        else:
            return pooling(*kwargs, kernel_size, stride)

    def register(self, strategy: str, creator: torch.nn.Module):
        self._pooling_strategies[strategy] = creator


class NormalizationLayerFactory(AbstractLayerFactory):

    def __init__(self):
        self._normalization_strategies = {
            "GroupNorm": torch.nn.GroupNorm,
            "BatchNorm3d": torch.nn.BatchNorm3d
        }

    def get_layer(self, strategy: str, *kwargs):
        norm = self._normalization_strategies.get(strategy)
        if not norm:
            raise ValueError(strategy)
        return norm(*kwargs)

    def register(self, strategy: str, creator: torch.nn.Module):
        self._normalization_strategies[strategy] = creator
