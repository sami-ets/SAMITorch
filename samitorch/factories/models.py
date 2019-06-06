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

from enum import Enum

from samitorch.models.resnet3d import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from samitorch.models.unet3d import UNet3D

from samitorch.configs.model_configurations import ModelConfiguration


class AbstractModelFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_model(self, name: str, config: ModelConfiguration):
        pass


class ModelFactory(AbstractModelFactory):
    """
    Object to instantiate a model.
    """

    def __init__(self):
        self._models = {
            'ResNet18': ResNet18,
            'ResNet34': ResNet34,
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'ResNet152': ResNet152,
            'UNet3D': UNet3D
        }

    def create_model(self, model_name: Enum, config: ModelConfiguration) -> torch.nn.Module:
        """
        Instantiate a new support model.

        Args:
            model_name (str): The model's name (e.g. 'UNet3D').
            config (:obj:`samitorch.configs.model_configuration.ModelConfiguration): An object containing model's
                parameters.

        Returns:
            :obj:`torch.nn.Module`: A PyTorch model.
        """
        model = self._models.get(model_name.name)
        if not model:
            raise ValueError("Model {} is not supported.".format(model_name.name))
        return model(config)
