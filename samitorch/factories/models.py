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

from samitorch.models.resnet3d import resnet18, resnet34, resnet50, resnet101, resnet152
from samitorch.models.unet3d import UNet3D
from samitorch.configs.model_configurations import ModelConfiguration


class AbstractModelFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_model(self, name: str, config: ModelConfiguration):
        pass


class ModelFactory(AbstractModelFactory):
    """
    Object to instantiate a model.
    """
    def __init__(self):
        self._models = {
            'ResNet18': resnet18,
            'ResNet34': resnet34,
            'ResNet50': resnet50,
            'ResNet101': resnet101,
            'ResNet152': resnet152,
            'UNet3d': UNet3D
        }

    def get_model(self, name: str, config: ModelConfiguration):
        """
        Instantiate a new support model.

        Args:
            name (str): The model's name (e.g. 'unet3d').
            config (:obj:`samitorch.configs.model_configuration.ModelConfiguration): An object containing model's
                parameters.

        Returns:
            :obj:`torch.nn.Module`: A PyTorch model.
        """
        model = self._models.get(name)
        if not model:
            raise ValueError(name)
        return model(config)
