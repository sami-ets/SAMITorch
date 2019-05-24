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
from samitorch.configs.model import ModelConfiguration


class AbstractModelFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_model(self, name: str, config: ModelConfiguration):
        pass


class ResNetModelFactory(AbstractModelFactory):

    def __init__(self):
        self._resnet_models = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
        }

    def get_model(self, name: str, config: ModelConfiguration):
        model = self._resnet_models.get(name)
        if not model:
            raise ValueError(name)
        return model(config)
