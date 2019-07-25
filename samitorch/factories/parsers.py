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
import yaml
import logging

from samitorch.configs.configurations import UNetModelConfiguration, ResNetModelConfiguration, ModelConfiguration


class AbstractConfigurationParserFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def parse(self, path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, model_type: str, configuration_class):
        raise NotImplementedError


class ModelConfigurationParserFactory(AbstractConfigurationParserFactory):

    def __init__(self) -> None:
        self._supported_model_configuration = {
            "UNet3D": UNetModelConfiguration,
            "ResNet3D": ResNetModelConfiguration
        }

    def parse(self, path: str) -> ModelConfiguration:
        """
        Parse a model configuration file.

        Args:
            path (str): Configuration YAML file path.

        Returns:
            :obj:`ModelConfiguration`: An object containing model's properties.

        """
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                model_type = config["model"]["type"]
                model_configuration = self._supported_model_configuration.get(model_type)
                return model_configuration(config["model"])
            except ValueError as e:
                logging.error("Model type {} not supported.".format(model_type, e))
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the training config file: {} with error {}".format(path, e))

    def register(self, model_type: str, model_configuration_class: ModelConfiguration) -> None:
        """
        Register a new type of model.

        Args:
            model_type (str): The generic model type (e.g. 'unet3d' or 'resnet3d').
            model_configuration_class (:obj:`samitorch.config.model.ModelConfiguration`): The class defining model's
                properties.
        """
        self._supported_model_configuration[model_type] = model_configuration_class
