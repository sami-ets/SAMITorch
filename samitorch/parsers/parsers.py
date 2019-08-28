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
import logging

import yaml

from samitorch.configs.configurations import UNetModelConfiguration, ResNetModelConfiguration, ModelConfiguration


class ModelConfigurationParser(object):
    LOGGER = logging.getLogger("ModelConfigurationParser")

    def __init__(self) -> None:
        pass

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
                return config["model"]["params"]
            except yaml.YAMLError as e:
                ModelConfigurationParser.LOGGER.warning(
                    "Unable to read the configuration file: {} with error {}".format(path, e))
