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


import yaml
import logging

from samitorch.configs.model import UNetModelConfiguration, ResNetModelConfiguration


class UNetYamlConfigurationParser(object):

    @staticmethod
    def parse(training_config_file_path: str):
        with open(training_config_file_path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                return UNetModelConfiguration(config['model'])
            except yaml.YAMLError as e:
                logging.warning(
                    "Unable to read the training config file: {} with error {}".format(training_config_file_path, e))


class ResNetYamlConfigurationParser(object):

    @staticmethod
    def parse(training_config_file_path: str):
        with open(training_config_file_path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                return ResNetModelConfiguration(config['model'])
            except yaml.YAMLError as e:
                logging.warning(
                    "Unable to read the training config file: {} with error {}".format(training_config_file_path, e))
