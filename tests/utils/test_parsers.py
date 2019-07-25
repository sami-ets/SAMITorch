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

import unittest

from hamcrest import *

from samitorch.factories.parsers import ModelConfigurationParserFactory

from samitorch.configs.configurations import UNetModelConfiguration


class ConfigurationParserFactoryTest(unittest.TestCase):
    VALID_CONFIGURATION_PATH = "samitorch/configs/unet3d.yaml"
    INVALID_CONFIGURATION_PATH = "samitorch/configs/invalid.yaml"
    INCORRECT_MODEL_TYPE_CONFIGURATION_PATH = "invalid_model_type_configuration.yaml"
    INCORRECT_MODEL_PROPERTY_CONFIGURATION_PATH = "invalid_model_property_configuration.yaml"

    def setUp(self):
        self.configuration_parser_factory = ModelConfigurationParserFactory()

    def test_instantiating_model_configuration_should_success_with_valid_path(self):
        model_configuration = self.configuration_parser_factory.parse(self.VALID_CONFIGURATION_PATH)
        assert_that(model_configuration, is_not(None))
        assert_that(model_configuration, instance_of(UNetModelConfiguration))

    def test_instantiating_model_configuration_should_fail_with_invalid_path(self):
        assert_that(calling(self.configuration_parser_factory.parse).with_args(self.INVALID_CONFIGURATION_PATH),
                    raises(FileNotFoundError))

    def test_instantiating_model_configuration_should_fail_with_incomplete_yaml_file(self):
        assert_that(
            calling(self.configuration_parser_factory).with_args(self.INCORRECT_MODEL_TYPE_CONFIGURATION_PATH)), raises(
            ValueError)

    def test_instantionating_model_configuration_should_fail_with_incorrect_model_property(self):
        assert_that(
            calling(self.configuration_parser_factory.parse).with_args(
                self.INCORRECT_MODEL_PROPERTY_CONFIGURATION_PATH)), raises(
            KeyError)
