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

from enum import Enum

from hamcrest import *

from samitorch.factories.factories import ModelFactory

from samitorch.factories.parsers import ModelConfigurationParserFactory

from samitorch.factories.enums import UNetModels, ResNetModels


class IncorrectModels(Enum):
    Incorrect = "Incorrect"


class ModelFactoryTest(unittest.TestCase):
    RESNET_MODEL = ResNetModels.ResNet18
    RESNET_CONFIGURATION_PATH = "samitorch/configs/resnet3d.yaml"
    UNET_MODEL = UNetModels.UNet3D
    UNET_CONFIGURATION_PATH = "samitorch/configs/unet3d.yaml"
    UNKNOWN_MODEL = IncorrectModels.Incorrect
    INVALID_CONFIG_PATH = "samitorch/configs/invalid.yaml"

    def setUp(self):
        self.model_factory = ModelFactory()
        self.configurationParserFactory = ModelConfigurationParserFactory()
        self.resnet_config = self.configurationParserFactory.parse(self.RESNET_CONFIGURATION_PATH)
        self.unet_config = self.configurationParserFactory.parse(self.UNET_CONFIGURATION_PATH)

    def test_should_instantiate_restnet_model(self):
        model = self.model_factory.create_model(self.RESNET_MODEL, self.resnet_config)
        assert_that(model, is_not(None))

    def test_should_instantiate_unet_model(self):
        model = self.model_factory.create_model(self.UNET_MODEL, self.unet_config)
        assert_that(model, is_not(None))

    def test_should_fail_with_unknown_model(self):
        model_factory = ModelFactory()
        assert_that(calling(model_factory.create_model).with_args(self.UNKNOWN_MODEL, self.resnet_config),
                    raises(ValueError))
