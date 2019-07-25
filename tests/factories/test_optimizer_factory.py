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

from samitorch.factories.factories import OptimizerFactory
from samitorch.factories.enums import Optimizers, UNetModels
from samitorch.factories.factories import ModelFactory
from samitorch.factories.parsers import ModelConfigurationParserFactory


class IncorrectMetrics(Enum):
    Incorrect = "Incorrect"


class ModelFactoryTest(unittest.TestCase):
    UNET_CONFIGURATION_PATH = "samitorch/configs/unet3d.yaml"

    def setUp(self):
        self.optimizer_factory = OptimizerFactory()
        self.configurationParserFactory = ModelConfigurationParserFactory()
        self.unet_config = self.configurationParserFactory.parse(self.UNET_CONFIGURATION_PATH)

    def test_should_instantiate_optimizer(self):
        model = ModelFactory().create_model(UNetModels.UNet3D, self.unet_config)
        optim = self.optimizer_factory.create_optimizer(Optimizers.SGD, model.parameters(), lr=0.01)
        assert_that(optim, is_not(None))
