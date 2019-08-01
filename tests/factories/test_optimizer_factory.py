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

from samitorch.optimizers.optimizers import OptimizerFactory, Optimizer
from samitorch.models.unet3d import UNetModel, UNet3DModelFactory
from samitorch.parsers.parsers import ModelConfigurationParserFactory


class IncorrectMetrics(Enum):
    Incorrect = "Incorrect"


class ModelFactoryTest(unittest.TestCase):
    UNET_CONFIGURATION_PATH = "samitorch/configs/unet3d.yaml"

    def setUp(self):
        self.optimizer_factory = OptimizerFactory()
        self.configurationParserFactory = ModelConfigurationParserFactory()
        self.unet_config = self.configurationParserFactory.parse(self.UNET_CONFIGURATION_PATH)

    def test_should_instantiate_optimizer(self):
        model = UNet3DModelFactory().create_model(UNetModel.UNet3D, self.unet_config)
        optim = self.optimizer_factory.create(Optimizer.SGD, model.parameters(), lr=0.01)
        assert_that(optim, is_not(None))
