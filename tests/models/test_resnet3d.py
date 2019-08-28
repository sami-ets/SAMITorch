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

import torch

from samitorch.parsers.parsers import ModelConfigurationParser
from samitorch.models.resnet3d import ResNet101, ResNetModel, ResNet3D
from tests.models.model_helper_test import TestModelHelper


class ResNet3DTest(unittest.TestCase):
    CONFIGURATION_PATH = "samitorch/configs/resnet3d.yaml"

    def setUp(self):
        self.configurationParserFactory = ModelConfigurationParser()
        self.config = self.configurationParserFactory.parse(self.CONFIGURATION_PATH)
        self.input = torch.rand((2, 1, 32, 32, 32))
        self.y = torch.randint(low=0, high=2, size=(2, 1)).float()

    def test_model_should_be_created_with_config(self):
        model = ResNet101(self.config)
        assert isinstance(model, ResNet3D)

    def test_model_should_update_vars(self):
        model = ResNet101(self.config)
        helper = TestModelHelper(model, torch.nn.BCEWithLogitsLoss(), torch.optim.SGD(model.parameters(), lr=0.01))
        helper.assert_vars_change((self.input, self.y))
