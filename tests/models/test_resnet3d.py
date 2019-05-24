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

import torch
import unittest

from samitorch.utils.parsers import ResNetYamlConfigurationParser
from samitorch.models.resnet3d import ResNet3D
from samitorch.factories.model_factories import ResNetModelFactory

from tests.models.model_helper_test import TestModelHelper


def load_test_config():
    return ResNetYamlConfigurationParser.parse("samitorch/configs/resnet3d.yaml")


class ResNet3DTest(unittest.TestCase):

    def setUp(self):
        self.config = load_test_config()
        self.input = torch.rand((2, 1, 32, 32, 32))
        self.y = torch.randint(low=0, high=2, size=(2, 1)).float()
        self.factory = ResNetModelFactory()

    def test_model_should_be_created_with_config(self):
        model = self.factory.get_model("resnet101", self.config)
        assert isinstance(model, ResNet3D)

    def test_model_should_update_vars(self):
        model = self.factory.get_model("resnet101", self.config)
        helper = TestModelHelper(model, torch.nn.BCEWithLogitsLoss(),
                                 torch.optim.SGD(model.parameters(), lr=0.01))
        helper.assert_vars_change((self.input, self.y))