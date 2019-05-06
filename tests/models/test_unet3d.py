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
import yaml
import unittest

from models.unet3d import UNet3D, SingleConv


def load_test_config():
    return yaml.load(open("configs/unet3d.yaml"))


class Unet3DTest(unittest.TestCase):

    def setUp(self):
        self._config = load_test_config()

    def test_model_should_be_created_with_config(self):
        # model = UNet3D(self._config["model"])
        pass


class SingleConvTest(unittest.TestCase):

    def setUp(self):
        self.in_channels = 1
        self.out_channels = 32
        self.kernel_size = 3
        self.num_groups = 8
        self.padding = (1, 1, 1, 1, 1, 1)
        self.relu_activation = "relu"
        self.leaky_relu_activation = "leaky_relu"

    def test_should_give_correct_dimensions(self):
        block = SingleConv(self.in_channels, self.out_channels, self.kernel_size, self.num_groups, self.padding,
                           self.relu_activation)
        input = torch.rand((2, 1, 32, 32, 32))
        output = block(input)
        assert output.size() == torch.Size([2, 32, 32, 32, 32])

    def test_should_give_correct_instances(self):
        block = SingleConv(self.in_channels, self.out_channels, self.kernel_size, self.num_groups, self.padding,
                           self.relu_activation)

        assert isinstance(block.padding, torch.nn.ReplicationPad3d)
        assert isinstance(block.norm, torch.nn.GroupNorm)
        assert isinstance(block.activation, torch.nn.ReLU)

        block = SingleConv(self.in_channels, self.out_channels, self.kernel_size, None, None,
                           self.leaky_relu_activation)

        assert block.padding is None
        assert isinstance(block.norm, torch.nn.BatchNorm3d)
        assert isinstance(block.activation, torch.nn.LeakyReLU)
