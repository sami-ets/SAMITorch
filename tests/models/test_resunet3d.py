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
import numpy as np
from hamcrest import *

from models.vnet3d import VNet, SingleConv, Encoder, Decoder, ResNetBlock1Conv, ResNetBlock2Conv, ResNetBlock3Conv


def load_test_config():
    return yaml.load(open("configs/resunet3d.yaml"))


class ResUNet3d(unittest.TestCase):

    def setUp(self):
        self.config = load_test_config()
        self.input = torch.rand((2, 1, 32, 32, 32))

    def test_model_should_be_created_with_config(self):
        model = VNet(self.config["model"])
        assert isinstance(model, VNet)

    def test_model_should_complete_one_forward_pass(self):
        model = VNet(self.config["model"])
        output = model(self.input)
        assert isinstance(output, torch.Tensor)

    def test_model_output_should_have_same_dimensions_than_input(self):
        input_dim = np.array(list(self.input.size()))
        model = VNet(self.config["model"])
        output = model(self.input)
        output_dim = np.array(list(output.size()))
        np.testing.assert_array_equal(input_dim, output_dim)


class ResNet3Block3ConvTest(unittest.TestCase):

    def setUp(self):
        self.config = load_test_config()
        self.input = torch.rand((2, 1, 32, 32, 32))
        self.input_high_channels = torch.rand(2, 128, 16, 16, 16)
        self.in_channels = 1
        self.out_channels = 64
        self.kernel_size = 5
        self.num_groups = 8
        self.padding = (2, 2, 2, 2, 2, 2)
        self.prelu_activation = "PReLU"

    def test_resnet1conv_with_padding_should_give_correct_dimensions(self):
        block = ResNetBlock1Conv(self.in_channels, self.out_channels, self.kernel_size, self.num_groups, self.padding,
                                 self.prelu_activation)
        output = block(self.input)
        assert output.size() == torch.Size([2, 64, 32, 32, 32])

    def test_resnet2conv_with_padding_should_give_correct_dimensions(self):
        block = ResNetBlock2Conv(self.in_channels, self.out_channels, self.kernel_size, self.num_groups, self.padding,
                                 self.prelu_activation)
        output = block(self.input)
        assert output.size() == torch.Size([2, 64, 32, 32, 32])

    def test_resnet3conv_with_padding_should_give_correct_dimensions(self):
        block = ResNetBlock3Conv(self.in_channels, self.out_channels, self.kernel_size, self.num_groups,
                                 self.padding, self.prelu_activation)
        output = block(self.input)
        assert output.size() == torch.Size([2, 64, 32, 32, 32])
