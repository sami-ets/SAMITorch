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

from hamcrest import *
from samitorch.factories.factories import PaddingFactory, ActivationFunctionsFactory, PoolingFactory, \
    NormalizationLayerFactory


class PaddingFactoryTest(unittest.TestCase):
    CORRECT_PADDING_STRATEGY = "ReplicationPad3d"
    INCORRECT_PADDING_STRATEGY = "unknownPad"

    def setUp(self):
        self.factory = PaddingFactory()

    def test_should_instantiate_padding_layer(self):
        padding_layer = self.factory.get_layer(self.CORRECT_PADDING_STRATEGY, dims=(1, 1, 1, 1, 1, 1))
        assert padding_layer is not None

    def test_should_instantiate_correct_padding_layer(self):
        padding_layer = self.factory.get_layer(self.CORRECT_PADDING_STRATEGY, dims=(1, 1, 1, 1, 1, 1))
        assert_that(padding_layer, instance_of(torch.nn.ReplicationPad3d))

    def test_should_raise_value_error_exception_with_unknown_padding_strategy(self):
        assert_that(
            calling(self.factory.get_layer).with_args(self.INCORRECT_PADDING_STRATEGY, dims=(1, 1, 1, 1, 1, 1)),
            raises(ValueError))

    def test_registering_new_padding_strategy_should_append_to_supported_strategies(self):
        new_padding_strategy = "ReplicationPad1d"
        creator = torch.nn.ReplicationPad1d
        self.factory.register(new_padding_strategy, creator)
        assert_that(self.factory._padding_strategies, has_entry(new_padding_strategy, creator))


class ActivationFunctionFactory(unittest.TestCase):
    CORRECT_ACTIVATION_FUNCTION = "ReLU"
    INCORRECT_ACTIVATION_FUNCTION = "unknown_activation"

    def setUp(self):
        self.factory = ActivationFunctionsFactory()

    def test_should_instantiate_activation_layer(self):
        activation_function = self.factory.get_layer(self.CORRECT_ACTIVATION_FUNCTION, True)
        assert activation_function is not None

    def test_should_instantiate_correct_activation_layer(self):
        activation_function = self.factory.get_layer(self.CORRECT_ACTIVATION_FUNCTION, True)
        assert_that(activation_function, instance_of(torch.nn.ReLU))

    def test_should_raise_value_error_exception_with_unknown_activation_function(self):
        assert_that(
            calling(self.factory.get_layer).with_args(self.INCORRECT_ACTIVATION_FUNCTION, True),
            raises(ValueError))

    def test_registering_new_activation_function_should_append_to_supported_function(self):
        new_activation_function = "ELU"
        creator = torch.nn.ELU
        self.factory.register(new_activation_function, creator)
        assert_that(self.factory._activation_functions, has_entry(new_activation_function, creator))


class PoolingFactoryTest(unittest.TestCase):
    CORRECT_POOLING_STRATEGY = "MaxPool3d"
    INCORRECT_POOLING_STRATEGY = "unknown_pooling"
    KERNEL_SIZE = 2

    def setUp(self):
        self.factory = PoolingFactory()

    def test_should_instantiate_pooling_layer(self):
        pooling_layer = self.factory.get_layer(self.CORRECT_POOLING_STRATEGY, kernel_size=self.KERNEL_SIZE)
        assert pooling_layer is not None

    def test_should_instantiate_correct_pooling_layer(self):
        pooling_layer = self.factory.get_layer(self.CORRECT_POOLING_STRATEGY, kernel_size=self.KERNEL_SIZE)
        assert_that(pooling_layer, instance_of(torch.nn.MaxPool3d))

    def test_should_raise_value_error_exception_with_unknown_pooling_strategy(self):
        assert_that(
            calling(self.factory.get_layer).with_args(self.INCORRECT_POOLING_STRATEGY, kernel_size=self.KERNEL_SIZE),
            raises(ValueError))

    def test_should_instantiate_Conv3d_pooling_layer(self):
        pooling_layer = self.factory.get_layer("Conv3d", 2, 2, 16, 32)
        assert_that(pooling_layer, instance_of(torch.nn.Conv3d))
        assert_that(pooling_layer, has_property("stride", (2, 2, 2)))
        assert_that(pooling_layer, has_property("in_channels", 16))
        assert_that(pooling_layer, has_property("out_channels", 32))

    def test_registering_new_pooling_strategy_should_append_to_supported_strategies(self):
        new_pooling_strategy = "MaxPool1d"
        creator = torch.nn.MaxPool1d
        self.factory.register(new_pooling_strategy, creator)
        assert_that(self.factory._pooling_strategies, has_entry(new_pooling_strategy, creator))


class NormalizationFactoryTest(unittest.TestCase):
    GROUP_NORM = "GroupNorm"
    BATCH_NORM = "BatchNorm3d"
    INCORRECT_NORM_LAYER = "unknown_norm_layer"
    NUM_FEATURES = 256
    NUM_GROUPS = 8

    def setUp(self):
        self.factory = NormalizationLayerFactory()

    def test_should_instantiate_normalization_layer(self):
        normalization_layer = self.factory.get_layer(self.GROUP_NORM, self.NUM_GROUPS,
                                                     self.NUM_FEATURES)
        assert normalization_layer is not None

    def test_should_instantiate_group_normalization(self):
        normalization_layer = self.factory.get_layer(self.GROUP_NORM, self.NUM_GROUPS,
                                                     self.NUM_FEATURES)
        assert_that(normalization_layer, instance_of(torch.nn.GroupNorm))

    def test_should_instantiate_batch_normalization(self):
        normalization_layer = self.factory.get_layer(self.BATCH_NORM, self.NUM_FEATURES)
        assert_that(normalization_layer, instance_of(torch.nn.BatchNorm3d))
        assert_that(normalization_layer, has_property("num_features", 256))

    def test_should_raise_value_error_exception_with_unknown_normalization_strategy(self):
        assert_that(
            calling(self.factory.get_layer).with_args(self.INCORRECT_NORM_LAYER, self.NUM_FEATURES),
            raises(ValueError))

    def test_registering_new_padding_strategy_should_append_to_supported_strategies(self):
        new_normalization_layer = "BatchNorm2d"
        creator = torch.nn.BatchNorm2d
        self.factory.register(new_normalization_layer, creator)
        assert_that(self.factory._normalization_strategies, has_entry(new_normalization_layer, creator))
