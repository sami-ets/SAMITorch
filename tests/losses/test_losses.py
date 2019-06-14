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
import numpy as np

from hamcrest import *
from losses.losses import DiceLoss, GeneralizedDiceLoss, WeightedCrossEntropyLoss


def get_y_true_y_pred():
    # Generate an image with labels 0 (background), 1, 2
    # 3 classes:
    y_true = np.zeros((30, 30), dtype=np.int)
    y_true[1:11, 1:11] = 1
    y_true[15:25, 15:25] = 2

    y_pred = np.zeros((30, 30), dtype=np.int)
    y_pred[5:15, 1:11] = 1
    y_pred[20:30, 20:30] = 2
    return y_true, y_pred


def compute_tensor_y_true_y_logits(y_true, y_pred):
    # Create torch.tensor from numpy
    y_true_tensor = torch.from_numpy(y_true).unsqueeze(0)
    # Create logits torch.tensor:
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    y_probas = np.ones((num_classes,) + y_true.shape) * 0.02
    for i in range(num_classes):
        y_probas[i, (y_pred == i)] = 0.98
    y_logits = torch.from_numpy(y_probas).unsqueeze(0).type(torch.float32)
    return y_true_tensor, y_logits


def compute_dice_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        true_res[index] = 2 * intersection.sum() / (bin_y_pred.sum() + bin_y_true.sum())
    return true_res


def compute_generalized_dice_loss_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        weights = (1.0 / (np.sum(bin_y_true) * np.sum(bin_y_true) + 1e-15))
        intersection = (bin_y_true & bin_y_pred)
        true_res[index] = 2 * intersection.sum() * weights / (((bin_y_pred.sum() + bin_y_true.sum()) * weights) + 1e-15)
    return true_res


class TestDiceLoss(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11
    INVALID_REDUCTION = "sum"

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.dice = compute_dice_truth(self.y_true, self.y_pred)
        self.mean_dice_loss = np.subtract(1.0, np.mean(self.dice))

    def test_should_raise_exception_with_bad_values(self):
        assert_that(calling(DiceLoss).with_args(reduction=self.INVALID_REDUCTION), raises(AssertionError))

        dice_loss = DiceLoss()

        assert_that(calling(dice_loss.forward).with_args(inputs=None, targets=None),
                    raises(AttributeError))
        assert_that(calling(dice_loss.forward).with_args(inputs=self.y_logits, targets=None),
                    raises(AttributeError))
        assert_that(calling(dice_loss.forward).with_args(inputs=None, targets=self.y_true_tensor),
                    raises(AttributeError))
        assert_that(calling(dice_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                         ignore_index=self.INVALID_VALUE_1), raises(AssertionError))
        assert_that(calling(dice_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                         ignore_index=self.INVALID_VALUE_2), raises(TypeError))
        assert_that(calling(dice_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                         ignore_index=self.INVALID_VALUE_3), raises(AssertionError))
        assert_that(calling(dice_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                         ignore_index=self.INVALID_VALUE_4), raises(AssertionError))

    def test_should_compute_dice(self):
        dice_loss = DiceLoss()
        loss = dice_loss.forward(self.y_logits, self.y_true_tensor)

        assert_that(loss, equal_to(self.mean_dice_loss))

    def test_should_compute_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            dice_loss = DiceLoss()
            res = dice_loss.forward(self.y_logits, self.y_true_tensor, ignore_index=ignore_index)
            true_res = np.subtract(1.0, np.mean(self.dice[:ignore_index] + self.dice[ignore_index + 1:]))
            np.testing.assert_array_equal(res.numpy(), true_res), "{}: {} vs {}".format(ignore_index, res, true_res)


class TestGeneralizedDiceLoss(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11
    INVALID_REDUCTION = "sum"

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.generalized_dice_loss = compute_generalized_dice_loss_truth(self.y_true, self.y_pred)
        self.mean_generalized_dice_loss = np.subtract(1.0, np.mean(self.generalized_dice_loss))

    def test_should_raise_exception_with_bad_values(self):
        assert_that(calling(GeneralizedDiceLoss).with_args(reduction=self.INVALID_REDUCTION), raises(AssertionError))

        generalized_dice_loss = GeneralizedDiceLoss()

        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=None, targets=None),
                    raises(AttributeError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=self.y_logits, targets=None),
                    raises(AttributeError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=None, targets=self.y_true_tensor),
                    raises(AttributeError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                                     ignore_index=self.INVALID_VALUE_1),
                    raises(AssertionError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                                     ignore_index=self.INVALID_VALUE_2),
                    raises(TypeError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                                     ignore_index=self.INVALID_VALUE_3),
                    raises(AssertionError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                                     ignore_index=self.INVALID_VALUE_4),
                    raises(AssertionError))

    def test_should_compute_dice(self):
        generalized_dice_loss = GeneralizedDiceLoss()
        loss = generalized_dice_loss.forward(self.y_logits, self.y_true_tensor)
        assert_that(loss, equal_to(self.mean_generalized_dice_loss))

    def test_should_compute_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            generalized_dice_loss = GeneralizedDiceLoss()
            res = generalized_dice_loss.forward(self.y_logits, self.y_true_tensor, ignore_index=ignore_index)
            true_res = np.subtract(1.0, np.mean(
                self.generalized_dice_loss[:ignore_index] + self.generalized_dice_loss[ignore_index + 1:]))
            np.testing.assert_array_equal(res.numpy(), true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_compute_generalized_dice(self):
        generalized_dice_loss = GeneralizedDiceLoss()
        loss = generalized_dice_loss.forward(self.y_logits, self.y_true_tensor)
        assert_that(loss, equal_to(self.mean_generalized_dice_loss))


class TestWeightedCrossEntropy(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11
    INVALID_REDUCTION = "sum"

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)

    def test_should_raise_exception_with_bad_values(self):
        assert_that(calling(WeightedCrossEntropyLoss).with_args(reduction=self.INVALID_REDUCTION),
                    raises(AssertionError))

        weighted_cross_entropy_loss = WeightedCrossEntropyLoss()

        assert_that(calling(weighted_cross_entropy_loss.forward).with_args(inputs=None, targets=None),
                    raises(AttributeError))
        assert_that(calling(weighted_cross_entropy_loss.forward).with_args(inputs=self.y_logits, targets=None),
                    raises(AttributeError))
        assert_that(calling(weighted_cross_entropy_loss.forward).with_args(inputs=None, targets=self.y_true_tensor),
                    raises(AttributeError))
        assert_that(
            calling(weighted_cross_entropy_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                                   ignore_index=self.INVALID_VALUE_1),
            raises(AssertionError))
        assert_that(
            calling(weighted_cross_entropy_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                                   ignore_index=self.INVALID_VALUE_2),
            raises(TypeError))
        assert_that(
            calling(weighted_cross_entropy_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                                   ignore_index=self.INVALID_VALUE_3),
            raises(AssertionError))
        assert_that(
            calling(weighted_cross_entropy_loss.forward).with_args(inputs=self.y_logits, targets=self.y_true_tensor,
                                                                   ignore_index=self.INVALID_VALUE_4),
            raises(AssertionError))

    def test_should_compute_weights(self):
        weighted_cross_entropy_loss = WeightedCrossEntropyLoss()
        weights = weighted_cross_entropy_loss._compute_class_weights(self.y_logits)
        np.testing.assert_almost_equal(weights.numpy(), np.array([0.3043478, 6.8947377, 6.8947353]), decimal=7)
