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
import numpy as np
import unittest

from samitorch.metrics.metrics import compute_dice_coefficient, compute_mean_dice_coefficient, \
    compute_generalized_dice_coefficient, compute_mean_generalized_dice_coefficient, validate_weights_size, \
    validate_num_classes, validate_ignore_index, Dice, GeneralizedDice
from ignite.metrics.confusion_matrix import ConfusionMatrix
from hamcrest import *


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
    y_probs = np.ones((num_classes,) + y_true.shape) * -10
    for i in range(num_classes):
        y_probs[i, (y_pred == i)] = 720
    y_logits = torch.from_numpy(y_probs).unsqueeze(0)
    return y_true_tensor, y_logits


def compute_dice_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        true_res[index] = 2 * intersection.sum() / (bin_y_pred.sum() + bin_y_true.sum())
    return true_res


def compute_generalized_dice_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    weights = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        weights[index] = (1.0 / (np.sum(bin_y_true) * np.sum(bin_y_true) + 1e-15))
        intersection = (bin_y_true & bin_y_pred)
        true_res[index] = 2 * intersection.sum() * weights[index] / (
                ((bin_y_pred.sum() + bin_y_true.sum()) * weights[index]) + 1e-15)
    return true_res, weights


class TestValidationMethods(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = 10
    INVALID_VALUE_3 = 11
    NUM_CLASSES = 5

    def setUp(self):
        pass

    def test_should_raise_assertion_error_with_bad_values(self):
        assert_that(calling(validate_ignore_index).with_args(self.INVALID_VALUE_1), raises(AssertionError))
        assert_that(
            calling(validate_num_classes).with_args(ignore_index=self.INVALID_VALUE_2, num_classes=self.NUM_CLASSES),
            raises(AssertionError))
        assert_that(calling(validate_weights_size).with_args(weights_size=self.INVALID_VALUE_2,
                                                             num_classes=self.INVALID_VALUE_3), raises(AssertionError))


class TestDiceMetric(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.dice_truth = compute_dice_truth(self.y_true, self.y_pred)
        self.mean_dice_truth = np.mean(self.dice_truth)

    def test_should_raise_exception_with_bad_values(self):
        confusion_matrix = ConfusionMatrix(num_classes=10)

        assert_that(calling(compute_dice_coefficient).with_args(cm=None), raises(AttributeError))
        assert_that(calling(compute_dice_coefficient).with_args(cm=confusion_matrix, ignore_index=self.INVALID_VALUE_1),
                    raises(AssertionError))
        assert_that(calling(compute_dice_coefficient).with_args(cm=confusion_matrix, ignore_index=self.INVALID_VALUE_2),
                    raises(TypeError))
        assert_that(calling(compute_dice_coefficient).with_args(cm=confusion_matrix, ignore_index=self.INVALID_VALUE_3),
                    raises(AssertionError))
        assert_that(calling(compute_dice_coefficient).with_args(cm=confusion_matrix, ignore_index=self.INVALID_VALUE_4),
                    raises(AssertionError))

    def test_should_compute_dice_for_multiclass(self):
        cm = ConfusionMatrix(num_classes=3)
        dice_coefficient = compute_dice_coefficient(cm)
        cm.update((self.y_logits, self.y_true_tensor))
        res = dice_coefficient.compute().numpy()
        assert np.all(res == self.dice_truth)

    def test_should_compute_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            cm = ConfusionMatrix(num_classes=3)
            dice_coefficient = compute_dice_coefficient(cm, ignore_index=ignore_index)
            cm.update((self.y_logits, self.y_true_tensor))
            res = dice_coefficient.compute().numpy()
            true_res = self.dice_truth[:ignore_index] + self.dice_truth[ignore_index + 1:]
            assert np.all(res == true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_compute_mean_dice(self):
        cm = ConfusionMatrix(num_classes=3)
        mean_dice_coefficient = compute_mean_dice_coefficient(cm)
        cm.update((self.y_logits, self.y_true_tensor))
        res = mean_dice_coefficient.compute().numpy()
        assert_that(res, equal_to(self.mean_dice_truth))

    def test_should_compute_mean_dice_with_ignored_index(self):
        for ignore_index in range(3):
            cm = ConfusionMatrix(num_classes=3)
            mean_dice_coefficient = compute_mean_dice_coefficient(cm, ignore_index=ignore_index)
            cm.update((self.y_logits, self.y_true_tensor))
            res = mean_dice_coefficient.compute().numpy()
            true_res = np.mean(self.dice_truth[:ignore_index] + self.dice_truth[ignore_index + 1:])
            assert_that(res, equal_to(true_res)), "{}: {} vs {}".format(ignore_index, res, true_res)


class DiceTest(unittest.TestCase):
    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.dice_truth = compute_dice_truth(self.y_true, self.y_pred)
        self.mean_dice_truth = np.mean(self.dice_truth)

    def test_should_compute_dice_metric(self):
        self.dice = Dice(num_classes=3)
        self.dice.update((self.y_logits, self.y_true_tensor))
        res = self.dice.compute()
        np.testing.assert_almost_equal(res, self.dice_truth)

    def test_should_compute_dice_metric_with_sample_average(self):
        self.dice = Dice(num_classes=3, average="samples")
        self.dice.update((self.y_logits, self.y_true_tensor))
        res = self.dice.compute()
        np.testing.assert_almost_equal(res, self.dice_truth)

    def test_should_compute_mean_dice_metric(self):
        self.dice = Dice(num_classes=3, reduction="mean")
        self.dice.update((self.y_logits, self.y_true_tensor))
        res = self.dice.compute()
        assert_that(res, instance_of(torch.Tensor))
        assert_that(res.dtype, is_(torch.float64))
        np.testing.assert_almost_equal(res.mean(), self.mean_dice_truth)


class GeneralizedDiceMetricTest(unittest.TestCase):
    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.generalized_dice_truth, weights = compute_generalized_dice_truth(self.y_true, self.y_pred)
        self.weights = torch.from_numpy(np.array(weights))
        self.generalized_mean_dice_truth = np.mean(self.generalized_dice_truth)

    def test_should_compute_generalized_dice_for_multiclass(self):
        self.dice = GeneralizedDice(num_classes=3)
        self.dice.update((self.y_logits, self.y_true_tensor), self.weights)
        res = self.dice.compute()
        np.testing.assert_almost_equal(res, self.generalized_dice_truth)

    def test_should_compute_mean_generalized_dice_for_multiclass(self):
        self.dice = GeneralizedDice(num_classes=3, reduction="mean")
        self.dice.update((self.y_logits, self.y_true_tensor), self.weights)
        res = self.dice.compute()
        assert_that(res, instance_of(torch.Tensor))
        assert_that(res.dtype, is_(torch.float64))
        np.testing.assert_almost_equal(res.mean(), self.generalized_mean_dice_truth)


class TestGeneralizedDiceMetric(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.generalized_dice_truth, weights = compute_generalized_dice_truth(self.y_true, self.y_pred)
        self.weights = torch.from_numpy(np.array(weights))
        self.generalized_mean_dice_truth = np.mean(self.generalized_dice_truth)

    def test_should_raise_exception_with_bad_values(self):
        confusion_matrix = ConfusionMatrix(num_classes=10)
        assert_that(calling(compute_generalized_dice_coefficient).with_args(cm=None, weights=self.weights),
                    raises(AttributeError))
        assert_that(calling(compute_generalized_dice_coefficient).with_args(cm=confusion_matrix,
                                                                            weights=self.weights,
                                                                            ignore_index=self.INVALID_VALUE_1),
                    raises(AssertionError))
        assert_that(calling(compute_generalized_dice_coefficient).with_args(cm=confusion_matrix,
                                                                            weights=self.weights,
                                                                            ignore_index=self.INVALID_VALUE_2),
                    raises(TypeError))
        assert_that(calling(compute_generalized_dice_coefficient).with_args(cm=confusion_matrix,
                                                                            weights=self.weights,
                                                                            ignore_index=self.INVALID_VALUE_3),
                    raises(AssertionError))
        assert_that(calling(compute_generalized_dice_coefficient).with_args(cm=confusion_matrix,
                                                                            weights=self.weights,
                                                                            ignore_index=self.INVALID_VALUE_4),
                    raises(AssertionError))

    def test_should_compute_dice_for_multiclass(self):
        cm = ConfusionMatrix(num_classes=3)
        generalized_dice_coefficient = compute_generalized_dice_coefficient(cm, weights=self.weights)
        cm.update((self.y_logits, self.y_true_tensor))
        res = generalized_dice_coefficient.compute().numpy()
        assert np.all(res == self.generalized_dice_truth)

    def test_should_compute_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            cm = ConfusionMatrix(num_classes=3)
            generalized_dice_coefficient = compute_generalized_dice_coefficient(cm, self.weights,
                                                                                ignore_index=ignore_index)
            cm.update((self.y_logits, self.y_true_tensor))
            res = generalized_dice_coefficient.compute().numpy()
            true_res = self.generalized_dice_truth[:ignore_index] + self.generalized_dice_truth[ignore_index + 1:]
            assert np.all(res == true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_compute_mean_dice(self):
        cm = ConfusionMatrix(num_classes=3)
        generalized_dice_coefficient = compute_mean_generalized_dice_coefficient(cm, weights=self.weights)
        cm.update((self.y_logits, self.y_true_tensor))
        res = generalized_dice_coefficient.compute().numpy()
        assert_that(res, equal_to(self.generalized_mean_dice_truth))

    def test_should_compute_mean_dice_with_ignored_index(self):
        for ignore_index in range(3):
            cm = ConfusionMatrix(num_classes=3)
            mean_generalized_dice_coefficient = compute_mean_generalized_dice_coefficient(cm, self.weights,
                                                                                          ignore_index=ignore_index)
            cm.update((self.y_logits, self.y_true_tensor))
            res = mean_generalized_dice_coefficient.compute().numpy()
            true_res = np.mean(
                self.generalized_dice_truth[:ignore_index] + self.generalized_dice_truth[ignore_index + 1:])
            assert_that(res, equal_to(true_res)), "{}: {} vs {}".format(ignore_index, res, true_res)
