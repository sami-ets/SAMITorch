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

from metrics.metrics import dice_coefficient, mean_dice_coefficient
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
    y_probas = np.ones((num_classes,) + y_true.shape) * -10
    for i in range(num_classes):
        y_probas[i, (y_pred == i)] = 720
    y_logits = torch.from_numpy(y_probas).unsqueeze(0)
    return y_true_tensor, y_logits


def compute_dice_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        true_res[index] = 2 * intersection.sum() / (bin_y_pred.sum() + bin_y_true.sum())
    return true_res


class TestDiceMetric(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.dice = compute_dice_truth(self.y_true, self.y_pred)
        self.mean_dice = np.mean(self.dice)

    def test_should_raise_exception(self):
        confusion_matrix = ConfusionMatrix(num_classes=10)

        assert_that(calling(dice_coefficient).with_args(cm=None), raises(AttributeError))
        assert_that(calling(dice_coefficient).with_args(cm=confusion_matrix, ignore_index=self.INVALID_VALUE_1),
                    raises(AssertionError))
        assert_that(calling(dice_coefficient).with_args(cm=confusion_matrix, ignore_index=self.INVALID_VALUE_2),
                    raises(TypeError))
        assert_that(calling(dice_coefficient).with_args(cm=confusion_matrix, ignore_index=self.INVALID_VALUE_3),
                    raises(AssertionError))
        assert_that(calling(dice_coefficient).with_args(cm=confusion_matrix, ignore_index=self.INVALID_VALUE_4),
                    raises(AssertionError))

    def test_should_equal_dice_for_each_class(self):
        cm = ConfusionMatrix(num_classes=3)
        dice_metric = dice_coefficient(cm)
        cm.update((self.y_logits, self.y_true_tensor))
        res = dice_metric.compute().numpy()
        assert np.all(res == self.dice)

    def test_should_equal_dice_for_each_class_with_ignored_index(self):
        for ignore_index in range(3):
            cm = ConfusionMatrix(num_classes=3)
            dice_metric = dice_coefficient(cm, ignore_index=ignore_index)
            cm.update((self.y_logits, self.y_true_tensor))
            res = dice_metric.compute().numpy()
            true_res = self.dice[:ignore_index] + self.dice[ignore_index + 1:]
            assert np.all(res == true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_equal_mean_dice(self):
        cm = ConfusionMatrix(num_classes=3)
        mean_dice_metric = mean_dice_coefficient(cm)
        cm.update((self.y_logits, self.y_true_tensor))
        res = mean_dice_metric.compute().numpy()
        assert_that(res, equal_to(self.mean_dice))

    def test_should_equal_mean_dice_with_ignored_index(self):
        for ignore_index in range(3):
            cm = ConfusionMatrix(num_classes=3)
            mean_dice_metric = mean_dice_coefficient(cm, ignore_index=ignore_index)
            cm.update((self.y_logits, self.y_true_tensor))
            res = mean_dice_metric.compute().numpy()
            true_res = np.mean(self.dice[:ignore_index] + self.dice[ignore_index + 1:])
            assert_that(res, equal_to(true_res)), "{}: {} vs {}".format(ignore_index, res, true_res)


if __name__ == 'main':
    unittest.main()
