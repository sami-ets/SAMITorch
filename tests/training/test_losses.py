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

import pytest
import torch
import numpy as np

from training.losses import DiceLoss


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


def compute_th_y_true_y_logits(y_true, y_pred):
    # Create torch.tensor from numpy
    th_y_true = torch.from_numpy(y_true).unsqueeze(0)
    # Create logits torch.tensor:
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    y_probas = np.ones((num_classes,) + y_true.shape) * -10
    for i in range(num_classes):
        y_probas[i, (y_pred == i)] = 720
    th_y_logits = torch.from_numpy(y_probas).unsqueeze(0)
    return th_y_true, th_y_logits


def test_dice_loss():
    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        true_res[index] = 2 * intersection.sum() / (bin_y_pred.sum() + bin_y_true.sum())

    true_res_ = 1.0 - np.mean(true_res)

    dice_loss = DiceLoss()
    loss = dice_loss.forward(th_y_logits, th_y_true)

    assert loss == true_res_

    for ignore_index in range(3):
        dice_loss = DiceLoss(ignore_index=ignore_index)
        res = dice_loss.forward(th_y_logits, th_y_true)
        # Update metric
        true_res_ = 1.0 - np.mean(true_res[:ignore_index] + true_res[ignore_index + 1:])
        assert np.all(res == true_res_), "{}: {} vs {}".format(ignore_index, res, true_res_)
