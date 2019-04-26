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

import torch.nn

from ignite.metrics.confusion_matrix import ConfusionMatrix
from metrics.metrics import mean_dice_coefficient, dice_coefficient


class DiceLoss(torch.nn.Module):

    def __init__(self, ignore_index=None, reduction="mean"):
        super(DiceLoss).__init__()
        self._ignore_index = ignore_index
        self._reduction = reduction

    def forward(self, inputs, targets):
        """
        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
           inputs (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The model's prediction on which the loss has to
                be computed.
           targets: a tensor of shape (B, C, ..). The ground truth.
       Returns:
           float: the Sørensen–Dice loss.
        """
        cm = ConfusionMatrix(num_classes=inputs.shape[1])

        if self._reduction == "mean":
            dice_loss = mean_dice_coefficient(cm, ignore_index=self._ignore_index)
        else:
            dice_loss = dice_coefficient(cm, ignore_index=self._ignore_index)

        output = (inputs, targets)
        cm.update(output)

        return 1.0 - dice_loss.compute().numpy()
