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
from torch.autograd import Variable

from samitorch.metrics.metrics import compute_mean_dice_coefficient, compute_mean_generalized_dice_coefficient
from samitorch.utils.utils import flatten, to_onehot

SUPPORTED_REDUCTIONS = ["mean"]


class DiceLoss(torch.nn.Module):

    def __init__(self, reduction: str = "mean"):
        super(DiceLoss).__init__()
        assert reduction in SUPPORTED_REDUCTIONS, "Reduction type not supported."
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = None):
        """
        Computes the Sørensen–Dice loss.

        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
            inputs (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The model prediction on which the loss has to be computed.
            targets (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The ground truth.
            ignore_index (int): An index to ignore for computation.

        Returns:
            float: The Sørensen–Dice loss.
         """

        cm = ConfusionMatrix(num_classes=inputs.shape[1])

        if self._reduction == "mean":
            dice_coefficient = compute_mean_dice_coefficient(cm, ignore_index=ignore_index)

        cm.update((inputs, targets))

        return 1.0 - dice_coefficient.compute().numpy()


class GeneralizedDiceLoss(torch.nn.Module):

    def __init__(self, reduction="mean"):
        super(GeneralizedDiceLoss, self).__init__()
        assert reduction in SUPPORTED_REDUCTIONS, "Reduction is not supported."
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = None):
        """
        Computes the Generalized Dice Loss as described in https://arxiv.org/pdf/1707.03237.pdf
        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
           inputs (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The model's prediction on which the loss has to be computed.
           targets (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The ground truth.
           ignore_index (int): An index to ignore for computation.

       Returns:
           float: the Generalized Dice Loss.
        """
        num_classes = inputs.shape[1]

        cm = ConfusionMatrix(num_classes)

        flattened_targets = flatten(to_onehot(targets, num_classes))

        weights = Variable(1.0 / torch.pow(flattened_targets.sum(-1), 2).clamp(min=1e-15), requires_grad=False).type(
            torch.float64)

        if self._reduction == "mean":
            generalized_dice = compute_mean_generalized_dice_coefficient(cm, weights, ignore_index=ignore_index)

        cm.update((inputs, targets))

        return 1.0 - generalized_dice.compute().numpy()


class WeightedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, reduction="mean"):
        super(WeightedCrossEntropyLoss, self).__init__()
        assert reduction in SUPPORTED_REDUCTIONS, "Reduction is not supported."
        self._reduction = reduction

    def _validate_ignore_index(self, ignore_index: int, input_shape: int):
        """
        Validate the `ignore_index` variable.

        Args:
            ignore_index (int): An index of a class to ignore for computation.
            input_shape (int): Input tensor.
        """
        assert ignore_index >= 0, "ignore_index must be non-negative, but given {}".format(ignore_index)
        assert ignore_index < input_shape, "ignore index must be lower than the number of classes in " \
                                           "confusion matrix, but {} was given.".format(ignore_index)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = None):
        """
        Computes the Weighted Cross Entropy Loss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf

        Args:
            nputs (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The model's prediction on which the loss has to be computed.
           targets (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The ground truth.
           ignore_index (int): An index to ignore for computation.

       Returns:
           float: the weighted Cross-Entropy loss value.

        """
        if ignore_index is not None:
            self._validate_ignore_index(ignore_index, inputs.shape[1])
        else:
            ignore_index = -100

        class_weights = self._compute_class_weights(inputs)

        return torch.nn.functional.cross_entropy(inputs, targets, weight=class_weights,
                                                 ignore_index=ignore_index).numpy()

    @staticmethod
    def _compute_class_weights(inputs: torch.Tensor):
        """
        Compute weights for each class as described in https://arxiv.org/pdf/1707.03237.pdf

        Args:
            inputs: (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The model's prediction on which the loss has to be computed.

        Returns:
            :obj:`torch.Variable`: A variable containing class weights.
        """
        flattened_inputs = flatten(inputs)
        class_weights = Variable((flattened_inputs.shape[1] - flattened_inputs.sum(-1)) / flattened_inputs.sum(-1),
                                 requires_grad=False)
        return class_weights
