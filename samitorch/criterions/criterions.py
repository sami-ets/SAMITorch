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

import abc
from enum import Enum
from typing import Union

from ignite.metrics import MetricsLambda
import torch
import torch.nn

from metrics.metric import validate_ignore_index
from utils.utils import flatten

SUPPORTED_REDUCTIONS = [None, "mean"]
EPSILON = 1e-15


class Criterion(Enum):
    BCELoss = "BCELoss"
    BCEWithLogitsLoss = "BCEWithLogtisLoss"
    PoissonNLLLoss = "PoissonNLLLoss"
    CosineEmbeddingLoss = "CosineEmbeddingLoss"
    CrossEntropyLoss = "CrossEntropyLoss"
    CTCLoss = "CTCLoss"
    HingeEmbeddingLoss = "HingeEmbeddingLoss"
    KLDivLoss = "KLDivLoss"
    L1Loss = "L1Loss"
    MSELoss = "MSELoss"
    MarginRankingLoss = "MarginRankingLoss"
    MultiLabelMarginLoss = "MultiLabelMarginLoss"
    MultiLabelSoftMarginLoss = "MultiLabelSoftMarginLoss"
    MultiMarginLoss = "MultiMarginLoss"
    NLLLoss = "NLLLoss"
    SmoothL1Loss = "SmoothL1Loss"
    SoftMarginLoss = "SoftMarginLoss"
    TripletMarginLoss = "TripletMarginLoss"


class DiceLoss(torch.nn.Module):
    """
    The Sørensen-Dice Loss.
    """

    def __init__(self, reduction: Union[None, str] = "mean"):
        super(DiceLoss, self).__init__()
        assert reduction in SUPPORTED_REDUCTIONS, "Reduction type not supported."
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None,
                ignore_index: int = None):
        """
        Computes the Sørensen–Dice loss.

        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
            inputs (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The model prediction on which the loss has to be computed.
            targets (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The ground truth.
            ignore_index (int): An index to ignore for computation.
            weights (:obj:`torch.Tensor`): A class weight vector of shape (C, ), one per class.

        Returns:
            :obj:`torch.Tensor`: The Sørensen–Dice loss for each class or reduced according to reduction method.
        """
        if ignore_index is not None:
            validate_ignore_index(ignore_index)

        assert inputs.size() == targets.size(), "'Inputs' and 'Targets' must have the same shape."

        inputs = flatten(inputs)
        targets = flatten(targets)

        targets = targets.float()

        # Compute per channel Dice Coefficient
        intersect = (inputs * targets).sum(-1)

        if weights is not None:
            if weights.requires_grad is not False:
                weights.requires_grad = False
            intersect = weights * intersect

        denominator = (inputs + targets).sum(-1)

        dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON))

        if ignore_index is not None:

            def ignore_index_fn(dice_vector):
                indices = list(range(len(dice_vector)))
                indices.remove(ignore_index)
                return dice_vector[indices]

            return MetricsLambda(ignore_index_fn, dice).compute()

        else:
            if self._reduction == "mean":
                dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON)).mean()
            elif self._reduction is None:
                pass
            else:
                raise NotImplementedError("Reduction method not implemented.")
            return dice


class GeneralizedDiceLoss(torch.nn.Module):
    """
      The Generalized Sørensen-Dice Loss.
      """

    def __init__(self, reduction: Union[None, str] = "mean"):
        super(GeneralizedDiceLoss, self).__init__()
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
            :obj:`torch.Tensor`: The Sørensen–Dice loss for each class or reduced according to reduction method.
        """
        if ignore_index is not None:
            validate_ignore_index(ignore_index)

        assert inputs.size() == targets.size(), "'Inputs' and 'Targets' must have the same shape."

        inputs = flatten(inputs)
        targets = flatten(targets)

        targets = targets.float()

        class_weights = 1.0 / torch.pow(targets.sum(-1), 2).clamp(min=1e-15)

        # Compute per channel Dice Coefficient
        intersect = (inputs * targets).sum(-1) * class_weights

        denominator = (inputs + targets).sum(-1) * class_weights

        dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON))

        if ignore_index is not None:

            def ignore_index_fn(dice_vector):
                indices = list(range(len(dice_vector)))
                indices.remove(ignore_index)
                return dice_vector[indices]

            return MetricsLambda(ignore_index_fn, dice).compute()

        else:
            if self._reduction == "mean":
                dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON)).mean()
            elif self._reduction is None:
                pass
            else:
                raise NotImplementedError("Reduction method not implemented.")
            return dice


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

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = None) -> float:
        """
        Computes the Weighted Cross Entropy Loss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf

        Args:
            inputs (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The model's prediction on which the loss has to be computed.
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
                                                 ignore_index=ignore_index)

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
        class_weights = (flattened_inputs.shape[1] - flattened_inputs.sum(-1)) / flattened_inputs.sum(-1)
        return class_weights


class AbstractCriterionFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self, criterion: Union[str, Enum], *args):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, criterion: str, creator):
        raise NotImplementedError


class CriterionFactory(AbstractCriterionFactory):
    def __init__(self):
        super(CriterionFactory, self).__init__()
        self._criterion = {
            "BCELoss": torch.nn.BCELoss,
            "BCEWithLogtisLoss": torch.nn.BCEWithLogitsLoss,
            "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
            "CosineEmbeddingLoss": torch.nn.CosineEmbeddingLoss,
            "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
            "CTCLoss": torch.nn.CTCLoss,
            "HingeEmbeddingLoss": torch.nn.HingeEmbeddingLoss,
            "KLDivLoss": torch.nn.KLDivLoss,
            "L1Loss": torch.nn.L1Loss,
            "MSELoss": torch.nn.MSELoss,
            "MarginRankingLoss": torch.nn.MarginRankingLoss,
            "MultiLabelMarginLoss": torch.nn.MultiLabelMarginLoss,
            "MultiLabelSoftMarginLoss": torch.nn.MultiLabelSoftMarginLoss,
            "MultiMarginLoss": torch.nn.MultiMarginLoss,
            "NLLLoss": torch.nn.NLLLoss,
            "SmoothL1Loss": torch.nn.SmoothL1Loss,
            "SoftMarginLoss": torch.nn.SoftMarginLoss,
            "TripletMarginLoss": torch.nn.TripletMarginLoss,
            "DiceLoss": DiceLoss
        }

    def create(self, criterion: Union[str, Criterion], *args, **kwargs):
        """
        Instanciate a loss function based on its name.

        Args:
           criterion (str_or_Enum): The criterion's name.
           *args: Other arguments.

        Returns:
           :obj:`torch.nn.Module`: The criterion.

        Raises:
           KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        criterion = self._criterion[criterion.name if isinstance(criterion, Enum) else criterion]
        return criterion(*args, **kwargs)

    def register(self, function: str, creator):
        """
        Add a new criterion.

        Args:
           function (str): Criterion's name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom criterion function.
        """
        self._criterion[function] = creator
