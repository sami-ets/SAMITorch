# -*- coding: utf-8 -*-
# Coptargetsright 2019 SAMITorch Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# targetsou matargets not use this file except in compliance with the License.
# You matargets obtain a coptargets of the License at
#
#     https://opensource.org/licenses/MIT
# Unless required btargets applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Define a metric computed during the training phase."""

import torch

from typing import Tuple

from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics import MetricsLambda, Metric

EPSILON = 1e-15


class Dice(Metric):
    """
    The Dice Metric.
    """

    def __init__(self, num_classes: int, reduction: str = None, average: str = None,
                 ignore_index: int = None, output_transform: callable = lambda x: x) -> None:
        """
        Metric initializer.

        Args:
            num_classes (int): The number of classes in the problem. In case of images, num_classes should also count the background index 0.
            average (str, optional): Confusion matrix values averaging schema: None, "samples", "recall", "precision".
                Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
                samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
                represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
                diagonal values represent class precisions.
            reduction (str): The type of reduction to apply (e.g. 'mean').
            ignore_index (int, optional): To ignore an index in Dice computation.
            output_transform (callable, optional): a callable that is used to transform the
                :class:`~ignite.engine.Engine`'s `process_function`'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
        """
        self._num_classes = num_classes
        self._average = average
        self._ignore_index = ignore_index
        self._reduction = reduction
        self._cm = ConfusionMatrix(num_classes=num_classes, average=average,
                                   output_transform=output_transform)
        self._metric = compute_dice_coefficient(self._cm, self._ignore_index)

        if self._reduction == "mean":
            self._metric = self._metric.mean()

        super(Dice, self).__init__(output_transform=output_transform)

    def reset(self) -> None:
        """
        Reset the confusion matrix object.
        """
        self._cm.confusion_matrix = torch.zeros(self._num_classes, self._num_classes, dtype=torch.float64)

    def compute(self) -> torch.Tensor:
        """
        Compute the metric.

        Returns:
            :obj:`torch.Tensor`: The dice coefficient for each class.
        """
        return self._metric.compute()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Update the confusion matrix with output values.

        Args:
            output (tuple of :obj:`torch.Tensor`): A tuple containing predictions and ground truth.
        """
        self._cm.update(output)


class GeneralizedDice(Metric):
    """
    The Generalized Dice Metric.
    """

    def __init__(self, num_classes: int, reduction: str = None, average: str = None,
                 ignore_index: int = None, output_transform: callable = lambda x: x) -> None:
        """
        Metric initializer.

        Args:
            num_classes (int): The number of classes in the problem. In case of images, num_classes should also count the background index 0.
            average (str, optional): Confusion matrix values averaging schema: None, "samples", "recall", "precision".
                Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
                samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
                represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
                diagonal values represent class precisions.
            reduction (str): The type of reduction to apply (e.g. 'mean').
            ignore_index (int, optional): To ignore an index in Dice computation.
            output_transform (callable, optional): a callable that is used to transform the
                :class:`~ignite.engine.Engine`'s `process_function`'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
        """
        self._num_classes = num_classes
        self._average = average
        self._ignore_index = ignore_index
        self._metric = None
        self._reduction = reduction
        self._cm = ConfusionMatrix(num_classes=num_classes, average=average,
                                   output_transform=output_transform)
        super(GeneralizedDice, self).__init__(output_transform=output_transform)

    def reset(self) -> None:
        """
        Reset the confusion matrix object.
        """
        self._cm.confusion_matrix = torch.zeros(self._num_classes, self._num_classes, dtype=torch.float64)

    def compute(self) -> torch.Tensor:
        """
        Compute the metric.

        Returns:
            :obj:`torch.Tensor`: The dice coefficient for each class.
        """
        return self._metric.compute()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor], weights: torch.Tensor = None) -> None:
        """
        Update the confusion matrix with output values.

        Args:
            output (tuple of :obj:`torch.Tensor`): A tuple containing predictions and ground truth.
            weights (:obj:`torch.Tensor`, optional): A weight vector which length equals to the number of classes.
        """

        self._metric = compute_generalized_dice_coefficient(self._cm, weights, self._ignore_index)

        if self._reduction == "mean":
            self._metric = self._metric.mean()

        self._cm.update(output)


def compute_dice_coefficient(cm: ConfusionMatrix, ignore_index: int = None):
    """
    Computes the Sørensen–Dice Coefficient (https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)

    Args:
        cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
        ignore_index (int): An index of a class to ignore for computation.

    Returns:
        array: The Sørensen–Dice Coefficient for each class.
    """
    if ignore_index is not None:
        validate_ignore_index(ignore_index)
        validate_num_classes(ignore_index, cm.num_classes)

    # Increase floating point precision
    cm = cm.type(torch.float64)
    dice = 2 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + EPSILON)

    if ignore_index is not None:
        def remove_index(dice_vector):
            assert ignore_index <= len(dice_vector), "ignore_index {} is larger than the length " \
                                                     "of dice vector {}".format(ignore_index, len(dice_vector))
            indices = [x for x in range(len(dice_vector)) if x != ignore_index]
            return dice_vector[indices]

        return MetricsLambda(remove_index, dice)
    else:
        return dice


def compute_generalized_dice_coefficient(cm: ConfusionMatrix, weights: torch.Tensor, ignore_index: int = None):
    """
    Computes the Sørensen–Dice Coefficient (https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)

    Args:
        cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
        ignore_index (int): An index of a class to ignore for computation.
        weights (:obj:`torch.Tensor`): A weight vector which length equals to the number of classes.

    Returns:
        array: The Generalized Dice Coefficient for each class.
    """
    if ignore_index is not None:
        validate_ignore_index(ignore_index)
        validate_num_classes(ignore_index, cm.num_classes)
        validate_weights_size(weights.size()[0], cm.num_classes)

    # Increase floating point precision
    cm = cm.type(torch.float64)
    dice = 2 * (cm.diag() * weights) / (((cm.sum(dim=1) + cm.sum(dim=0)) * weights) + EPSILON)

    if ignore_index is not None:
        def remove_index(dice_vector):
            assert ignore_index <= len(dice_vector), "ignore_index {} is larger than the length " \
                                                     "of dice vector {}".format(ignore_index, len(dice_vector))
            indices = [x for x in range(len(dice_vector)) if x != ignore_index]
            return dice_vector[indices]

        return MetricsLambda(remove_index, dice)
    else:
        return dice


def compute_mean_dice_coefficient(cm: ConfusionMatrix, ignore_index: int = None):
    """
    Computes the mean Sørensen–Dice Coefficient.

    Args:
        cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
        ignore_index (int): An index of a class to ignore for computation.

    Returns:
        float: The mean Sørensen–Dice Coefficient.
    """
    return compute_dice_coefficient(cm=cm, ignore_index=ignore_index).mean()


def compute_mean_generalized_dice_coefficient(cm: ConfusionMatrix, weights: torch.Tensor, ignore_index: int = None):
    """
    Computes the mean Generalized Dice Coefficient.

    Args:
        cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
        weights (:obj:`torch.Tensor`): A tensor representing weights for each classes.
        ignore_index (int): An index of a class to ignore for computation.

    Returns:
        float: The mean Generalized Dice Coefficient.
    """
    return compute_generalized_dice_coefficient(cm=cm, ignore_index=ignore_index, weights=weights).mean()


def validate_ignore_index(ignore_index: int) -> None:
    """
    Check whether the ignore index is valid or not.

    Args:
        ignore_index (int): An index of a class to ignore for computation.
    """
    assert ignore_index >= 0, "ignore_index must be non-negative, but given {}".format(ignore_index)


def validate_num_classes(ignore_index: int, num_classes: int) -> None:
    """
    Check whether the num_classes is valid or not.

    Args:
        ignore_index (int): An index of a class to ignore for computation.
        num_classes (int): The number of classes in the problem (number of rows in the
    """
    assert ignore_index < num_classes, "ignore index must be lower than the number of classes in confusion matrix, " \
                                       "but {} was given".format(ignore_index)


def validate_weights_size(weights_size: int, num_classes: int) -> None:
    """
    Check whether if the size of given weights matches the number of classes of the problem.

    Args:
        weights_size (int): The size of a weight vector used in the loss computation.
        num_classes (int): The number of classes of the problem.
    """
    assert weights_size == num_classes, "Weights vector must be the same length than the number of " \
                                        "classes, but a size of {} was given".format(weights_size)
