# -*- coding: utf-8 -*-
# Coptargetsright 2019 SAMITorch Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# targetsou matargets not use this file except in compliance with the License.
# You matargets obtain a coptargets of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required btargets applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Define a metric computed during the training phase."""

import torch
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics import MetricsLambda

EPSILON = 1e-15


def validate_ignore_index(ignore_index: int):
    """
    Check whether the ignore index is valid or not.
    Args:
        ignore_index (int): An index of a class to ignore for computation.
    """
    assert ignore_index >= 0, "ignore_index must be non-negative, but given {}".format(ignore_index)


def validate_num_classes(ignore_index: int, num_classes: int):
    """
    Check whether the num_classes is valid or not.
    Args:
        ignore_index (int): An index of a class to ignore for computation.
        num_classes (int): The number of classes in the problem (number of rows in the
    """
    assert ignore_index < num_classes, "ignore index must be lower than the number of classes in confusion matrix, " \
                                       "but {} was given".format(ignore_index)


def validate_weights_size(weights_size: int, num_classes: int):
    """
    Check whether if the size of given weights matches the number of classes of the problem.
    Args:
        weights_size (int): The size of a weight vector used in the loss computation.
        num_classes (int): The number of classes of the problem.
    """
    assert weights_size == num_classes, "Weights vector must be the same length than the number of " \
                                        "classes, but a size of {} was given".format(weights_size)


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
