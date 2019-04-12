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

import abc
import torch
import math

from torch.nn.functional import pairwise_distance
from utils.utils import to_onehot


class Metric(object):
    """Base class of all metrics objects."""

    def __init__(self, is_multilabel=False):
        """Class constructor"""
        self._type = None
        self._num_classes = None
        self._is_multilabel = is_multilabel

    @abc.abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute a specific metric.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    def _check_shapes(self, predictions: torch.Tensor, targets: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Check if shape of predictions and targets matches.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Raises:
            ValueError: If targets and predictions have incompatible shapes.
        """
        if targets.ndimension() > 1 and targets.shape[1] == 1:
            # (N, 1, ...) -> (N, ...)
            targets = targets.squeeze(dim=1)

        if predictions.ndimension() > 1 and predictions.shape[1] == 1:
            # (N, 1, ...) -> (N, ...)
            predictions = predictions.squeeze(dim=1)

        if not (
                targets.ndimension() == predictions.ndimension() or targets.ndimension() + 1 == predictions.ndimension()):
            raise ValueError("targets must have shape of (batch_size, ...) and predictions must have shape of "
                             "(batch_size, num_categories, ...) or (batch_size, ...), "
                             "but given {} vs {}.".format(targets.shape, predictions.shape))

        targets_shape = targets.shape
        predictions_shape = predictions.shape

        if targets.ndimension() + 1 == predictions.ndimension():
            predictions_shape = (predictions_shape[0],) + predictions_shape[2:]

        if not (targets_shape == predictions_shape):
            raise ValueError("targets and predictions must have compatible shapes.")

        if self._is_multilabel and not (
                targets.shape == predictions.shape and targets.ndimension() > 1 and targets.shape[1] != 1):
            raise ValueError("targets and predictions must have same shape of (batch_size, num_categories, ...).")

        return predictions, targets

    def _select_metric_type(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Select the right type of metric between binary, multiclass or multilabel.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Raises:
            ValueError: If in binary cases the prediction is not comprised of 0's and 1's or if number of classes
                changed since creation of the metric's object.
            RuntimeError: If predictions and targets' shapes are incompatible or if data type changed since
                creation of the metric's object.
        """
        if targets.ndimension() + 1 == predictions.ndimension():
            update_type = "multiclass"
            num_classes = predictions.shape[1]
        elif targets.ndimension() == predictions.ndimension():
            if not torch.equal(targets, targets ** 2):
                raise ValueError("For binary cases, y must be comprised of 0's and 1's.")

            if not torch.equal(predictions, predictions ** 2):
                raise ValueError("For binary cases, predictions must be comprised of 0's and 1's.")

            if self._is_multilabel:
                update_type = "multilabel"
                num_classes = predictions.shape[1]
            else:
                update_type = "binary"
                num_classes = 1
        else:
            raise RuntimeError("Invalid shapes of y (shape={}) and predictions (shape={}), check documentation."
                               " for expected shapes of y and predictions.".format(targets.shape, predictions.shape))
        if self._type is None:
            self._type = update_type
            self._num_classes = num_classes
        else:
            if self._type != update_type:
                raise RuntimeError("Input data type has changed from {} to {}.".format(self._type, update_type))
            if self._num_classes != num_classes:
                raise ValueError("Input data number of classes has changed from {} to {}"
                                 .format(self._num_classes, num_classes))


class BasePrecisionRecall(Metric):
    def __init__(self, average=False, is_multilabel=False):
        self._average = average
        self._true_positives = None
        self._positives = None
        self._epsilon = 1e-20
        super(BasePrecisionRecall, self).__init__(is_multilabel=is_multilabel)

    @abc.abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute a Precision or Recall metric.

         Args:
             predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
             targets (:obj:`torch.Tensor`): The ground truth.

         Raises:
             NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()


class Accuracy(Metric):
    """Calculate Accuracy."""

    def __init__(self, is_multilabel=False):
        """Class constructor."""
        super().__init__(is_multilabel=is_multilabel)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's accuracy.
        """
        _num_correct = 0

        predictions, targets = self._check_shapes(predictions, targets)
        self._select_metric_type(predictions, targets)

        if self._type == "binary":
            correct = torch.eq(predictions.type(targets.type()), targets).view(-1)
        elif self._type == "multiclass":
            indices = torch.argmax(predictions, dim=1)
            correct = torch.eq(indices, targets).view(-1)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (N x ..., C)
            num_classes = predictions.size(1)
            last_dim = predictions.ndimension()
            y_pred = torch.transpose(predictions, 1, last_dim - 1).reshape(-1, num_classes)
            y = torch.transpose(targets, 1, last_dim - 1).reshape(-1, num_classes)
            correct = torch.all(y == y_pred.type_as(y), dim=-1)

        _num_correct = torch.sum(correct).item()

        if correct.shape[0] == 0:
            raise ValueError('Accuracy metric must have at least one example before it can be computed.')

        return _num_correct / correct.shape[0]


class TopKCategoricalAccuracy(Metric):
    """Calculate the top-k categorical accuracy."""

    def __init__(self, k=5):
        super(TopKCategoricalAccuracy, self).__init__()
        self._k = k

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Top-K Accuracy.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's top K categorical accuracy.
        """
        predictions, targets = self._check_shapes(predictions, targets)

        sorted_indices = torch.topk(predictions, self._k, dim=1)[1]
        expanded_targets = targets.view(-1, 1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_targets), dim=1)
        _num_correct = torch.sum(correct).item()
        _num_examples = correct.shape[0]

        if _num_examples == 0:
            raise ValueError("TopKCategoricalAccuracy must have at"
                             "least one example before it can be computed.")
        return _num_correct / _num_examples


class MeanSquaredError(Metric):
    """Calculate the Mean Squared Error."""

    def __init__(self):
        """Class constructor."""
        super().__init__()

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Squared Error.

              Args:
                  predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
                  targets (:obj:`torch.Tensor`): The ground truth.

              Returns:
                  float: The batch's mean squared error.
        """
        predictions, targets = self._check_shapes(predictions, targets)

        squared_errors = torch.pow(predictions - targets.view_as(predictions), 2)
        _sum_of_squared_errors = torch.sum(squared_errors).item()
        _num_examples = targets.shape[0]

        if _num_examples == 0:
            raise ValueError('MeanSquaredError must have at least one example before it can be computed.')

        return _sum_of_squared_errors / _num_examples


class RootMeanSquaredError(MeanSquaredError):
    """Calculate the Root Mean Squared Error."""

    def __init__(self):
        """Class constructor."""
        super().__init__()

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Root Mean Squared Error.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's root mean squared error.
        """
        mse = super(RootMeanSquaredError, self).compute(predictions, targets)
        return math.sqrt(mse)


class MeanAbsoluteError(Metric):
    """Calculate the Mean Absolute Error."""

    def __init__(self):
        """Class constructor."""
        super().__init__()

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Absolute Error.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's mean absolute error..
        """
        predictions, targets = self._check_shapes(predictions, targets)

        absolute_errors = torch.abs(predictions - targets.view_as(predictions))
        _sum_of_absolute_errors = torch.sum(absolute_errors).item()
        _num_examples = targets.shape[0]

        if _num_examples == 0:
            raise ValueError('MeanAbsoluteError must have at least one example before it can be computed.')

        return _sum_of_absolute_errors / _num_examples


class MeanPairwiseDistance(Metric):
    """Calculate the Mean Pairwise Distance."""

    def __init__(self, p=2, epsilon=1e-6):
        """Class constructor."""
        super(MeanPairwiseDistance, self).__init__()
        self._p = p
        self._epsilon = epsilon

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Pairwise Distance.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's accuracy.
        """
        predictions, targets = self._check_shapes(predictions, targets)

        distances = pairwise_distance(predictions, targets, p=self._p, eps=self._epsilon)
        _sum_of_distances = torch.sum(distances).item()
        _num_examples = targets.shape[0]

        if _num_examples == 0:
            raise ValueError('MeanAbsoluteError must have at least one example before it can be computed.')

        return _sum_of_distances / _num_examples


class DiceCoefficient(Metric):
    """Metric defining the commonly used Dice Coefficient."""

    def __init__(self, epsilon=1e-6, ignore_index=None):
        """Class constructor."""
        super().__init__()
        self._epsilon = epsilon
        self._ignore_index = ignore_index

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Dice coefficient.

       Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's Dice coefficient.
        """
        predictions, targets = self._check_shapes(predictions, targets)


class MeanIOU(Metric):
    """Calculate the Mean IOU"""

    def __init__(self, skip_channels=None, ignore_index=None):
        """Class constructor."""
        super().__init__()
        self._skip_channels = skip_channels
        self._ignore_index = ignore_index

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Intersection Over Union.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's mean IOU.
        """
        predictions, targets = self._check_shapes(predictions, targets)


class Precision(BasePrecisionRecall):
    def __init__(self, average=False, is_multilabel=False):
        super(Precision, self).__init__(average=average, is_multilabel=is_multilabel)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Precision.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's precision.
        """
        predictions, targets = self._check_shapes(predictions, targets)

        self._select_metric_type(predictions, targets)

        if self._type == "binary":
            predictions = predictions.view(-1)
            targets = targets.view(-1)
        elif self._type == "multiclass":
            num_classes = predictions.size(1)
            if targets.max() + 1 > num_classes:
                raise ValueError("predictions contains less classes than targets. Number of predicted classes is {}"
                                 " and element in targets has invalid class = {}.".format(num_classes,
                                                                                          targets.max().item() + 1))
            targets = to_onehot(targets.view(-1), num_classes=num_classes)
            indices = torch.argmax(predictions, dim=1).view(-1)
            predictions = to_onehot(indices, num_classes=num_classes)
        elif self._type == "multilabel":
            # if targets, predictions shape is (N, C, ...) -> (C, N x ...)
            num_classes = predictions.size(1)
            predictions = torch.transpose(predictions, 1, 0).reshape(num_classes, -1)
            targets = torch.transpose(targets, 1, 0).reshape(num_classes, -1)

        targets = targets.type_as(predictions)
        correct = targets * predictions
        all_positives = predictions.sum(dim=0).type(torch.DoubleTensor)  # Convert from int cuda/cpu to double cpu

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=0)
        # Convert from int cuda/cpu to double cpu
        # We need double precision for the division true_positives / all_positives
        true_positives = true_positives.type(torch.DoubleTensor)

        if self._type == "multilabel":
            if not self._average:
                self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)
                self._positives = torch.cat([self._positives, all_positives], dim=0)
            else:
                self._true_positives += torch.sum(true_positives / (all_positives + self.eps))
                self._positives += len(all_positives)
        else:
            self._true_positives = true_positives
            self._positives = all_positives

        if not (isinstance(self._positives, torch.Tensor) or self._positives > 0):
            raise ValueError("{} must have at least one example before"
                             " it can be computed.".format(self.__class__.__name__))

        result = self._true_positives / (self._positives + self._epsilon)

        if self._average:
            return result.mean().item()
        else:
            return result


class Recall(BasePrecisionRecall):
    def __init__(self, average=False, is_multilabel=False):
        super(Recall, self).__init__(average=average, is_multilabel=is_multilabel)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Recall.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.

        Returns:
            float: The batch's recall.
        """
        predictions, targets = self._check_shapes(predictions, targets)
        self._select_metric_type(predictions, targets)

        if self._type == "binary":
            predictions = predictions.view(-1)
            targets = targets.view(-1)
        elif self._type == "multiclass":
            num_classes = predictions.size(1)
            if targets.max() + 1 > num_classes:
                raise ValueError("predictions contains less classes than targets. Number of predicted classes is {}"
                                 " and element in targets has invalid class = {}.".format(num_classes,
                                                                                          targets.max().item() + 1))
            targets = to_onehot(targets.view(-1), num_classes=num_classes)
            indices = torch.argmax(predictions, dim=1).view(-1)
            predictions = to_onehot(indices, num_classes=num_classes)
        elif self._type == "multilabel":
            # if targets, predictions shape is (N, C, ...) -> (C, N x ...)
            num_classes = predictions.size(1)
            predictions = torch.transpose(predictions, 1, 0).reshape(num_classes, -1)
            targets = torch.transpose(targets, 1, 0).reshape(num_classes, -1)

        targets = targets.type_as(predictions)
        correct = targets * predictions
        actual_positives = targets.sum(dim=0).type(torch.DoubleTensor)  # Convert from int cuda/cpu to double cpu

        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual_positives)
        else:
            true_positives = correct.sum(dim=0)

        # Convert from int cuda/cpu to double cpu
        # We need double precision for the division true_positives / actual_positives
        true_positives = true_positives.type(torch.DoubleTensor)

        if self._type == "multilabel":
            if not self._average:
                self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)
                self._positives = torch.cat([self._positives, actual_positives], dim=0)
            else:
                self._true_positives += torch.sum(true_positives / (actual_positives + self.eps))
                self._positives += len(actual_positives)
        else:
            self._true_positives += true_positives
            self._positives += actual_positives

        if not (isinstance(self._positives, torch.Tensor) or self._positives > 0):
            raise ValueError("{} must have at least one example before"
                             " it can be computed.".format(self.__class__.__name__))

        result = self._true_positives / (self._positives + self.eps)

        if self._average:
            return result.mean().item()
        else:
            return result
