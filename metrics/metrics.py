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


class Metric(object):
    """Base class of all metrics objects."""

    def __init__(self, is_multilabel=False):
        """Class constructor"""
        self._type = None
        self._num_classes = None
        self._is_multilabel = is_multilabel

    @abc.abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
        """Compute a specific metric.

        Args:
            predictions (:obj:`torch.Tensor`): The model's predictions on which the metric has to be computed.
            targets (:obj:`torch.Tensor`): The ground truth.
            **kwargs (dict): keyword arguments.

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


class Accuracy(Metric):
    """Metric defining accuracy."""

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
