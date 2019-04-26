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
import numbers
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics import MetricsLambda


def dice_coefficient(cm, ignore_index=None):
    if not isinstance(cm, ConfusionMatrix):
        raise TypeError("Argument cm should be instance of ConfusionMatrix, but given {}".format(type(cm)))

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError("ignore_index should be non-negative integer, but given {}".format(ignore_index))

    # Increase floating point precision
    cm = cm.type(torch.float64)
    dice = 2 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(dice_vector):
            if ignore_index >= len(dice_vector):
                raise ValueError("ignore_index {} is larger than the length of IoU vector {}"
                                 .format(ignore_index, len(dice_vector)))
            indices = list(range(len(dice_vector)))
            indices.remove(ignore_index)
            return dice_vector[indices]

        return MetricsLambda(ignore_index_fn, dice)
    else:
        return dice


def mean_dice(cm, ignore_index=None):
    return dice_coefficient(cm=cm, ignore_index=ignore_index).mean()
