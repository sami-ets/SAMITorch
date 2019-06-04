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


class Gauge(object):
    """Base class for all Metrics."""

    def __init__(self):
        """Class constructor."""
        self.reset()

    @abc.abstractmethod
    def update(self, value: float, **kwargs):
        """Update the gauge's state.

        Args:
            value (float): The value with which to update the gauge.
            **kwargs (dict): keyword arguments.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        """Reset gauge's data.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()


class RunningAverageGauge(Gauge):
    """A moving accuracy gauge."""

    def __init__(self) -> None:
        """Class constructor."""
        self.count = 0
        self.sum = 0
        self.average = 0
        super(RunningAverageGauge).__init__()

    def update(self, value: float, n_data: int = 1) -> None:
        """Update the moving average.

        Args:
            value (float): The value with which to update the average.
            n_data (int): The number of data passed to this moving average.
        """
        self.count += n_data
        self.sum += value * n_data
        self.average = self.sum / self.count

    def reset(self) -> None:
        """Reset the moving average."""
        self.count = 0
        self.sum = 0
        self.average = 0
