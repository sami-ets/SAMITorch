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

import unittest

import numpy as np
from hamcrest import *

from samitorch.metrics.gauges import RunningAverageGauge


class AccuracyGaugeTest(unittest.TestCase):
    VALID_VALUE_1 = 0.50
    VALID_VALUE_2 = 0.70
    VALID_VALUE_3 = 0.90
    INVALID_VALUE = "DARTH VADER"

    def setUp(self):
        self.gauge = RunningAverageGauge()
        self.gauge.reset()

    def tearDown(self):
        self.gauge.reset()

    def test_should_keep_a_valid_average_and_reset(self):
        expected_average = np.array([self.VALID_VALUE_1, self.VALID_VALUE_2, self.VALID_VALUE_3]).mean()
        expected_reset_average = 0

        self.gauge.update(self.VALID_VALUE_1)
        self.gauge.update(self.VALID_VALUE_2, 2)
        self.gauge.update(self.VALID_VALUE_3)

        assert_that(self.gauge.average, close_to(expected_average, 0.00001))

        self.gauge.reset()

        assert_that(self.gauge.average, equal_to(expected_reset_average))

    def test_average_should_be_zero_for_empty_gauge(self):
        expected_average = 0

        assert_that(self.gauge.average, equal_to(expected_average))

    def test_should_reset(self):
        assert_that(calling(self.gauge.update).with_args(self.INVALID_VALUE, 1), raises(TypeError))


if __name__ == 'main':
    unittest.main()
