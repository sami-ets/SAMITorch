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

from hamcrest import *
from metrics.gauges import AccuracyGauge


class AccuracyGaugeTest(unittest.TestCase):

    def setUp(self):
        self.the_gauge = AccuracyGauge()
        self.the_gauge.reset()

    def testEquals(self):
        self.the_gauge.reset()
        self.the_gauge.update(0.50, 2)
        self.the_gauge.update(0.70, 2)
        self.the_gauge.update(0.90, 2)

        assert_that(round(self.the_gauge.average, 2), equal_to(0.70))


if __name__ == 'main':
    unittest.main()
