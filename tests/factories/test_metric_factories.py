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

from enum import Enum

from hamcrest import *

from samitorch.factories.factories import MetricsFactory
from samitorch.factories.enums import Metrics


class IncorrectMetrics(Enum):
    Incorrect = "Incorrect"


class ModelFactoryTest(unittest.TestCase):

    def setUp(self):
        self.metric_factory = MetricsFactory()

    def test_should_instantiate_dice_metric(self):
        metric = self.metric_factory.create_metric(Metrics.Dice, num_classes=3)
        assert_that(metric, is_not(None))
