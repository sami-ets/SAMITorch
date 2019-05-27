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
import torchvision
import numpy as np

from samitorch.transformers.transformers import RemapClassIDs


class RemapClassIDsTest(unittest.TestCase):

    def setUp(self):
        self.image = self._build_image(ids=[10, 150])
        self.correct_image = self._build_image(ids=[1, 2])
        self.transformer = RemapClassIDs([10, 150], [1, 2])
        self.composed = torchvision.transforms.Compose([self.transformer])

    def _build_image(self, ids: list):
        image = np.zeros((30, 30), dtype=np.int)
        image[1:11, 1:11] = ids[0]
        image[15:25, 15:25] = ids[1]
        return image

    def test_should_pass_with_replacing_initial_ids_with_new_ones(self):
        transformed_image = self.composed(self.image)
        np.testing.assert_array_equal(transformed_image, self.correct_image)
