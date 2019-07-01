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
import torch
import os

from hamcrest import *

from samitorch.utils.utils import to_onehot, flatten, glob_imgs


class UtilsTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")

    def setUp(self):
        pass

    @staticmethod
    def test_to_onehot():
        indices = torch.LongTensor([0, 1, 2, 3])
        actual = to_onehot(indices, 4)
        expected = torch.eye(4)
        assert actual.equal(expected)

        y = torch.randint(0, 21, size=(1000,))
        y_ohe = to_onehot(y, num_classes=21)
        y2 = torch.argmax(y_ohe, dim=1)
        assert y.equal(y2)

        y = torch.randint(0, 21, size=(4, 250, 255))
        y_ohe = to_onehot(y, num_classes=21)
        y2 = torch.argmax(y_ohe, dim=1)
        assert y.equal(y2)

        y = torch.randint(0, 21, size=(4, 150, 155, 4, 6))
        y_ohe = to_onehot(y, num_classes=21)
        y2 = torch.argmax(y_ohe, dim=1)
        assert y.equal(y2)

    @staticmethod
    def test_flatten():
        x = torch.randint(0, 255, size=(15, 3, 10, 10))
        flattened_x = flatten(x)
        assert flattened_x.size() == torch.Size([3, 1500])

    def test_glob_imgs(self):
        all_paths = [os.path.join(self.TEST_DATA_FOLDER_PATH, "DTI.nii"),
                     os.path.join(self.TEST_DATA_FOLDER_PATH, "FA.nii"),
                     os.path.join(self.TEST_DATA_FOLDER_PATH, "Mask.nii"),
                     os.path.join(self.TEST_DATA_FOLDER_PATH, "T1.nii"),
                     os.path.join(self.TEST_DATA_FOLDER_PATH, "T1_1mm.nii")]

        paths = glob_imgs(self.TEST_DATA_FOLDER_PATH)

        assert_that(paths, equal_to(all_paths))
