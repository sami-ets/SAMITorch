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
import os
import numpy as np
import nibabel as nib

from hamcrest import *

from samitorch.inputs.datasets import NiftiDataset, MultimodalNiftiDataset


class NiftiDatasetTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    TEST_SOURCE_IMAGE = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "T1/Normalized_Processed_subject-1-T1.nii")).get_fdata()
    TEST_LABEL_IMAGE = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "label/Normalized_Processed_subject-1-label.nii")).get_fdata()

    def setUp(self):
        pass

    def test_should_instantiate_non_cached_dataset(self):
        dataset = NiftiDataset(source_dir=self.PATH_TO_SOURCE, target_dir=self.PATH_TO_TARGET, preload=False)
        assert_that(dataset, is_not(None))

    def test_should_instantiate_in_memmap_dataset(self):
        dataset = NiftiDataset(source_dir=self.PATH_TO_SOURCE, target_dir=self.PATH_TO_TARGET, preload=True)
        assert_that(dataset, is_not(None))
        assert_that(len(dataset._images), is_not(None))

    def test_should_give_a_tuple_of_training_elements(self):
        dataset = NiftiDataset(source_dir=self.PATH_TO_SOURCE, target_dir=self.PATH_TO_TARGET, preload=False)

        source, label = dataset.__getitem__(0)

        np.testing.assert_array_equal(source, self.TEST_SOURCE_IMAGE)
        np.testing.assert_array_equal(label, self.TEST_LABEL_IMAGE)


class MultimodalNiftiDatasetTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE_T1 = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATH_TO_SOURCE_T2 = os.path.join(TEST_DATA_FOLDER_PATH, "T2")
    PATHS_TO_SOURCES = [PATH_TO_SOURCE_T1, PATH_TO_SOURCE_T2]
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    PATHS_TO_TARGETS = [PATH_TO_TARGET]
    TEST_SOURCE_IMAGE_T1 = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "T1/Normalized_Processed_subject-1-T1.nii")).get_fdata()
    TEST_SOURCE_IMAGE_T2 = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "T2/Normalized_Processed_subject-1-T2.nii")).get_fdata()
    TEST_LABEL_IMAGE = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "label/Normalized_Processed_subject-1-label.nii")).get_fdata()

    def setUp(self):
        pass

    def test_should_instantiate_non_cached_dataset(self):
        dataset = MultimodalNiftiDataset(source_dirs=self.PATHS_TO_SOURCES, target_dirs=self.PATHS_TO_TARGETS)
        assert_that(dataset, is_not(None))

    def test_should_give_a_tuple_of_training_elements(self):
        dataset = MultimodalNiftiDataset(source_dirs=self.PATHS_TO_SOURCES, target_dirs=self.PATHS_TO_TARGETS)

        source, label = dataset.__getitem__(0)

        source_t1, source_t2 = source

        np.testing.assert_array_equal(source_t1, self.TEST_SOURCE_IMAGE_T1)
        np.testing.assert_array_equal(source_t2, self.TEST_SOURCE_IMAGE_T2)
        np.testing.assert_array_equal(np.squeeze(label, 0), self.TEST_LABEL_IMAGE)
