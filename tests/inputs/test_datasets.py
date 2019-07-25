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

import os
import unittest

import nibabel as nib
import numpy as np
import torch
from hamcrest import *
from torchvision.transforms import transforms

from samitorch.inputs.datasets import SegmentationDataset, PatchDataset, MultimodalSegmentationDataset
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, ToNDTensor, ToNifti1Image, NiftiToDisk


class SegmentationDatasetTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    TEST_SOURCE_IMAGE = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "T1/1-T1.nii")).get_fdata()
    TEST_LABEL_IMAGE = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "label/1-Labels.nii")).get_fdata()

    def setUp(self):
        pass

    def test_should_instantiate_non_cached_dataset(self):
        dataset = SegmentationDataset(source_dir=self.PATH_TO_SOURCE, target_dir=self.PATH_TO_TARGET)
        assert_that(dataset, is_not(None))

    def test_should_give_a_tuple_of_training_elements(self):
        dataset = SegmentationDataset(source_dir=self.PATH_TO_SOURCE, target_dir=self.PATH_TO_TARGET)

        sample = dataset.__getitem__(0)

        np.testing.assert_array_equal(nib.load(sample.x).get_fdata(), self.TEST_SOURCE_IMAGE)
        np.testing.assert_array_equal(nib.load(sample.y).get_fdata(), self.TEST_LABEL_IMAGE)


class MultimodalSegmentationDatasetTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE_T1 = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATH_TO_SOURCE_T2 = os.path.join(TEST_DATA_FOLDER_PATH, "T2")
    PATHS_TO_SOURCES = [PATH_TO_SOURCE_T1, PATH_TO_SOURCE_T2]
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    PATHS_TO_TARGETS = [PATH_TO_TARGET]
    TEST_SOURCE_IMAGE_T1 = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "T1/1-T1.nii")).get_fdata()
    TEST_SOURCE_IMAGE_T2 = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "T2/1-T2.nii")).get_fdata()
    TEST_LABEL_IMAGE = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "label/1-Labels.nii")).get_fdata()

    def setUp(self):
        pass

    def test_should_instantiate_non_cached_dataset(self):
        dataset = MultimodalSegmentationDataset(source_dirs=self.PATHS_TO_SOURCES, target_dirs=self.PATHS_TO_TARGETS)
        assert_that(dataset, is_not(None))

    def test_should_give_a_tuple_of_training_elements(self):
        dataset = MultimodalSegmentationDataset(source_dirs=self.PATHS_TO_SOURCES, target_dirs=self.PATHS_TO_TARGETS)

        sample = dataset.__getitem__(0)

        source_t1, source_t2 = sample.x[0], sample.x[1]
        label = sample.y[0]

        np.testing.assert_array_equal(nib.load(source_t1).get_fdata(), self.TEST_SOURCE_IMAGE_T1)
        np.testing.assert_array_equal(nib.load(source_t2).get_fdata(), self.TEST_SOURCE_IMAGE_T2)
        np.testing.assert_array_equal(nib.load(label).get_fdata(), self.TEST_LABEL_IMAGE)


class SegmentationDatasetWithTransformsTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE_T1 = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATHS_TO_SOURCES = [PATH_TO_SOURCE_T1]
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    PATHS_TO_TARGETS = [PATH_TO_TARGET]

    def setUp(self):
        pass

    def test_should_instantiate_dataset_with_transforms(self):
        transforms_ = transforms.Compose([ToNumpyArray()])
        dataset = MultimodalSegmentationDataset(source_dirs=self.PATHS_TO_SOURCES, target_dirs=self.PATHS_TO_TARGETS,
                                                transform=transforms_)

        assert_that(dataset, is_not(None))
        assert_that(dataset._transform, is_not(None))

    def test_should_return_a_sample_of_Numpy_ndarrays_with_respective_transform(self):
        transforms_ = transforms.Compose([ToNumpyArray()])
        dataset = MultimodalSegmentationDataset(source_dirs=self.PATHS_TO_SOURCES, target_dirs=self.PATHS_TO_TARGETS,
                                                transform=transforms_)
        sample = dataset.__getitem__(0)

        assert_that(sample.x, instance_of(np.ndarray))
        assert_that(sample.x.ndim, is_(4))
        assert_that(sample.x.shape[0], is_(1))
        assert_that(sample.y, instance_of(np.ndarray))
        assert_that(sample.y.shape[0], is_(1))


class MultimodalSegmentationDatasetWithTransformsTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE_T1 = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATH_TO_SOURCE_T2 = os.path.join(TEST_DATA_FOLDER_PATH, "T2")
    PATHS_TO_SOURCES = [PATH_TO_SOURCE_T1, PATH_TO_SOURCE_T2]
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    PATHS_TO_TARGETS = [PATH_TO_TARGET]

    def setUp(self):
        pass

    def test_should_instantiate_dataset_with_transforms(self):
        transforms_ = transforms.Compose([ToNumpyArray()])
        dataset = MultimodalSegmentationDataset(source_dirs=self.PATHS_TO_SOURCES, target_dirs=self.PATHS_TO_TARGETS,
                                                transform=transforms_)

        assert_that(dataset, is_not(None))
        assert_that(dataset._transform, is_not(None))

    def test_should_return_a_sample_of_Numpy_ndarrays_with_respective_transform(self):
        transforms_ = transforms.Compose([ToNumpyArray()])

        dataset = MultimodalSegmentationDataset(source_dirs=self.PATHS_TO_SOURCES, target_dirs=self.PATHS_TO_TARGETS,
                                                transform=transforms_)
        sample = dataset.__getitem__(0)

        assert_that(sample.x, instance_of(np.ndarray))
        assert_that(sample.x.ndim, is_(4))
        assert_that(sample.x.shape[0], is_(2))
        assert_that(sample.y, instance_of(np.ndarray))
        assert_that(sample.y.shape[0], is_(1))


class PatchDatasetWithTransformsTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    TEST_SOURCE_IMAGE = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "T1/1-T1.nii")).get_fdata()
    TEST_LABEL_IMAGE = nib.load(
        os.path.join(TEST_DATA_FOLDER_PATH, "label/1-Labels.nii")).get_fdata()
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/NiftiPatchDataset")
    PATH_TO_SAVE_X = os.path.join(OUTPUT_DATA_FOLDER_PATH, "nifti_patch_x.nii")
    PATH_TO_SAVE_Y = os.path.join(OUTPUT_DATA_FOLDER_PATH, "nifti_patch_y.nii")

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_should_instantiate_dataset_with_transforms(self):
        transforms_ = transforms.Compose([ToNDTensor()])
        dataset = PatchDataset(source_dir=self.PATH_TO_SOURCE, target_dir=self.PATH_TO_TARGET,
                               patch_shape=(1, 32, 32, 32), step=(1, 8, 8, 8),
                               transform=transforms_)

        assert_that(dataset, is_not(None))
        assert_that(dataset._transform, is_not(None))

    def test_should_return_a_sample_of_Numpy_ndarrays_with_respective_transform(self):
        transforms_ = transforms.Compose([ToNDTensor()])
        dataset = PatchDataset(source_dir=self.PATH_TO_SOURCE, target_dir=self.PATH_TO_TARGET,
                               patch_shape=(1, 32, 32, 32), step=(1, 32, 32, 32),
                               transform=transforms_)
        sample = dataset.__getitem__(0)

        assert_that(sample.x, instance_of(torch.Tensor))
        assert_that(sample.x.ndimension(), is_(4))
        assert_that(sample.x.shape[0], is_(1))
        assert_that(sample.y.ndimension(), is_(4))
        assert_that(sample.y, instance_of(torch.Tensor))
        assert_that(sample.y.shape[0], is_(1))

    def test_should_return_a_sample_of_Numpy_ndarrays_for_inspection(self):
        transforms_ = transforms.Compose([ToNDTensor()])
        dataset = PatchDataset(source_dir=self.PATH_TO_SOURCE, target_dir=self.PATH_TO_TARGET,
                               patch_shape=(1, 24, 120, 120), step=(1, 24, 120, 120),
                               transform=transforms_)
        sample = dataset[0]

        sample = Sample(x=sample.x.numpy(), y=sample.y.numpy(), is_labeled=True)

        transforms_ = transforms.Compose(
            [ToNifti1Image([None, None]), NiftiToDisk([self.PATH_TO_SAVE_X, self.PATH_TO_SAVE_Y])])
        transforms_(sample)
