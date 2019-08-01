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

import torch
from hamcrest import *

from samitorch.inputs.batch import ImageBatch, PatchBatch
from samitorch.inputs.images import Modality
from samitorch.inputs.utils import sample_collate, patch_collate
from samitorch.inputs.datasets import PatchDatasetFactory, SegmentationDatasetFactory


class SegmentationDataLoaderTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self):
        self._training_patch_dataset, self._validation_patch_dataset = SegmentationDatasetFactory.create_train_test(
            self.TEST_DATA_FOLDER_PATH,
            self.PATH_TO_TARGET,
            Modality.T1,
            dataset_id=0,
            test_size=0.2
        )

    def test_should_initialize_dataloader_with_dataset(self):
        dataloader = torch.utils.data.DataLoader(dataset=self._training_patch_dataset,
                                                 batch_size=2,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 collate_fn=sample_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

    def test_should_initialize_dataloader_and_get_batch(self):
        dataloader = torch.utils.data.DataLoader(dataset=self._training_patch_dataset,
                                                 batch_size=2,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 collate_fn=sample_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

        batch = next(iter(dataloader))

        assert_that(batch, instance_of(ImageBatch))


class PatchDataLoaderTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self):
        self._training_patch_dataset, self._validation_patch_dataset = PatchDatasetFactory.create_train_test(
            self.TEST_DATA_FOLDER_PATH,
            self.PATH_TO_TARGET,
            Modality.T1,
            patch_size=(1, 32, 32, 32),
            step=(1, 32, 32, 32),
            dataset_id=0,
            test_size=0.2
        )

    def test_should_initialize_dataloader_with_dataset(self):
        dataloader = torch.utils.data.DataLoader(dataset=self._training_patch_dataset,
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 collate_fn=patch_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

    def test_should_initialize_dataloader_and_get_batch(self):
        dataloader = torch.utils.data.DataLoader(dataset=self._training_patch_dataset,
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 collate_fn=patch_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

        batch = next(iter(dataloader))

        assert_that(batch, instance_of(PatchBatch))
        assert_that(batch.x, instance_of(torch.Tensor))
        assert_that(batch.x.size(0), is_(32))


class MultimodalSegmentationDataLoaderTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self):
        self._training_patch_dataset, self._validation_patch_dataset = SegmentationDatasetFactory.create_multimodal_train_test(
            self.TEST_DATA_FOLDER_PATH,
            self.PATH_TO_TARGET,
            Modality.T1,
            Modality.T2,
            dataset_id=0,
            test_size=0.2
        )

    def test_should_initialize_dataloader_with_dataset(self):
        dataloader = torch.utils.data.DataLoader(dataset=self._training_patch_dataset,
                                                 batch_size=2,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 collate_fn=sample_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

    def test_should_initialize_dataloader_and_get_batch(self):
        dataloader = torch.utils.data.DataLoader(dataset=self._training_patch_dataset,
                                                 batch_size=2,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 collate_fn=sample_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

        batch = next(iter(dataloader))

        assert_that(batch, instance_of(ImageBatch))
        assert_that(batch.x, instance_of(torch.Tensor))
        assert_that(batch.x.size(0), is_(2))


class MultimodalPatchDataLoaderTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self):
        self._training_patch_dataset, self._validation_patch_dataset = PatchDatasetFactory.create_multimodal_train_test(
            self.TEST_DATA_FOLDER_PATH,
            self.PATH_TO_TARGET,
            Modality.T1,
            Modality.T2,
            patch_size=(1, 32, 32, 32),
            step=(1, 32, 32, 32),
            dataset_id=0,
            test_size=0.2
        )

    def test_should_initialize_dataloader_with_dataset(self):
        dataloader = torch.utils.data.DataLoader(dataset=self._training_patch_dataset,
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 collate_fn=patch_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

    def test_should_initialize_dataloader_and_get_batch(self):
        dataloader = torch.utils.data.DataLoader(dataset=self._training_patch_dataset,
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 collate_fn=patch_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

        batch = next(iter(dataloader))

        assert_that(batch, instance_of(PatchBatch))
        assert_that(batch.x, instance_of(torch.Tensor))
        assert_that(batch.x.size(0), is_(32))
