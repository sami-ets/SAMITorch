#  -*- coding: utf-8 -*-
#  Copyright 2019 SAMITorch Authors. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import os
import unittest

import torch
from hamcrest import *

from samitorch.inputs.datasets import PatchDataset, MultimodalPatchDataset, SegmentationDataset, \
    MultimodalSegmentationDataset, PatchDatasetFactory, SegmentationDatasetFactory
from samitorch.inputs.images import Modality


class TestPatchDatasetFactory(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self) -> None:
        pass

    def test_should_instantiate_both_training_and_test_dataset(self):
        training_dataset, test_dataset = PatchDatasetFactory._create_single_modality_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modality=Modality.T1,
            patch_size=(1, 32, 32, 32),
            step=(1, 32, 32, 32),
            dataset_id=0,
            test_size=0.2)
        assert_that(training_dataset, instance_of(PatchDataset))
        assert_that(test_dataset, instance_of(PatchDataset))

    def test_should_produce_a_single_modality_input_with_one_channel(self):
        training_dataset, test_dataset = PatchDatasetFactory._create_single_modality_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modality=Modality.T1,
            patch_size=(1, 32, 32, 32),
            step=(1, 32, 32, 32),
            dataset_id=0,
            test_size=0.2)
        sample = training_dataset[0]

        assert_that(sample.x.slice.size(), is_(torch.Size([1, 32, 32, 32])))


class TestMultimodalPatchDatasetFactory(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self) -> None:
        pass

    def test_should_instantiate_both_training_and_test_dataset(self):
        training_dataset, test_dataset = PatchDatasetFactory.create_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modalities=[Modality.T1, Modality.T2],
            patch_size=(1, 32, 32, 32),
            step=(1, 32, 32, 32),
            dataset_id=0,
            test_size=0.2)
        assert_that(training_dataset, instance_of(MultimodalPatchDataset))
        assert_that(test_dataset, instance_of(MultimodalPatchDataset))

    def test_should_produce_a_multi_modal_input_with_two_channels(self):
        training_dataset, test_dataset = PatchDatasetFactory.create_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modalities=[Modality.T1, Modality.T2],
            patch_size=(1, 32, 32, 32),
            step=(1, 32, 32, 32),
            dataset_id=0,
            test_size=0.2)

        sample = training_dataset[0]

        assert_that(sample.x.slice.size(), is_(torch.Size([2, 32, 32, 32])))


class TestSegmentationDatasetFactory(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self) -> None:
        pass

    def test_should_instantiate_both_training_and_test_dataset(self):
        training_dataset, test_dataset = SegmentationDatasetFactory.create_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modalities=Modality.T1,
            dataset_id=0,
            test_size=0.2)
        assert_that(training_dataset, instance_of(SegmentationDataset))
        assert_that(test_dataset, instance_of(SegmentationDataset))

    def test_should_produce_a_single_modality_input_with_one_channel(self):
        training_dataset, test_dataset = SegmentationDatasetFactory.create_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modalities=Modality.T1,
            dataset_id=0,
            test_size=0.2)
        sample = training_dataset[0]

        assert_that(sample.x.size(), is_(torch.Size([1, 48, 240, 240])))


class TestMultimodalSegmentationDatasetFactory(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self) -> None:
        pass

    def test_should_instantiate_both_training_and_test_dataset(self):
        training_dataset, test_dataset = SegmentationDatasetFactory.create_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modalities=[Modality.T1, Modality.T2],
            dataset_id=0,
            test_size=0.2)
        assert_that(training_dataset, instance_of(MultimodalSegmentationDataset))
        assert_that(test_dataset, instance_of(MultimodalSegmentationDataset))

    def test_should_produce_a_multimodal_input_with_two_channels(self):
        training_dataset, test_dataset = SegmentationDatasetFactory.create_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modalities=[Modality.T1, Modality.T2],
            dataset_id=0,
            test_size=0.2)
        sample = training_dataset[0]

        assert_that(sample.x.size(), is_(torch.Size([2, 48, 240, 240])))
