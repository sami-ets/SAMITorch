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
        training_dataset, test_dataset = PatchDatasetFactory.create_train_test(source_dir=self.TEST_DATA_FOLDER_PATH,
                                                                               target_dir=self.PATH_TO_TARGET,
                                                                               modality=Modality.T1,
                                                                               patch_size=(1, 32, 32, 32),
                                                                               step=(1, 32, 32, 32),
                                                                               dataset_id=0,
                                                                               test_size=0.2)
        assert_that(training_dataset, instance_of(PatchDataset))
        assert_that(test_dataset, instance_of(PatchDataset))


class TestMultimodalPatchDatasetFactory(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self) -> None:
        pass

    def test_should_instantiate_both_training_and_test_dataset(self):
        training_dataset, test_dataset = PatchDatasetFactory.create_multimodal_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modality_1=Modality.T1,
            modality_2=Modality.T2,
            patch_size=(1, 32, 32, 32),
            step=(1, 32, 32, 32),
            dataset_id=0,
            test_size=0.2)
        assert_that(training_dataset, instance_of(MultimodalPatchDataset))
        assert_that(test_dataset, instance_of(MultimodalPatchDataset))


class TestSegmentationDatasetFactory(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self) -> None:
        pass

    def test_should_instantiate_both_training_and_test_dataset(self):
        training_dataset, test_dataset = SegmentationDatasetFactory.create_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modality=Modality.T1,
            dataset_id=0,
            test_size=0.2)
        assert_that(training_dataset, instance_of(SegmentationDataset))
        assert_that(test_dataset, instance_of(SegmentationDataset))


class TestMultimodalSegmentationDatasetFactory(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")

    def setUp(self) -> None:
        pass

    def test_should_instantiate_both_training_and_test_dataset(self):
        training_dataset, test_dataset = SegmentationDatasetFactory.create_multimodal_train_test(
            source_dir=self.TEST_DATA_FOLDER_PATH,
            target_dir=self.PATH_TO_TARGET,
            modality_1=Modality.T1,
            modality_2=Modality.T2,
            dataset_id=0,
            test_size=0.2)
        assert_that(training_dataset, instance_of(MultimodalSegmentationDataset))
        assert_that(test_dataset, instance_of(MultimodalSegmentationDataset))
