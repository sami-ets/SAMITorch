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

from hamcrest import *

from samitorch.inputs.datasets import MultimodalNiftiDataset, NiftiPatchDataset
from samitorch.inputs.dataloaders import DataLoader
from samitorch.inputs.utils import sample_collate
from torchvision.transforms import Compose
from samitorch.inputs.transformers import ToNDTensor
from samitorch.inputs.batch import Batch


class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        transforms_ = Compose([ToNDTensor()])
        self._dataset = MultimodalNiftiDataset(source_dirs=["tests/data/test_dataset/T1",
                                                            "tests/data/test_dataset/T2"],
                                               target_dirs=["tests/data/test_dataset/label"])

        self._patch_dataset = NiftiPatchDataset(source_dir="tests/data/test_dataset/T1",
                                                target_dir="tests/data/test_dataset/label",
                                                patch_shape=(1, 32, 32, 32),
                                                step=(1, 8, 8, 8),
                                                transform=transforms_)

    def test_should_initialize_dataloader(self):
        dataloader = DataLoader(dataset=self._dataset,
                                batch_size=2,
                                shuffle=True,
                                validation_split=1,
                                num_workers=2,
                                collate_fn=sample_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

    def test_should_initialize_dataloader_with_validation_split(self):
        dataloader = DataLoader(dataset=self._dataset,
                                batch_size=2,
                                shuffle=True,
                                validation_split=1,
                                num_workers=2,
                                collate_fn=sample_collate)
        valid_dataloader = dataloader.get_validation_dataloader()

        assert_that(dataloader, is_not(None))
        assert_that(valid_dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))
        assert_that(valid_dataloader, instance_of(torch.utils.data.dataloader.DataLoader))
        assert valid_dataloader.sampler.indices not in dataloader.sampler.indices

    def test_should_initialize_patch_dataloader(self):
        dataloader = DataLoader(dataset=self._patch_dataset,
                                batch_size=4,
                                shuffle=True,
                                validation_split=1,
                                num_workers=2,
                                collate_fn=sample_collate)

        assert_that(dataloader, is_not(None))
        assert_that(dataloader, instance_of(torch.utils.data.dataloader.DataLoader))

        batch = next(iter(dataloader))

        assert_that(batch, instance_of(Batch))
