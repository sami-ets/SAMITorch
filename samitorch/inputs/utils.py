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
import torch

from samitorch.inputs.batch import ImageBatch, PatchBatch


def sample_collate(batch: list):
    batch = ImageBatch(samples=batch)
    return batch.x, [batch.y, batch.dataset_id]


def augmented_sample_collate(batch: list):
    batch = ImageBatch(samples=batch)
    return [batch.x, batch.augmented_x], [batch.y, batch.dataset_id]


def patch_collate(batch: list):
    batch = PatchBatch(samples=batch)
    return batch.x, [batch.y, batch.dataset_id]
