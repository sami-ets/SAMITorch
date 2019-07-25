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

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms.transforms import Compose

from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, PadToPatchShape
from samitorch.utils.slice_builder import SliceBuilder
from samitorch.utils.utils import extract_file_paths


class SegmentationDataset(Dataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_dir: str, target_dir: str, dataset_id: int = None,
                 transform: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_dir (str): Path to source images.
            target_dir (str): Path to target (labels) images.
            transform (Callable): transform to apply to both source and target images.
        """
        self._source_dir, self._target_dir = source_dir, target_dir
        self._source_paths, self._target_paths = extract_file_paths(source_dir), extract_file_paths(target_dir)
        self._dataset_id = dataset_id
        self._transform = transform

        if len(self._source_paths) != len(self._target_paths) or len(self._source_paths) == 0:
            raise ValueError("Number of source and target images must be equal and non-zero.")

    def __len__(self):
        return len(self._source_paths)

    def __getitem__(self, idx: int):
        source_path, target_path = self._source_paths[idx], self._target_paths[idx]
        sample = Sample(x=source_path, y=target_path, dataset_id=self._dataset_id, is_labeled=True)

        if self._transform is not None:
            sample = self._transform(sample)
        return sample


class PatchDataset(SegmentationDataset):
    """
    Create a dataset of patches in PyTorch for reading NiFTI files and slicing them into fixed shape.
    """

    def __init__(self, source_dir: str, target_dir: str, patch_shape: Tuple[int, int, int, int],
                 step: Tuple[int, int, int, int], dataset_id: int = None, transform: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_dir (str): Path to source images.
            target_dir (str): Path to target (labels) images.
            transform (Callable): transform to apply to both source and target images.
        """
        super().__init__(source_dir, target_dir, dataset_id, transform)
        self._source_dir, self._target_dir = source_dir, target_dir
        self._source_paths, self._target_paths = extract_file_paths(source_dir), extract_file_paths(target_dir)
        self._dataset_id = dataset_id
        self._transform = transform

        if len(self._source_paths) != len(self._target_paths) or len(self._source_paths) == 0:
            raise ValueError("Number of source and target images must be equal and non-zero.")

        pre_transforms = Compose([ToNumpyArray(), PadToPatchShape(patch_shape, step)])

        self._images = list()
        self._labels = list()

        for idx in range(len(self._source_paths)):
            source_path, target_path = self._source_paths[idx], self._target_paths[idx]
            sample = Sample(x=source_path, y=target_path, dataset_id=self._dataset_id, is_labeled=True)
            transformed_sample = pre_transforms(sample)
            self._images.append(transformed_sample.x)
            self._labels.append(transformed_sample.y)

        self._slice_builder = SliceBuilder(self._images[0].shape, patch_size=patch_shape, step=step)

        slices = []
        for i in range(len(self._images)):
            slices.append(list(
                filter(lambda slice: np.count_nonzero(self._images[i][slice]) > 0, self._slice_builder.image_slices)))

        self._slices = [j for sub in slices for j in sub]
        self._slices = np.array(self._slices)
        self._num_patches = self._slices.shape[0]

    @property
    def slices(self):
        return self._slices

    @slices.setter
    def slices(self, slices):
        self._slices = slices

    @property
    def num_patches(self):
        return self._num_patches

    @num_patches.setter
    def num_patches(self, num_patches):
        self._num_patches = num_patches

    def __len__(self):
        return self._num_patches

    def __getitem__(self, idx: int):
        id = random.randint(0, len(self._images) - 1)
        slice = self._slices[idx]
        img = self._images[id]
        label = self._labels[id]
        x, y = img[tuple(slice)], label[tuple(slice)]

        sample = Sample(x=x, y=y, dataset_id=self._dataset_id, is_labeled=True)

        if self._transform is not None:
            sample = self._transform(sample)
        return sample


class MultimodalSegmentationDataset(Dataset):
    """
    Base class for Multimodal Dataset.
    """

    def __init__(self, source_dirs: List[str], target_dirs: List[str], dataset_id: int = None,
                 transform: Optional[Callable] = None) -> None:

        """
        Multimodal Dataset initializer.

        Args:
            source_dirs (List[str]): Paths to source images.
            target_dirs (List[str]): paths to target (labels) images.
            transform (Callable): transform to apply to both source and target images.w
        """
        self._source_dirs, self._target_dirs = source_dirs, target_dirs
        self._source_paths, self._target_paths = [extract_file_paths(sd) for sd in source_dirs], [extract_file_paths(td)
                                                                                                  for td
                                                                                                  in
                                                                                                  target_dirs]
        self._dataset_id = dataset_id
        self._transform = transform

        if any([len(self._source_paths[0]) != len(sfn) for sfn in self._source_paths]) or \
                any([len(self._target_paths[0]) != len(tfn) for tfn in self._target_paths]) or \
                len(self._source_paths[0]) != len(self._target_paths[0]) or \
                len(self._source_paths[0]) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')

    def __len__(self):
        return len(self._source_paths[0])

    def __getitem__(self, idx: int):
        source_image, target_image = [source_path[idx] for source_path in self._source_paths], [target_path[idx] for
                                                                                                target_path in
                                                                                                self._target_paths]
        sample = Sample(x=source_image, y=target_image, dataset_id=self._dataset_id, is_labeled=True)

        if self._transform is not None:
            sample = self._transform(sample)
        return sample
