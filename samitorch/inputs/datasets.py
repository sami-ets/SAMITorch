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

import numpy as np
import random

from typing import Callable, List, Optional, Tuple, Union

from torch.utils.data.dataset import Dataset

from samitorch.utils.utils import glob_imgs
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, ToNDArrayPatches, ToNDTensor, PadToPatchShape
from torchvision.transforms.transforms import Compose

from samitorch.utils.slice_builder import SliceBuilder


class NiftiPatchDataset(Dataset):
    """
    Create a dataset of patches in PyTorch for reading NiFTI files and slicing them into fixed shape.
    """

    def __init__(self, source_dir: str, target_dir: str, patch_shape: Tuple[int, int, int, int],
                 step: Tuple[int, int, int, int],
                 transform: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_dir (str): Path to source images.
            target_dir (str): Path to target (labels) images.
            transform (Callable): transform to apply to both source and target images.
        """
        self._source_dir, self._target_dir = source_dir, target_dir
        self._source_paths, self._target_paths = glob_imgs(source_dir), glob_imgs(target_dir)
        self._transform = transform

        if len(self._source_paths) != len(self._target_paths) or len(self._source_paths) == 0:
            raise ValueError("Number of source and target images must be equal and non-zero.")

        pre_transforms = Compose([ToNumpyArray(), PadToPatchShape(patch_shape, step)])

        self._images = list()
        self._labels = list()

        for idx in range(len(self._source_paths)):
            source_path, target_path = self._source_paths[idx], self._target_paths[idx]
            sample = Sample(x=source_path, y=target_path, is_labeled=True)
            transformed_sample = pre_transforms(sample)
            self._images.append(transformed_sample.x)
            self._labels.append(transformed_sample.y)

        self._slice_builder = SliceBuilder(self._images[0].shape, patch_size=patch_shape, step=step)

        slices = []
        for i in range(len(self._images)):
            slices.append(list(
                filter(lambda slice: np.count_nonzero(self._images[i][slice]) > 0, self._slice_builder.image_slices)))

        self._slices = [j for sub in slices for j in sub]

        self._num_patches = len(self._slices)

    def __len__(self):
        return self._num_patches

    def __getitem__(self, idx: int):
        id = random.randint(0, len(self._images)-1)
        slice = self._slices[idx]
        img = self._images[id]
        label = self._labels[id]
        x, y = img[slice], label[slice]

        sample = Sample(x=x, y=y, is_labeled=True)

        if self._transform is not None:
            sample = self._transform(sample)
        return sample.unpack()


class NiftiDataset(Dataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_dir: str, target_dir: str, transform: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_dir (str): Path to source images.
            target_dir (str): Path to target (labels) images.
            transform (Callable): transform to apply to both source and target images.
        """
        self._source_dir, self._target_dir = source_dir, target_dir
        self._source_paths, self._target_paths = glob_imgs(source_dir), glob_imgs(target_dir)
        self._transform = transform

        if len(self._source_paths) != len(self._target_paths) or len(self._source_paths) == 0:
            raise ValueError("Number of source and target images must be equal and non-zero.")

    def __len__(self):
        return len(self._source_paths)

    def __getitem__(self, idx: int):
        source_path, target_path = self._source_paths[idx], self._target_paths[idx]
        sample = Sample(x=source_path, y=target_path, is_labeled=True)

        if self._transform is not None:
            sample = self._transform(sample)
        return sample.unpack()


class MultimodalDataset(Dataset):
    """
    Base class for Multimodal NifTI Dataset.
    """

    def __init__(self, source_dirs: List[str], target_dirs: List[str], transform: Optional[Callable] = None) -> None:
        """
        Multimodal Dataset initializer.

        Args:
            source_dirs (List[str]): Paths to source images.
            target_dirs (List[str]): paths to target (labels) images.
            transform (Callable): transform to apply to both source and target images.w
        """
        self._source_dirs, self._target_dirs = source_dirs, target_dirs
        self._source_paths, self._target_paths = [glob_imgs(sd) for sd in source_dirs], [glob_imgs(td) for td
                                                                                         in
                                                                                         target_dirs]
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
        sample = Sample(x=source_image, y=target_image, is_labeled=True)

        if self._transform is not None:
            sample = self._transform(sample)
        return sample.unpack()


class MultimodalNiftiDataset(MultimodalDataset):
    """
    Create a dataset class in PyTorch for reading N types of NIfTI files to M types of output NIfTI files
    Note that all images must have the same dimensions.
    """

    def __init__(self, source_dirs: List[str], target_dirs: List[str], transform: Optional[Callable] = None) -> None:
        super(MultimodalNiftiDataset, self).__init__(source_dirs, target_dirs, transform)
