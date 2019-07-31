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
from typing import Callable, List, Optional, Tuple

import numpy as np
from torchvision.transforms import Compose
from torch.utils.data.dataset import Dataset

from samitorch.inputs.transformers import ToNumpyArray, PadToPatchShape
from samitorch.inputs.sample import Sample
from samitorch.inputs.patch import Patch, CenterCoordinate
from samitorch.inputs.images import Modality


class SegmentationDataset(Dataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_paths: List[str], target_paths: List[str], samples: List[Sample], modality: Modality,
                 dataset_id: int = None, transforms: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_paths (List of str): Path to source images.
            target_paths (List of str): Path to target (labels) images.
            samples (list of :obj:`samitorch.inputs.sample.Sample`): A list of Sample objects.
            modality (:obj:`samitorch.inputs.images.Modalities`): The modality of the data set.
            dataset_id (int): An integer representing the ID of the data set.
            transforms (Callable): transform to apply to both source and target images.
        """
        self._source_paths = source_paths
        self._target_paths = target_paths
        self._samples = samples
        self._modality = modality
        self._dataset_id = dataset_id
        self._transform = transforms

        if len(self._source_paths) != len(self._target_paths) or len(self._source_paths) == 0:
            raise ValueError("Number of source and target images must be equal and non-zero.")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        sample = self._samples[idx]

        if self._transform is not None:
            sample = self._transform(sample)
        return sample


class MultimodalSegmentationDataset(Dataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_paths: List[str], target_paths: List[str], samples: List[Sample], modality_1: Modality,
                 modality_2: Modality, dataset_id: int = None, transforms: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_paths (List of str): Path to source images.
            target_paths (List of str): Path to target (labels) images.
            samples (list of :obj:`samitorch.inputs.sample.Sample`): A list of Sample objects.
            modality_1 (:obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
            modality_2 (:obj:`samitorch.inputs.images.Modalities`): The second modality of the data set.
            dataset_id (int): An integer representing the ID of the data set.
            transforms (Callable): transform to apply to both source and target images.
        """
        self._source_paths = source_paths
        self._target_paths = target_paths
        self._samples = samples
        self._modality_1 = modality_1
        self._modality_2 = modality_2
        self._dataset_id = dataset_id
        self._transform = transforms

        if len(self._source_paths) != len(self._target_paths) or len(self._source_paths) == 0:
            raise ValueError("Number of source and target images must be equal and non-zero.")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        sample = self._samples[idx]

        if self._transform is not None:
            sample = self._transform(sample)
        return sample


class PatchDataset(SegmentationDataset):
    """
    Create a dataset of patches in PyTorch for reading NiFTI files and slicing them into fixed shape.
    """

    def __init__(self, source_paths: List[str], target_paths: List[str], samples: List[Sample],
                 patch_size: Tuple[int, int, int, int],
                 step: Tuple[int, int, int, int], modality: Modality,
                 dataset_id: int = None, transforms: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_paths (List of str): Path to source images.
            target_paths (List of str): Path to target (labels) images.
            samples (list of :obj:`samitorch.inputs.sample.Sample`): A list of Sample objects.
            patch_size (Tuple of int): A tuple representing the desired patch size.
            step (Tuple of int): A tuple representing the desired step between two patches.
            modality (:obj:`samitorch.inputs.images.Modalities`): The modality of the data set.
            dataset_id (int): An integer representing the ID of the data set.
            transforms (Callable): transform to apply to both source and target images.
        """
        super(PatchDataset, self).__init__(source_paths, target_paths, samples, modality, dataset_id, transforms)
        self._pre_transform = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def modality(self):
        return self._modality

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        # Get the sample according to the id.
        sample = self._samples[idx]
        image_id = sample.x.image_id

        image = self._source_paths[image_id]
        target = self._target_paths[image_id]

        image = self._pre_transform(image)
        target = self._pre_transform(target)

        # Get the patch and its slice out of this sample.
        patch_x, patch_y = sample.x, sample.y
        slice_x, slice_y = patch_x.slice, patch_y.slice

        slice_x, slice_y = image[tuple(slice_x)], target[tuple(slice_y)]

        center_coordinate = CenterCoordinate(slice_x, slice_y)
        transformed_patch_x = Patch(slice_x, image_id, center_coordinate)
        transformed_patch_y = Patch(slice_y, image_id, center_coordinate)

        sample.x = transformed_patch_x
        sample.y = transformed_patch_y

        if self._transform is not None:
            sample = self._transform(sample)
        return sample


class MultimodalPatchDataset(MultimodalSegmentationDataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_paths: List[str], target_paths: List[str], samples: List[Sample],
                 patch_size: Tuple[int, int, int, int], step: Tuple[int, int, int, int], modality_1: Modality,
                 modality_2: Modality, dataset_id: int = None, transforms: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_paths (List of str): Path to source images.
            target_paths (List of str): Path to target (labels) images.
            samples (list of :obj:`samitorch.inputs.sample.Sample`): A list of Sample objects.
            modality_1 (:obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
            modality_2 (:obj:`samitorch.inputs.images.Modalities`): The second modality of the data set.
            dataset_id (int): An integer representing the ID of the data set.
            transforms (Callable): transform to apply to both source and target images.
        """
        super(MultimodalPatchDataset, self).__init__(source_paths, target_paths, samples, modality_1, modality_2,
                                                     dataset_id, transforms)
        self._pre_transform = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        # Get the sample according to the id.
        sample = self._samples[idx]
        image_id = sample.x.image_id

        image_modality_1 = self._source_paths[image_id][0]
        image_modality_2 = self._source_paths[image_id][1]
        target = self._target_paths[image_id]

        image_modality_1 = self._pre_transform(image_modality_1)
        image_modality_2 = self._pre_transform(image_modality_2)
        target = self._pre_transform(target)

        # Get the patch and its slice out of this sample.
        patch_x, patch_y = sample.x, sample.y
        slice_x, slice_y = patch_x.slice, patch_y.slice

        # Get the real image data for each modality and target.
        x_modality_1 = image_modality_1[tuple(slice_x)]
        x_modality_2 = image_modality_2[tuple(slice_x)]
        y = target[tuple(slice_y)]

        # Concatenate on channel axis both modalities.
        slice_x = np.concatenate((x_modality_1, x_modality_2), axis=0)
        slice_y = y

        center_coordinate = CenterCoordinate(slice_x, slice_y)
        transformed_patch_x = Patch(slice_x, image_id, center_coordinate)
        transformed_patch_y = Patch(slice_y, image_id, center_coordinate)

        sample.x = transformed_patch_x
        sample.y = transformed_patch_y

        if self._transform is not None:
            sample = self._transform(sample)
        return sample
