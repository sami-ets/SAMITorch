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
import abc
import copy
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torchvision.transforms import Compose
from torch.utils.data.dataset import Dataset

from samitorch.inputs.augmentation.strategies import DataAugmentationStrategy
from samitorch.inputs.transformers import ToNumpyArray, PadToPatchShape, ToNDTensor
from samitorch.inputs.sample import Sample
from samitorch.inputs.patch import Patch, CenterCoordinate
from samitorch.inputs.images import Modality
from samitorch.utils.slice_builder import SliceBuilder
from samitorch.utils.files import extract_file_paths, split_filename


class AbstractDatasetFactory(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def create_train_test(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def create_train_valid_test(*args, **kwargs):
        raise NotImplementedError


class SegmentationDataset(Dataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_paths: List[str], target_paths: List[str], samples: List[Sample],
                 modalities: Union[Modality, List[Modality]], dataset_id: int = None,
                 transforms: Optional[Callable] = None,
                 augment: DataAugmentationStrategy = None) -> None:
        """
        Dataset initializer.

        Args:
            source_paths (List of str): Path to source images.
            target_paths (List of str): Path to target (labels) images.
            samples (list of :obj:`samitorch.inputs.sample.Sample`): A list of Sample objects.
            modality (:obj:`samitorch.inputs.images.Modalities`): The modality of the data set.
            dataset_id (int): An integer representing the ID of the data set.
            transforms (Callable): transform to apply to both source and target images.
            augment (Callable): transform to apply as data augmentation on source image.
        """
        self._source_paths = source_paths
        self._target_paths = target_paths
        self._samples = samples
        self._modalities = modalities
        self._dataset_id = dataset_id
        self._transform = transforms
        self._augment = augment

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        sample = self._samples[idx]

        if self._transform is not None:
            sample = self._transform(sample)

        if self._augment is not None:
            sample.augmented_x = self._augment(sample.x)
        else:
            sample.augmented_x = sample.x

        return sample


class MultimodalSegmentationDataset(Dataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_paths: List[str], target_paths: List[str], samples: List[Sample],
                 modalities: List[Modality], dataset_id: int = None, transforms: Optional[Callable] = None,
                 augment: DataAugmentationStrategy = None) -> None:
        """
        Dataset initializer.

        Args:
            source_paths (List of str): Path to source images.
            target_paths (List of str): Path to target (labels) images.
            samples (list of :obj:`samitorch.inputs.sample.Sample`): A list of Sample objects.
            modalities (List of :obj:`samitorch.inputs.images.Modalities`): The list of modalities of the data set.
            dataset_id (int): An integer representing the ID of the data set.
            transforms (Callable): transform to apply to both source and target images.
            augment (Callable): transform to apply as data augmentation on source image.
        """
        self._source_paths = source_paths
        self._target_paths = target_paths
        self._samples = samples
        self._modalities = modalities
        self._dataset_id = dataset_id
        self._transform = transforms
        self._augment = augment

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        sample = self._samples[idx]

        if self._transform is not None:
            sample = self._transform(sample)

        if self._augment is not None:
            sample.augmented_x = self._augment(sample.x)
        else:
            sample.augmented_x = sample.x

        return sample


class PatchDataset(SegmentationDataset):
    """
    Create a dataset of patches in PyTorch for reading NiFTI files and slicing them into fixed shape.
    """

    def __init__(self, source_paths: List[str], target_paths: List[str], samples: List[Sample],
                 patch_size: Tuple[int, int, int, int],
                 step: Tuple[int, int, int, int], modality: Modality,
                 dataset_id: int = None, transforms: Optional[Callable] = None,
                 augment: DataAugmentationStrategy = None) -> None:
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
            augment (Callable): transform to apply as data augmentation on source image.
        """
        super(PatchDataset, self).__init__(source_paths, target_paths, samples, modality, dataset_id, transforms,
                                           augment)
        self._pre_transform = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def modality(self):
        return self._modalities

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

        if self._augment is not None:
            sample = self._augment(sample)

        return sample


class MultimodalPatchDataset(MultimodalSegmentationDataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_paths: List[str], target_paths: List[str], samples: List[Sample],
                 patch_size: Tuple[int, int, int, int], step: Tuple[int, int, int, int], modalities: List[Modality],
                 dataset_id: int = None, transforms: Optional[Callable] = None,
                 augment: DataAugmentationStrategy = None) -> None:
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
            augment (Callable): transform to apply as data augmentation on source image.
        """
        super(MultimodalPatchDataset, self).__init__(source_paths, target_paths, samples, modalities, dataset_id,
                                                     transforms, augment)
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

        if self._augment is not None:
            sample = self._augment(sample)
        return sample


class PatchDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create_train_test(source_dir: str, target_dir: str, modalities: Union[Modality, List[Modality]],
                          patch_size: Tuple[int, int, int, int],
                          step: Tuple[int, int, int, int], dataset_id: int, test_size: float,
                          keep_centered_on_foreground: bool = True,
                          augmentation_strategy: DataAugmentationStrategy = None):
        """
            Create a SegmentationDataset object for both training and validation.

            Args:
               source_dir (str): Path to source directory.
               target_dir (str): Path to target directory.
               modalities (List of :obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
               dataset_id (int): An integer representing the ID of the data set.
               test_size (float): The size in percentage of the validation set over total number of samples.

            Returns:
               Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """
        if isinstance(modalities, list):
            return PatchDatasetFactory._create_multimodal_train_test(source_dir, target_dir, modalities,
                                                                     patch_size, step, dataset_id, test_size,
                                                                     keep_centered_on_foreground,
                                                                     augmentation_strategy)
        else:
            return PatchDatasetFactory._create_single_modality_train_test(source_dir, target_dir, modalities,
                                                                          patch_size, step, dataset_id, test_size,
                                                                          keep_centered_on_foreground,
                                                                          augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, target_dir: str, modality: Modality,
                                           patch_size: Tuple[int, int, int, int],
                                           step: Tuple[int, int, int, int], dataset_id: int, test_size: float,
                                           keep_centered_on_foreground: bool = True,
                                           augmentation_strategy: DataAugmentationStrategy = None):
        """
        Create a PatchDataset object for both training and validation.

        Args:
            source_dir (str): Path to source directory.
            target_dir (str): Path to target directory.
            modality (:obj:`samitorch.inputs.images.Modalities`): The modality of the data set.
            patch_size (Tuple of int): A tuple representing the desired patch size.
            step (Tuple of int): A tuple representing the desired step between two patches.
            dataset_id (int): An integer representing the ID of the data set.
            test_size (float): The size in percentage of the validation set over total number of samples.
            keep_centered_on_foreground (bool): Keep only patches which center coordinates belongs to a foreground class.
            augmentation_strategy (:obj:`samitorch.inputs.augmentation.strategies.DataAugmentationStrategy`): Data
                augmentation strategy to apply on inputs.

        Returns:
            Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """

        source_dir = os.path.join(source_dir, str(modality))
        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(extract_file_paths(target_dir))

        transforms = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

        patches = PatchDatasetFactory.get_patches(source_paths, target_paths, patch_size, step, transforms,
                                                  keep_centered_on_foreground)
        label_patches = copy.deepcopy(patches)

        train_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(range(len(patches)), list(
                map(lambda patch: patch.class_id, patches))))

        train_samples = list(
            map(lambda source, target: Sample(source, target, dataset_id=dataset_id, is_labeled=True),
                patches[train_ids], label_patches[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, dataset_id=dataset_id, is_labeled=True),
                patches[test_ids], label_patches[test_ids]))

        training_dataset = PatchDataset(list(source_paths), list(target_paths), train_samples, patch_size, step,
                                        modality, dataset_id, Compose([ToNDTensor()]), augmentation_strategy)

        test_dataset = PatchDataset(list(source_paths), list(target_paths), test_samples, patch_size, step, modality,
                                    dataset_id, Compose([ToNDTensor()]), augmentation_strategy)

        return training_dataset, test_dataset

    @staticmethod
    def _create_multimodal_train_test(source_dir: str, target_dir: str, modalities: List[Modality],
                                      patch_size: Tuple[int, int, int, int], step: Tuple[int, int, int, int],
                                      dataset_id: int, test_size: float, keep_centered_on_foreground: bool = False,
                                      augmentation_strategy: DataAugmentationStrategy = None):
        """
        Create a MultimodalPatchDataset object for both training and validation.

        Args:
            source_dir (str): Path to source directory.
            target_dir (str): Path to target directory.
            modality_1 (:obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
            modality_2 (:obj:`samitorch.inputs.images.Modalities`): The second modality of the data set.
            patch_size (Tuple of int): A tuple representing the desired patch size.
            step (Tuple of int): A tuple representing the desired step between two patches.
            dataset_id (int): An integer representing the ID of the data set.
            test_size (float): The size in percentage of the validation set over total number of samples.
            keep_centered_on_foreground (bool): Keep only patches which center coordinates belongs to a foreground class.
            augmentation_strategy (:obj:`samitorch.inputs.augmentation.strategies.DataAugmentationStrategy`): Data
                augmentation strategy to apply on inputs.

        Returns:
            Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """
        source_dirs = list(map(lambda modality: os.path.join(source_dir, str(modality)), modalities))
        source_paths = list(map(lambda source_dir: np.array(extract_file_paths(source_dir)), source_dirs))
        target_paths = np.array(extract_file_paths(target_dir))

        source_paths = np.stack((modality for modality in source_paths), axis=1)

        transforms = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

        patches = PatchDatasetFactory.get_patches(source_paths, target_paths, patch_size, step, transforms,
                                                  keep_centered_on_foreground)
        label_patches = copy.deepcopy(patches)

        train_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(range(len(patches)), list(
                map(lambda patch: patch.class_id, patches))))

        train_samples = list(
            map(lambda source, target: Sample(source, target, dataset_id=dataset_id, is_labeled=True),
                patches[train_ids],
                label_patches[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, dataset_id=dataset_id, is_labeled=True),
                patches[test_ids],
                label_patches[test_ids]))

        training_dataset = MultimodalPatchDataset(list(source_paths), list(target_paths), train_samples, patch_size,
                                                  step, modalities, dataset_id, Compose([ToNDTensor()]),
                                                  augmentation_strategy)

        test_dataset = MultimodalPatchDataset(list(source_paths), list(target_paths), test_samples, patch_size, step,
                                              modalities, dataset_id, Compose([ToNDTensor()]),
                                              augmentation_strategy)

        return training_dataset, test_dataset

    @staticmethod
    def get_patches(source_paths: np.ndarray, target_paths: np.ndarray, patch_size: Tuple[int, int, int, int],
                    step: Tuple[int, int, int, int], transforms: Callable, keep_centered_on_foreground: bool = False):

        patches = list()

        for idx in range(len(source_paths)):
            source_path, target_path = source_paths[idx], target_paths[idx]
            sample = Sample(x=source_path, y=target_path, dataset_id=None, is_labeled=True)
            transformed_sample = transforms(sample)
            slices = SliceBuilder(transformed_sample.x.shape, patch_size=patch_size, step=step).build_slices()
            for slice in slices:
                if np.count_nonzero(transformed_sample.x[slice]) > 0:
                    center_coordinate = CenterCoordinate(transformed_sample.x[slice], transformed_sample.y[slice])
                    patches.append(
                        Patch(slice, idx, center_coordinate))
                else:
                    pass

        if keep_centered_on_foreground:
            patches = list(filter(lambda patch: patch.center_coordinate.is_foreground, patches))

        return np.array(patches)


class SegmentationDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create_train_test(source_dir: str, target_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int,
                          test_size: float, augmentation_strategy: DataAugmentationStrategy = None):
        """
            Create a SegmentationDataset object for both training and validation.

            Args:
               source_dir (str): Path to source directory.
               target_dir (str): Path to target directory.
               modalities (List of :obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
               dataset_id (int): An integer representing the ID of the data set.
               test_size (float): The size in percentage of the validation set over total number of samples.

            Returns:
               Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """
        if isinstance(modalities, list):
            return SegmentationDatasetFactory._create_multimodal_train_test(source_dir, target_dir, modalities,
                                                                            dataset_id, test_size,
                                                                            augmentation_strategy)
        else:
            return SegmentationDatasetFactory._create_single_modality_train_test(source_dir, target_dir, modalities,
                                                                                 dataset_id, test_size,
                                                                                 augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, target_dir: str, modality: Modality, dataset_id: int,
                                           test_size: float,
                                           augmentation_strategy: DataAugmentationStrategy = None):
        """
        Create a SegmentationDataset object for both training and validation.

        Args:
           source_dir (str): Path to source directory.
           target_dir (str): Path to target directory.
           modality (:obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
           dataset_id (int): An integer representing the ID of the data set.
           test_size (float): The size in percentage of the validation set over total number of samples.
           augmentation_strategy (:obj:`samitorch.inputs.augmentation.strategies.DataAugmentationStrategy`): Data
                augmentation strategy to apply on inputs.

        Returns:
           Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """
        source_dir = os.path.join(source_dir, str(modality))

        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(extract_file_paths(target_dir))

        train_ids, test_ids = next(ShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, target_paths))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[train_ids],
                target_paths[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[test_ids],
                target_paths[test_ids]))

        training_dataset = SegmentationDataset(list(source_paths), list(target_paths), train_samples, modality,
                                               dataset_id, Compose([ToNumpyArray(), ToNDTensor()]),
                                               augmentation_strategy)

        test_dataset = SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id,
                                           Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        return training_dataset, test_dataset

    @staticmethod
    def _create_multimodal_train_test(source_dir: str, target_dir: str, modalities: List[Modality],
                                      dataset_id: int, test_size: float,
                                      augmentation_strategy: DataAugmentationStrategy = None):
        """
        Create a MultimodalDataset object for both training and validation.

        Args:
           source_dir (str): Path to source directory.
           target_dir (str): Path to target directory.
           modality_1 (:obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
           modality_2 (:obj:`samitorch.inputs.images.Modalities`): The second modality of the data set.
           dataset_id (int): An integer representing the ID of the data set.
           test_size (float): The size in percentage of the validation set over total number of samples.
           augmentation_strategy (:obj:`samitorch.inputs.augmentation.strategies.DataAugmentationStrategy`): Data
                augmentation strategy to apply on inputs.

        Returns:
           Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """
        source_dirs = list(map(lambda modality: os.path.join(source_dir, str(modality)), modalities))
        source_paths = list(map(lambda source_dir: np.array(extract_file_paths(source_dir)), source_dirs))
        target_paths = np.array(extract_file_paths(target_dir))

        source_paths = np.stack((modality for modality in source_paths), axis=1)

        train_ids, test_ids = next(ShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, target_paths))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[train_ids],
                target_paths[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[test_ids],
                target_paths[test_ids]))

        training_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), train_samples,
                                                         modalities, dataset_id,
                                                         Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        test_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), test_samples,
                                                     modalities, dataset_id, Compose([ToNumpyArray(), ToNDTensor()]),
                                                     augmentation_strategy)

        return training_dataset, test_dataset
