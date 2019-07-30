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

import abc
import os
import copy
from enum import Enum
from typing import Union, Tuple, Callable

import torch
import numpy as np
from torchvision.transforms import Compose
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from samitorch.configs.configurations import ModelConfiguration
from samitorch.inputs.datasets import PatchDataset, MultimodalPatchDataset, SegmentationDataset, \
    MultimodalSegmentationDataset
from samitorch.utils.utils import extract_file_paths
from samitorch.inputs.sample import Sample
from samitorch.inputs.patch import Patch
from samitorch.factories.enums import *
from samitorch.inputs.transformers import ToNumpyArray, PadToPatchShape, ToNDTensor
from samitorch.utils.slice_builder import SliceBuilder


class AbstractLayerFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_layer(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, function: str, creator: torch.nn.Module):
        raise NotImplementedError


class AbstractCriterionFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_criterion(self, criterion: Union[str, Enum], *args):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, criterion: str, creator):
        raise NotImplementedError


class AbstractMetricFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_metric(self, function: Union[str, Metrics], *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, function: str, creator):
        raise NotImplementedError


class AbstractModelFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_model(self, name: Union[str, ResNetModels, UNetModels], config: ModelConfiguration):
        pass

    @abc.abstractmethod
    def register(self, model: str, creator):
        raise NotImplementedError


class AbstractOptimizerFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_optimizer(self, function: Union[str, Optimizers], *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, function: str, creator):
        raise NotImplementedError


class AbstractDatasetFactory(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def create_train_test(*args, **kwargs):
        raise NotImplementedError


class OptimizerFactory(AbstractOptimizerFactory):

    def __init__(self):
        super(OptimizerFactory, self).__init__()

        self._optimizers = {
            "Adam": torch.optim.Adam,
            "Adagrad": torch.optim.Adagrad,
            "SGD": torch.optim.SGD,
            "Adadelta": torch.optim.Adadelta,
            "SparseAdam": torch.optim.SparseAdam,
            "Adamax": torch.optim.Adamax,
            "Rprop": torch.optim.Rprop,
            "RMSprop": torch.optim.RMSprop,
            "ASGD": torch.optim.ASGD
        }

    def create_optimizer(self, function: Union[str, Optimizers], *args, **kwargs):
        """
        Instanciate an optimizer based on its name.

        Args:
            function (Enum): The optimizer's name.
            *args: Other arguments.

        Returns:
            :obj:`torch.optim.Optimizer`: The optimizer.

        Raises:
            KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        optimizer = self._optimizers[function.name if isinstance(function, Optimizers) else function]
        return optimizer(*args, **kwargs)

    def register(self, function: str, creator: torch.optim.Optimizer):
        """
        Add a new activation layer.

        Args:
           function (str): Activation layer name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom optimizer function.
        """
        self._optimizers[function] = creator


class ModelFactory(AbstractModelFactory):
    """
    Object to instantiate a model.
    """

    def __init__(self):
        from samitorch.models.resnet3d import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        from samitorch.models.unet3d import UNet3D
        self._models = {
            'ResNet18': ResNet18,
            'ResNet34': ResNet34,
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'ResNet152': ResNet152,
            'UNet3D': UNet3D
        }

    def create_model(self, model_name: Union[str, Enum], config: ModelConfiguration) -> torch.nn.Module:
        """
        Instantiate a new support model.

        Args:
            model_name (str): The model's name (e.g. 'UNet3D').
            config (:obj:`samitorch.configs.model_configuration.ModelConfiguration`): An object containing model's parameters.

        Returns:
            :obj:`torch.nn.Module`: A PyTorch model.
        """
        model = self._models.get(model_name.name if isinstance(model_name, Enum) else model_name)
        if not model:
            raise ValueError("Model {} is not supported.".format(model_name))
        return model(config)

    def register(self, model: str, creator):
        """
        Add a new model.

        Args:
          model (str): Model's name.
          creator: A torch module object wrapping the new custom model.
        """
        self._models[model] = creator


class MetricsFactory(AbstractMetricFactory):
    def __init__(self):
        import ignite
        from samitorch.metrics.metrics import Dice

        super(MetricsFactory, self).__init__()
        self._metrics = {
            "Dice": Dice,
            "Accuracy": ignite.metrics.Accuracy,
            "Precision": ignite.metrics.Precision,
            "MeanAbsoluteError": ignite.metrics.MeanAbsoluteError,
            "MeanPairwiseDistance": ignite.metrics.MeanPairwiseDistance,
            "MeanSquaredError": ignite.metrics.MeanSquaredError,
            "Recall": ignite.metrics.Recall,
            "RootMeanSquaredError": ignite.metrics.RootMeanSquaredError,
            "TopKCategoricalAccuracy": ignite.metrics.TopKCategoricalAccuracy,
            "IoU": ignite.metrics.IoU,
            "mIoU": ignite.metrics.mIoU
        }

    def create_metric(self, metric: Union[str, Metrics], *args, **kwargs):
        """
        Instanciate an optimizer based on its name.

        Args:
            metric (str_or_Enum): The optimizer's name.
            *args: Other arguments.

        Returns:
            The metric, a torch or ignite module.

        Raises:
            KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        metric = self._metrics[metric.name if isinstance(metric, Metrics) else metric]
        return metric(*args, **kwargs)

    def register(self, metric: str, creator):
        """
        Add a new activation layer.

        Args:
           metric (str): Metric's name.
           creator: A torch or ignite module object wrapping the new custom metric function.
        """
        self._metrics[metric] = creator


class ActivationLayerFactory(AbstractLayerFactory):
    """
    Object to instantiate an activation layer.
    """

    def __init__(self):
        super(ActivationLayerFactory, self).__init__()

        self._activation_functions = {
            "ReLU": torch.nn.ReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            'PReLU': torch.nn.PReLU
        }

    def create_layer(self, function: Union[str, ActivationLayers], *args, **kwargs):
        """
        Instantiate an activation layer based on its name.

        Args:
            function (str): The activation layer's name.
            *args: Optional arguments for the respective activation function.

        Returns:
            :obj:`torch.nn.Module`: The activation layer.

        Raises:
            KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        activation_function = self._activation_functions[
            function.name if isinstance(function, ActivationLayers) else function]
        return activation_function(*args, **kwargs)

    def register(self, function: str, creator: torch.nn.Module):
        """
        Add a new activation layer.

        Args:
            function (str): Activation layer name.
            creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom activation function.
        """
        self._activation_functions[function] = creator


class PaddingLayerFactory(AbstractLayerFactory):
    """
    Object to instantiate a padding layer.
    """

    def __init__(self):
        super(PaddingLayerFactory, self).__init__()
        self._padding_strategies = {
            "ReplicationPad3d": torch.nn.ReplicationPad3d
        }

    def create_layer(self, strategy: Union[str, PaddingLayers], dims: tuple, *args, **kwargs):
        """
        Instantiate a new padding layer.

        Args:
            strategy (:obj:samitorch.models.layers.PaddingLayers): The padding strategy.
            dims (tuple): The number of  where to apply padding.
            *args: Optional arguments for the respective activation function.

        Returns:
            :obj:`torch.nn.Module`: The padding layer.

        Raises:
            KeyError: Raises KeyError Exception if Padding Function is not found.
        """
        padding = self._padding_strategies[strategy.name if isinstance(strategy, PaddingLayers) else strategy]
        return padding(dims, *args, **kwargs)

    def register(self, strategy: str, creator: torch.nn.Module):
        """
        Add a new padding strategy.

        Args:
            strategy (str): The padding strategy name.
            creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom padding layer.
        """
        self._padding_strategies[strategy] = creator


class PoolingLayerFactory(AbstractLayerFactory):
    """
    An object to instantiate pooling layers.
    """

    def __init__(self):
        super(PoolingLayerFactory, self).__init__()
        self._pooling_strategies = {
            "MaxPool3d": torch.nn.MaxPool3d,
            "AvgPool3d": torch.nn.AvgPool3d,
            "Conv3d": torch.nn.Conv3d
        }

    def create_layer(self, strategy: Union[str, PoolingLayers], kernel_size: int, stride: int = None, *args,
                     **kwargs):
        """
        Instantiate a new pooling layer with mandatory parameters.

        Args:
            strategy (str): The pooling strategy name.
            kernel_size (int): Pooling kernel's size.
            stride (int): The size of the stride of the window.
            *kwargs: Optional keywords.

        Returns:
            :obj:`torch.nn.Module`: The pooling layer.

        Raises:
            KeyError: Raises KeyError Exception if Pooling Function is not found.
        """
        pooling = self._pooling_strategies[strategy.name if isinstance(strategy, PoolingLayers) else strategy]

        if not "Conv3d" in strategy.name:
            return pooling(kernel_size, stride, *args, **kwargs)

        else:
            return pooling(*args, kernel_size, stride, **kwargs)

    def register(self, strategy: str, creator: torch.nn.Module):
        """
        Add a new pooling layer.

        Args:
            strategy (str): The pooling strategy name.
            creator (obj:`torch.nn.Module`): A torch module object wrapping the new custom pooling layer.
        """
        self._pooling_strategies[strategy] = creator


class NormalizationLayerFactory(AbstractLayerFactory):
    """
    An object to instantiate normalization layers.
    """

    def __init__(self):
        super(NormalizationLayerFactory, self).__init__()
        self._normalization_strategies = {
            "GroupNorm": torch.nn.GroupNorm,
            "BatchNorm3d": torch.nn.BatchNorm3d
        }

    def create_layer(self, strategy: Union[str, NormalizationLayers], *args, **kwargs):
        """
        Instantiate a new normalization layer.

        Args:
            strategy (str_or_Enum): The normalization strategy layer.
            *args: Optional keywords arguments.

        Returns:
            :obj:`torch.nn.Module`: The normalization layer.

        Raises:
            KeyError: Raises KeyError Exception if Normalization Function is not found.
        """
        norm = self._normalization_strategies[
            strategy.name if isinstance(strategy, NormalizationLayers) else strategy]
        return norm(*args, **kwargs)

    def register(self, strategy: str, creator: torch.nn.Module):
        """
        Add a new normalization layer.

        Args:
            strategy (str): The normalization strategy name.
            creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom normalization layer.
        """
        self._normalization_strategies[strategy] = creator


class CriterionFactory(AbstractCriterionFactory):
    def __init__(self):
        super(CriterionFactory, self).__init__()
        from samitorch.losses.losses import DiceLoss
        self._criterion = {
            "DiceLoss": DiceLoss,
            "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
            "NLLLoss": torch.nn.NLLLoss,
            "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
            "BCELoss": torch.nn.BCELoss,
            "MSELoss": torch.nn.MSELoss,
        }

    def create_criterion(self, criterion: Union[str, Enum], *args, **kwargs):
        """
        Instanciate a loss function based on its name.

        Args:
           criterion (str_or_Enum): The criterion's name.
           *args: Other arguments.

        Returns:
           :obj:`torch.nn.Module`: The criterion.

        Raises:
           KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        criterion = self._criterion[criterion.name if isinstance(criterion, Enum) else criterion]
        return criterion(*args, **kwargs)

    def register(self, function: str, creator):
        """
        Add a new criterion.

        Args:
           function (str): Criterion's name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom criterion function.
        """
        self._criterion[function] = creator


class PatchDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create_train_test(source_dir, target_dir, modality, patch_size, step, dataset_id, test_size):
        source_dir = os.path.join(source_dir, str(modality.value))
        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(extract_file_paths(target_dir))

        transforms = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

        patches = PatchDatasetFactory._get_patches(source_paths, target_paths, patch_size, step, transforms)
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
                                        modality, dataset_id,
                                        Compose([ToNDTensor()]))

        test_dataset = PatchDataset(list(source_paths), list(target_paths), test_samples, patch_size, step, modality,
                                    dataset_id,
                                    Compose([ToNDTensor()]))

        return training_dataset, test_dataset

    @staticmethod
    def _get_patches(source_paths: np.ndarray, target_paths: np.ndarray, patch_size: Tuple[int, int, int, int],
                     step: Tuple[int, int, int, int], transforms: Callable):

        patches = list()

        for idx in range(len(source_paths)):
            source_path, target_path = source_paths[idx], target_paths[idx]
            sample = Sample(x=source_path, y=target_path, dataset_id=None, is_labeled=True)
            transformed_sample = transforms(sample)
            slices = SliceBuilder(transformed_sample.x.shape, patch_size=patch_size, step=step).image_slices
            for slice in slices:
                if np.count_nonzero(transformed_sample.x[slice]) > 0:
                    patches.append(
                        Patch(slice, idx, transformed_sample.x[slice], transformed_sample.y[slice]))
                else:
                    pass

        return np.array(patches)


class MultimodalPatchDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create_train_test(source_dir, target_dir, modality_1, modality_2, patch_size, step, dataset_id, test_size):
        source_dir_modality_1 = os.path.join(source_dir, str(modality_1.value))
        source_dir_modality_2 = os.path.join(source_dir, str(modality_2.value))

        source_paths_modality_1, target_paths = np.array(extract_file_paths(source_dir_modality_1)), np.array(
            extract_file_paths(target_dir))
        source_paths_modality_2, target_paths = np.array(extract_file_paths(source_dir_modality_2)), np.array(
            extract_file_paths(target_dir))

        source_paths = np.stack((source_paths_modality_1, source_paths_modality_2), axis=1)

        transforms = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

        patches = MultimodalPatchDatasetFactory._get_patches(source_paths, target_paths, patch_size, step, transforms)
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
                                                  step, modality_1, modality_2, dataset_id, Compose([ToNDTensor()]))

        test_dataset = MultimodalPatchDataset(list(source_paths), list(target_paths), test_samples, patch_size, step,
                                              modality_1, modality_2, dataset_id, Compose([ToNDTensor()]))

        return training_dataset, test_dataset

    @staticmethod
    def _get_patches(source_paths: np.ndarray, target_paths: np.ndarray, patch_size: Tuple[int, int, int, int],
                     step: Tuple[int, int, int, int], transforms: Callable):

        patches = list()

        for idx in range(len(source_paths)):
            source_path, target_path = source_paths[idx], target_paths[idx]
            sample = Sample(x=source_path, y=target_path, dataset_id=None, is_labeled=True)
            transformed_sample = transforms(sample)
            slices = SliceBuilder(transformed_sample.x.shape, patch_size=patch_size, step=step).image_slices
            for slice in slices:
                if np.count_nonzero(transformed_sample.x[slice]) > 0:
                    patches.append(
                        Patch(slice, idx, transformed_sample.x[slice], transformed_sample.y[slice]))
                else:
                    pass

        return np.array(patches)


class SegmentationDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create_train_test(source_dir, target_dir, modality, dataset_id, test_size):
        source_dir = os.path.join(source_dir, str(modality.value))

        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(extract_file_paths(target_dir))

        train_ids, test_ids = next(ShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, target_paths))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[train_ids],
                target_paths[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[test_ids],
                target_paths[test_ids]))

        training_dataset = SegmentationDataset(list(source_paths), list(target_paths), train_samples, modality,
                                               dataset_id,
                                               Compose([ToNumpyArray(), ToNDTensor()]))

        test_dataset = SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id,
                                           Compose([ToNumpyArray(), ToNDTensor()]))

        return training_dataset, test_dataset


class MultimodalSegmentationDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create_train_test(source_dir, target_dir, modality_1, modality_2, dataset_id, test_size):
        source_dir_modality_1 = os.path.join(source_dir, str(modality_1.value))
        source_dir_modality_2 = os.path.join(source_dir, str(modality_2.value))

        source_paths_modality_1, target_paths = np.array(extract_file_paths(source_dir_modality_1)), np.array(
            extract_file_paths(target_dir))
        source_paths_modality_2, target_paths = np.array(extract_file_paths(source_dir_modality_2)), np.array(
            extract_file_paths(target_dir))

        source_paths = np.stack((source_paths_modality_1, source_paths_modality_2), axis=1)

        train_ids, test_ids = next(ShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, target_paths))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[train_ids],
                target_paths[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[test_ids],
                target_paths[test_ids]))

        training_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), train_samples,
                                                         modality_1.value,
                                                         modality_2.value, dataset_id,
                                                         Compose([ToNumpyArray(), ToNDTensor()]))

        test_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), test_samples,
                                                     modality_1.value,
                                                     modality_2.value, dataset_id,
                                                     Compose([ToNumpyArray(), ToNDTensor()]))

        return training_dataset, test_dataset
