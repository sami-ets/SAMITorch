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
import torch

from samitorch.factories.enums import ActivationLayers, PoolingLayers


class Configuration(metaclass=abc.ABCMeta):
    """
    A standard Configuration object.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        """
        Initialize a Configuration object.

        An implementation of Configuration object must take a dictionary as parameter. The __init__ method places the
        content of the dictionary into Python variables.

        Examples::
            self._attribute = config["attribute"]
            self._attribute = config.get("attribute", default=None)
        """
        pass


class DatasetConfiguration(Configuration, metaclass=abc.ABCMeta):
    """
    A base class for storing dataset configuration as an object.
    """

    @abc.abstractmethod
    def __init__(self, config: dict) -> None:
        super(DatasetConfiguration, self).__init__()
        pass


class ModelConfiguration(Configuration, metaclass=abc.ABCMeta):
    """
    A base class for storing model configuration as an object.
    """

    @abc.abstractmethod
    def __init__(self, config: dict) -> None:
        super(ModelConfiguration, self).__init__()
        pass


class TrainingConfiguration(Configuration, metaclass=abc.ABCMeta):
    """
    A base class for storing training configuration as an object.
    """

    @abc.abstractmethod
    def __init__(self, config: dict) -> None:
        super(TrainingConfiguration, self).__init__()
        pass


class MetricConfiguration(Configuration, metaclass=abc.ABCMeta):
    """
    A base class for storing metric configuration as an object.
    """

    @abc.abstractmethod
    def __init__(self, config: dict) -> None:
        super(MetricConfiguration, self).__init__()
        pass


class VariableConfiguration(Configuration, metaclass=abc.ABCMeta):
    """
    A base class for storing various variables using during training as an object.
    """

    @abc.abstractmethod
    def __init__(self, config: dict) -> None:
        super(VariableConfiguration, self).__init__()
        pass


class LoggerConfiguration(Configuration, metaclass=abc.ABCMeta):
    """
    A base class for storing logger attributes.
    """

    @abc.abstractmethod
    def __init__(self, config: dict) -> None:
        super(LoggerConfiguration, self).__init__()
        pass


class RunningConfiguration(Configuration):
    def __init__(self, config: dict):
        super(RunningConfiguration, self).__init__()
        self._opt_level = config.get("opt_level", default="00")
        self._num_workers = config.get("num_workers", default=0)
        self._local_rank = config.get("local_rank", default=0)
        self._sync_batch_norm = config.get("sync_batch_norm", None)
        self._keep_batch_norm_fp32 = config.get("keep_batch_norm_fp32", None)
        self._loss_scale = config.get("loss_scale", None)
        self._num_gpus = config.get("num_gpus", 0)
        self._log_path = config.get("log", "/tmp")
        self._device = torch.device("cuda:" + str(self._local_rank)) if torch.cuda.is_available() else torch.device(
            "cpu")

    @property
    def opt_level(self) -> str:
        """
        The optimization level.
            - 00: FP32 training
            - 01: Mixed Precision (recommended)
            - 02: Almost FP16 Mixed Precision
            - 03: FP16 Training.

        Returns:
            str: The optimization level.
        """
        return self._opt_level

    @property
    def num_workers(self) -> int:
        """
        The number of data loading workers (default: 4).

        Returns:
            int: The number of parallel threads.
        """
        return self._num_workers

    @property
    def local_rank(self) -> int:
        """
        The local rank of the distributed node.

        Returns:
            int: The local rank.
        """
        return self._local_rank

    @property
    def sync_batch_norm(self) -> bool:
        """
        Enables the APEX sync of batch normalization.

        Returns:
            bool: Whether if synchronization is enabled or not.
        """
        return self._sync_batch_norm

    @property
    def keep_batch_norm_fp32(self) -> bool:
        """
        Whether to keep the batch normalization in 32-bit floating point (Mixed Precision).

        Returns:
            bool: True will keep 32-bit FP batch norm, False will convert it to 16-bit FP.
        """
        return self._keep_batch_norm_fp32

    @property
    def loss_scale(self) -> str:
        """
        The loss scale in Mixed Precision training.

        Returns:
            The loss scale.
        """
        return self._loss_scale

    @property
    def num_gpus(self) -> int:
        """
        The number of GPUs on the Node to be used for training.

        Returns:
            int: The number of allowed GPU to be used for training.
        """
        return self._num_gpus

    @property
    def log_path(self) -> str:
        """
        The path where logs are saved.

        Returns:
            str: The path to log directory.
        """
        return self._log_path

    @property
    def device(self) -> torch.device:
        """
        Get the device where Tensors are going to be transfered.

        Returns:
            :obj:`torch.device`: A Torch Device object.
        """
        return self._device


class DiceMetricConfiguration(MetricConfiguration):
    def __init__(self, config):
        super(DiceMetricConfiguration, self).__init__(config)

        self._num_classes = config["num_classes"]
        self._reduction = config["reduction"]
        self._ignore_index = config["ignore_index"]
        self._average = config["average"]

    @property
    def num_classes(self) -> int:
        """
        int: The number of classes of the problem.
        """
        return self._num_classes

    @property
    def reduction(self) -> str:
        """
        str: The reduction method (e.g. "mean").
        """
        return self._reduction

    @property
    def ignore_index(self) -> int:
        """
        int: The ignored index in a problem.
        """
        return self._ignore_index

    @property
    def average(self) -> str:
        """
        str: The average method in case the metric uses a Confusion Matrix
        """
        return self._average


class UNetModelConfiguration(ModelConfiguration):
    """
    Configuration properties for a UNet model.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate a UnetModelConfiguration object regrouping every UNet3D model's hyper-parameters.

        Args:
            config (dict): A dictionary containing model's hyper-parameters.
        """
        super(UNetModelConfiguration, self).__init__(config)

        self._feature_maps = config["feature_maps"]
        self._in_channels = config["in_channels"]
        self._out_channels = config["out_channels"]
        self._num_levels = config["num_levels"]
        self._conv_kernel_size = config["conv_kernel_size"]
        self._pool_kernel_size = config["pool_kernel_size"]
        self._pooling_type = PoolingLayers(config["pooling_type"])
        self._num_groups = config["num_groups"]
        self._padding = config["padding"]
        self._activation = ActivationLayers(config["activation"])
        self._interpolation = config["interpolation"]
        self._scale_factor = config["scale_factor"]

    @property
    def feature_maps(self) -> int:
        """
        int: Number of feature maps of first UNet level.
        """
        return self._feature_maps

    @property
    def in_channels(self) -> int:
        """
        int: Number of input channels (modality).
        """
        return self._in_channels

    @property
    def out_channels(self) -> int:
        """
        int: Number of output channels.
        """
        return self._out_channels

    @property
    def num_levels(self) -> int:
        """
        int: Number of levels in the UNet architecture.
        """
        return self._num_levels

    @property
    def conv_kernel_size(self) -> int:
        """
        int: The convolution kernel size as integer.
        """
        return self._conv_kernel_size

    @property
    def pool_kernel_size(self) -> int:
        """
        int: The pooling kernel size as integer.
        """
        return self._pool_kernel_size

    @property
    def pooling_type(self) -> PoolingLayers:
        """
        :obj:`samitorch.models.enums.PoolingLayers`: The pooling type.
        """
        return self._pooling_type

    @property
    def num_groups(self) -> int:
        """
        int: The number of groups in group normalization.
        """
        return self._num_groups

    @property
    def padding(self) -> tuple:
        """
        tuple: The padding size of each dimension.
        """
        return self._padding

    @property
    def activation(self) -> ActivationLayers:
        """
        :obj:`samitorch.models.enums.ActivationLayers`: The activation function as a string.
        """
        return self._activation

    @property
    def interpolation(self) -> bool:
        """
        bool: Whether the decoder is doing interpolation (True) or transposed convolution (False).
        """
        return self._interpolation

    @property
    def scale_factor(self) -> tuple:
        """
        tuple: The scale factor (or stride in the transposed convolution) in the decoding path.
        """
        return self._scale_factor


class ResNetModelConfiguration(ModelConfiguration):
    """
    Configuration properties for a ResNet model.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate a ResNetModelConfiguration object regrouping every ResNet3D model's hyper-parameters.

        Args:
            config (dict): A dictionary containing model's hyper-parameters.
        """
        super(ResNetModelConfiguration, self).__init__(config)

        self._in_channels = config["in_channels"]
        self._out_channels = config["out_channels"]
        self._num_groups = config["num_groups"]
        self._conv_groups = config["conv_groups"]
        self._width_per_group = config["width_per_group"]
        self._padding = config["padding"]
        self._activation = ActivationLayers(config["activation"])
        self._zero_init_residual = config["zero_init_residual"]
        self._replace_stride_with_dilation = config["replace_stride_with_dilation"]

    @property
    def in_channels(self) -> int:
        """
        int: Number of input channels (modality).
        """
        return self._in_channels

    @property
    def out_channels(self) -> int:
        """
        int: Number of output channels.
        """
        return self._out_channels

    @property
    def num_groups(self) -> int:
        """
        int: The number of groups in group normalization.
        """
        return self._num_groups

    @property
    def conv_groups(self) -> int:
        """
        int: The number of groups that control the connections between inputs and outputs convolutions.
        """
        return self._conv_groups

    @property
    def width_per_group(self) -> int:
        """
        int: The width of convolution groups.

        Notes:
            Variable used in the width calculation formula of ResNet convolutional groups: `width = int(planes * (width_per_group / 64.)) * groups`
            And then used in torch.nn.conv3d operations as in_channels and out_channels parameters.
        """
        return self._width_per_group

    @property
    def padding(self) -> tuple:
        """
        tuple: The padding size of each dimensions.
        """
        return self._padding

    @property
    def activation(self) -> ActivationLayers:
        """
        :obj:`samitorch.models.enums.ActivationLayers`: The activation function as a string.
        """
        return self._activation

    @property
    def zero_init_residual(self) -> str:
        """
        bool:  Zero-initialize the last batch normalization layer in each residual branch.
        """
        return self._zero_init_residual

    @property
    def replace_stride_with_dilation(self):
        """
        tuple: A tuple of boolean where each element in the tuple indicates if we should replace
            the 2x2 stride with a dilated convolution instead
        """
        return self._replace_stride_with_dilation
