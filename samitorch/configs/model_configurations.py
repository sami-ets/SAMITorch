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

from samitorch.factories.enums import ActivationLayers, PoolingLayers


class ModelConfiguration(metaclass=abc.ABCMeta):
    pass


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
        super(UNetModelConfiguration, self).__init__()

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
    def pooling_type(self) -> str:
        """
        str: The pooling type.
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
        super(ResNetModelConfiguration, self).__init__()

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
            Variable used in the width calculation formula of ResNet convolutional groups:
                `width = int(planes * (width_per_group / 64.)) * groups`
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
