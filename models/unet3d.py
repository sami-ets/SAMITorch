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

import torch

SUPPORTED_POOL_TYPE = ["max", "avg"]
SUPPORTED_ACTIVATIONS = ["relu", "leaky_relu"]


def validate_config(config):
    """
    Validate model configuration parameters.
    Args:
        config (dict): The model's configuration dictionary.
    """
    assert isinstance(config["feature_maps"], int), "'feature_maps' must be an instance of int."
    assert isinstance(config["num_levels"], int), "'num_levels' must be an instance of int."
    assert isinstance(config["in_channels"], int), "'num_levels' must be an instance of int."


def get_activation_fn(function: str):
    assert function in SUPPORTED_ACTIVATIONS

    if function == "relu":
        return torch.nn.ReLU(inplace=True)
    elif function == "leaky_relu":
        return torch.nn.LeakyReLU(inplace=True)


class UNet3D(torch.nn.Module):
    """
     3DUnet model from `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        config (dict): Dictionary containing the various model's configuration.
    """

    @staticmethod
    def _get_feature_maps(starting_feature_maps, num_levels):
        return [starting_feature_maps * 2 ** k for k in range(num_levels)]

    @staticmethod
    def _build_encoder(in_channels, feature_maps, config_encoder):
        encoders = []
        for i, num_out_features in enumerate(feature_maps):
            if i == 0:
                encoder = Encoder(in_channels, num_out_features, DoubleConv, config_encoder,
                                  apply_pooling=False)
            else:
                encoder = Encoder(feature_maps[i - 1], num_out_features, DoubleConv, config_encoder,
                                  apply_pooling=True)
            encoders.append(encoder)
        return encoders

    @staticmethod
    def _build_decoder(feature_maps):
        decoders = []
        reversed_f_maps = list(reversed(feature_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num)
            decoders.append(decoder)
        return decoders

    def __init__(self, config: dict):
        super(UNet3D, self).__init__()

        validate_config(config)

        feature_maps = self._get_feature_maps(config["feature_maps"], config["num_levels"])

        self.encoder = torch.nn.ModuleList(self._build_encoder(config, feature_maps, config["encoder"]))

        self.decoders = torch.nn.ModuleList(self._build_decoder(feature_maps))

        self.final_conv = torch.nn.Conv3d(feature_maps[0], config["out_channels"], 1)

        self.final_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoder:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        x = self.final_activation(x)

        return x


class Encoder(torch.nn.Module):
    """
    A single module for the UNet encoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        basic_module (:obj:`torch.nn.Module`): either ResNetBlock or DoubleConv
        config (dict): Dictionary containing standard Encoder configuration.
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int, basic_module: torch.nn.Module, config: dict,
                 apply_pooling: bool = True):
        super(Encoder, self).__init__()
        assert config["pool_type"] in SUPPORTED_POOL_TYPE

        if apply_pooling:
            if config["pool_type"] == "max":
                self.pooling = torch.nn.MaxPool3d(kernel_size=config["pool_kernel_size"])
            elif config["pool_type"] == "avg":
                self.pooling = torch.nn.AvgPool3d(kernel_size=config["pool_kernel_size"])
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         is_in_encoder=True,
                                         kernel_size=config["conv_kernel_size"],
                                         num_groups=config["num_groups"])

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class DoubleConv(torch.nn.Module):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels: int, out_channels: int, is_in_encoder: bool, kernel_size: int = 3,
                 num_groups: int = 8, padding: str = "replicate"):
        super(DoubleConv, self).__init__()
        self._num_groups = num_groups
        self._padding = padding

        if is_in_encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2

            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels

            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels

        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.conv1 = torch.nn.Conv3d(conv1_in_channels, conv1_out_channels, kernel_size)
        self.conv2 = torch.nn.Conv3d(conv2_in_channels, conv2_out_channels, kernel_size)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1), self._padding)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SingleConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, num_groups: int = None,
                 padding: tuple = None, activation: str = None):
        super(SingleConv, self).__init__()
        self._padding = padding
        self._activation = activation

        if padding is not None:
            self.padding = torch.nn.ReplicationPad3d(padding)

        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)

        if num_groups is not None:
            self.norm = torch.nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm = torch.nn.BatchNorm3d(out_channels)

        if activation is not None:
            self.activation = get_activation_fn(activation)

    def forward(self, x):

        if self._padding is not None:
            x = self.padding(x)

        x = self.conv(x)
        x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x
