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


class UNet3D(torch.nn.Module):
    """
     3DUnet model from `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        config (dict): Dictionary containing the various model's configuration.
    """

    def __init__(self, config: dict):
        super(UNet3D, self).__init__()

        assert isinstance(config["feature_maps"], int), "'feature_maps' must be an instance of int."
        assert isinstance(config["num_levels"], int), "'num_levels' must be an instance of int."

        f_maps = [config["feature_maps"] * 2 ** k for k in range(config["n_levels"])]

        encoders = []
        for i, num_out_features in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(config["in_channels"], num_out_features, DoubleConv, config["encoder"],
                                  apply_pooling=False)
            else:
                encoder = Encoder(f_maps[i - 1], num_out_features, DoubleConv, config["encoder"], apply_pooling=True)
            encoders.append(encoder)
        self.encoders = torch.nn.ModuleList(encoders)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num)
            decoders.append(decoder)
        self.decoders = torch.nn.ModuleList(decoders)

        self.final_conv = torch.nn.Conv3d(f_maps[0], config["out_channels"], 1)

        self.final_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
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
                 num_groups: int = 8):
        super(DoubleConv, self).__init__()
        if is_in_encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.conv1 = torch.nn.Conv3d()

        self.conv2 = torch.nn.Conv3d()

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
