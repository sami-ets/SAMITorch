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


from enum import Enum


class ActivationLayers(Enum):
    ReLU = "ReLU"
    LeakyReLU = "LeakyReLU"
    PReLU = "PReLU"

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member


class PaddingLayers(Enum):
    ReplicationPad3d = "ReplicatonPad3d"

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member


class PoolingLayers(Enum):
    MaxPool3d = "MaxPool3d"
    AvgPool3d = "AvgPool3d"
    Conv3d = "Conv3d"

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member


class NormalizationLayers(Enum):
    GroupNorm = "GroupNorm"
    BatchNorm3d = "BatchNorm3d"

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member


class ResNetModels(Enum):
    ResNet18 = "ResNet18"
    ResNet34 = "ResNet34"
    ResNet50 = "ResNet50"
    ResNet101 = "ResNet101"
    ResNet152 = "ResNet152"

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member


class UNetModels(Enum):
    UNet3D = "UNet3D"

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member
