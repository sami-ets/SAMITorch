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

from torchvision.transforms import Compose


class DataAugmentationStrategy(metaclass=abc.ABCMeta):
    def __init__(self, transform: Compose) -> None:
        super().__init__()
        self._transform = transform

    @abc.abstractmethod
    def apply(self, X, y):
        raise NotImplementedError


class AugmentDuplicatedInstance(DataAugmentationStrategy):

    def __init__(self, transform: Compose) -> None:
        super().__init__(transform)
        self._seen_instance = []

    def apply(self, X, y):
        pass
