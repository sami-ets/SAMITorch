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

"""A base model class.

Declares methods a Model must have.
"""

from abc import abstractmethod


class BaseModel(object):
    def __init__(self, config):
        """Class initializer.

        Args:
            config: a dictionary containing configuration.
        """
        super(BaseModel, self).__init__()
        self.config = config

    @abstractmethod
    def save(self, file_name="checkpoint.pytorch", is_best=False, **kwargs):
        """Save a checkpoint.

        Args:
            file_name: the file name to save a PyTorch model checkpoint.
            is_best: boolean flag to indicate whether current checkpoint's metric is the best so far.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, file_name, **kwargs):
        """Load a checkpoint.

        Args:
            file_name: the file name from which to load a PyTorch model checkpoint.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()
