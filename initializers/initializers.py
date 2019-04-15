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

"""A base initializer class.

Declare methods an Initializer must have.
"""

import abc


class Initializer(object):

    def __init__(self):
        """Class constructor."""

    pass

    @abc.abstractmethod
    def initialize(self, *args, **kwargs):
        """Initialize a PyTorch torch.nn layer/operation.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()
