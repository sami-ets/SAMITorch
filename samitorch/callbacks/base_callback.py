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

"""A base callback class.

Declare methods a Callback must have.
"""
import abc


class Callback(object):
    """Implements abstract callback interface.

    All callbacks should be derived from this class

    See Also:
        class:`Trainer`
    """

    def __init__(self, *args, **kwargs):
        """ Class initializer.
        Args:
            *args: positional arguments
            **kwargs: keyword arguments
        """
        pass

    @abc.abstractmethod
    def at_epoch_begin(self, **kwargs):
        """Function which will be executed at begin of each epoch

        Args:
            **kwargs: additional keyword arguments

        Returns:
            dict: modified trainer attributes, where the name must correspond to the trainer's attribute name
        """
        return {}

    @abc.abstractmethod
    def at_epoch_end(self, **kwargs):
        """Function which will be executed at end of each epoch

        Args:
            **kwargs: additional keyword arguments

        Returns:
            dict: modified trainer attributes, where the name must correspond to the trainer's attribute name
        """
        return {}
