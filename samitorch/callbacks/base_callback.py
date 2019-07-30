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


class Callback(object):
    """Implements abstract callback interface.

    All callbacks should be derived from this class
    """

    def on_training_begin(self, *args, **kwargs):
        """Function which will be executed at the beginning of the training.

        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        """

    def on_training_end(self, *args, **kwargs):
        """Function which will be executed at the end of the training.

        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        """

    def on_epoch_begin(self, epoch_num, **kwargs):
        """Function which will be executed at beginning of each epoch

        Args:
            epoch_num: The current epoch number
            **kwargs: additional keyword arguments
        """
        pass

    def on_epoch_end(self, epoch_num, **kwargs):
        """Function which will be executed at the end of each epoch

        Args:
            epoch_num: The current epoch number
            **kwargs: additional keyword arguments
        """
        pass

    def on_batch_begin(self, *args, **kwargs):
        """Function which will be executed at the beginning of each batch

        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        pass

    def on_batch_end(self, *args, **kwargs):
        """Function which will be executed at the end of each epoch

        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        pass
