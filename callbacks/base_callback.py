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

Declares methods a Callback must have.
"""


class BaseCallback(object):
    """Implements abstract callback interface.
    All callbacks should be derived from this class
    See Also:
        class:`AbstractNetworkTrainer`
    """

    def __init__(self, *args, **kwargs):
        """ Class initializer.
        Args:
            *args : positional arguments
            **kwargs : keyword arguments
        """
        pass

    def at_epoch_begin(self, trainer, **kwargs):
        """Function which will be executed at begin of each epoch

        Args:
            trainer : :class:`AbstractNetworkTrainer`
            **kwargs : additional keyword arguments

        Returns:
            dict: modified trainer attributes, where the name must correspond to the trainer's attribute name
        """
        return {}

    def at_epoch_end(self, trainer, **kwargs):
        """Function which will be executed at end of each epoch

        Args:
            trainer : :class:`AbstractNetworkTrainer`
            **kwargs : additional keyword arguments

        Returns:
            dict: modified trainer attributes, where the name must correspond to the trainer's attribute name
        """
        return {}
