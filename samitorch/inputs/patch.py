#  -*- coding: utf-8 -*-
#  Copyright 2019 SAMITorch Authors. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

"""
Class to describe a Patch object for patch segmentation purpose.
"""
import numpy as np

from typing import Union


class CenterCoordinate(object):
    """
    Represent the center coordinate of a Path.
    """

    def __init__(self, source_array: np.ndarray, target_array: np.ndarray):
        """
        Class constructor.

        Args:
            source_array (:obj:`numpy.ndarray`): A 4D Numpy array representing the source image.
            target_array (:obj:`numpy.ndarray`): 4 4D Numpy array representing the target image.
        """
        channels = source_array.shape[0]
        self._center_x, self._center_y, self._center_z = self.get_center_coordinate(source_array)
        self._value = np.array([self._get_center_value(source_array, channel) for channel in range(channels)])
        self._class_id = int(self._get_center_value(target_array, 0))
        self._is_foreground = True if self._class_id > 0 else False

    @property
    def center_x(self):
        return self._center_x

    @property
    def center_y(self):
        return self._center_y

    @property
    def center_z(self):
        return self._center_z

    @property
    def value(self):
        return self._value

    @property
    def class_id(self):
        return self._class_id

    @property
    def is_foreground(self):
        return self._is_foreground

    def _get_center_value(self, array: np.ndarray, channel: int):
        """
        Get the center value for the specified channel.

        Args:
            array (:obj:`numpy.ndarray`): A 4D Numpy array.
            channel (int): The channel of the image.

        Returns:
            int: The value.
        """
        return array[channel][self._center_x][self._center_y][self._center_z]

    @staticmethod
    def get_center_coordinate(array: np.ndarray):
        """
        Get the center coordinate of a Patch.

        Args:
            array (:obj:`numpy.ndarray`): A 4D Numpy array.

        Returns:
            tuple: A tuple representing the X, Y, Z coordinates of the center.
        """
        return array.shape[1] // 2, array.shape[2] // 2, array.shape[3] // 2


class Patch(object):

    def __init__(self, slice: Union[slice, np.ndarray], image_id: int, center_coordinate: CenterCoordinate):
        """
        Class constructor.

        Args:
            slice (slice or :obj:`numpy.ndarray`): A slice or a 4D Numpy array containing patch's values.
            image_id (int): The parent image ID to which the patch belongs to.
            center_coordinate (:obj:`samitorch.inputs.patch.CenterCoordinate`): A CenterCoordinate object.
        """
        self._slice = slice
        self._image_id = image_id
        self._center_coordinate = center_coordinate

    @property
    def class_id(self):
        return self._center_coordinate.class_id

    @property
    def image_id(self):
        return self._image_id

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, slice):
        self._slice = slice

    @property
    def center_coordinate(self):
        return self._center_coordinate

    def update(self, patch):
        """
        Update an existing Patch from another Patch.

        Args:
            patch (:obj:`samitorch.inputs.patch.Patch`): Takes the properties of this Patch to update an existing
                Patch.

        Returns:
            :obj:`samitorch.inputs.patch.Patch`: The updated Patch.
        """
        self._slice = patch.slice
        self._center_coordinate = patch.center_coordinate
        return self

    @classmethod
    def from_patch(cls, patch):
        """
        Create a new Patch from an existing Patch passed in parameter.

        Args:
            patch (:obj:`samitorch.inputs.patch.Patch`): A template Patch.

        Returns:
            :obj:`samitorch.inputs.patch.Patch`: A new patch object with same properties as the one passed in
                parameter.
        """
        return cls(patch.slice, patch.image_id, patch.center_coordinate)
