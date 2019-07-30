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


class CenterCoordinate(object):

    def __init__(self, source_array: np.ndarray, target_array: np.ndarray):
        self._center_x, self._center_y, self._center_z = self._get_center_coordinate(source_array)
        self._value = self._get_center_value(source_array)
        self._class_id = int(self._get_center_value(target_array))
        self._is_foreground = True if self._class_id > 0 else False

    @property
    def value(self):
        return self._value

    @property
    def class_id(self):
        return self._class_id

    @property
    def is_foreground(self):
        return self._is_foreground

    def _get_center_value(self, array: np.ndarray):
        return array[0][self._center_x][self._center_y][self._center_z]

    @staticmethod
    def _get_center_coordinate(array: np.ndarray):
        return array.shape[1] // 2, array.shape[2] // 2, array.shape[3] // 2


class Patch(object):

    def __init__(self, slice, source_image_id, source_slice, target_slice, ):
        self._slice = slice
        self._image_id = source_image_id
        self._center_coordinate = CenterCoordinate(source_slice, target_slice)

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
