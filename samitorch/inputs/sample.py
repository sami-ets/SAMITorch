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


"""
Class to describe a Sample object, which contains a source element associated to a ground truth.
"""


class Sample(object):

    def __init__(self, x=None, y=None, template=None, is_labeled=False):
        """
        Sample initializer.

        Args:
            x (any): Source element.
            y (any): Associated ground truth.
            template (any): A template image (for resampling).
            is_labeled (bool): Whether or not the sample has a label image (y).
        """
        self._x = x
        self._y = y
        self._template = template
        self._is_labeled = is_labeled

    @property
    def is_labeled(self):
        """
        Define if Sample is labeled.

        Returns:
            bool: True if sample is labeled (has a non-None `y` property, else False.
        """
        return self._is_labeled

    @property
    def x(self):
        """
        The `x` property of the Sample (generally an image).

        Returns:
            The `x` property.
        """
        return self._x

    @property
    def y(self):
        """
        The `y` property of the Sample (generally an image).

        Returns:
           The `y` property.
        """
        return self._y

    @property
    def template(self):
        """
        The `template` property of the Sample (generally an image from which a transformer is based to resample the `x` image).

        Returns:
           The `template` property.
        """
        return self._template

    @x.setter
    def x(self, x):
        self._x = x

    @y.setter
    def y(self, y):
        self._y = y

    @template.setter
    def template(self, template):
        self._template = template

    @is_labeled.setter
    def is_labeled(self, is_labeled):
        self._is_labeled = is_labeled

    def update(self, sample):
        """
        Update an existing sample from another Sample.

        Args:
            sample (:obj:`samitorch.inputs.sample.Sample`): Takes the properties of this Sample to update an existing
                Sample.

        Returns:
            :obj:`samitorch.inputs.sample.Sample`: The updated Sample.
        """
        self._x = sample.x
        self._y = sample.y
        self._is_labeled = sample.is_labeled
        return self

    def unpack(self) -> tuple:
        """
        Unpack a Sample.

        Returns:
            tuple: A Tuple of elements representing the (X, y) properties of the Sample.
        """
        return self._x, self._y

    @classmethod
    def from_sample(cls, sample):
        """
        Create a new Sample from an existing Sample passed in parameter.

        Args:
            sample (:obj:`samitorch.inputs.sample.Sample`): A template Sample.

        Returns:
            :obj:`samitorch.inputs.sample.Sample`: A new Sample object with same properties as the one passed in
                parameter.
        """
        return cls(sample.x, sample.y, sample.template, sample.is_labeled)
