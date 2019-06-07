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
        """
        self._x = x
        self._y = y
        self._template = template
        self._is_labeled = is_labeled

    @property
    def is_labeled(self):
        return self._is_labeled

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def template(self):
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
        self._x = sample.x
        self._y = sample.y
        self._is_labeled = sample.is_labeled
        return self

    def unpack(self):
        return self._x, self._y

    @classmethod
    def from_sample(cls, sample):
        return cls(sample.x, sample.y, sample.template, sample.is_labeled)
