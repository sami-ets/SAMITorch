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


import numpy as np


class RemapClassIDs(object):

    def __init__(self, initial_ids: list, final_ids: list):
        if not isinstance(initial_ids, list) and isinstance(final_ids, list):
            raise TypeError(
                "Initial and final IDs must be a list of integers, but got {} and {}".format(type(initial_ids),
                                                                                             type(final_ids)))
        self._initial_ids = initial_ids
        self._final_ids = final_ids

    def __call__(self, nd_array: np.ndarray):
        for i, id in enumerate(self._initial_ids):
            nd_array[nd_array == id] = self._final_ids[i]

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'
