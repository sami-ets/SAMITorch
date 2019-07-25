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

from typing import Tuple


class SliceBuilder(object):
    """
    Build slices for a data set.
    """

    def __init__(self, image_shape: Tuple[int, int, int, int], patch_size: Tuple[int, int, int, int],
                 step: Tuple[int, int, int, int]):
        """

        Args:
            image_shape (tuple of int): The shape of a dataset image.
            patch_size(tuple of int): The size of the patch to produce.
            step (tuple of int): The size of the stride between patches.
        """

        self._image_slices = self._build_slices(image_shape, patch_size, step)

    @property
    def image_slices(self) -> list:
        """
        Image's slices.

        Returns:
            list: List of image's slices.
        """
        return self._image_slices

    @staticmethod
    def _build_slices(image_shape: Tuple[int, int, int, int], patch_size: Tuple[int, int, int, int],
                      step: Tuple[int, int, int, int]) -> list:
        """
        Iterates over a given n-dim dataset patch-by-patch with a given step and builds an array of slice positions.

        Args:
            image_shape (tuple of int): The shape of a dataset image.
            patch_size(tuple of int): The size of the patch to produce.
            step (tuple of int): The size of the stride between patches.

        Returns:
            list: list of slices.
        """
        slices = []
        channels, i_z, i_y, i_x = image_shape
        k_c, k_z, k_y, k_x = patch_size
        s_c, s_z, s_y, s_x = step
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if len(image_shape) == 4:
                        slice_idx = (slice(0, channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i: int, k: int, s: int):
        """
        Generate slice indices.

        Args:
            i (int): image's coordinate.
            k (int): patch size.
            s (int): step size.

        Returns:
            generator: A generator of indices.
        """
        assert i >= k, 'Sample size has to be bigger than the patch size.'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k
