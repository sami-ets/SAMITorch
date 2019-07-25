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
import random
import nibabel as nib
import nrrd
import math
import torch
import os
import cv2

from typing import Optional, Tuple, Union, List

from nilearn.image.resampling import resample_to_img

from sklearn.feature_extraction.image import extract_patches

from samitorch.inputs.images import ImageTypes, Image, Extensions
from samitorch.inputs.sample import Sample


class ToNDTensor(object):
    """
    Creates a torch.Tensor object from a numpy array.

    The transformer supports 3D and 4D numpy arrays. The numpy arrays are transposed in order to create tensors with
    dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.

    The dimensions are D: Depth, H: Height, W: Width, C: Channels.
    """

    # noinspection PyArgumentList
    def __call__(self, sample: Sample) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
           sample (:obj:`samitorch.inputs.sample.Sample`): A Sample object containing Numpy ndarray as `x`.

        Returns:
            Tuple[:obj:`torch.Tensor`]: A Tuple containing two torch.Tensor of size (DxHxW) or (CxDxHxW)
        """
        transformed_sample = Sample.from_sample(sample)

        if not isinstance(sample.x, np.ndarray):
            raise TypeError("Only {} are supported.".format(np.ndarray))

        if sample.x.ndim == 3:
            transformed_sample.x = torch.Tensor(sample.x.reshape(sample.x.shape + (1,)))
        elif sample.x.ndim == 4:
            transformed_sample.x = torch.Tensor(sample.x)
        else:
            raise NotImplementedError("Only 3D or 4D arrays are supported.")

        if sample.is_labeled:
            if not isinstance(sample.y, np.ndarray):
                raise TypeError("Only {} are supported.".format(np.ndarray))
            if sample.y.ndim == 1:
                transformed_sample.y = torch.Tensor(sample.y)
            elif sample.y.ndim == 3:
                transformed_sample.y = torch.Tensor(sample.y.reshape(sample.x.shape + (1,)))
            elif sample.y.ndim == 4:
                transformed_sample.y = torch.Tensor(sample.y)
            else:
                raise NotImplementedError("Only 3D or 4D arrays are supported.")

        return sample.update(transformed_sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNumpyArray(object):
    """
    Creates a Numpy ndarray from a given a Sample containing at least one file path as X property.

    The Numpy array is transposed to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    """

    def __call__(self, input: Union[str, nib.Nifti1Image, nib.Nifti2Image, Sample]) -> Union[np.ndarray, Sample]:
        """
        Returns a new Sample with updated `x` property and `y` property if labeled.

        Args:
            input (str or :obj:`nibabel.Nifti1Image` or :obj:`nibabel.Nifti2Image` or :obj:`samitorch.inputs.sample.Sample`): A Sample object.

        Returns:
            :obj:`samitorch.inputs.sample.Sample`_or_:obj:`Numpy.ndarray`: An updated Sample object.
        """

        if isinstance(input, nib.Nifti1Image) or isinstance(input, nib.Nifti2Image):
            nd_array = input.get_fdata().__array__()
            nd_array = self._expand_dims(nd_array)
            return nd_array

        elif isinstance(input, str):
            if Image.is_nifti(input):
                nd_array = nib.load(input).get_fdata().__array__()
            elif Image.is_nrrd(input):
                nd_array, header = nrrd.read(input)
            else:
                raise NotImplementedError("Only {} files are supported".format(ImageTypes.ALL.value))

            nd_array = self._expand_dims(nd_array)
            nd_array = self._transpose(nd_array)
            return nd_array

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            if isinstance(sample.x, list):
                x = list()
                for path in sample.x:
                    if not os.path.exists(path):
                        raise FileNotFoundError("Provided image path is not valid.")

                    if Image.is_nifti(path):
                        nd_array = nib.load(path).get_fdata().__array__()
                    elif Image.is_nrrd(path):
                        nd_array, header = nrrd.read(path)
                    else:
                        raise NotImplementedError(
                            "Only {} files are supported, but got {}".format(ImageTypes.ALL,
                                                                             os.path.splitext(sample.x)[1]))
                    if nd_array.ndim == 3:
                        nd_array = self._expand_dims(nd_array)
                    nd_array = self._transpose(nd_array)
                    x.append(nd_array)

                transformed_sample.x = np.concatenate(x, axis=0)

            elif isinstance(sample.x, str):
                if Image.is_nifti(sample.x):
                    x = nib.load(sample.x).get_fdata().__array__()
                elif Image.is_nrrd(sample.x):
                    x, header = nrrd.read(sample.x)
                else:
                    raise NotImplementedError(
                        "Only {} files are supported, but got {}".format(ImageTypes.ALL, os.path.splitext(sample.x)[1]))

                if x.ndim == 3:
                    x = self._expand_dims(x)
                x = self._transpose(x)

                transformed_sample.x = x

            else:
                raise ValueError("X in Sample must either be a String or List of Strings.")

            if sample.is_labeled:
                if isinstance(sample.y, list):
                    y = list()
                    for path in sample.y:
                        if not os.path.exists(path):
                            raise FileNotFoundError("Provided image path is not valid.")

                        if Image.is_nifti(path):
                            nd_array = nib.load(path).get_fdata().__array__()

                        elif Image.is_nrrd(path):
                            nd_array, header = nrrd.read(path)

                        else:
                            raise NotImplementedError(
                                "Only {} files are supported, but got {}".format(ImageTypes.ALL.value,
                                                                                 os.path.splitext(sample.x)[1]))
                        if nd_array.ndim == 3:
                            nd_array = self._expand_dims(nd_array)
                        nd_array = self._transpose(nd_array)
                        y.append(nd_array)

                    transformed_sample.y = np.concatenate(y, axis=0)

                elif isinstance(sample.y, str):
                    if not os.path.exists(sample.y):
                        raise FileNotFoundError("Provided image path is not valid.")
                    if Image.is_nifti(sample.y):
                        nd_array = nib.load(sample.y).get_fdata().__array__()

                    elif Image.is_nrrd(sample.y):
                        nd_array, header = nrrd.read(sample.y)

                    elif isinstance(sample.y, np.ndarray) and sample.y.ndim == 1:
                        nd_array = sample.y

                    else:
                        raise NotImplementedError(
                            "Only Nifti, NRRD or Numpy ndarray types are supported but got {}".format(type(sample.y)))

                    if nd_array.ndim == 3:
                        nd_array = self._expand_dims(nd_array)
                    nd_array = self._transpose(nd_array)
                    transformed_sample.y = nd_array

            return sample.update(transformed_sample)

    @staticmethod
    def _expand_dims(nd_darray: np.ndarray) -> np.ndarray:
        """
        Expands an Numpy ndarray (DxHxW) from 3D to 4D array.

        Args:
            nd_darray (:obj:`Numpy.ndarray`): A Numpy array.

        Returns:
            :obj:`Numpy.ndarray`: An expanded Numpy array.
        """
        if nd_darray.ndim == 3:
            nd_darray = np.expand_dims(nd_darray, 3)
        elif nd_darray.ndim == 4:
            pass
        else:
            raise ValueError("Numpy ndarray must be 3D or 4D.")
        return nd_darray

    @staticmethod
    def _transpose(nd_darray: np.ndarray) -> np.ndarray:
        """
        Transpose axes of an Numpy ndarray to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.

        Args:
            nd_darray (:obj:`Numpy.ndarray`): A Numpy array.

        Returns:
            :obj:`Numpy.ndarray`: A transposed, expanded Numpy array.
        """
        if nd_darray.ndim == 3:
            nd_darray = np.transpose(nd_darray, (2, 1, 0))
        elif nd_darray.ndim == 4:
            nd_darray = np.transpose(nd_darray, (3, 2, 1, 0))
        else:
            raise ValueError("Numpy ndarray must be 3D or 4D.")
        return nd_darray

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PadToPatchShape(object):
    def __init__(self, patch_size: Union[int, Tuple[int, int, int, int]], step: Union[int, Tuple[int, int, int, int]]):
        """
        Transformer initializer.

        Args:
            patch_size (int or Tuple of int):  The size of the patch to produce.
            step (int or Tuple of int):  The size of the stride between patches.
        """
        self._patch_size = patch_size
        self._step = step

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[np.ndarray, Sample]:
        if isinstance(input, np.ndarray):

            for i in range(1, input.ndim):
                if not input.shape[i] >= self._patch_size[i]:
                    raise ValueError("Shape incompatible with patch_size parameter.")

            c, d, h, w, = input.shape

            pad_d, pad_h, pad_w = 0, 0, 0

            if d % self._patch_size[1] != 0:
                pad_d = int((self._patch_size[1] - d % self._patch_size[1]) / 2)
            if h % self._patch_size[2] != 0:
                pad_h = int((self._patch_size[2] - h % self._patch_size[2]) / 2)
            if w % self._patch_size[3] != 0:
                pad_w = int((self._patch_size[3] - w % self._patch_size[3]) / 2)

            if pad_d != 0 or pad_h != 0 or pad_w != 0:
                input = np.pad(input, ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)), mode="constant",
                               constant_values=0)

            return input

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            c, d, h, w, = sample.x.shape

            pad_d, pad_h, pad_w = 0, 0, 0

            if d % self._patch_size[1] != 0:
                pad_d = int((self._patch_size[1] - d % self._patch_size[1]) / 2)
            if h % self._patch_size[2] != 0:
                pad_h = int((self._patch_size[2] - h % self._patch_size[2]) / 2)
            if w % self._patch_size[3] != 0:
                pad_w = int((self._patch_size[3] - w % self._patch_size[3]) / 2)

            if pad_d != 0 or pad_h != 0 or pad_w != 0:
                transformed_sample.x = np.pad(transformed_sample.x,
                                              ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                                              mode="constant",
                                              constant_values=0)

            if sample.is_labeled:
                if pad_d != 0 or pad_h != 0 or pad_w != 0:
                    transformed_sample.y = np.pad(transformed_sample.y,
                                                  ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                                                  mode="constant",
                                                  constant_values=0)

            return sample.update(transformed_sample)
        

class ToNDArrayPatches(object):
    """
    Produces patches (slices) of a Numpy ndarray.
    """

    def __init__(self, patch_size: Union[int, Tuple[int, int, int, int]], step: Union[int, Tuple[int, int, int, int]]):
        """
        Transformer initializer.

        Args:
            patch_size (int or Tuple of int):  The size of the patch to produce.
            step (int or Tuple of int):  The size of the stride between patches.
        """
        self._patch_size = patch_size
        self._step = step

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[np.ndarray, Sample]:
        if isinstance(input, np.ndarray):

            for i in range(1, input.ndim):
                if not input.shape[i] >= self._patch_size[i]:
                    raise ValueError("Shape incompatible with patch_size parameter.")

            c, d, h, w, = input.shape

            pad_d, pad_h, pad_w = 0, 0, 0

            if d % self._patch_size[1] != 0:
                pad_d = int((self._patch_size[1] - d % self._patch_size[1]) / 2)
            if h % self._patch_size[2] != 0:
                pad_h = int((self._patch_size[2] - h % self._patch_size[2]) / 2)
            if w % self._patch_size[3] != 0:
                pad_w = int((self._patch_size[3] - w % self._patch_size[3]) / 2)

            if pad_d != 0 or pad_h != 0 or pad_w != 0:
                input = np.pad(input, ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)), mode="constant",
                               constant_values=0)

            patches = extract_patches(input, patch_shape=self._patch_size, extraction_step=self._step)
            patches = patches.reshape([-1] + list(self._patch_size))

            return patches

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            c, d, h, w, = sample.x.shape

            pad_d, pad_h, pad_w = 0, 0, 0

            if d % self._patch_size[1] != 0:
                pad_d = int((self._patch_size[1] - d % self._patch_size[1]) / 2)
            if h % self._patch_size[2] != 0:
                pad_h = int((self._patch_size[2] - h % self._patch_size[2]) / 2)
            if w % self._patch_size[3] != 0:
                pad_w = int((self._patch_size[3] - w % self._patch_size[3]) / 2)

            if pad_d != 0 or pad_h != 0 or pad_w != 0:
                transformed_sample.x = np.pad(transformed_sample.x,
                                              ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                                              mode="constant",
                                              constant_values=0)

            transformed_sample.x = extract_patches(transformed_sample.x, patch_shape=self._patch_size,
                                                   extraction_step=self._step)
            transformed_sample.x = transformed_sample.x.reshape([-1] + list(self._patch_size))

            if sample.is_labeled:
                if pad_d != 0 or pad_h != 0 or pad_w != 0:
                    transformed_sample.y = np.pad(transformed_sample.y,
                                                  ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                                                  mode="constant",
                                                  constant_values=0)
                transformed_sample.y = extract_patches(transformed_sample.y, patch_shape=self._patch_size,
                                                       extraction_step=self._step)
                transformed_sample.y = transformed_sample.y.reshape([-1] + list(self._patch_size))

            return sample.update(transformed_sample)


class ToTensorPatches(object):
    """
    Produces patches (slices) of a Tensor.
    """

    def __init__(self, patch_size: Tuple[int, int, int, int], step: Tuple[int, int, int, int]) -> None:
        """
        Transformer initializer.

        Args:
            patch_size(tuple of int): The size of the patch to produce.
            step (tuple of int): The size of the stride between patches.
        """
        self._patch_size = patch_size
        self._step = step

    def __call__(self, input: Union[torch.Tensor, Sample]) -> Union[torch.Tensor, Sample]:
        """
        Slices a Tensor or Sample of tensors.

        Args:
            input (:obj:`samitorch.inputs.sample.Sample` or :obj:`torch.Tensor`): Input object to slice.

        Returns:
            :obj:`samitorch.inputs.sample.Sample` or :obj:`torch.Tensor`): Sliced tensor or a Sample containing sliced
                tensors.
        """

        if isinstance(input, torch.Tensor):

            for i in range(1, input.ndimension()):
                if not input.shape[i] >= self._patch_size[i - 1]:
                    raise ValueError("Shape incompatible with patch_size parameter.")

            c, d, h, w, = input.shape

            pad_d, pad_h, pad_w = 0, 0, 0

            if d % self._patch_size[1] != 0:
                pad_d = int((self._patch_size[1] - d % self._patch_size[1]) / 2)
            if h % self._patch_size[2] != 0:
                pad_h = int((self._patch_size[2] - h % self._patch_size[2]) / 2)
            if w % self._patch_size[3] != 0:
                pad_w = int((self._patch_size[3] - w % self._patch_size[3]) / 2)

            if pad_d != 0 or pad_h != 0 or pad_w != 0:
                input = torch.nn.functional.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))

            patches = input.data.unfold(1, self._patch_size[0], self._step[0]). \
                unfold(1, self._patch_size[1], self._step[1]). \
                unfold(2, self._patch_size[2], self._step[2]). \
                unfold(3, self._patch_size[3], self._step[3])

            patches = patches.reshape([-1] + list(self._patch_size))

            return patches

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            c, d, h, w, = sample.x.shape

            pad_d, pad_h, pad_w = 0, 0, 0

            if d % self._patch_size[1] != 0:
                pad_d = int((self._patch_size[1] - d % self._patch_size[1]) / 2)
            if h % self._patch_size[2] != 0:
                pad_h = int((self._patch_size[2] - h % self._patch_size[2]) / 2)
            if w % self._patch_size[3] != 0:
                pad_w = int((self._patch_size[3] - w % self._patch_size[3]) / 2)

            if pad_d != 0 or pad_h != 0 or pad_w != 0:
                transformed_sample.x = torch.nn.functional.pad(transformed_sample.x,
                                                               (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))

            transformed_sample.x = transformed_sample.x.data.data.unfold(1, self._patch_size[0], self._step[0]). \
                unfold(1, self._patch_size[1], self._step[1]). \
                unfold(2, self._patch_size[2], self._step[2]). \
                unfold(3, self._patch_size[3], self._step[3])

            transformed_sample.x = transformed_sample.x.reshape([-1] + list(self._patch_size))

            if sample.is_labeled:
                if pad_d != 0 or pad_h != 0 or pad_w != 0:
                    transformed_sample.y = torch.nn.functional.pad(transformed_sample.y,
                                                                   (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))

                transformed_sample.y = transformed_sample.y.data.data.unfold(1, self._patch_size[0], self._step[0]). \
                    unfold(1, self._patch_size[1], self._step[1]). \
                    unfold(2, self._patch_size[2], self._step[2]). \
                    unfold(3, self._patch_size[3], self._step[3])

                transformed_sample.y = transformed_sample.y.reshape([-1] + list(self._patch_size))

            return sample.update(transformed_sample)


class ToNrrdFile(object):
    """
    Create a .NRRD file and save it at the given path.

    The numpy arrays are transposed to respect the standard NRRD dimensions (WxHxDxC)
    """

    def __init__(self, file_path: Union[str, list]) -> None:
        """
        Transformer initializer.

        Args:
            file_path (list): A list of file paths where to save a Sample.
        """
        self._file_path = file_path

    def __call__(self, input: Union[np.ndarray, Sample]) -> None:
        """
        Write an NRRD file.

        Args:
            input (:obj:`Numpy.ndarray`_or_:obj:samitorch.inputs.sample.Sample`): Input object to serialze as NRRD file.

        Raises:
            TypeError: If input doesn't match required type and/or dimensions.
        """
        if isinstance(input, np.ndarray):
            if not input.ndim == 4:
                raise TypeError("Only 4D (CxDxHxW) {} are supported.".format(np.ndarray))

            header = self._create_header_from(input)
            nrrd.write(self._file_path, input.transpose((3, 2, 1, 0)), header=header)

        elif isinstance(input, Sample):
            sample = input

            if not isinstance(sample.x, np.ndarray) or not sample.x.ndim == 4:
                raise TypeError("Only 4D (CxDxHxW) {} ndarrays are supported.".format(np.ndarray))

            if isinstance(self._file_path, str) and not sample.is_labeled:
                header = self._create_header_from(sample.x)
                nrrd.write(self._file_path, sample.x.transpose((3, 2, 1, 0)), header=header)

            elif isinstance(self._file_path, list) and sample.is_labeled:
                header = self._create_header_from(sample.x)
                nrrd.write(self._file_path[0], sample.x.transpose((3, 2, 1, 0)), header=header)

            if sample.is_labeled:
                if isinstance(sample.y, np.ndarray) and sample.y.ndim == 4:
                    assert isinstance(self._file_path,
                                      list), "Provided file paths parameter not sufficient to store Y from Sample."
                    header = self._create_header_from(sample.y)
                    nrrd.write(self._file_path[1], sample.y.transpose((3, 2, 1, 0)), header=header)
                elif isinstance(sample.y, np.ndarray) and sample.y.ndim == 1:
                    pass
                else:
                    raise TypeError("Only 4D (CxDxHxW) or 1D ndarrays are supported.")

        else:
            raise TypeError("Input type must be of type Numpy ndarray or a Sample, but got {}".format(type(input)))

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def _create_header_from(nd_array: np.ndarray) -> dict:
        """

        Args:
            nd_array (:obj:`Numpy.ndarray`): A Numpy ndarray to transform as NRRD file.

        Returns:
            dict: NRRD header.

        Raises:
            TypeError: If input doesn't match required type and/or dimensions.
        """
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) {} are supported".format(np.ndarray))

        return {
            'type': nd_array.dtype,
            'dimension': nd_array.ndim,
            'sizes': nd_array.shape,
            'kinds': ['domain', 'domain', 'domain', '3D-matrix'] if nd_array.ndim == 4 else ['domain', 'domain',
                                                                                             'domain'],
            'endian': 'little',
            'encoding': 'raw'
        }


class LoadNifti(object):
    """
    Load a Nibabel Nifti Image from a given Nifti file path.
    """

    def __call__(self, input: Union[str, Sample]) -> Union[nib.Nifti1Image, Sample]:
        """
        Load a Nifti Image.

        Args:
             input (str or :obj:`samitorch.inputs.sample.Sample`): An input, either a string or a Sample.

        Returns:
            :obj:`nibabel.Nifti1Image or :obj:`samitorch.inputs.sample.Sample`: A Nifti1Image or a transformed Sample.

        Raises:
            FileNotFoundError: If provided file path isn't found.
            NoteImplementedError: If file extension isn't in supported extensions.
                See :obj:`samitorch.inputs.images.Extensions`.
        """
        if isinstance(input, str):
            if not os.path.exists(input):
                raise FileNotFoundError("Provided image path is not valid.")
            if Image.is_nifti(input):
                return nib.load(input)
            else:
                raise NotImplementedError("Only {} files are supported.".format(ImageTypes.NIFTI.name))

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            if not os.path.exists(sample.x):
                raise FileNotFoundError("Provided image path is not valid.")
            if Image.is_nifti(sample.x):
                x = nib.load(sample.x)
            else:
                raise NotImplementedError(
                    "Only {} files are supported but got {}".format(ImageTypes.NIFTI.name,
                                                                    os.path.splitext(sample.x)[1]))
            transformed_sample.x = x

            if sample.is_labeled:
                if not os.path.exists(sample.x):
                    raise FileNotFoundError("Provided image path is not valid.")
                elif Image.is_nifti(sample.y):
                    y = nib.load(sample.y)
                elif isinstance(sample.y, np.ndarray) and sample.y.ndim == 1:
                    y = sample.y
                else:
                    raise NotImplementedError(
                        "Only Nifti, NRRD or 1D Numpy arrays types are supported but got {}".format(type(sample.y)))
                transformed_sample.y = y

            return sample.update(transformed_sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NiftiImageToNumpy(object):
    """
    Create a NiftiImage to Numpy NDArray.

    The Numpy array is transposed to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    """

    def __call__(self, input: Union[nib.Nifti1Image, nib.Nifti2Image, Sample]) -> Union[np.ndarray, Sample]:
        """
        Transforms a NiftiImage to Numpy NDArray.

        Args:
            input (:obj:`nibabel.Nifti1Image`_or_:obj:`nibabel.Nifti2Image`_or_:obj:`samitorch.inputs.sample.Sample`):
                An input, either a Nibabel Nifti image or a Sample.

        Returns:
            :obj:`Numpy.ndarray`_or_:obj:`samitorch.inputs.sample.Sample`: A Numpy NDArray or transformed Sample.

        Raises:
            TypeError: If the type isn't supported.
        """
        if isinstance(input, nib.Nifti1Image) or isinstance(input, nib.Nifti2Image):
            nd_array = input.get_fdata().__array__()
            nd_array = self._expand_dims_and_transpose(nd_array)
            return nd_array

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            if isinstance(sample.x, nib.Nifti1Image) or isinstance(sample.x, nib.Nifti2Image):
                x = sample.x.get_fdata().__array__()
                x = self._expand_dims_and_transpose(x)
                transformed_sample.x = x
            else:
                raise TypeError(
                    "Only Nifti1Images and Nifti2Images types are supported, but got {}".format(type(sample.x)))

            if sample.is_labeled:
                if isinstance(sample.y, nib.Nifti1Image) or isinstance(sample.y, nib.Nifti2Image):
                    y = sample.y.get_fdata().__array__()
                    y = self._expand_dims_and_transpose(y)
                elif isinstance(sample.y, np.ndarray) and sample.y.ndim == 1:
                    y = sample.y
                else:
                    raise TypeError(
                        "Only nibabel.Nifti1Images, nibabel.Nifti2Images or 1D Numpy ndarrays are supported, \
                            but got {}".format(type(sample.y)))

                transformed_sample.y = y

            return sample.update(transformed_sample)

        else:
            raise TypeError(
                "Only Nifti1Images, Nifti2Images or Sample types are supported, but got {}".format(type(input)))

    @staticmethod
    def _expand_dims_and_transpose(nd_darray: np.ndarray) -> np.ndarray:
        """
        Expands an Numpy ndarray and transpose its axes to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
        Args:
            nd_darray (:obj:`Numpy.ndarray`): A Numpy array.

        Returns:
            :obj:`Numpy.ndarray`: A transposed, expanded Numpy array.
        """
        if nd_darray.ndim == 3:
            nd_darray = np.expand_dims(nd_darray, 3).transpose((3, 2, 1, 0))
        elif nd_darray.ndim == 4:
            nd_darray = nd_darray.transpose((3, 2, 1, 0))
        return nd_darray

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ResampleNiftiImageToTemplate(object):
    """
    Resamples a Nifti Image to a template file using Nilearn.image.resampling module.
    """

    def __init__(self, interpolation: str, clip: bool,
                 template: Optional[Union[str, nib.Nifti1Image, nib.Nifti2Image]] = None) -> None:
        """
        Resampler initializer.

        Args:
            interpolation (str): Either `continuous`, `linear`, or `nearest`. See `nilearn.image.resample_to_img`.
            clip (bool): If False (default) no clip is preformed. If True all resampled image values above max(img) and under min(img) are clipped to min(img) and max(img).
            template (optional, str_or_nibabel.Nifti1Image): The target reference image.
        """
        self._interpolation = interpolation
        self._clip = clip
        self._template = template

    def __call__(self, input: Union[str, nib.Nifti1Image, nib.Nifti2Image, Sample]) -> Union[nib.Nifti1Image, Sample]:
        """
        Resample an image.

        Args:
            input (str or nib.Nifti1Image or :obj:`samitorch.inputs.sample.Sample`): An input to be resampled.

        Returns:
            nib.Nifti1Image or :obj:`samitorch.inputs.sample.Sample`): A transformed Nibabel Nifti1Image or transformed Sample.
        """
        if isinstance(input, str):
            if Image.is_nifti(input):
                input = nib.load(input)
            else:
                raise NotImplementedError("Only Nifti files are supported.")

        if isinstance(input, nib.Nifti1Image) or isinstance(input, nib.Nifti2Image):
            if isinstance(self._template, str):
                return resample_to_img(input, nib.load(self._template), interpolation=self._interpolation,
                                       clip=self._clip)
            elif isinstance(self._template, nib.Nifti1Image):
                return resample_to_img(input, self._template, interpolation=self._interpolation,
                                       clip=self._clip)
            else:
                raise TypeError(
                    "Template must be of type nibabel.Nifti1Image, nibabel.Nifti2Image or String, \
                        but got {}".format(type(self._template)))

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            if not isinstance(sample.x, nib.nifti1.Nifti1Image) or isinstance(sample.x, nib.nifti2.Nifti2Image):
                raise TypeError(
                    "Only Nifti1Images and Nifti2Images typles are supported, but got {}".format(type(sample.x)))

            if isinstance(sample.template, str):
                if not os.path.exists(sample.template):
                    raise FileNotFoundError("Provided image path is not valid.")
                template = nib.load(sample.template)
            elif isinstance(sample.template, nib.Nifti1Image) or isinstance(sample.template, nib.Nifti1Image):
                template = sample.template
            else:
                raise TypeError("Template must be a String, nibabel.Nifti1Image or nibabel.Nifti2Image.")

            transformed_sample.x = resample_to_img(sample.x, template, interpolation=self._interpolation,
                                                   clip=self._clip)

            if sample.is_labeled:
                if not isinstance(sample.y, nib.nifti1.Nifti1Image) or isinstance(sample.y, nib.nifti2.Nifti2Image):
                    raise TypeError(
                        "Only Nifti1Images and Nifti2Images typles are supported, but got {}".format(type(sample.x)))

                transformed_sample.y = resample_to_img(sample.y, template, interpolation="linear", clip=True)

            return sample.update(transformed_sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NiftiToDisk(object):
    """
    Write to disk a Nifti image.
    """

    def __init__(self, file_path: Union[str, list]) -> None:
        """
        Transformer initializer.

        Args:
            file_path (str_or_list): The file path to save the Nifti file.
        """
        self._file_path = file_path

    def __call__(self, input: Union[nib.Nifti1Image, nib.Nifti2Image, Sample]) -> None:
        """
        Write a NiftiImage to disk.

        Args:
            input (:obj:`nibabel.Nifti1Image`_or_:obj:`nibabel.Nifti2Image`_or_:obj:`samitorch.inputs.sample.Sample`):
                An input image or sample to write to disk.

        Raises:
            TypeError: If type of objects doesn't match expected types.
        """
        if isinstance(input, nib.Nifti1Image) or isinstance(input, nib.Nifti2Image):
            if not isinstance(self._file_path, str):
                raise TypeError(
                    "You passed an instance of a single Nibabel Nifti1Image or Nifti2Image,\
                        but file_path isn't of type {}.".format(type(str)))
            nib.save(input, self._file_path)

        elif isinstance(input, Sample):
            sample = input
            if isinstance(self._file_path, str) and not sample.is_labeled:
                assert Extensions.NIFTI.value in self._file_path, "Bad file extension."
                nib.save(sample.x, self._file_path)

            elif isinstance(self._file_path, list) and sample.is_labeled:
                assert len(self._file_path) == 2, "Length of provided file path list doesn't match Sample to save."
                if isinstance(sample.x, nib.Nifti1Image) or isinstance(sample.x, nib.Nifti2Image):
                    nib.save(sample.x, self._file_path[0])
                if isinstance(sample.y, nib.Nifti1Image) or isinstance(sample.y, nib.Nifti2Image):
                    nib.save(sample.y, self._file_path[1])
                elif isinstance(sample.y, np.ndarray) and sample.y.ndim == 1:
                    pass
                else:
                    raise TypeError("Only nibabel.Nifti1Image and nibabel.Nifti2Image are supported.")

            else:
                raise ValueError("Impossible to save Nifti file with provided parameters.")

        else:
            raise TypeError("Input type must be nibabel.Nifti1Image, nibabel.Nifti2Image or a Sample, \
                                but got {}".format(type(input)))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ApplyMask(object):
    """
    Apply a mask by a given mask/label/ROI image file over an input image.
    """

    def __init__(self, input_mask: Optional[Union[str, nib.Nifti1Image, nib.Nifti2Image, np.ndarray]] = None) -> None:
        """
        Initialize transformer to register a mask image.

        Args:
            input_mask (str or :obj:`nibabel.Nifti1Image` or :obj:`nibabel.Nifti2Image` or :obj:`Numpy.ndarray`): The
                mask to apply.
        """
        if input_mask is not None:
            if isinstance(input_mask, str):

                if not os.path.exists(input_mask):
                    raise FileNotFoundError("Provided image path is not valid.")

                if Image.is_nifti(input_mask):
                    mask = nib.load(input_mask).get_fdata().__array__()
                elif Image.is_nrrd(input_mask):
                    mask, header = nrrd.read(input_mask)
                else:
                    raise NotImplementedError(
                        "Only {} files are supported but got {}".format(ImageTypes.ALL,
                                                                        os.path.splitext(input_mask)[1]))

            elif isinstance(input_mask, nib.Nifti1Image) or isinstance(input_mask, nib.Nifti2Image):
                mask = input_mask.get_fdata().__array__()

            elif isinstance(input_mask, np.ndarray):
                mask = input_mask

            else:
                raise TypeError(
                    "Provided mask must be type Nifti1Image, Nifti2Image, Numpy ndarray, or String, \
                    but got {}".format(type(input_mask)))

            mask[mask >= 1] = 1
            self._mask = mask

    def __call__(self, input: Union[nib.Nifti1Image, nib.Nifti2Image, np.ndarray, Sample]) -> Union[
        nib.Nifti1Image, np.ndarray, Sample]:
        """
        Apply mask over an image (matrix multiplication).

        Args:
            input (str or :obj:`nibabel.Nifti1Image` or :obj:`nibabel.Nifti2Image` or :obj:`Numpy.ndarray`): The
                input image or sample on which to apply mask.

        Returns:
            :obj:`nibabel.Nifti1Image` or :obj:`nibabel.Nifti2Image` or :obj:`Numpy.ndarray`: The transformed image or
                Sample.
        """
        if isinstance(input, nib.Nifti1Image) or isinstance(input, nib.Nifti2Image):
            assert self._mask is not None
            nd_array = input.get_fdata().__array__()
            return nib.Nifti1Image(np.multiply(nd_array, self._mask), input.affine, input.header)

        elif isinstance(input, np.ndarray):
            return np.multiply(input, self._mask)

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            if ((isinstance(sample.x, nib.nifti1.Nifti1Image) and isinstance(sample.y,
                                                                             nib.nifti1.Nifti1Image)) or (
                    isinstance(sample.x, nib.nifti2.Nifti2Image) and isinstance(sample.y,
                                                                                nib.nifti2.Nifti2Image))):
                transformed_sample.x = nib.Nifti1Image(
                    np.multiply(sample.x.get_fdata().__array__(), sample.y.get_fdata().__array__()),
                    sample.x.affine, sample.x.header)

                return sample.update(transformed_sample)

            elif isinstance(sample.x, np.ndarray) and isinstance(sample.y, np.ndarray):

                if not sample.x.ndim not in [3, 4] and sample.y.ndim not in [3, 4]:
                    raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) {} are supported.".format(np.ndarray))

                transformed_sample.x = np.multiply(sample.x, sample.y)
                return sample.update(transformed_sample)

            else:
                raise TypeError(
                    "Sample must contain Numpy ndarrays, but got X={}, Y={}".format(type(sample.x), type(sample.y)))

        else:
            raise TypeError(
                "Image type must be Nifti1Image, Nifti2Image or a Sample containing Numpy ndarrays,\
                 but got {}".format(type(input)))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RemapClassIDs(object):
    """
    Remap the class IDs of a Numpy ndarray to a given list of IDs.
    """

    def __init__(self, initial_ids: list, final_ids: list) -> None:
        """
        Transformer initializer.

        Args:
            initial_ids (list): A list of integers of the original label values to remap.
            final_ids (list): A list of integers of final label values.
        """
        if not isinstance(initial_ids, list) and isinstance(final_ids, list):
            raise TypeError(
                "Initial and final IDs must be a list of integers, but got {} and {}".format(type(initial_ids),
                                                                                             type(final_ids)))

        if not all(isinstance(class_id, int) for class_id in initial_ids) or not all(
                isinstance(class_id, int) for class_id in final_ids):
            raise ValueError("Lists of IDs must contain only Integers.")

        self._initial_ids = initial_ids
        self._new_ids = final_ids

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[np.ndarray, Sample]:
        if isinstance(input, np.ndarray):
            return self._remap(input)

        elif isinstance(input, Sample):
            sample = input
            assert sample.is_labeled
            transformed_sample = Sample.from_sample(sample)

            if isinstance(sample.y, nib.Nifti1Image) or isinstance(sample.y, nib.Nifti2Image):
                y = self._remap(sample.y.get_fdata().__array__())
                y = nib.Nifti1Image(y, affine=sample.y.affine, header=sample.y.header)

            elif isinstance(sample.y, np.ndarray):
                y = self._remap(sample.y)
            else:
                raise TypeError(
                    "Sample must contain Nifti1Image or 3D/4D Numpy ndarray, but got {}".format(type(sample.y)))

            transformed_sample.y = y

            return sample.update(transformed_sample)

        else:
            raise TypeError("Input parameter must but of type Numpy ndarray or Sample, but got {}".format(type(input)))

    def _remap(self, nd_array: np.ndarray) -> np.ndarray:
        new_nd_array = nd_array.copy()

        for class_id, new_id in zip(self._initial_ids, self._new_ids):
            np.place(new_nd_array, nd_array == class_id, new_id)
        return new_nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToPNGFile(object):
    """
       Write to disk a PNG image.
       """

    def __init__(self, file_path: Union[str, list], coeff: Optional[int] = 50) -> None:
        """
        Transformer initializer.

        Args:
            file_path (str): The file path to save the Nifti file.
            coeff (Optional, int): A multiplication factor to save labels.
        """
        self._file_path = file_path
        self._coeff = coeff

    def __call__(self, input: Union[np.ndarray, Sample]) -> None:

        if isinstance(input, np.ndarray):
            if not isinstance(self._file_path, str):
                raise TypeError(
                    "You passed an instance of a single Numpy array, but file_path isn't of type String.")
            if input.ndim == 3:
                cv2.imwrite(self._file_path, input[-1, ...])
            elif input.ndim == 2:
                cv2.imwrite(self._file_path, input)
            else:
                raise TypeError("Only 2D (HxW) or 3D (CxHxW) ndarrays are supported")

        elif isinstance(input, Sample):
            sample = input
            if not isinstance(self._file_path, list) and sample.is_labeled:
                raise TypeError(
                    "You passed an instance of a Sample containing an X and Y, but file_path isn't of type List.")
            if isinstance(self._file_path, str) and not sample.is_labeled:
                if sample.x.ndim == 3:
                    cv2.imwrite(self._file_path, input[-1, ...])
                elif sample.x.ndim == 2:
                    cv2.imwrite(self._file_path, input)
                else:
                    raise TypeError("Only 2D (HxW) or 3D (CxHxW) ndarrays are supported")

            elif isinstance(self._file_path, list) and sample.is_labeled:
                if sample.x.ndim == 3:
                    cv2.imwrite(self._file_path[0], sample.x[-1, ...])
                    cv2.imwrite(self._file_path[1], sample.y[-1, ...] * self._coeff)
                elif sample.x.ndim == 2:
                    cv2.imwrite(self._file_path[0], sample.x)
                    cv2.imwrite(self._file_path[1], sample.y * self._coeff)
                else:
                    raise TypeError("Only 2D (HxW) or 3D (CxHxW) ndarrays are supported")
        else:
            raise TypeError("Input must be a Numpy ndarray or a Sample, but got {}".format(type(input)))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class To2DNifti1Image(object):
    """
    Creates a Nifti1Image from a given Numpy ndarray.

    The Numpy arrays are transposed to respect the standard Nifti dimensions (WxHxDxC)
    """

    def __init__(self, header: Union[nib.Nifti1Header, List[Union[nib.Nifti1Header, nib.Nifti2Header, None]]] = None,
                 affine: Union[List[np.ndarray], np.ndarray, None] = None) -> None:
        """
        Transformer initializer.

        Args:
            header (:obj:`nibabel.Nifti1Header`): The Nifti image header.
        """

        self._header = header
        self._affine = affine

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[nib.Nifti1Image, Sample]:
        """
        Convert a 2D Numpy ndarray to a nibabel.Nifti1Image.

        Args:
           input (:obj:`Numpy.ndarray` or :obj:`samitorch.inputs.sample.Sample`): The input image or sample to convert
            to a Nifti1Image.

        Returns:
            :obj:`nibabel.Nifti1Image` or :obj:`samitorch.inputs.sample.Sample`: The transformed image or Sample.
        """

        if isinstance(input, np.ndarray) and (
                isinstance(self._header, nib.Nifti1Header) or isinstance(self._header, nib.Nifti2Header)):
            if not isinstance(input, np.ndarray) and not input.ndim == 3:
                raise TypeError("Only 3D (CxHxW) ndarrays are supported")

            return nib.Nifti1Image(input.transpose((2, 1, 0)), self._affine, self._header)

        elif isinstance(input, Sample):

            sample = input
            transformed_sample = Sample.from_sample(sample)

            if not sample.x.ndim == 3:
                raise TypeError("Numpy ndarray with be 3D (CxHxW).")

            transformed_sample.x = nib.Nifti1Image(sample.x.transpose((2, 1, 0)), None,
                                                   self._header[0] if self._header is not None else None)
            if sample.is_labeled:
                if isinstance(sample.y, np.ndarray) and sample.y.ndim == 3:

                    y = nib.Nifti1Image(sample.y.transpose((2, 1, 0)), None,
                                        self._header[1] if self._header is not None else None)

                elif isinstance(sample.y, np.ndarray) and sample.y.ndim == 1:
                    y = sample.y
                else:
                    raise TypeError("Numpy ndarray must be 3D (CxHxW) or 1D.")

                transformed_sample.y = y

            return sample.update(transformed_sample)

        else:
            raise ValueError("Incorrect parameters.")

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNifti1Image(object):
    """
    Creates a Nifti1Image from a given Numpy ndarray.

    The Numpy arrays are transposed to respect the standard Nifti dimensions (WxHxDxC)
    """

    def __init__(self,
                 header: Union[nib.Nifti1Header, List[Union[nib.Nifti1Header, nib.Nifti2Header, None]]] = None,
                 affine: Union[List[np.ndarray], np.ndarray, None] = None) -> None:
        """
        Transformer initializer.

        Args:
            header (:obj:`nibabel.Nifti1Header`): The Nifti image header.
        """
        self._header = header
        self._affine = affine

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[nib.Nifti1Image, Sample]:
        """
        Convert 3D or 4D Numpy arrays to Nifti1Image.

        Args:
            input: (:obj:`Numpy.ndarray`_or_:obj:`samitorch.inputs.sample.Sample`): The input image or sample to convert
            to a Nifti1Image.

        Returns:
            :obj:`nibabel.Nifti1Image`_or_:obj:`samitorch.inputs.sample.Sample`: The transformed image or Sample.
        """
        if isinstance(input, np.ndarray):
            if not isinstance(input, np.ndarray) or (input.ndim not in [3, 4]):
                raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

            return nib.Nifti1Image(input.transpose((3, 2, 1, 0)), self._affine, self._header)

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            if not (isinstance(sample.x, np.ndarray) and (sample.x.ndim in [3, 4])):
                raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

            if not isinstance(self._header, list):
                raise TypeError("A sample requires a list of headers.")

            transformed_sample.x = nib.Nifti1Image(sample.x.transpose((3, 2, 1, 0)),
                                                   self._affine[0] if self._affine is not None else None,
                                                   self._header[0] if self._header is not None else None)

            if sample.is_labeled:
                transformed_sample.y = nib.Nifti1Image(sample.y.transpose((3, 2, 1, 0)),
                                                       self._affine[1] if self._affine is not None else None,
                                                       self._header[1] if self._header is not None else None)

            return sample.update(transformed_sample)

        else:
            raise ValueError("Incorrect parameters.")

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropToContent(object):
    """
    Crops the image to its content.

    The content's bounding box is defined by the first non-zero slice in each direction (D, H, W)
    """

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[np.ndarray, Sample]:
        if isinstance(input, np.ndarray):

            if not input.ndim is 4:
                raise TypeError("Only 4D (CxDxHxW) ndarrays are supported")

            d_min, d_max, h_min, h_max, w_min, w_max = self.extract_content_bounding_box_from(input)

            return input[:, d_min:d_max, h_min:h_max, w_min:w_max]

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            d_min, d_max, h_min, h_max, w_min, w_max = self.extract_content_bounding_box_from(sample.x)

            transformed_sample.x = sample.x[:, d_min:d_max, h_min:h_max,
                                   w_min:w_max]

            transformed_sample.y = sample.y[:, d_min:d_max, h_min:h_max,
                                   w_min:w_max]

            return sample.update(transformed_sample)

        else:
            raise TypeError("Only Numpy ndarrays and Sample objects are supported.")

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def extract_content_bounding_box_from(nd_array: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """
        Computes the D, H, W min and max values defining the content bounding box.

        Args:
            nd_array (:obj:`numpy.ndarray`): The input image.

        Returns:
            tuple: The D, H, W min and max values of the bounding box.
        """

        depth_slices = np.any(nd_array, axis=(2, 3))
        height_slices = np.any(nd_array, axis=(1, 3))
        width_slices = np.any(nd_array, axis=(1, 2))

        d_min, d_max = np.where(depth_slices)[1][[0, -1]]
        h_min, h_max = np.where(height_slices)[1][[0, -1]]
        w_min, w_max = np.where(width_slices)[1][[0, -1]]

        return d_min, d_max, h_min, h_max, w_min, w_max


class PadToShape(object):
    """
    Pad an image to a given target shape.
    """

    def __init__(self, target_shape: tuple, padding_value: int = 0) -> None:
        self._target_shape = target_shape
        self._padding_value = padding_value

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[np.ndarray, Sample]:
        if not isinstance(input, np.ndarray) or (input.ndim not in [3, 4]):
            raise ValueError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        if isinstance(input, np.ndarray):
            return self._apply(input, self._target_shape, self._padding_value)

        elif isinstance(input, Sample):
            sample = input
            transformed_sample = Sample.from_sample(sample)

            transformed_sample.x = self._apply(sample.x, self._target_shape, self._padding_value)
            transformed_sample.y = self._apply(sample.y, self._target_shape, self._padding_value)

            return sample.update(transformed_sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def _apply(nd_array: np.ndarray, target_shape: tuple, padding_value: int) -> np.ndarray:
        """
        Apply padding to a Numpy ndarray.

        Args:
            nd_array (:obj:`Numpy.ndarray`): The image to pad.
            target_shape (tuple): The desired target shape.
            padding_value (int): The value to fill padding.

        Returns:
            :obj:`Numpy.ndarray`: The padded array.
        """
        deltas = tuple(max(0, target - current) for target, current in zip(target_shape, nd_array.shape))

        if nd_array.ndim == 3:
            nd_array = np.pad(nd_array, ((math.floor(deltas[0] / 2), math.ceil(deltas[0] / 2)),
                                         (math.floor(deltas[1] / 2), math.ceil(deltas[1] / 2)),
                                         (math.floor(deltas[2] / 2), math.ceil(deltas[2] / 2))),
                              'constant', constant_values=padding_value)
        elif nd_array.ndim == 4:
            nd_array = np.pad(nd_array, ((0, 0),
                                         (math.floor(deltas[1] / 2), math.ceil(deltas[1] / 2)),
                                         (math.floor(deltas[2] / 2), math.ceil(deltas[2] / 2)),
                                         (math.floor(deltas[3] / 2), math.ceil(deltas[3] / 2))),
                              'constant', constant_values=padding_value)
        else:
            raise ValueError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        return nd_array

    @staticmethod
    def _undo(nd_array, original_shape):
        """
        Undo padding and restore original shape.

        Args:
            nd_array (:obj:`Numpy.ndarray`): The image on which to undo padding.
            original_shape (tuple): The target original shape.

        Returns:
            :obj:`Numpy.ndarray`: The unpadded image.

        """
        deltas = tuple(max(0, current - target) for target, current in zip(original_shape, nd_array.shape))

        if nd_array.ndim == 3:
            nd_array = nd_array[
                       math.floor(deltas[0] / 2):-math.ceil(deltas[0] / 2),
                       math.floor(deltas[1] / 2):-math.ceil(deltas[1] / 2),
                       math.floor(deltas[2] / 2):-math.ceil(deltas[2] / 2)]
        elif nd_array.ndim == 4:
            nd_array = nd_array[
                       :,
                       math.floor(deltas[1] / 2):-math.ceil(deltas[1] / 2),
                       math.floor(deltas[2] / 2):-math.ceil(deltas[2] / 2),
                       math.floor(deltas[3] / 2):-math.ceil(deltas[3] / 2)]
        else:
            raise ValueError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        return nd_array


class CropBase(object):
    """
    Base class for crop transform.
    """

    def __init__(self, out_dim: int, output_size: Union[tuple, int, list], threshold: Optional[float] = None) -> None:
        """
        Provide the common functionality for RandomCrop2D and RandomCrop3D.

        Args:
            out_dim (int): The number of dimensions the output must have.
            output_size (tuple_or_int_or_list): The output size of the cropped imge.
            threshold (float): Optional threshold on pixel intensities where to crop the image.

        """
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self._output_size = (output_size,)
            for _ in range(out_dim - 1):
                self._output_size += (output_size,)
        else:
            assert len(output_size) == out_dim
            self._output_size = output_size
        self._out_dim = out_dim
        self._threshold = threshold

    def _get_sample_idxs(self, nd_array: np.ndarray) -> Tuple[int, int, int]:
        """
        Get the set of indices from which to sample (foreground).
        """
        mask = np.where(
            nd_array >= (
                nd_array.mean() if self._threshold is None else self._threshold))  # returns a tuple of length 3
        c = random.randint(0, len(mask[0]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask]  # pull out the chosen idxs
        return h, w, d

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomCrop(CropBase):
    """
    Randomly crop a 2-D slice/patch from a 3-D image

    Original source: https://github.com/jcreinhold/niftidataset/blob/master/niftidataset/transforms.py
    """

    def __init__(self, output_size: Union[int, tuple, list], axis: Optional[int] = 0,
                 include_neighbors: bool = False, threshold: Optional[float] = None) -> None:
        """
        RandomCrop initializer.

        Args:
            output_size (tuple or int): Desired output size. If int, cube crop is made.
            axis (int or None): Axis on which should the patch/slice be extracted. Provide None for random axis
            include_neighbors (bool): Extract 3 neighboring slices instead of just 1
            threshold (float): Optional threshold on pixel intensities where to crop the image.
        """
        if axis is not None:
            if not 0 <= axis <= 2:
                raise ValueError("Axis must be in range [0, 2], but got {}".format(axis))

        super().__init__(2, output_size, threshold)
        self._axis = axis
        self._include_neighbors = include_neighbors

    def __call__(self, sample: Sample) -> Sample:
        """
        Randomly crop an input.

        Args:
            sample (:obj:`samitorch.inputs.sample.Sample`): An input Sample to crop.

        Returns:
            :obj:`samitorch.inputs.sample.Sample`: A cropped Sample.
        """
        transformed_sample = Sample.from_sample(sample)

        axis = self._axis if self._axis is not None else random.randint(0, 3)
        x, y = sample.x, sample.y
        *cs, h, w, d = x.shape
        *ct, _, _, _ = x.shape
        new_h, new_w = self._output_size
        max_idxs = (np.inf, w - new_h // 2, d - new_w // 2) if axis == 0 else \
            (h - new_h // 2, np.inf, d - new_w // 2) if axis == 1 else \
                (h - new_h // 2, w - new_w // 2, np.inf)
        min_idxs = (-np.inf, new_h // 2, new_w // 2) if axis == 0 else \
            (new_h // 2, -np.inf, new_w // 2) if axis == 1 else \
                (new_h // 2, new_w // 2, -np.inf)
        s = x[0] if len(cs) > 0 else x  # use the first image to determine sampling if multimodal
        s_idxs = super()._get_sample_idxs(s)
        idxs = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        transformed_sample.x = self._get_slice(x, idxs, axis).squeeze()
        transformed_sample.y = self._get_slice(y, idxs, axis).squeeze()
        if len(cs) == 0 or transformed_sample.x.ndim == 2: transformed_sample.x = transformed_sample.x[
            np.newaxis, ...]  # add channel axis if empty
        if len(ct) == 0 or transformed_sample.y.ndim == 2: transformed_sample.y = transformed_sample.y[np.newaxis, ...]

        return sample.update(transformed_sample)

    def _get_slice(self, nd_array: np.ndarray, idxs: Tuple[int, int, int], axis: int) -> np.ndarray:
        h, w = self._output_size
        n = 1 if self._include_neighbors else 0
        oh = 0 if h % 2 == 0 else 1
        ow = 0 if w % 2 == 0 else 1
        i, j, k = idxs
        s = nd_array[..., i - n:i + 1 + n, j - h // 2:j + h // 2 + oh, k - w // 2:k + w // 2 + ow] if axis == 0 else \
            nd_array[..., i - h // 2:i + h // 2 + oh, j - n:j + 1 + n, k - w // 2:k + w // 2 + ow] if axis == 1 else \
                nd_array[..., i - h // 2:i + h // 2 + oh, j - w // 2:j + w // 2 + ow, k - n:k + 1 + n]
        if self._include_neighbors:
            s = np.transpose(s, (0, 1, 2)) if axis == 0 else \
                np.transpose(s, (1, 0, 2)) if axis == 1 else \
                    np.transpose(s, (2, 0, 1))
        return s


class RandomCrop3D(CropBase):
    """
    Randomly crop a 3-D patch from a pair of 3-D images.

    Original source: https://github.com/jcreinhold/niftidataset/blob/master/niftidataset/transforms.py
    """

    def __init__(self, output_size: Union[tuple, int, list], threshold: Optional[float] = None) -> None:
        """
        Random Crop 3D Initializer.

        Args:
            output_size (tuple or int): Desired output size. If int, cube crop is made.
            threshold (float): Optional threshold on pixel intensities where to crop the image.
        """
        super().__init__(3, output_size, threshold)

    def __call__(self, sample: Sample) -> Sample:
        """
        Randomly crop an input.

        Args:
           sample (:obj:`samitorch.inputs.sample.Sample`): An input Sample to crop.

        Returns:
           :obj:`samitorch.inputs.sample.Sample`: A cropped Sample.
        """
        transformed_sample = Sample.from_sample(sample)
        x, y = sample.x, sample.y
        *cs, h, w, d = x.shape
        *ct, _, _, _ = y.shape
        hh, ww, dd = self._output_size
        max_idxs = (h - hh // 2, w - ww // 2, d - dd // 2)
        min_idxs = (hh // 2, ww // 2, dd // 2)
        s = x[0] if len(cs) > 0 else x  # Use the first image to determine sampling if multimodal
        s_idxs = super()._get_sample_idxs(s)
        i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                   for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        oh = 0 if hh % 2 == 0 else 1
        ow = 0 if ww % 2 == 0 else 1
        od = 0 if dd % 2 == 0 else 1
        transformed_sample.x = x[..., i - hh // 2:i + hh // 2 + oh, j - ww // 2:j + ww // 2 + ow,
                               k - dd // 2:k + dd // 2 + od]
        transformed_sample.y = y[..., i - hh // 2:i + hh // 2 + oh, j - ww // 2:j + ww // 2 + ow,
                               k - dd // 2:k + dd // 2 + od]
        if len(cs) == 0: transformed_sample.x = transformed_sample.x[np.newaxis, ...]  # Add channel axis if empty
        if len(ct) == 0: transformed_sample.y = transformed_sample.y[np.newaxis, ...]

        return sample.update(transformed_sample)


class RandomSlice(object):
    """
    Take a random 2-D slice from an image given a sample axis.

    Original source: https://github.com/jcreinhold/niftidataset/blob/master/niftidataset/transforms.py
    """

    def __init__(self, axis: int = 0, div: float = 2) -> None:
        """
        Random slice initializer.

        Args:
            axis (int): Axis on which to take a slice
            div (float): Divide the mean by this value in the calculation of mask.
                The higher this value, the more background will be "valid".
        """
        if not 0 <= axis <= 2:
            raise ValueError("Axis must be in range of [0, 2], but got {}".format(axis))

        self._axis = axis
        self._div = div

    def __call__(self, sample: Sample) -> Sample:
        """
        Randomly slice an input.

        Args:
           sample (:obj:`samitorch.inputs.sample.Sample`): An input Sample to crop.

        Returns:
           :obj:`samitorch.inputs.sample.Sample`: A sliced Sample.
        """
        x, y = sample.x, sample.y
        transformed_sample = Sample.from_sample(sample)

        *cs, _, _, _ = x.shape
        *ct, _, _, _ = y.shape
        s = x[0] if len(cs) > 0 else x  # Use the first image to determine sampling if multimodal
        idx = random.choice(self._valid_idxs(s)[self._axis])
        s = self._get_slice(x, idx)
        t = self._get_slice(y, idx)
        if len(cs) == 0: s = s[np.newaxis, ...]  # Add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis, ...]
        transformed_sample.x = s
        transformed_sample.y = t
        return sample.update(transformed_sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def _get_slice(self, nd_array: np.ndarray, idx: int) -> np.ndarray:
        """
        Get slices of an nd_array.

        Args:
            nd_array (:obj:`Numpy.ndarray`): The ndarray on which to extract slices.
            idx (int): The slice idx.

        Returns:
            :obj:`Numpy.ndarray`: The extracted slice.
        """
        s = nd_array[..., idx, :, :] if self._axis == 0 else \
            nd_array[..., :, idx, :] if self._axis == 1 else \
                nd_array[..., :, :, idx]
        return s

    def _valid_idxs(self, nd_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the set of indices from which to sample (foreground).

        Args:
             nd_array (:obj:`Numpy.ndarray`): The ndarray on which to extract idxs.
        """
        mask = np.where(nd_array > nd_array.mean() / self._div)  # returns a tuple of length 3
        h, w, d = [np.arange(np.min(m), np.max(m) + 1) for m in mask]  # pull out the valid idx ranges
        return h, w, d


class Normalize(object):
    """
    Normalize a Numpy ndarray input.
    """

    def __init__(self, mean: float, std: float) -> None:
        """
        Normalization initializer.

        Args:
            mean (float): A global mean.
            std (float): A global standard deviation.
        """
        if std <= 0:
            raise ValueError("Invalid standard deviation, must be greater than 0, but got {}".format(std))

        self._mean = mean
        self._std = std

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[np.ndarray, Sample]:
        """
        Normalize an Numpy ndarray or Sample.

        Args:
            input (:obj:`Numpy.ndarray`_or_:obj:`samitorch.inputs.sample.Sample`): An Numpy ndarray or Sample to normalize.

        Returns:
            :obj:`Numpy.ndarray`_or_:obj:`samitorch.inputs.sample.Sample`: A normalized Numpy ndarray or Sample.

        Raises:
            TypeError: If sample doesn't contain Numpy ndarrays.
        """
        if isinstance(input, np.ndarray):
            return (input - self._mean) / self._std
        elif isinstance(input, Sample):
            sample = input

            if not isinstance(sample.x, np.ndarray):
                raise TypeError("Only Numpy arrays are supported.")

            transformed_sample = Sample.from_sample(sample)
            transformed_sample.x = (sample.x - self._mean) / self._std
            return sample.update(transformed_sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class IntensityScaler(object):
    """
    Scale a Numpy ndarray in a default range of 0 to 1.
    """

    def __init__(self, scale: float = 1) -> None:
        """
        Scaler initializer.

        Args:
            scale (float): The scale of the range.
        """
        self._scale = scale

    def __call__(self, input: Union[np.ndarray, Sample]) -> Union[np.ndarray, Sample]:
        """
         Scale an Numpy ndarray or Sample.

        Args:
            input (:obj:`Numpy.ndarray`_or_:obj:`samitorch.inputs.sample.Sample`): An Numpy ndarray or Sample to scale.

        Returns:
            :obj:`Numpy.ndarray`_or_:obj:`samitorch.inputs.sample.Sample`: A scaled Numpy ndarray or Sample.

        Raises:
            TypeError: If sample doesn't contain Numpy ndarrays.
        """
        if isinstance(input, np.ndarray):
            return self._scale * ((input - input.min()) / (input.max() - input.min()))
        elif isinstance(input, Sample):
            sample = input

            if not isinstance(sample.x, np.ndarray):
                raise TypeError("Only Numpy arrays are supported.")

            transformed_sample = Sample.from_sample(sample)
            transformed_sample.x = self._scale * ((sample.x - sample.x.min()) / (sample.x.max() - sample.x.min()))
            return sample.update(transformed_sample)
