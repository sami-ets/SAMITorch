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
import nibabel as nib
import nrrd
import math
import torch
import os

from nilearn.image.resampling import resample_to_img

from samitorch.inputs.images import ImageType, Image


class ToNDTensor(object):
    """
    Creates a torch.Tensor object from a numpy array.

    The transformer supports 3D and 4D numpy arrays. The numpy arrays are transposed in order to create tensors with
    dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.

    The dimensions are D: Depth, H: Height, W: Width, C: Channels.
    """

    # noinspection PyArgumentList
    def __call__(self, nd_array: np.ndarray):
        """
        Args:
            nd_array (:obj:`Numpy.ndarray`):  A 3D or 4D numpy array to convert to torch.Tensor

        Returns:
            :obj:`torch.Tensor`: A torch.Tensor of size (DxHxW) or (CxDxHxW)
        """

        if not isinstance(nd_array, np.ndarray):
            raise TypeError("Only {} are supporter".format(np.ndarray))

        if nd_array.ndim == 3:
            nd_tensor = torch.Tensor(nd_array.reshape(nd_array.shape + (1,)).transpose((3, 2, 1, 0)))
        elif nd_array.ndim == 4:
            nd_tensor = torch.Tensor(nd_array.transpose((3, 2, 1, 0)))
        else:
            raise NotImplementedError("Only 3D or 4D arrays are supported.")

        return nd_tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNumpyArray(object):
    """
    Creates a Numpy ndarray from a given Nifti or NRRD image file path.

    The Numpy array is transposed to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    """

    def __call__(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError("Provided image path is not valid.")

        if Image.is_nifti(image_path):
            nd_array = nib.load(image_path).get_fdata().__array__()
        elif Image.is_nrrd(image_path):
            nd_array, header = nrrd.read(image_path)
        else:
            raise NotImplementedError(
                "Only {} files are supported but got {}".format(ImageType.ALL, os.path.splitext(image_path)[1]))

        if nd_array.ndim == 3:
            nd_array = np.expand_dims(nd_array, 3).transpose((3, 2, 1, 0))
        elif nd_array.ndim == 4:
            nd_array = nd_array.transpose((3, 2, 1, 0))

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNrrdFile(object):
    """
    Create a .NRRD file and save it at the given path.

    The numpy arrays are transposed to respect the standard NRRD dimensions (WxHxDxC)
    """

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError("Provided NRRD file path is not valid.")

        self._file_path = file_path

    def __call__(self, nd_array: np.ndarray):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) {} are supported".format(np.ndarray))

        header = self._create_header_from(nd_array)
        nrrd.write(self._file_path, nd_array.transpose((3, 2, 1, 0)), header=header)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def _create_header_from(nd_array: np.ndarray):
        """

        Args:
            nd_array (:obj:`Numpy.ndarray`): A Numpy ndarray to transform as NRRD file.

        Returns:
            dict: NRRD header.
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

    def __call__(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError("Provided image path is not valid.")

        if Image.is_nifti(image_path):
            return nib.load(image_path)
        else:
            raise NotImplementedError("Only {} files are supported !".format(ImageType.NIFTI))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NiftiImageToNumpy(object):
    """
    Create a Numpy ndarray from Nifti image.

    The Numpy array is transposed to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    """

    def __call__(self, nifti_image):
        if isinstance(nifti_image, nib.Nifti1Image) or isinstance(nifti_image, nib.Nifti2Image):
            nd_array = nifti_image.get_fdata().__array__()
        else:
            raise TypeError("Image type must be Nifti1Image or Nifti2Image, but got {}".format(type(nifti_image)))

        if nd_array.ndim == 3:
            nd_array = np.expand_dims(nd_array, 3).transpose((3, 2, 1, 0))
        elif nd_array.ndim == 4:
            nd_array = nd_array.transpose((3, 2, 1, 0))

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ResampleNiftiImgToTemplate(object):
    """
    Resamples a Nifti Image to a template file using Nilearn.image.resampling module.
    """

    def __init__(self, template: str, interpolation: str, clip: bool):
        self._template = template
        self._interpolation = interpolation
        self._clip = clip

    def __call__(self, nifti_image: nib.Nifti1Image):
        return resample_to_img(nifti_image, nib.load(self._template), interpolation=self._interpolation,
                               clip=self._clip)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ApplyMaskToNiftiImage(object):
    """
    Apply a mask by a given mask/label/ROI image file.
    """

    def __init__(self, mask_path: str):
        if not os.path.exists(mask_path):
            raise FileNotFoundError("Provided image path is not valid.")

        if Image.is_nifti(mask_path):
            mask = nib.load(mask_path).get_fdata().__array__()
        elif Image.is_nrrd(mask_path):
            mask, header = nrrd.read(mask_path)
        else:
            raise NotImplementedError(
                "Only {} files are supported but got {}".format(ImageType.ALL, os.path.splitext(mask_path)[1]))

        mask[mask >= 1] = 1
        self._mask = mask

    def __call__(self, nifti_image):
        if isinstance(nifti_image, nib.Nifti1Image) or isinstance(nifti_image, nib.Nifti2Image):
            nd_array = nifti_image.get_fdata().__array__()
            header = nifti_image.header
        else:
            raise TypeError("Image type must be Nifti1Image or Nifti2Image, but got {}".format(type(nifti_image)))

        return nib.Nifti1Image(np.multiply(nd_array, self._mask), None, header)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ApplyMaskToTensor(object):
    """
    Multiply a Numpy ndarray by a given mask/label/ROI image file.

    The mask is transposed to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    """

    def __init__(self, mask_path: str):
        if not os.path.exists(mask_path):
            raise FileNotFoundError("Provided image path is not valid.")

        if Image.is_nifti(mask_path):
            nd_array = nib.load(mask_path).get_fdata().__array__()
        elif Image.is_nrrd(mask_path):
            nd_array, header = nrrd.read(mask_path)
        else:
            raise NotImplementedError(
                "Only {} files are supported but got {}".format(ImageType.ALL, os.path.splitext(mask_path)[1]))

        if nd_array.ndim == 3:
            nd_array = np.expand_dims(nd_array, 3).transpose((3, 2, 1, 0))
        elif nd_array.ndim == 4:
            nd_array = nd_array.transpose((3, 2, 1, 0))

        nd_array[nd_array >= 1] = 1
        self._mask = nd_array

    def __call__(self, nd_array: np.ndarray):
        return np.multiply(nd_array, self._mask)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RemapClassIDs(object):
    """
    Remap the class IDs of a Numpy ndarray to a given list of IDs.
    """

    def __init__(self, initial_ids: list, final_ids: list):
        if not isinstance(initial_ids, list) and isinstance(final_ids, list):
            raise TypeError(
                "Initial and final IDs must be a list of integers, but got {} and {}".format(type(initial_ids),
                                                                                             type(final_ids)))

        if not all(isinstance(class_id, int) for class_id in initial_ids) or not all(
                isinstance(class_id, int) for class_id in final_ids):
            raise ValueError("Lists of IDs must contain only Integers.")

        self._initial_ids = initial_ids
        self._new_ids = final_ids

    def __call__(self, nd_array: np.ndarray):
        new_nd_array = nd_array.copy()

        indexes = [np.where(nd_array == class_id) for class_id in self._initial_ids]

        for indexes, new_id in zip(indexes, self._new_ids):
            new_nd_array[indexes] = new_id

        return new_nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NiftiToDisk(object):
    """
    Write to disk a Nifti image.
    """

    def __init__(self, file_path: str):
        self._file_path = file_path

    def __call__(self, image):
        if isinstance(image, nib.Nifti1Image) or isinstance(image, nib.Nifti2Image):
            nib.save(image, self._file_path)
        else:
            raise TypeError("Image type must be Nifti1Image or Nifti2Image, but got {}".format(type(image)))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNifti1Image(object):
    """
    Creates a Nifti1Image from a given Numpy ndarray.

    The Numpy arrays are transposed to respect the standard Nifti dimensions (WxHxDxC)
    """

    def __init__(self, file_path: str, header: nib.Nifti1Header):
        self._file_path = file_path
        self._header = header

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        return nib.Nifti1Image(nd_array.transpose((3, 2, 1, 0)), None, self._header)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropToContent(object):
    """
    Crops the image to its content.

    The content's bounding box is defined by the first non-zero slice in each direction (D, H, W)
    """

    def __call__(self, nd_array: np.ndarray):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        d_min, d_max, h_min, h_max, w_min, w_max = self.extract_content_bounding_box_from(nd_array)

        return nd_array[:, d_min:d_max, h_min:h_max, w_min:w_max] if nd_array.ndim is 4 else \
            nd_array[d_min:d_max, h_min:h_max, w_min:w_max]

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def extract_content_bounding_box_from(nd_array: np.ndarray):
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

    def __init__(self, target_shape: tuple, padding_value: int = 0):
        self._target_shape = target_shape
        self._padding_value = padding_value

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        return self.apply(nd_array, self._target_shape, self._padding_value)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(nd_array: np.ndarray, target_shape: tuple, padding_value: int):
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
        return nd_array

    @staticmethod
    def undo(nd_array, original_shape):
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
        return nd_array
