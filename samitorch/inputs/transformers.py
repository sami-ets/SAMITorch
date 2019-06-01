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
import cv2

from typing import Optional, Tuple, Union

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

    def __init__(self, file_path: Union[str, list]) -> None:
        """
        Transformer initializer.

        Args:
            file_path (str): The file path to save the Nifti file.
        """
        self._file_path = file_path

    def __call__(self, nifti_image: Union[nib.Nifti1Image, nib.Nifti2Image, tuple]) -> None:
        if isinstance(nifti_image, tuple) and isinstance(self._file_path, list):
            for element, file_name in zip(nifti_image, self._file_path):
                nib.save(element, file_name)
        elif isinstance(nifti_image, nib.Nifti1Image) or isinstance(nifti_image, nib.Nifti2Image):
            nib.save(nifti_image, self._file_path)
        else:
            raise TypeError("Image type must be Nifti1Image or Nifti2Image, but got {}".format(type(nifti_image)))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToPNGFile(object):
    """
       Write to disk a PNG image.
       """

    def __init__(self, file_path: Union[str, list]) -> None:
        """
        Transformer initializer.

        Args:
            file_path (str): The file path to save the Nifti file.
        """
        self._file_path = file_path

    def __call__(self, nd_array: Union[np.ndarray, tuple]) -> None:
        if isinstance(nd_array, tuple):
            for element, file_name in zip(nd_array, self._file_path):
                if element.ndim == 3:
                    cv2.imwrite(file_name, element[-1, ...])
                elif element.ndim == 2:
                    cv2.imwrite(file_name, element)
                else:
                    raise TypeError("Only 2D (HxW) or 3D (CxHxW) ndarrays are supported")

        elif isinstance(nd_array, np.ndarray):
            if not isinstance(nd_array, np.ndarray) and (nd_array.ndim not in [2, 3]):
                raise TypeError("Only 2D (HxW) or 3D (CxHxW) ndarrays are supported")
            if nd_array.ndim == 3:
                cv2.imwrite(self._file_path, nd_array[-1, ...])
            elif nd_array.ndim == 2:
                cv2.imwrite(self._file_path, nd_array)
            else:
                raise TypeError("Only 2D (HxW) or 3D (CxHxW) ndarrays are supported")

    def __repr__(self):
        return self.__class__.__name__ + '()'


class To2DNifti1Image(object):
    """
    Creates a Nifti1Image from a given Numpy ndarray.

    The Numpy arrays are transposed to respect the standard Nifti dimensions (WxHxDxC)
    """

    def __init__(self, header: Union[nib.Nifti1Header, list] = None) -> None:
        """
        Transformer initializer.

        Args:
            header (:obj:`nibabel.Nifti1Header): The Nifti image header.
        """
        if isinstance(header, list):
            if not len(header) == 2:
                raise ValueError("List of headers must contain at most 2 elements.")

            self._header = header
        else:
            self._header = header

    def __call__(self, nd_array: Union[np.ndarray, tuple]) -> Union[nib.Nifti1Image, tuple]:
        if isinstance(nd_array, tuple):
            return (nib.Nifti1Image(nd_array[0].transpose((2, 1, 0)), None,
                                    self._header[0] if self._header is not None else None),
                    nib.Nifti1Image(nd_array[1].transpose((2, 1, 0)), None,
                                    self._header[1] if self._header is not None else None))

        elif isinstance(nd_array, np.ndarray) and isinstance(self._header, nib.Nifti1Header):
            if not isinstance(nd_array, np.ndarray) and (nd_array.ndim not in [3, 4]):
                raise TypeError("Only 2D (HxW) or 3D (CxHxW) ndarrays are supported")

            return nib.Nifti1Image(nd_array.transpose((2, 1, 0)), None, self._header)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNifti1Image(object):
    """
    Creates a Nifti1Image from a given Numpy ndarray.

    The Numpy arrays are transposed to respect the standard Nifti dimensions (WxHxDxC)
    """

    def __init__(self, header: Union[nib.Nifti1Header, list, None]) -> None:
        """
        Transformer initializer.

        Args:
            header (:obj:`nibabel.Nifti1Header): The Nifti image header.
        """
        if isinstance(header, list):
            if not len(header) <= 2:
                raise ValueError("List of headers must contain at most 2 elements.")

            self._header = header
        else:
            self._header = header

    def __call__(self, nd_array: Union[np.ndarray, tuple]) -> Union[nib.Nifti1Image, tuple]:

        if isinstance(nd_array, tuple) and isinstance(self._header, list):
            return (nib.Nifti1Image(nd_array[0].transpose((3, 2, 1, 0)), None, self._header[0]),
                    nib.Nifti1Image(nd_array[1].transpose((3, 2, 1, 0)), None, self._header[1]))

        elif isinstance(nd_array, np.ndarray) and isinstance(self._header, nib.Nifti1Header):
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

    def __call__(self, nd_array: np.ndarray) -> np.ndarray:
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        d_min, d_max, h_min, h_max, w_min, w_max = self.extract_content_bounding_box_from(nd_array)

        return nd_array[:, d_min:d_max, h_min:h_max, w_min:w_max] if nd_array.ndim is 4 else \
            nd_array[d_min:d_max, h_min:h_max, w_min:w_max]

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

    def __call__(self, nd_array) -> np.ndarray:
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        return self.apply(nd_array, self._target_shape, self._padding_value)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(nd_array: np.ndarray, target_shape: tuple, padding_value: int) -> np.ndarray:
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
        """ get the set of indices from which to sample (foreground) """
        mask = np.where(
            nd_array >= (
                nd_array.mean() if self._threshold is None else self._threshold))  # returns a tuple of length 3
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
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
        self.axis = axis
        self.include_neighbors = include_neighbors

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        axis = self.axis if self.axis is not None else np.random.randint(0, 3)
        x, y = sample
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
        s = self._get_slice(x, idxs, axis).squeeze()
        t = self._get_slice(y, idxs, axis).squeeze()
        if len(cs) == 0 or s.ndim == 2: s = s[np.newaxis, ...]  # add channel axis if empty
        if len(ct) == 0 or t.ndim == 2: t = t[np.newaxis, ...]
        return s, t

    def _get_slice(self, nd_array: np.ndarray, idxs: Tuple[int, int, int], axis: int) -> np.ndarray:
        h, w = self._output_size
        n = 1 if self.include_neighbors else 0
        oh = 0 if h % 2 == 0 else 1
        ow = 0 if w % 2 == 0 else 1
        i, j, k = idxs
        s = nd_array[..., i - n:i + 1 + n, j - h // 2:j + h // 2 + oh, k - w // 2:k + w // 2 + ow] if axis == 0 else \
            nd_array[..., i - h // 2:i + h // 2 + oh, j - n:j + 1 + n, k - w // 2:k + w // 2 + ow] if axis == 1 else \
                nd_array[..., i - h // 2:i + h // 2 + oh, j - w // 2:j + w // 2 + ow, k - n:k + 1 + n]
        if self.include_neighbors:
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

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = sample
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
        s = x[..., i - hh // 2:i + hh // 2 + oh, j - ww // 2:j + ww // 2 + ow, k - dd // 2:k + dd // 2 + od]
        t = y[..., i - hh // 2:i + hh // 2 + oh, j - ww // 2:j + ww // 2 + ow, k - dd // 2:k + dd // 2 + od]
        if len(cs) == 0: s = s[np.newaxis, ...]  # Add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis, ...]
        return s, t


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

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = sample
        *cs, _, _, _ = x.shape
        *ct, _, _, _ = y.shape
        s = x[0] if len(cs) > 0 else x  # Use the first image to determine sampling if multimodal
        idx = np.random.choice(self._valid_idxs(s)[self._axis])
        s = self._get_slice(x, idx)
        t = self._get_slice(y, idx)
        if len(cs) == 0: s = s[np.newaxis, ...]  # Add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis, ...]
        return s, t

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def _get_slice(self, nd_array: np.ndarray, idx: int):
        """
        Get slices of an nd_array.

        Args:
            nd_array (:obj:`Numpy.ndarray`): The ndarray on which to extract slices.
            idx (int): The slice idx.

        Returns:
            slice: The extracted slice.
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

    def __call__(self, nd_array: np.ndarray) -> np.ndarray:
        return (nd_array - self._mean) / self._std

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

    def __call__(self, nd_array: np.ndarray) -> np.ndarray:
        return self._scale * ((nd_array - nd_array.min()) / (nd_array.max() - nd_array.min()))
