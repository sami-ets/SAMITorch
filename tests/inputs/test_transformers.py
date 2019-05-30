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

import unittest
import os
import nibabel as nib
import numpy as np
from torchvision.transforms import transforms

from hamcrest import *

from samitorch.inputs.transformers import LoadNifti, ToNifti1Image, RemapClassIDs, ApplyMaskToNiftiImage, \
    ApplyMaskToTensor, \
    ToNumpyArray


class ToNiftiImageTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "FA.nii")
    VALID_4D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "DTI.nii")
    INVALID_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "invalid.nii")

    def setUp(self):
        pass

    def test_should_return_nifti_image_with_valid_3d_nifti_file(self):
        nifti_3d_image = LoadNifti().__call__(self.VALID_3D_NIFTI_FILE)
        assert_that(nifti_3d_image, is_not(None))
        assert_that(nifti_3d_image, is_(nib.Nifti1Image))

    def test_should_return_nifti_image_with_valid_4d_nifti_file(self):
        nifti_4d_image = LoadNifti().__call__(self.VALID_4D_NIFTI_FILE)
        assert_that(nifti_4d_image, is_not(None))
        assert_that(nifti_4d_image, is_(nib.Nifti1Image))

    def test_should_raise_exception_with_invalid_file_path(self):
        assert_that(calling(LoadNifti().__call__).with_args(self.INVALID_FILE), raises(FileNotFoundError))


class ApplyMaskToNiftiImageTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    INVALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_mask.nii")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")

    def setUp(self):
        pass

    def test_should_initialize_mask_correctly(self):
        transformer = ApplyMaskToNiftiImage(self.VALID_MASK_FILE)
        np.testing.assert_array_less(transformer._mask, 1.0 + 1e-10)
        assert_that(len(transformer._mask.shape), is_(3))

    def test_returned_image_should_be_Nifti1Image_type(self):
        transformer = ApplyMaskToNiftiImage(self.VALID_MASK_FILE)
        nifti = nib.load(self.VALID_3D_NIFTI_FILE)
        transformed_nifti = transformer.__call__(nifti)
        assert_that(transformed_nifti, instance_of(nib.Nifti1Image))

    def test_should_return_image_with_same_shape_as_original(self):
        transformer = ApplyMaskToNiftiImage(self.VALID_MASK_FILE)
        nifti = nib.load(self.VALID_3D_NIFTI_FILE)
        transformed_nifti = transformer.__call__(nifti)
        np.testing.assert_array_equal(transformed_nifti.shape, nifti.shape)


class ApplyMaskToNiftiTensor(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    INVALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_mask.nii")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")

    def setUp(self):
        pass

    def test_should_initialize_mask_correctly(self):
        transformer = ApplyMaskToTensor(self.VALID_MASK_FILE)
        np.testing.assert_array_less(transformer._mask, 1.0 + 1e-10)
        assert_that(len(transformer._mask.shape), is_(4))

    def test_returned_image_should_be_ndarray_type(self):
        transforms_ = transforms.Compose([ToNumpyArray(),
                                          ApplyMaskToTensor(self.VALID_MASK_FILE)])
        masked_nd_array = transforms_(self.VALID_3D_NIFTI_FILE)
        assert_that(masked_nd_array, instance_of(np.ndarray))

    def test_should_return_image_with_same_shape_as_original(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        transforms_ = transforms.Compose([ToNumpyArray(),
                                          ApplyMaskToTensor(self.VALID_MASK_FILE)])
        masked_nd_array = transforms_(self.VALID_3D_NIFTI_FILE)
        np.testing.assert_array_equal(masked_nd_array.shape, nd_array.shape)

    def test_should_raise_FileNotFoundError_exception_with_invalid_file(self):
        assert_that(calling(ApplyMaskToTensor).with_args(self.INVALID_MASK_FILE), raises(FileNotFoundError))


class RemapClassIDsTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")

    def setUp(self):
        pass

    def test_should_remap_labels_correctly(self):
        nd_array = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        transforms_ = transforms.Compose([ToNumpyArray(),
                                          RemapClassIDs([1, 2, 3], [4, 8, 12])])

        remapped_nd_array = transforms_(self.VALID_MASK_FILE)

        expected_nd_array = nd_array * 4
        assert_that(np.isnan(np.all(remapped_nd_array)), is_(False))
        assert_that(np.isnan(np.all(remapped_nd_array)), is_(False))
        assert_that(np.isnan(np.all(expected_nd_array)), is_(False))
        assert_that(np.isnan(np.all(expected_nd_array)), is_(False))
        np.testing.assert_array_equal(remapped_nd_array, expected_nd_array)

    def test_should_fail_with_non_integer_class_ids(self):
        assert_that(calling(RemapClassIDs).with_args([1, 2, 3], [1.5, 4, 8]), raises(ValueError))
