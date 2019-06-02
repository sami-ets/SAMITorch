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
import collections

from torchvision.transforms import transforms

from hamcrest import *

from samitorch.inputs.transformers import LoadNifti, ToNifti1Image, RemapClassIDs, ApplyMaskToNiftiImage, \
    ApplyMaskToTensor, ToNumpyArray, RandomCrop, NiftiToDisk, To2DNifti1Image, ToPNGFile, RandomCrop3D


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


class RandomCropTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/RandomCrop")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    CROPPED_NIFTI_IMAGE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_nd_array.nii")
    CROPPED_NIFTI_LABELS = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_labels.nii")
    CROPPED_PNG_IMAGE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_nd_array.png")
    CROPPED_PNG_LABELS = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_labels.png")

    ALL = [CROPPED_NIFTI_IMAGE, CROPPED_NIFTI_LABELS, CROPPED_PNG_IMAGE, CROPPED_PNG_LABELS]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
                yield from cls._flatten(el)
            else:
                yield el

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        all_files = cls._flatten(cls.ALL)
        for element in all_files:
            if os.path.exists(element):
                os.remove(element)
            else:
                print('File does not exists')

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

        print("Files deleted.")

    def test_transformer_should_return_ndarray(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        transform_ = transforms.Compose([RandomCrop(output_size=32)])

        cropped_nd_array, cropped_label = transform_((nd_array, labels))

        assert_that(cropped_nd_array, instance_of(np.ndarray))
        assert_that(cropped_label, instance_of(np.ndarray))
        assert_that(cropped_nd_array.ndim, equal_to(3))
        assert_that(cropped_label.ndim, equal_to(3))
        assert_that(cropped_nd_array.shape[0], equal_to(1))
        assert_that(cropped_label.shape[0], equal_to(1))

    def test_transformer_should_save_files_as_nifti_for_inspection(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        file_names = [self.CROPPED_NIFTI_IMAGE, self.CROPPED_NIFTI_LABELS]

        transform_ = transforms.Compose([RandomCrop(output_size=32),
                                         To2DNifti1Image(),
                                         NiftiToDisk(
                                             [os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_path) for file_path in
                                              file_names])])
        transform_((nd_array, labels))

        for file_name in file_names:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))

    def test_transformer_should_save_files_as_png_for_inspection(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        file_names = [self.CROPPED_PNG_IMAGE, self.CROPPED_PNG_LABELS]

        transform_ = transforms.Compose([RandomCrop(output_size=32),

                                         ToPNGFile(
                                             [os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_path) for file_path in
                                              file_names])])
        transform_((nd_array.astype(np.float32), 75 * labels.astype(np.uint8)))

        for file_name in file_names:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))


class RandomCrop3DTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/RandomCrop3D")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    CROPPED_NIFTI_IMAGE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_nd_array.nii")
    CROPPED_NIFTI_LABELS = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_labels.nii")
    CROPPED_PNG_IMAGE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_nd_array.png")
    CROPPED_PNG_LABELS = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_labels.png")

    ALL = [CROPPED_NIFTI_IMAGE, CROPPED_NIFTI_LABELS, CROPPED_PNG_IMAGE, CROPPED_PNG_LABELS]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
                yield from cls._flatten(el)
            else:
                yield el

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        all_files = cls._flatten(cls.ALL)
        for element in all_files:
            if os.path.exists(element):
                os.remove(element)
            else:
                print('File does not exists')

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

        print("Files deleted.")

    def test_transformer_should_return_ndarray(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        transform_ = transforms.Compose([RandomCrop3D(output_size=32)])

        cropped_nd_array, cropped_label = transform_((nd_array, labels))

        assert_that(cropped_nd_array, instance_of(np.ndarray))
        assert_that(cropped_label, instance_of(np.ndarray))
        assert_that(cropped_nd_array.ndim, equal_to(4))
        assert_that(cropped_label.ndim, equal_to(4))
        assert_that(cropped_nd_array.shape[0], equal_to(1))
        assert_that(cropped_label.shape[0], equal_to(1))

    def test_transformer_should_save_files_as_nifti_for_inspection(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        file_names = [self.CROPPED_NIFTI_IMAGE, self.CROPPED_NIFTI_LABELS]

        transform_ = transforms.Compose([RandomCrop3D(output_size=32),
                                         ToNifti1Image(None),
                                         NiftiToDisk(
                                             [os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_path) for file_path in
                                              file_names])])
        transform_((nd_array, labels))

        for file_name in file_names:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))


class ToPNGFileTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/ToPNGFile")
    PNG_FILE_2D = os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFile2D.png")
    PNG_FILE_3D = os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFile3D.png")
    PNG_FILE_4D = os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFile4D.png")
    PNG_TUPLE_2D = [os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleHxW_1.png"),
                    os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleHxW_2.png")]
    PNG_TUPLE_3D = [os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleCxHxW_1.png"),
                    os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleCxHxW_2.png")]
    PNG_TUPLE_4D = [os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleCxDxHxW_1.png"),
                    os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleCxDxHxW_2.png")]
    ALL = [PNG_FILE_2D, PNG_FILE_3D, PNG_FILE_4D, PNG_TUPLE_2D, PNG_TUPLE_3D, PNG_TUPLE_4D]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
                yield from cls._flatten(el)
            else:
                yield el

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        all_files = cls._flatten(cls.ALL)
        for element in all_files:
            if os.path.exists(element):
                os.remove(element)
            else:
                print('File does not exists')

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

        print("Files deleted.")

    def test_should_raise_exception_when_passing_4D_ndarray(self):
        nd_array = np.random.randint(0, 1000, size=(1, 32, 32, 32))

        transform_ = ToPNGFile(self.PNG_FILE_4D)

        assert_that(calling(transform_.__call__).with_args(nd_array), raises(TypeError))

    def test_should_raise_exception_when_passing_tuple_of_4D_ndarray(self):
        nd_array = np.random.randint(0, 1000, size=(1, 32, 32, 32))

        transform_ = ToPNGFile(self.PNG_TUPLE_4D)

        assert_that(calling(transform_.__call__).with_args((nd_array, nd_array)), raises(TypeError))

    def test_should_pass_when_passing_2D_ndarray(self):
        nd_array = np.random.randint(0, 1000, size=(32, 32))

        ToPNGFile(self.PNG_FILE_2D).__call__(nd_array)

        assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, self.PNG_FILE_2D)))

    def test_should_pass_when_passing_3D_ndarray(self):
        nd_array = np.random.randint(0, 1000, size=(1, 32, 32))

        ToPNGFile(self.PNG_FILE_3D).__call__(nd_array)

        assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, self.PNG_FILE_3D)))

    def test_should_pass_when_passing_tuple_2D_nd_array(self):

        nd_array = np.random.randint(0, 1000, size=(32, 32))

        ToPNGFile(self.PNG_TUPLE_2D).__call__((nd_array, nd_array))

        for file_name in self.PNG_TUPLE_2D:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))

    def test_should_pass_when_passing_tuple_3D_nd_array(self):

        nd_array = np.random.randint(0, 1000, size=(1, 32, 32))

        ToPNGFile(self.PNG_TUPLE_3D).__call__((nd_array, nd_array))

        for file_name in self.PNG_TUPLE_3D:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))


class NiftiToDiskTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/NiftiToDisk")
    NIFTI_FILE_FROM_TUPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "NiftiToDiskTuple.png")
    NIFTI_FILE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "NiftiToDisk.nii")
    VALID_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")

    ALL = [NIFTI_FILE_FROM_TUPLE, NIFTI_FILE]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
                yield from cls._flatten(el)
            else:
                yield el

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        all_files = cls._flatten(cls.ALL)
        for element in all_files:
            if os.path.exists(element):
                os.remove(element)
            else:
                print('File does not exists')

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

        print("Files deleted.")

    def test_should_pass_when_saving_one_nifti_image_to_disk(self):
        header = LoadNifti().__call__(self.VALID_NIFTI_FILE).header

        transform_ = transforms.Compose([ToNumpyArray(),
                                         ToNifti1Image(header),
                                         NiftiToDisk(self.NIFTI_FILE)])

        transform_(self.VALID_NIFTI_FILE)

        assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, self.NIFTI_FILE)))
