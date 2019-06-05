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

from samitorch.inputs.sample import Sample

from samitorch.inputs.transformers import LoadNifti, ToNifti1Image, RemapClassIDs, ApplyMaskToNiftiImage, \
    ApplyMaskToTensor, ToNumpyArray, RandomCrop, NiftiToDisk, To2DNifti1Image, ToPNGFile, RandomCrop3D, \
    NiftiImageToNumpy, ResampleNiftiImageToTemplate


class ToNumpyArrayTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_3D_LABELS = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    INVALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_T1.nii")
    INVALID_3D_LABELS = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_mask.nii")

    def setUp(self):
        pass

    def test_should_return_a_sample_containing_both_Numpy_ndarrays(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_3D_LABELS, is_labeled=True)

        transformed_sample = ToNumpyArray().__call__(sample)

        assert_that(transformed_sample.x, instance_of(np.ndarray))
        assert_that(transformed_sample.y, instance_of(np.ndarray))

    def test_should_return_a_single_Numpy_ndarray_from_sample(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE)
        transformed_sample = ToNumpyArray().__call__(sample)

        assert_that(transformed_sample.x, instance_of(np.ndarray))
        assert_that(transformed_sample.y, is_(None))

    def test_should_return_a_single_Numpy_ndarray_from_string(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_LABELS)
        assert_that(nd_array, instance_of(np.ndarray))

    def test_should_return_a_single_Numpy_ndarray_from_Nifti1Image(self):
        nifti_image = nib.load(self.VALID_3D_LABELS)

        nd_array = ToNumpyArray().__call__(nifti_image)
        assert_that(nd_array, instance_of(np.ndarray))

    def test_should_raise_FileNotFound_with_x_invalid_type(self):
        sample = Sample(x=self.INVALID_3D_NIFTI_FILE)

        assert_that(calling(ToNumpyArray()).with_args(sample), raises(FileNotFoundError))

    def test_should_raise_FileNotFound_with_y_invalid_type(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.INVALID_3D_LABELS, is_labeled=True)

        assert_that(calling(ToNumpyArray()).with_args(sample), raises(FileNotFoundError))

    def test_should_raise_FileNotFound_with_two_invalid_path(self):
        sample = Sample(x=self.INVALID_3D_NIFTI_FILE, y=self.INVALID_3D_LABELS, is_labeled=True)

        assert_that(calling(ToNumpyArray()).with_args(sample), raises(FileNotFoundError))


class LoadNiftiTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_3D_LABELS = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    VALID_4D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "FA.nii")
    INVALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_T1.nii")
    INVVALID_3D_LABELS = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_mask.nii")

    def setUp(self):
        pass

    def test_should_return_a_sample_containing_both_Nifti1Image(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_3D_LABELS, is_labeled=True)

        transformed_sample = LoadNifti().__call__(sample)

        assert_that(transformed_sample.x, instance_of(nib.Nifti1Image))
        assert_that(transformed_sample.y, instance_of(nib.Nifti1Image))

    def test_should_return_a_single_Nifti1Image(self):
        nifti1Image = LoadNifti().__call__(self.VALID_3D_NIFTI_FILE)

        assert_that(nifti1Image, instance_of(nib.Nifti1Image))

    def test_should_return_a_sample_containing_one_Nifti1Image(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE)
        transformed_sample = LoadNifti().__call__(sample)

        assert_that(transformed_sample.x, instance_of(nib.Nifti1Image))
        assert_that(transformed_sample.y, is_(None))

    def test_should_raise_FileNotFoundError_with_x_invalid_path(self):
        sample = Sample(x=self.INVALID_3D_NIFTI_FILE)

        assert_that(calling(LoadNifti()).with_args(sample), raises(FileNotFoundError))

    def test_should_raise_FileNotFoundError_with_y_invalid_path(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.INVVALID_3D_LABELS, is_labeled=True)

        assert_that(calling(LoadNifti()).with_args(sample), raises(FileNotFoundError))

    def test_should_raise_FileNotFoundError_with_two_invalid_path(self):
        sample = Sample(x=self.INVALID_3D_NIFTI_FILE, y=self.INVVALID_3D_LABELS, is_labeled=True)

        assert_that(calling(LoadNifti()).with_args(sample), raises(FileNotFoundError))

    def test_should_raise_FileNotFoundError_with_one_invalid_path(self):
        assert_that(calling(LoadNifti()).with_args(self.INVALID_3D_NIFTI_FILE), raises(FileNotFoundError))

    def test_should_return_nifti_image_with_valid_4d_nifti_file(self):
        nifti_4d_image = LoadNifti().__call__(self.VALID_4D_NIFTI_FILE)
        assert_that(nifti_4d_image, is_not(None))
        assert_that(nifti_4d_image, is_(nib.Nifti1Image))


class NiftiImageToNumpyTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_3D_LABELS = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    INVALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_T1.nii")
    INVVALID_3D_LABELS = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_mask.nii")

    def setUp(self):
        pass

    def test_should_return_a_sample_containing_both_Numpy_ndarrays(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_3D_LABELS, is_labeled=True)

        transformed_sample = LoadNifti().__call__(sample)
        transformed_sample = NiftiImageToNumpy().__call__(transformed_sample)

        assert_that(transformed_sample.x, instance_of(np.ndarray))
        assert_that(transformed_sample.y, instance_of(np.ndarray))

    def test_should_return_a_single_Numpy_ndarray(self):
        nifti1Image = LoadNifti().__call__(self.VALID_3D_NIFTI_FILE)
        nd_array = NiftiImageToNumpy().__call__(nifti1Image)

        assert_that(nd_array, instance_of(np.ndarray))

    def test_should_return_a_sample_containing_one_Numpy_ndarray(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE)
        transformed_sample = LoadNifti().__call__(sample)
        transformed_sample = NiftiImageToNumpy().__call__(transformed_sample)

        assert_that(transformed_sample.x, instance_of(np.ndarray))
        assert_that(transformed_sample.y, is_(None))

    def test_should_raise_TypeError_with_x_invalid_type(self):
        sample = Sample(x=self.INVALID_3D_NIFTI_FILE)

        assert_that(calling(NiftiImageToNumpy()).with_args(sample), raises(TypeError))

    def test_should_raise_TypeError_with_y_invalid_type(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.INVVALID_3D_LABELS, is_labeled=True)

        assert_that(calling(NiftiImageToNumpy()).with_args(sample), raises(TypeError))

    def test_should_raise_TypeError_with_two_invalid_path(self):
        sample = Sample(x=self.INVALID_3D_NIFTI_FILE, y=self.INVVALID_3D_LABELS, is_labeled=True)

        assert_that(calling(NiftiImageToNumpy()).with_args(sample), raises(TypeError))

    def test_should_raise_TypeError_with_one_invalid_type(self):
        assert_that(calling(NiftiImageToNumpy()).with_args(self.INVALID_3D_NIFTI_FILE), raises(TypeError))


class ResampleNiftiImageToTemplateTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/resample")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/ResampleNiftiImageToTemplate")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_3D_1MM_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1_1mm.nii")
    INVALID_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "invalid.nii")
    RESAMPLED_NIFTI_FROM_NIFTI = os.path.join(OUTPUT_DATA_FOLDER_PATH, "resampled_nifti_from_nifti.nii")
    RESAMPLED_NIFTI_FROM_STRING = os.path.join(OUTPUT_DATA_FOLDER_PATH, "resampled_nifti_from_string.nii")
    RESAMPLED_NIFTI_FROM_STRING_TEMPLATE = os.path.join(OUTPUT_DATA_FOLDER_PATH,
                                                        "resampled_nifti_from_string_template.nii")

    ALL = [RESAMPLED_NIFTI_FROM_NIFTI, RESAMPLED_NIFTI_FROM_STRING, RESAMPLED_NIFTI_FROM_STRING_TEMPLATE]

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

    def test_should_return_resampled_Nifti1Image_from_Nifti1Image(self):
        transform_ = ResampleNiftiImageToTemplate(interpolation="continuous",
                                                  clip=False,
                                                  template=nib.load(self.VALID_3D_1MM_NIFTI_FILE))
        nifti_image = transform_(nib.load(self.VALID_3D_NIFTI_FILE))

        np.testing.assert_array_equal([256, 256, 192], nifti_image.shape)
        assert_that(nifti_image, instance_of(nib.Nifti1Image))
        transform_ = NiftiToDisk(self.RESAMPLED_NIFTI_FROM_NIFTI)
        transform_(nifti_image)

    def test_should_return_resampled_Nifti1Image_from_string(self):
        transform_ = ResampleNiftiImageToTemplate(interpolation="continuous",
                                                  clip=False,
                                                  template=nib.load(self.VALID_3D_1MM_NIFTI_FILE))
        nifti_image = transform_(self.VALID_3D_NIFTI_FILE)
        np.testing.assert_array_equal([256, 256, 192], nifti_image.shape)
        assert_that(nifti_image, instance_of(nib.Nifti1Image))
        transform_ = NiftiToDisk(self.RESAMPLED_NIFTI_FROM_STRING)
        transform_(nifti_image)

    def test_should_return_resampled_Nifti1Image_from_string_template(self):
        transform_ = ResampleNiftiImageToTemplate(interpolation="continuous",
                                                  clip=False,
                                                  template=self.VALID_3D_1MM_NIFTI_FILE)
        nifti_image = transform_(self.VALID_3D_NIFTI_FILE)
        np.testing.assert_array_equal([256, 256, 192], nifti_image.shape)
        assert_that(nifti_image, instance_of(nib.Nifti1Image))
        transform_ = NiftiToDisk(self.RESAMPLED_NIFTI_FROM_STRING_TEMPLATE)
        transform_(nifti_image)

    def test_should_return_resampled_Nifti1Image_from_Nifti1Image_from_string_template(self):
        transform_ = ResampleNiftiImageToTemplate(interpolation="continuous",
                                                  clip=False,
                                                  template=self.VALID_3D_1MM_NIFTI_FILE)
        nifti_image = transform_(nib.load(self.VALID_3D_NIFTI_FILE))
        np.testing.assert_array_equal([256, 256, 192], nifti_image.shape)
        assert_that(nifti_image, instance_of(nib.Nifti1Image))
        transform_ = NiftiToDisk(self.RESAMPLED_NIFTI_FROM_STRING_TEMPLATE)
        transform_(nifti_image)

    def test_should_return_resampled_Nifti1Image_from_sample_Nifti1Image(self):
        sample = Sample(x=nib.load(self.VALID_3D_NIFTI_FILE), template=nib.load(self.VALID_3D_1MM_NIFTI_FILE))
        transform_ = ResampleNiftiImageToTemplate(interpolation="continuous",
                                                  clip=False)
        transformed_sample = transform_(sample)
        np.testing.assert_array_equal([256, 256, 192], transformed_sample.x.shape)
        assert_that(transformed_sample.x, instance_of(nib.Nifti1Image))


class ToNifti1ImageTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_3D_LABELS = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")

    def setUp(self):
        pass

    def test_should_return_a_Nifti1Image_from_Numpy_ndarray(self):
        header = nib.load(self.VALID_3D_LABELS).header
        nd_array = ToNumpyArray().__call__(self.VALID_3D_LABELS)
        transform_ = ToNifti1Image(header)
        new_nifti_image = transform_(nd_array)
        assert_that(new_nifti_image, instance_of(nib.nifti1.Nifti1Image))

    def test_should_return_a_Nifti1Image_Sample_from_Sample(self):
        header_image = nib.load(self.VALID_3D_LABELS).header
        header_labels = nib.load(self.VALID_3D_LABELS).header
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image([header_image, header_labels])
        nifti_image_sample = transform_(nd_array_sample)

        assert_that(nifti_image_sample.x, instance_of(nib.Nifti1Image))
        assert_that(nifti_image_sample.y, instance_of(nib.Nifti1Image))

    def test_should_return_a_Nifti1Image_Sample_from_single_element_Sample(self):
        header_image = nib.load(self.VALID_3D_LABELS).header
        sample = Sample(x=self.VALID_3D_LABELS)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image(header_image)
        nifti_image_sample = transform_(nd_array_sample)

        assert_that(nifti_image_sample.x, instance_of(nib.Nifti1Image))
        assert_that(nifti_image_sample.y, is_(None))

    def test_should_return_a_Nifti1Image_Sample_from_single_element_Sample_without_header(self):
        sample = Sample(x=self.VALID_3D_LABELS)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image(None)
        nifti_image_sample = transform_(nd_array_sample)
        assert_that(nifti_image_sample.x, instance_of(nib.Nifti1Image))
        assert_that(nifti_image_sample.y, is_(None))

    def test_should_return_a_Nifti1Image_Sample_from_Sample_without_header(self):
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image(None)
        nifti_image_sample = transform_(nd_array_sample)

        assert_that(nifti_image_sample.x, instance_of(nib.Nifti1Image))
        assert_that(nifti_image_sample.y, instance_of(nib.Nifti1Image))

    def test_should_raise_TypeErrorException_with_non_ndarray_sample(self):
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        transform_ = ToNifti1Image(None)

        assert_that(calling(transform_).with_args(sample), raises(TypeError))

    def test_should_raise_TypeErrorException_with_incorrect_dims_ndarray_in_x_sample(self):
        sample = Sample(x=np.random.randint(0, 255, (32, 32), dtype=np.int16))
        transform_ = ToNifti1Image(None)

        assert_that(calling(transform_).with_args(sample), raises(TypeError))

    def test_should_raise_TypeErrorException_with_incorrect_dims_ndarray_in_y_sample(self):
        sample = Sample(x=np.random.randint(0, 255, (32, 32), dtype=np.int16),
                        y=np.random.randint(0, 255, (32, 32), dtype=np.int16), is_labeled=True)
        transform_ = ToNifti1Image(None)

        assert_that(calling(transform_).with_args(sample), raises(TypeError))

    def test_should_raise_ValueError_with_non_list_header_while_sample_is_labeled(self):
        header_image = nib.load(self.VALID_3D_LABELS).header
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image(header_image)

        assert_that(calling(transform_).with_args(nd_array_sample), raises(ValueError))

    def test_should_raise_AssertionError_with_non_list_header_while_sample_is_mislabeled(self):
        header_image = nib.load(self.VALID_3D_LABELS).header
        header_labels = nib.load(self.VALID_3D_LABELS).header
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=False)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image([header_image, header_labels])

        assert_that(calling(transform_).with_args(nd_array_sample), raises(AssertionError))

    def test_should_raise_AssertionError_with_non_list_header_while_sample_has_bad_header(self):
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image(["bad_header", np.random.randint(0, 255, (32, 32))])

        assert_that(calling(transform_).with_args(nd_array_sample), raises(TypeError))


class NiftiToDiskTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/NiftiToDisk")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_3D_LABELS = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    SAVED_NIFTI_FILE_FROM_NIFTI = os.path.join(OUTPUT_DATA_FOLDER_PATH, "T1_saved_from_nifti.nii")
    SAVED_NIFTI_FILE_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "T1_saved_from_sample.nii")
    SAVED_LABELS_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "Mask_saved_from_sample.nii")
    ALL = [SAVED_NIFTI_FILE_FROM_NIFTI, SAVED_NIFTI_FILE_FROM_SAMPLE, SAVED_LABELS_FROM_SAMPLE]

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

    def test_should_save_nifti_file_from_Nifti1Image(self):
        nifti_image = nib.load(self.VALID_3D_NIFTI_FILE)
        transform_ = NiftiToDisk(self.SAVED_NIFTI_FILE_FROM_NIFTI)
        transform_(nifti_image)

        assert_that(os.path.exists(self.SAVED_NIFTI_FILE_FROM_NIFTI))

    def test_should_save_nifti_file_from_sample(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_3D_LABELS, is_labeled=True)
        header_image = nib.load(self.VALID_3D_NIFTI_FILE).header
        header_label = nib.load(self.VALID_3D_LABELS).header

        transformed_sample = ToNumpyArray().__call__(sample)
        transformed_sample = ToNifti1Image([header_image, header_label]).__call__(transformed_sample)
        transform_ = NiftiToDisk([self.SAVED_NIFTI_FILE_FROM_SAMPLE, self.SAVED_LABELS_FROM_SAMPLE])
        transform_(transformed_sample)

        assert_that(os.path.exists(self.SAVED_NIFTI_FILE_FROM_SAMPLE))
        assert_that(os.path.exists(self.SAVED_LABELS_FROM_SAMPLE))

    def test_should_raise_TypeError_exception_when_not_passing_a_list_of_file_paths(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_3D_LABELS, is_labeled=True)
        header_image = nib.load(self.VALID_3D_NIFTI_FILE).header
        header_label = nib.load(self.VALID_3D_LABELS).header

        transformed_sample = ToNumpyArray().__call__(sample)
        transformed_sample = ToNifti1Image([header_image, header_label]).__call__(transformed_sample)
        transform_ = NiftiToDisk("bad_input")

        assert_that(calling(transform_).with_args(transformed_sample), raises(TypeError))


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
