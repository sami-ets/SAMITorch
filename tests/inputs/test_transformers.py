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
import torch

from torchvision.transforms import transforms

from hamcrest import *

from samitorch.inputs.sample import Sample

from samitorch.inputs.transformers import LoadNifti, ToNifti1Image, RemapClassIDs, ApplyMask, \
    ToNumpyArray, RandomCrop, NiftiToDisk, To2DNifti1Image, ToPNGFile, RandomCrop3D, \
    NiftiImageToNumpy, ResampleNiftiImageToTemplate, CropToContent, ToTensorPatches, ToNDTensor, ToNDArrayPatches


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
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

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
        transform_ = ToNifti1Image([header_image])
        nifti_image_sample = transform_(nd_array_sample)

        assert_that(nifti_image_sample.x, instance_of(nib.Nifti1Image))
        assert_that(nifti_image_sample.y, is_(None))

    def test_should_return_sample_from_single_element_Sample_with_one_None_Header(self):
        sample = Sample(x=self.VALID_3D_LABELS)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image([None])
        transformed_sample = transform_(nd_array_sample)

        assert_that(transformed_sample.x, instance_of(nib.Nifti1Image))

    def test_should_return_a_Nifti1Image_Sample_from_Sample_without_header(self):
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image([None, None])
        nifti_image_sample = transform_(nd_array_sample)

        assert_that(nifti_image_sample.x, instance_of(nib.Nifti1Image))
        assert_that(nifti_image_sample.y, instance_of(nib.Nifti1Image))

    def test_should_raise_TypeErrorException_with_non_ndarray_sample(self):
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        transform_ = ToNifti1Image([None])

        assert_that(calling(transform_).with_args(sample), raises(TypeError))

    def test_should_raise_TypeErrorException_with_incorrect_dims_ndarray_in_x_sample(self):
        sample = Sample(x=np.random.randint(0, 255, (32, 32), dtype=np.int16))
        transform_ = ToNifti1Image([None])

        assert_that(calling(transform_).with_args(sample), raises(TypeError))

    def test_should_raise_TypeErrorException_with_incorrect_dims_ndarray_in_y_sample(self):
        sample = Sample(x=np.random.randint(0, 255, (32, 32), dtype=np.int16),
                        y=np.random.randint(0, 255, (32, 32), dtype=np.int16), is_labeled=True)
        transform_ = ToNifti1Image([None, None])

        assert_that(calling(transform_).with_args(sample), raises(TypeError))

    def test_should_return_a_nifti1Image_sample_with_non_list_header_while_sample_is_labeled(self):
        header_image = nib.load(self.VALID_3D_LABELS).header
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image(header_image)

        assert_that(calling(transform_).with_args(nd_array_sample), raises(TypeError))

    def test_should_raise_TypeError_with_non_list_header_with_Sample(self):
        header_image = nib.load(self.VALID_3D_NIFTI_FILE).header
        header_labels = nib.load(self.VALID_3D_LABELS).header
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_3D_LABELS, is_labeled=True)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image(header_image)

        assert_that(calling(transform_).with_args(nd_array_sample), raises(TypeError))

    def test_should_raise_AssertionError_with_non_list_header_while_sample_has_bad_header(self):
        sample = Sample(x=self.VALID_3D_LABELS, y=self.VALID_3D_NIFTI_FILE, is_labeled=True)
        nd_array_sample = ToNumpyArray().__call__(sample)
        transform_ = ToNifti1Image(["bad_header", np.random.randint(0, 255, (32, 32))])

        assert_that(calling(transform_).with_args(nd_array_sample), raises(AttributeError))


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
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

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

        assert_that(calling(transform_).with_args(transformed_sample), raises(ValueError))


class ApplyMaskTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/ApplyMaskToNiftiImage")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    INVALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "invalid_mask.nii")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    MASKED_T1_FILE_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "masked_t1_file_from_sample.nii")
    MASK_FILE_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "mask.nii")

    ALL = [MASKED_T1_FILE_FROM_SAMPLE, MASK_FILE_FROM_SAMPLE]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_transformer_should_initialize_with_str(self):
        transform_ = ApplyMask(self.VALID_MASK_FILE)
        assert_that(transform_, is_not(None))
        assert_that(transform_._mask, instance_of(np.ndarray))

    def test_transformer_should_initialize_with_Nifti1Image(self):
        transform_ = ApplyMask(nib.load(self.VALID_MASK_FILE))
        assert_that(transform_, is_not(None))
        assert_that(transform_._mask, instance_of(np.ndarray))

    def test_transformer_should_initialize_with_Numpy_ndarray(self):
        transform_ = ApplyMask(nib.load(self.VALID_MASK_FILE).get_fdata().__array__())
        assert_that(transform_, is_not(None))
        assert_that(transform_._mask, instance_of(np.ndarray))

    def test_should_initialize_mask_correctly(self):
        transformer = ApplyMask(self.VALID_MASK_FILE)
        np.testing.assert_array_less(transformer._mask, 1.0 + 1e-10)
        assert_that(len(transformer._mask.shape), is_(3))

    def test_returned_image_should_be_Nifti1Image_type(self):
        transformer = ApplyMask(self.VALID_MASK_FILE)
        nifti = nib.load(self.VALID_3D_NIFTI_FILE)
        transformed_nifti = transformer.__call__(nifti)
        assert_that(transformed_nifti, instance_of(nib.Nifti1Image))

    def test_should_return_image_with_same_shape_as_original(self):
        transformer = ApplyMask(self.VALID_MASK_FILE)
        nifti = nib.load(self.VALID_3D_NIFTI_FILE)
        transformed_nifti = transformer.__call__(nifti)
        np.testing.assert_array_equal(transformed_nifti.shape, nifti.shape)

    def test_should_return_sample_with_masked_image(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_MASK_FILE, is_labeled=True)
        transformed_sample = LoadNifti().__call__(sample)
        transformed_sample = ApplyMask().__call__(transformed_sample)

        assert_that(transformed_sample.x, instance_of(nib.Nifti1Image))

    def test_should_raise_TypeError_exception_with_invalid_masked_image(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_MASK_FILE, is_labeled=True)
        transformed_sample = LoadNifti().__call__(sample)
        transformed_sample.y = "incorrect_input"
        transform_ = ApplyMask()

        assert_that(calling(transform_).with_args(transformed_sample), raises(TypeError))

    def test_should_raise_TypeError_exception_with_invalid_argument(self):
        transform_ = ApplyMask()

        assert_that(calling(transform_).with_args("invalid_argument"), raises(TypeError))

    def test_should_return_sample_with_masked_image_for_inspection(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE, y=self.VALID_MASK_FILE, is_labeled=True)
        transformed_sample = LoadNifti().__call__(sample)
        transformed_sample = ApplyMask().__call__(transformed_sample)

        transform_ = NiftiToDisk([self.MASKED_T1_FILE_FROM_SAMPLE, self.MASK_FILE_FROM_SAMPLE])
        transform_(transformed_sample)

        assert_that(os.path.exists(self.MASKED_T1_FILE_FROM_SAMPLE))
        assert_that(os.path.exists(self.MASK_FILE_FROM_SAMPLE))

    def test_should_raise_FileNotFoundError_exception_with_invalid_file(self):
        assert_that(calling(ApplyMask).with_args(self.INVALID_MASK_FILE), raises(FileNotFoundError))

    def test_should_raise_TypeError_exception_when_passing_invalid_x_sample(self):
        sample = Sample(x="invalid", y=np.random.randint(0, 255, (32, 32)), is_labeled=True)
        transform_ = ApplyMask()

        assert_that(calling(transform_).with_args(sample), raises(TypeError))

    def test_shoud_raise_TypeError_exception_with_invalid_argument(self):
        transform_ = ApplyMask()

        assert_that(calling(transform_).with_args("invalid"), raises(TypeError))


class RemapClassIDsTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/RemapClassIDs")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    REMAPPED_LABELS_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "remapped_labels_from_sample.nii")
    UNTOUCHED_IMAGE_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "original_image_from_sample.nii")
    REMAPPED_LABELS_FROM_NDARRAY = os.path.join(OUTPUT_DATA_FOLDER_PATH, "remapped_labels_from_ndarray.nii")
    ALL = [UNTOUCHED_IMAGE_FROM_SAMPLE, REMAPPED_LABELS_FROM_SAMPLE, REMAPPED_LABELS_FROM_NDARRAY]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_should_remap_labels_correctly_from_Numpy_ndarray(self):
        nd_array = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        transform_ = RemapClassIDs([1, 2, 3], [4, 8, 12])

        remapped_nd_array = transform_(nd_array)

        expected_nd_array = nd_array * 4
        assert_that(np.isnan(np.all(remapped_nd_array)), is_(False))
        assert_that(np.isnan(np.all(remapped_nd_array)), is_(False))
        assert_that(np.isnan(np.all(expected_nd_array)), is_(False))
        assert_that(np.isnan(np.all(expected_nd_array)), is_(False))
        np.testing.assert_array_equal(remapped_nd_array, expected_nd_array)

    def test_should_fail_with_non_integer_class_ids(self):
        assert_that(calling(RemapClassIDs).with_args([1, 2, 3], [1.5, 4, 8]), raises(ValueError))

    def test_should_return_remapped_sample_from_Nifti1Image_sample(self):
        x_nifti = nib.load(self.VALID_3D_NIFTI_FILE)
        y_nifti = nib.load(self.VALID_MASK_FILE)
        sample = Sample(x=x_nifti, y=y_nifti, is_labeled=True)
        transform_ = RemapClassIDs([1, 2, 3], [4, 8, 12])

        remapped_sample = transform_(sample)

        expected_x = x_nifti
        expected_y = nib.Nifti1Image(y_nifti.get_fdata().__array__() * 4, affine=y_nifti.affine, header=y_nifti.header)

        np.testing.assert_array_equal(remapped_sample.y.get_fdata(), expected_y.get_fdata())
        np.testing.assert_array_equal(remapped_sample.x.get_fdata(), expected_x.get_fdata())

    def test_should_return_remapped_sample_from_Nifti1Image_for_inspection(self):
        x_nifti = nib.load(self.VALID_3D_NIFTI_FILE)
        y_nifti = nib.load(self.VALID_MASK_FILE)
        sample = Sample(x=x_nifti, y=y_nifti, is_labeled=True)
        transform_ = RemapClassIDs([1, 2, 3], [4, 8, 12])

        remapped_sample = transform_(sample)

        NiftiToDisk([self.UNTOUCHED_IMAGE_FROM_SAMPLE, self.REMAPPED_LABELS_FROM_SAMPLE]).__call__(remapped_sample)

    def test_should_return_remapped_sample_from_Numpy_ndarray_sample_for_inspection(self):
        x = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        y = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        header_image = nib.load(self.VALID_3D_NIFTI_FILE).header
        header_label = nib.load(self.VALID_MASK_FILE).header
        sample = Sample(x=x, y=y, is_labeled=True)

        transform_ = RemapClassIDs([1, 2, 3], [4, 8, 12])

        remapped_sample = transform_(sample)

        nifti_sample = ToNifti1Image([header_image, header_label]).__call__(remapped_sample)

        NiftiToDisk([self.UNTOUCHED_IMAGE_FROM_SAMPLE, self.REMAPPED_LABELS_FROM_SAMPLE]).__call__(nifti_sample)

    def test_should_remap_labels_correctly_from_Numpy_ndarray_for_inspection(self):
        nd_array = ToNumpyArray().__call__(self.VALID_MASK_FILE)
        nifti_image = nib.load(self.VALID_MASK_FILE)
        transform_ = RemapClassIDs([1, 2, 3], [4, 8, 12])
        remapped_nd_array = transform_(nd_array)
        expected_nd_array = nd_array * 4

        assert_that(np.isnan(np.all(remapped_nd_array)), is_(False))
        assert_that(np.isnan(np.all(remapped_nd_array)), is_(False))
        assert_that(np.isnan(np.all(expected_nd_array)), is_(False))
        assert_that(np.isnan(np.all(expected_nd_array)), is_(False))
        np.testing.assert_array_equal(remapped_nd_array, expected_nd_array)

        new_nifti_image = nib.Nifti1Image(remapped_nd_array.transpose((3, 2, 1, 0)), nifti_image.affine,
                                          nifti_image.header)
        nib.save(new_nifti_image, self.REMAPPED_LABELS_FROM_NDARRAY)

    def test_should_raise_TypeError_exception_with_2D_Numpy_ndarray(self):
        nd_array = np.random.randint(0, 3, (32, 32))
        transform_ = RemapClassIDs([1, 2, 3], [4, 8, 12])
        remapped_nd_array = transform_(nd_array)
        expected_nd_array = nd_array * 4

        assert_that(np.isnan(np.all(remapped_nd_array)), is_(False))
        assert_that(np.isnan(np.all(remapped_nd_array)), is_(False))
        assert_that(np.isnan(np.all(expected_nd_array)), is_(False))
        assert_that(np.isnan(np.all(expected_nd_array)), is_(False))
        np.testing.assert_array_equal(remapped_nd_array, expected_nd_array)


class To2DNifti1ImageTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/To2DNifti1Image")
    NIFTI_2D_IMAGE_FROM_NDARRAY = os.path.join(OUTPUT_DATA_FOLDER_PATH, "2d_nifti_image_from_nd_array.nii")
    NIFTI_2D_IMAGE_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "2d_nifti_image_from_sample.nii")
    NIFTI_2D_LABEL_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "2d_nifti_label_from_sample.nii")
    ALL = [NIFTI_2D_IMAGE_FROM_NDARRAY, NIFTI_2D_IMAGE_FROM_SAMPLE, NIFTI_2D_LABEL_FROM_SAMPLE]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_should_return_Nifti1Image_from_Numpy_array(self):
        nifti_image = nib.load(self.VALID_3D_NIFTI_FILE)
        nd_array = nib.load(self.VALID_3D_NIFTI_FILE).get_fdata().__array__()
        transform_ = To2DNifti1Image(nifti_image.header)
        nd_array_2d = transform_(nd_array)

        assert_that(nd_array_2d, instance_of(nib.Nifti1Image))

    def test_should_return_Sample_from_Sample_of_Numpy_array(self):
        header_x = nib.load(self.VALID_3D_NIFTI_FILE).header
        header_y = nib.load(self.VALID_MASK_FILE).header

        x = nib.load(self.VALID_3D_NIFTI_FILE).get_fdata().__array__()
        y = nib.load(self.VALID_MASK_FILE).get_fdata().__array__()

        sample = Sample(x=x, y=y, is_labeled=True)

        transform_ = To2DNifti1Image([header_x, header_y])
        transformed_sample = transform_(sample)

        assert_that(transformed_sample.x, instance_of(nib.Nifti1Image))
        assert_that(transformed_sample.y, instance_of(nib.Nifti1Image))

    def test_should_return_Sample_from_Sample_of_Numpy_array_for_inspection(self):
        header_x = nib.load(self.VALID_3D_NIFTI_FILE).header
        header_y = nib.load(self.VALID_MASK_FILE).header

        x = nib.load(self.VALID_3D_NIFTI_FILE).get_fdata().__array__()
        y = nib.load(self.VALID_MASK_FILE).get_fdata().__array__()

        sample = Sample(x=x, y=y, is_labeled=True)
        transform_ = RandomCrop(64)
        transformed_sample = transform_(sample)
        transform_ = To2DNifti1Image([header_x, header_y])
        transformed_sample = transform_(transformed_sample)

        assert_that(transformed_sample.x, instance_of(nib.Nifti1Image))
        assert_that(transformed_sample.y, instance_of(nib.Nifti1Image))

        NiftiToDisk([self.NIFTI_2D_IMAGE_FROM_SAMPLE, self.NIFTI_2D_LABEL_FROM_SAMPLE]).__call__(transformed_sample)


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
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_transformer_should_return_ndarray(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)

        sample = Sample(x=nd_array, y=labels, is_labeled=True)

        transform_ = transforms.Compose([RandomCrop(output_size=32)])

        transformed_sample = transform_(sample)

        assert_that(transformed_sample.x, instance_of(np.ndarray))
        assert_that(transformed_sample.y, instance_of(np.ndarray))
        assert_that(transformed_sample.x.ndim, equal_to(3))
        assert_that(transformed_sample.y.ndim, equal_to(3))
        assert_that(transformed_sample.x.shape[0], equal_to(1))
        assert_that(transformed_sample.y.shape[0], equal_to(1))

    def test_transformer_should_save_files_as_nifti_for_inspection(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)
        sample = Sample(x=nd_array, y=labels, is_labeled=True)
        file_names = [self.CROPPED_NIFTI_IMAGE, self.CROPPED_NIFTI_LABELS]
        transform_ = transforms.Compose([RandomCrop(output_size=32),
                                         To2DNifti1Image(),
                                         NiftiToDisk(
                                             [os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_path) for file_path in
                                              file_names])])
        transform_(sample)

        for file_name in file_names:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))

    def test_transformer_should_save_files_as_png_for_inspection(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)
        sample = Sample(x=nd_array, y=labels, is_labeled=True)
        file_names = [self.CROPPED_PNG_IMAGE, self.CROPPED_PNG_LABELS]

        transform_ = transforms.Compose([RandomCrop(output_size=32),
                                         ToPNGFile(
                                             [os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_path) for file_path in
                                              file_names])])
        transform_(sample)

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
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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

        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_transformer_should_return_ndarray(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)
        sample = Sample(x=nd_array, y=labels, is_labeled=True)
        transform_ = transforms.Compose([RandomCrop3D(output_size=32)])

        transformed_sample = transform_(sample)

        assert_that(transformed_sample.x, instance_of(np.ndarray))
        assert_that(transformed_sample.y, instance_of(np.ndarray))
        assert_that(transformed_sample.x.ndim, equal_to(4))
        assert_that(transformed_sample.y.ndim, equal_to(4))
        assert_that(transformed_sample.x.shape[0], equal_to(1))
        assert_that(transformed_sample.y.shape[0], equal_to(1))

    def test_transformer_should_save_files_as_nifti_for_inspection(self):
        nd_array = ToNumpyArray().__call__(self.VALID_3D_NIFTI_FILE)
        labels = ToNumpyArray().__call__(self.VALID_MASK_FILE)
        sample = Sample(x=nd_array, y=labels, is_labeled=True)
        file_names = [self.CROPPED_NIFTI_IMAGE, self.CROPPED_NIFTI_LABELS]

        transform_ = transforms.Compose([RandomCrop3D(output_size=32),
                                         ToNifti1Image([None, None]),
                                         NiftiToDisk(
                                             [os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_path) for file_path in
                                              file_names])])
        transform_(sample)

        for file_name in file_names:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))


class ToPNGFileTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/ToPNGFile")
    PNG_FILE_2D = os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFile2D.png")
    PNG_FILE_3D = os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFile3D.png")
    PNG_FILE_4D = os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFile4D.png")
    PNG_SAMPLE_2D = [os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleHxW_1.png"),
                     os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleHxW_2.png")]
    PNG_SAMPLE_3D = [os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleCxHxW_1.png"),
                     os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleCxHxW_2.png")]
    PNG_SAMPLE_4D = [os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleCxDxHxW_1.png"),
                     os.path.join(OUTPUT_DATA_FOLDER_PATH, "ToPNGFileTupleCxDxHxW_2.png")]
    ALL = [PNG_FILE_2D, PNG_FILE_3D, PNG_FILE_4D, PNG_SAMPLE_2D, PNG_SAMPLE_3D, PNG_SAMPLE_4D]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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
        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_should_raise_exception_when_passing_4D_ndarray(self):
        nd_array = np.random.randint(0, 1000, size=(1, 32, 32, 32))

        transform_ = ToPNGFile(self.PNG_FILE_4D)

        assert_that(calling(transform_.__call__).with_args(nd_array), raises(TypeError))

    def test_should_raise_exception_when_passing_Sample_of_4D_ndarray(self):
        x = np.random.randint(0, 1000, size=(1, 32, 32, 32))
        y = np.random.randint(0, 1000, size=(1, 32, 32, 32))

        sample = Sample(x=x, y=y, is_labeled=True)

        transform_ = ToPNGFile(self.PNG_SAMPLE_4D)

        assert_that(calling(transform_.__call__).with_args(sample), raises(TypeError))

    def test_should_pass_when_passing_2D_ndarray(self):
        nd_array = np.random.randint(0, 1000, size=(32, 32))

        ToPNGFile(self.PNG_FILE_2D).__call__(nd_array)

        assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, self.PNG_FILE_2D)))

    def test_should_pass_when_passing_3D_ndarray(self):
        nd_array = np.random.randint(0, 1000, size=(1, 32, 32))

        ToPNGFile(self.PNG_FILE_3D).__call__(nd_array)

        assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, self.PNG_FILE_3D)))

    def test_should_pass_when_passing_sample_2D_nd_array(self):

        x = np.random.randint(0, 1000, size=(32, 32))
        y = np.random.randint(0, 1000, size=(32, 32))

        sample = Sample(x=x, y=y, is_labeled=True)

        ToPNGFile(self.PNG_SAMPLE_2D).__call__(sample)

        for file_name in self.PNG_SAMPLE_2D:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))

    def test_should_pass_when_passing_tuple_3D_nd_array(self):

        x = np.random.randint(0, 1000, size=(1, 32, 32))
        y = np.random.randint(0, 1000, size=(1, 32, 32))

        sample = Sample(x=x, y=y, is_labeled=True)

        ToPNGFile(self.PNG_SAMPLE_3D).__call__(sample)

        for file_name in self.PNG_SAMPLE_3D:
            assert_that(os.path.exists(os.path.join(self.OUTPUT_DATA_FOLDER_PATH, file_name)))


class CropToContentTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/CropToContent")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1.nii")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    CROPPED_NIFTI_IMAGE_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_image_sample.nii")
    CROPPED_NIFTI_LABELS_FROM_SAMPLE = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_labels_sample.nii")
    CROPPED_NIFTI_IMAGE_FROM_NUMPY_NDARRAY = os.path.join(OUTPUT_DATA_FOLDER_PATH, "cropped_image_ndarray.nii")
    ALL = [CROPPED_NIFTI_IMAGE_FROM_NUMPY_NDARRAY, CROPPED_NIFTI_IMAGE_FROM_SAMPLE, CROPPED_NIFTI_LABELS_FROM_SAMPLE]

    @classmethod
    def _flatten(cls, l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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
        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_should_produce_cropped_to_content_from_sample(self):
        image = nib.load(self.VALID_3D_NIFTI_FILE)
        label = nib.load(self.VALID_MASK_FILE)

        image_data = np.expand_dims(image.get_data().__array__(), axis=-1).transpose((3, 2, 1, 0))
        label_data = np.expand_dims(label.get_data().__array__(), axis=-1).transpose((3, 2, 1, 0))

        sample = Sample(x=image_data, y=label_data, is_labeled=True)

        cropped_sample = CropToContent().__call__(sample)

        assert_that(cropped_sample.x.ndim, equal_to(image_data.ndim))
        assert_that(cropped_sample.x.shape, less_than_or_equal_to(sample.x.shape))

        nifti_image = nib.Nifti1Image(cropped_sample.x.transpose((3, 2, 1, 0)), None, image.header)
        nib.save(nifti_image, self.CROPPED_NIFTI_IMAGE_FROM_SAMPLE)
        nifti_image = nib.Nifti1Image(cropped_sample.y.transpose((3, 2, 1, 0)), None, label.header)
        nib.save(nifti_image, self.CROPPED_NIFTI_LABELS_FROM_SAMPLE)

    def test_should_produce_cropped_to_content_from_ndarray(self):
        image = nib.load(self.VALID_3D_NIFTI_FILE)
        image_data = image.get_data().__array__()
        image_data = np.expand_dims(image_data, axis=-1).transpose((3, 2, 1, 0))
        cropped = CropToContent().__call__(image_data)
        assert_that(cropped.ndim, equal_to(image_data.ndim))
        assert_that(cropped.shape, less_than_or_equal_to(image_data.shape))

        image = nib.Nifti1Image(cropped.transpose((3, 2, 1, 0)), None, image.header)
        nib.save(image, self.CROPPED_NIFTI_IMAGE_FROM_NUMPY_NDARRAY)


class ToTensorPatchesTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/ToTensorPatches")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1_1mm.nii")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    PATCH_SIZE = (1, 96, 128, 128)
    WRONG_PATCH_SIZE = (1, 256, 384, 384)
    STEP = (1, 96, 128, 128)

    def setUp(self) -> None:
        pass

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_should_slice_image(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE)

        transform_ = transforms.Compose([ToNumpyArray(),
                                         ToNDTensor(),
                                         ToTensorPatches(self.PATCH_SIZE, self.STEP)])

        transformed_sample = transform_.__call__(sample)

        assert_that(transformed_sample.x.ndimension(), is_(5))

    def test_should_produce_patches_from_tensor(self):
        transform_ = transforms.Compose([ToNumpyArray()])

        x = torch.from_numpy(transform_.__call__(self.VALID_3D_NIFTI_FILE))

        transformed_x = ToTensorPatches(self.PATCH_SIZE, self.STEP).__call__(x)

        assert_that(transformed_x.ndimension(), is_(5))

    def test_should_raise_value_error_with_patch_too_big(self):
        transform_ = transforms.Compose([ToNumpyArray()])

        x = torch.from_numpy(transform_.__call__(self.VALID_3D_NIFTI_FILE))

        transform_ = ToTensorPatches(self.WRONG_PATCH_SIZE, self.STEP)

        assert_that(calling(transform_).with_args(x), raises(ValueError))

    def test_should_produce_patches_from_sample(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE)

        transform_ = transforms.Compose([ToNumpyArray(),
                                         ToNDTensor(),
                                         ToTensorPatches(self.PATCH_SIZE, self.STEP)])

        transformed_sample = transform_.__call__(sample)

        for i in range(transformed_sample.x.shape[0]):
            img = nib.Nifti1Image(transformed_sample.x[i, 0].numpy(), None)
            nib.save(img, os.path.join(self.OUTPUT_DATA_FOLDER_PATH, "patch_{}".format(i)))


class ToNDArrayPatchesTest(unittest.TestCase):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/generated/ToNDArrayPatches")
    VALID_3D_NIFTI_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "T1_1mm.nii")
    VALID_MASK_FILE = os.path.join(TEST_DATA_FOLDER_PATH, "Mask.nii")
    PATCH_SIZE = (1, 96, 128, 128)
    WRONG_PATCH_SIZE = (1, 256, 384, 384)
    STEP = (1, 96, 128, 128)

    def setUp(self) -> None:
        pass

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def test_should_slice_image(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE)

        transform_ = transforms.Compose([ToNumpyArray(),
                                         ToNDArrayPatches(self.PATCH_SIZE, self.STEP)])

        transformed_sample = transform_.__call__(sample)

        assert_that(transformed_sample.x.ndim, is_(5))

    def test_should_produce_patches_from_ndarray(self):
        transform_ = transforms.Compose([ToNumpyArray(),
                                         ToNDArrayPatches(self.PATCH_SIZE, self.STEP)])

        transformed_x = transform_(self.VALID_3D_NIFTI_FILE)

        assert_that(transformed_x.ndim, is_(5))

    def test_should_raise_value_error_with_patch_too_big(self):
        transform_ = transforms.Compose([ToNumpyArray()])

        x = transform_.__call__(self.VALID_3D_NIFTI_FILE)

        transform_ = ToNDArrayPatches(self.WRONG_PATCH_SIZE, self.STEP)

        assert_that(calling(transform_).with_args(x), raises(ValueError))

    def test_should_produce_patches_from_sample(self):
        sample = Sample(x=self.VALID_3D_NIFTI_FILE)

        transform_ = transforms.Compose([ToNumpyArray(),
                                         ToNDArrayPatches(self.PATCH_SIZE, self.STEP)])

        transformed_sample = transform_.__call__(sample)

        for i in range(transformed_sample.x.shape[0]):
            img = nib.Nifti1Image(transformed_sample.x[i, 0], None)
            nib.save(img, os.path.join(self.OUTPUT_DATA_FOLDER_PATH, "patch_{}".format(i)))
