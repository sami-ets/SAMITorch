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

from enum import Enum


class ImageType(Enum):
    NIFTI = 'nifti'
    NRRD = 'nrrd'
    ALL = [NIFTI, NRRD]

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member

    def __str__(self):
        return self.value


class Extension(Enum):
    NIFTI = ".nii"
    NRRD = ".nrrd"

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member

    def __str__(self):
        return self.value


class Modality(Enum):
    DTI = 'DTI'
    T1 = 'T1'
    T2 = 'T2'
    MD = 'MD'
    FA = 'FA'
    ALL = [DTI, T1, T2, MD, FA]

    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.name == name:
                return member

    def __str__(self):
        return self.value


class Image(object):

    @staticmethod
    def is_processed_dti(file):
        return Image.is_dti(file) and Image.is_processed(file)

    @staticmethod
    def is_unprocessed_dti(file):
        return Image.is_dti(file) and not Image.is_processed(file)

    @staticmethod
    def is_unprocessed_fa(file):
        return Image.is_fa(file) and not Image.is_processed(file)

    @staticmethod
    def is_unprocessed_md(file):
        return Image.is_md(file) and not Image.is_processed(file)

    @staticmethod
    def is_processed_t1(file):
        return Image.is_t1(file) and Image.is_processed(file)

    @staticmethod
    def is_unprocessed_t1(file):
        return Image.is_t1(file) and not Image.is_processed(file)

    @staticmethod
    def is_nifti_t1_mask(file):
        return Image.is_nifti(file) and Image.is_t1(file) and 'mask' in file

    @staticmethod
    def is_nifti(file):
        return Extension.NIFTI.value in file

    @staticmethod
    def is_nrrd(file):
        return Extension.NRRD.value in file

    @staticmethod
    def is_t1(file):
        return Modality.T1 in file

    @staticmethod
    def is_dti(file):
        return Modality.DTI in file

    @staticmethod
    def is_fa(file):
        return Modality.FA in file

    @staticmethod
    def is_md(file):
        return Modality.MD in file

    @staticmethod
    def is_(modality, file):
        return modality in file and Image.is_nifti(file) and Image.is_unprocessed(file)

    @staticmethod
    def is_processed(file):
        return "Processed" in file

    @staticmethod
    def is_unprocessed(file):
        return not Image.is_processed(file) and (Image.is_nifti(file) or Image.is_nrrd(file))
