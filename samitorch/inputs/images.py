class ImageType(object):
    NIFTI = 'nifti'
    NRRD = 'nrrd'
    ALL = [NIFTI, NRRD]

    EXTENSION = {
        NIFTI: '.nii',
        NRRD: '.nrrd'
    }


class Modality(object):
    DTI = 'DTI'
    T1 = 'T1'
    T2 = 'T2'
    MD = 'MD'
    FA = 'FA'
    ALL = [DTI, T1, T2, MD, FA]


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
        return ImageType.EXTENSION[ImageType.NIFTI] in file

    @staticmethod
    def is_nrrd(file):
        return ImageType.EXTENSION[ImageType.NRRD] in file

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
    def is_processed(file):
        return "Processed" in file
