import pathlib

from setuptools import find_packages, setup

# The text of the README file
README_CONTENT = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name='SAMITorch',
    version='0.1.32',
    description='Deep Learning Framework For Medical Image Analysis',
    long_description=README_CONTENT,
    long_description_content_type='text/markdown',
    author='Benoit Anctil-Robitaille and Pierre-Luc Delisle',
    author_email='benoit.anctil-robitaille.1@ens.etsmtl.ca',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests",)),
    install_requires=['numpy>=1.16.1',
                      'torch>=1.1.0',
                      'torchvision>=0.2.2',
                      'nibabel>=2.4.0',
                      'nipype>=1.1.9',
                      'nilearn>=0.5.0',
                      'scikit-learn>=0.20.3',
                      'pytorch-ignite>=0.2.0',
                      'pynrrd>=0.4.0',
                      'opencv-python>=4.1.0']
)
