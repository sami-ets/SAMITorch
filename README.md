# <img src="/icons/artificial-intelligence.png" width="60" vertical-align="bottom"> SAMITorch

## Welcome to SAMITorch

[![Build Status](https://travis-ci.com/sami-ets/SAMITorch.svg?branch=master)](https://travis-ci.com/sami-ets/SAMITorch)
![GitHub All Releases](https://img.shields.io/github/downloads/sami-ets/SAMITorch/total.svg)
![GitHub issues](https://img.shields.io/github/issues/sami-ets/SAMITorch.svg)
![GitHub](https://img.shields.io/github/license/sami-ets/SAMITorch.svg)
![GitHub contributors](https://img.shields.io/github/contributors/sami-ets/SAMITorch.svg)


SAMITorch is a deep learning framework for *Shape Analysis in Medical Imaging* laboratory of [École de technologie supérieure](https://www.etsmtl.ca/) using [PyTorch](https://github.com/pytorch) library.
It implements an extensive set of loaders, transformers, models and data sets suited for deep learning in medical imaging.
Our objective is to build a tested, standard framework for quickly producing results in deep learning reasearch applied to medical imaging. 

# Table Of Contents

-  [Authors](#authors)
-  [References](#references)
-  [Project architecture](#project-architecture)
    -  [Folder structure](#folder-structure)
    -  [Main Components](#main-components)
        -  [Models](#models)
        -  [Transformers](#transformers)
        -  [Configuration](#configs)
        -  [Main](#main)
 -  [Contributing](#contributing)
 -  [Branch naming](#branch-naming)
 -  [Commits syntax](#commits-syntax)
 -  [Acknowledgments](#acknowledgments)
 
 
## Authors

* Pierre-Luc Delisle - [pldelisle](https://github.com/pldelisle) 
* Benoit Anctil-Robitaille - [banctilrobitaille](https://github.com/banctilrobitaille)

## References

#### Segmentation
```
@article{RN10,
   author = {Çiçek, Özgün and Abdulkadir, Ahmed and Lienkamp, Soeren S. and Brox, Thomas and Ronneberger, Olaf},
   title = {3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation},
   journal = {eprint arXiv:1606.06650},
   pages = {arXiv:1606.06650},
   url = {https://ui.adsabs.harvard.edu/\#abs/2016arXiv160606650C},
   year = {2016},
   type = {Journal Article}
}
```

#### Classification
```
@inproceedings{RN12,
   author = {He, K. and Zhang, X. and Ren, S. and Sun, J.},
   title = {Deep Residual Learning for Image Recognition},
   booktitle = {2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   pages = {770-778},
   ISBN = {1063-6919},
   DOI = {10.1109/CVPR.2016.90},
   type = {Conference Proceedings}
}
```

#### Diffusion imaging

#### Application


## Setup
> pip install -r [path/to/requirements.txt]  
> python3 <main_script>.py


## Project architecture
### Folder structure

```
── samitorch
|    ├── configs                 - This folder contains the YAML configuration files.
|    │   ├── configurations.py       - This file contains the definitions of different configuration classes.
|    │   |── resnet3d.yaml           - Standard ResNet 3D configuration file and model definition.
|    │   └── unet3d.yaml             - Standard UNet 3D configuration file and model definition.
|    |
|    ├── initializers            - This folder contains custom layer/op initializers.  
|    |   └── initializers.py
|    │
|    ├── inputs                  - This folder contains anything relative to inputs to a network.
|    |   |── batch.py                - Contains Batch definition object used in training. 
|    |   |── datasets.py             - Contains basic dataset definition for classification and segmentation.
|    |   |── images.py               - Contains Enums for various methods.
|    |   |── patch.py                - Contains Patch definition used in segmentation problems.
|    |   |── sample.py               - Contains a Sample object.
|    |   |── transformers.py         - Contains a series of common transformations.
|    |   └── utils.py                - Contains various utilitary methods.
|    |   
|    ├── models                  - This folder contains any standard and tested deep learning models.
|    │   |── layers.py               - Contains layer definitions. 
|    |   |── resnet3d.py             - Contains a standard ResNet 3D model.
|    |   └── unet3d.py               - Contains a standard UNet 3D model.                   
|    |
|    |── parsers                 - This folder contains parsers definition used in SAMITorch.
|    |
|    ├── preprocessing           - This folder contains anything relative to input preprocessing, and scripts that must be executed prior training.
|    |
|    └── utils                   - This folder contains any utils you may need.
|         |── files.py              - Contains file related utils methods.
|         |── slice_builder.py      - Contains an object to build slices out of a data sets (for image segmentation).
|         └── tensors.py            - Contains tensor related utils methods.            
── tests                   - Folder containing unit tests of the standard framework api and functions.

```

### Main components
(To be documented shortly...)
#### Models

#### Transformers

#### Configs

#### Main

## Contributing
If you find a bug or have an idea for an improvement, please first have a look at our [contribution guideline](https://github.com/sami-ets/SAMITorch/blob/master/CONTRIBUTING.md). Then,
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

## Branch naming

| Instance        | Branch                                              | Description, Instructions, Notes                   |
|-----------------|-----------------------------------------------------|----------------------------------------------------|
| Stable          | stable                                              | Accepts merges from Development and Hotfixes       |
| Development     | dev/ [Short description] [Issue number]             | Accepts merges from Features / Issues and Hotfixes |
| Features/Issues | feature/ [Short feature description] [Issue number] | Always branch off HEAD or dev/                     |
| Hotfix          | fix/ [Short feature description] [Issue number]     | Always branch off Stable                           |

## Commits syntax

##### Adding code:
> \+ Added [Short Description] [Issue Number]

##### Deleting code:
> \- Deleted [Short Description] [Issue Number]

##### Modifying code:
> \* Changed [Short Description] [Issue Number]

##### Merging branches:
> Y Merged [Short Description]

## To build documentation

SAMITorch uses Sphinx Documentation. To build doc, simply execute the following: 

> cd docs  
> sphinx-build -b html source build  


## Acknowledgment
Thanks to [École de technologie supérieure](https://www.etsmtl.ca/), [Hervé Lombaert](https://profs.etsmtl.ca/hlombaert/) and [Christian Desrosiers](https://www.etsmtl.ca/Professeurs/cdesrosiers/Accueil) for providing us a lab and helping us in our research activities.

Icons made by <a href="http://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>
