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
        -  [Trainers](#trainers)
        -  [Transformers](#transformers)
        -  [Logger](#logger)
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

#### Diffusion imaging

#### Application


## Setup
> pip install -r [path/to/requirements.txt]  
> python3 <main_script>.py


## Project architecture
### Folder structure

```
├── configs                 - This folder contains the YAML configuration files.
│   ├── train.yaml                  - This file contains your training configuration. MUST be a YAML file.
│   └── test.yaml                   - OPTIONAL. This file contains the testing configuration. MUST be a YAML file.
|
├── docker                  - Contains Dockerfile needed to provide a functional Docker environment for your publication.
|   └── dockerfile
|
├── icons                   - Contains project's artwork.
|
├── initializers            - This folder contains custom layer/op initializers.  
|   └── base_initializer.py
│
├── inputs                  - This folder contains anything relative to inputs to a network.
|   └── transformers.py  
|
├── metrics                  - This folder contains various metrics used to measure a training session of a model.
|   ├── gauges.py 
|   └── metrics.py
|   
├── models                  - This folder contains any standard model.
│   └── base_model.py                      
|
├── preprocessing           - This folder contains anything relative to input preprocessing, and scripts that must be executed prior training.
|
├── tests                   - Folder containing unit tests of the standard framework api and functions.
|   
├── training                - This folder contains trainers.
│   ├── base_trainer.py 
|   ├── losses.py  
|   └── trainer.py
│  
└── utils                   - This folder contains any utils you need.
     └── utils.py
```

### Main components
#### Models

#### Trainers

#### Transformers

#### Logger

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

## Acknowledgment
Thanks to [École de technologie supérieure](https://www.etsmtl.ca/), [Hervé Lombaert](https://profs.etsmtl.ca/hlombaert/) and [Christian Desrosiers](https://www.etsmtl.ca/Professeurs/cdesrosiers/Accueil) for providing us a lab and helping us in our research activities.

Icons made by <a href="http://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>
