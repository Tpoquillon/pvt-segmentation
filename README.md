# PVT Vesselness
 *PVT (Pulmonary Vascular Tree) Vesselness is a python package for pulmonary vascular tree segmentation. It allows users to build, evaluate and optimise a segmentation model based on random forest. This package was developed for covid patients with lung injuries.* (see french [internship report](/assets/text/Stage_Titouan_.pdf))

<div style="text-align: center">
<img src="/assets/images/pvt_couv.png" alt="front image" width="500"/>
</div>


## Table of Contents
1. [Instalation and Usage](#instalation-and-usage)
3. [Data Format](#data-format)
4. [Short Tutorial](#short-tutorial)


## Instalation and Usage

### Install
the pvtvesselness package require **python 3.7** or any later vesion.
For now, one can only install **pvtvesselness** package through wheel files:

*   From the [dist directory](/dist) download the .whl file with the desired version (for exemple: *pvtvesselness-1.0-py3-none-any.whl* ),
*   In the the downloaded file directory, install the package with [pip](https://pypi.org/project/pip/), 

        python -m pip install pvtvesselness-1.0-py3-none-any.whl

*   or

        python3 -m pip install pvtvesselness-1.0-py3-none-any.whl


The install may not work on linux without the freeglut package:

        sudo apt-get install freeglut3
   
### Usage

The **pvtvesselness** package is composed of 4 modules:
-  *vesselness* (the package main module), to build sgmentations model and performe cross-validations, optimisations and segmentations,
-  *data_preparation*, to build balanced data,
-  *evaluation* , to evaluate models segmentation compared to ground truth,
-  *features_3d*, to build and manage features needed for segmentation.
 


## Data Format

This library does not directly use a standard data format. All image file are stored as simple [numpy ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) matrix, as it is a verry basic and easy to process format. A module to convert anny kind of classical 3d immage format into nd array will be added soon. 



## Short Tutorial
Here is a quick tutorial to learn how to use the PVT Vesselness library for blood vessel segmentation. Yon can also directly load the [jupyter notebook](Tutorial.ipynb) file for the tutorial: [Totorial1](Tutorial1.ipynb).