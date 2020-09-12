---
date: '2020-09-12T17:13:00.795Z'
docname: features_3d
images: {}
path: /features-3-d
title: pvtseg.features_3d module
---

# pvtseg.features_3d module

Created on Fri Jul 17 08:44:31 2020

@author: titou


### pvtseg.features_3d.BuildFeatureFiles(x, features=['Gau', 'Lap', 'HGE'], sigma=1.0, dir_path=WindowsPath('.'))
Build all feature files of an imput nd array for a given sigma
smoothing.


* **Parameters**

    
    * **x** (*numpy nd-array*) – The imput array on wich the features will be computed


    * **features** (*list*) – name of the features to be computed


    * **sigma** (*float*) – The gaussian standar deviation (smoothing coeficient)


    * **dir_path** (*str*) – path of the directory in wich the feature files are to be stored



### pvtseg.features_3d.BuildFeatureFilesFromList(image_file, features_list, dir_path)
Build all feature files in a targeted directory from an imput
features list.


* **Parameters**

    
    * **image_file** (*Path** or **str*) – Path to the immage 3d array on wich the features will be computed


    * **features_list** (*list*) – name of the features to be computed.
    Each feature must be writen in the following format,
    <<feature_type>>_sigma<<range>>, ex: Gau_sigma2.0 or Lap_sigma1.0.
    Only compatible features are computed, non compatible features
    are to be separatly computed and mannualy added into the features
    directory as numpy ndarray.

    > Compatible feature type:
    > - “Gau” for Gaussian filter,
    > - “Lap” for Laplacian filter,
    > - “HGE1”, “HGE2”, “HGE3” for 1st, 2nd and 3rd Hessian eigenvalues,



    * **dir_path** (*str*) – path of the directory in wich the feature files are to be stored



### pvtseg.features_3d.GaussianSmoothing(x, sigma, mask=None)
Compute n-dimentional gaussian smoothing on nd array.


* **Parameters**

    
    * **x** (*numpy nd-array*) – The imput array to be smoothed


    * **sigma** (*float*) – The gaussian standar deviation (smoothing coeficient)


    * **mask** (*tuple of n 1d array of size l*) – coordonate of l points one wich the smoothing is wanted.



* **Returns**

    
    * **gaussian_s** (*numpy nd-array*) – The nd array smoothed with a coeficient sigma


    * **gaussian_s[mask]** (*numpy 1d array*) – if mask is not None, return an array of the value of the gaussian
    smoothing for all points in mask




### pvtseg.features_3d.HessianEigenvalues(x, mask=None)
Compute n-dimentional hessian eigenvalues on nd array.


* **Parameters**

    
    * **x** (*numpy nd-array*) – The imput array on wich the hessian eigenvalues will be computed


    * **mask** (*tuple of n 1d array of size l*) – coordonate of l points one wich the hessian eigenvalues are wanted.



* **Returns**

    
    * **eigenvalues** (*numpy nd-array*) – The nd array with all n eigenvalues (shape as x shape + 1)


    * **eigenvalues[mask]** (*numpy 2d array*) – if mask is not None, return an array with the value of the hessian
    eigenvalues for all points in mask




### pvtseg.features_3d.Laplacian(x, mask=None)
Compute n-dimentional laplacian on nd array.


* **Parameters**

    
    * **x** (*numpy nd-array*) – The imput array on wich the laplacian will be computed


    * **mask** (*tuple of n 1d array of size l*) – coordonate of l points one wich the laplacian is wanted.



* **Returns**

    
    * **laplacian** (*numpy nd-array*) – The nd array laplacian (same shape as x)


    * **laplacian[mask]** (*numpy 1d array*) – if mask is not None, return an array with the value of the laplacian
    for all points in mask




### pvtseg.features_3d.LoadFeaturesDir(dir_path, mask, features_list=None)
Assuming only feature files are in a target directory, build a
dataframe with the value of each feature at each points of the mask


* **Parameters**

    
    * **dir_path** (*str*) – path of the directory in wich the feature files are stored


    * **mask** (*tuple of n 1d array of size l*) – coordonate of l points one wich the features are wanted.


    * **features_list** (*list**, **optional*) – names of the features files to be loded without the “.npy”.
    Each feature must a nd numpy array the same size as the mask.

    The default is None
    if None, all feature files in the directory will be loded




* **Returns**

    **df** – a n \* m dataframe where n  is the number of points in mask and m the
    number of features



* **Return type**

    pandas DataFrame



### pvtseg.features_3d.MultiFilesBuildFeatureFilesFromList(image_list, features_list, dir_path)
For a list of image files, build in the tageted folder a list of
sub_folder in wich for each image, all the features in the feature file
list are built


* **Parameters**

    
    * **image_list** (*list*) – list of image files for wich the features are to be computed


    * **features_list** (*list*) – name of the features to be computed.
    Each feature must be writen in the following format,
    <<feature_type>>_sigma<<range>>, ex: Gau_sigma2.0 or Lap_sigma1.0.
    Only compatible features are computed, non compatible features
    are to be separatly computed and mannualy added into the features
    directory as numpy ndarray.

    > Compatible feature type:
    > - “Gau” for Gaussian filter,
    > - “Lap” for Laplacian filter,
    > - “HGE1”, “HGE2”, “HGE3” for 1st, 2nd and 3rd Hessian eigenvalues,



    * **dir_path** (*str** or **Path*) – path of the directory in wich the subfolder for each image feature files
    are to be stored
