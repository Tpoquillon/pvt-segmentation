---
date: '2020-09-12T17:13:00.795Z'
docname: vessel_seg
images: {}
path: /vessel-seg
title: pvtseg.vessel_seg module
---

# pvtseg.vessel_seg module


### pvtseg.vessel_seg.Batch_Processing(model, raw_file, mask_file=None, threshold=None, feature_dir=WindowsPath('featur_provisional'), clean=True)

* **Parameters**

    
    * **model** (*dict*) – Output of BuildModel methode, a dict with:

        -“model” an sklearn trained random forest classifier.
        -“features” the list of the model features.



    * **raw_file** (*Path** or **str*) – path to the raw data (numpy nd array of voxel intensities).


    * **mask_file** (*Path** or **str**, **optional*) – path to the mask file (numpy binary nd array of 1 and 0).
    The default is None.


    * **threshold** (*float**, **optional:*) – A float with values between 0 and 1. If None, Batch_Processing
    return a prediction array with for each voxel its probability to be
    a vessel. Else, return a binary (vessel: 1, non_vessel: 0)
    segmented array for the given threshold. The default is None


    * **feature_dir** (*Path** or **str**, **optional*) – directory in wich feature files are stored. If the directory already
    exist, Batch Processing will load the features inside. Else, the
    methode will build the dir ansd the feature files. At the end of the batch
    processing, this dire and all files inside are removed if clean option
    is True. The default is Path(“featur_provisional”).


    * **clean** (*boolean**, **optional*) – if True, the feature_dir is removed at the end of the
    Batch_Precessing. The default is True.



* **Returns**

    **results** – a prediction array or a binary segmentation array.



* **Return type**

    numpy ndarray



### pvtseg.vessel_seg.BuildDataFrame(path_featur_dir, annotations_file, points_set=None, feature_list=None)
Build a dataframe with, for each point, a row with features and label. If points
set is None, uses all annotations to build the dataframe, else, uses only points
in the points set.


* **Parameters**

    
    * **path_featur_dir** (*str** or **Path*) – Path to featured_dir, in wich features as been computed.


    * **annotations_file** (*str** or **Path*) – Path to annotation file.


    * **points_set** (*list**, **optional*) – List of point sets. Result of DataPreparation methode. if None uses all annotated
    points as a single set. The default is None


    * **feature_list** (*list**, **optional*) – If None, all files in the featur_dir are loaded as features, else
    only files in feature list are loaded.
    The default is None



* **Returns**

    **data** – dataframe with, for each point, a row with features and label.



* **Return type**

    dataframe



### pvtseg.vessel_seg.BuildDirectoryTree(path='My_data')
Build the directory structure for the segmentation project.
:param path: path to the project root
:type path: string, optional


* **Returns**

    
    * **path_s** (*Path*) – Path to the source directory


    * **path_f** (*Path*) – Path to the features directory


    * **path_r** (*Path*) – Path to the result directory




### pvtseg.vessel_seg.BuildModel(data, hyperparameters=None, shuffle=False)
From a dataframes, build a segmentation model


* **Parameters**

    
    * **data** (*dataframe*) – Dataframe with for each point, a row with features and label.


    * **hyperparameters** (*dict*) – Sklearn random forest classifier parameters. The default is None.

        If None, those parameter are going to be used:
        - “n_estimators”: 100,
        - “criterion”:’gini’,
        - “max_depth”:  None,
        - “min_samples_split”: 2,
        - “min_samples_leaf”: 1,
        - “max_features”: ‘auto’



    * **cv** (*int**, **optional*) – Number of folds for the cross-validation. The default is 5


    * **shuffle** (*boolean**, **optional*) – If the dataframe has to be shuffuled. The default is False



* **Returns**

    **model** –

    Output of BuildModel methode.

        A dictionary with:
        - “model” an sklearn trained random forest classifier.
        - “features” the list of the model features.




* **Return type**

    dict



### pvtseg.vessel_seg.CrossValidation(data, hyperparameters=None, cv=5, shuffle=False)
From a dataframes, performe n-fold cross validation (default is 5)


* **Parameters**

    
    * **data** (*dataframe*) – dataframe with, for each point, a row with features and label.


    * **hyperparameters** (*dict*) – best param to optimise forest.


    * **cv** (*int**, **optional*) – number of folds for the cross-validation. The default is 5


    * **shuffle** (*boolean**, **optional*) – if the dataframe has to be shuffuled.



* **Returns**

    **results_dict** – a dictionary with results all cross validation experiments,
    to be treated with Summarise.



* **Return type**

    dict



### pvtseg.vessel_seg.DFToXY(dataframe)
Divide a dataframe producted by BuildDataFrames methode between features
and labels


* **Parameters**

    **dataframe** (*pandas DataFrame*) – dataframe with features and labels



* **Returns**

    
    * **x** (*pandas DataFrame*) – dataframe of features.


    * **y** (*pandas Series*) – labels




### pvtseg.vessel_seg.FeatureSelection(data, hyperparameters=None, cv=5, shuffle=False, worst_features_set_size=4)
A methode that evaluate  sets of features for the random
forest classifier, by remooving worst features one by one.


* **Parameters**

    
    * **data** (*dataframe*) – dataframe with, for each point, a row with features and label.


    * **hyperparameters** (*dict*) – Optimised parameters for random forest.


    * **cv** (*int**, **optional*) – the nuber of folds for the cross validation. The default is 5


    * **shuffle** (*boolean**, **optional*) – if True, rows of data are shuffuled during the cross validation


    * **worst_features_set_size** (*int**, **optional*) – the number of worst feature to be tested before remooving
    The default is 4



* **Returns**

    a dictionary of metrics showing their evolution with the
    number of features and the features ranked by increassing relevance for
    the model.



* **Return type**

    dict



### pvtseg.vessel_seg.LoadModel(filname)
Load a model using pickle


* **Parameters**

    **filname** (*str*) – model file name.



* **Returns**

    **model** –

    Output of BuildModel methode, a dict with:

        -“model” an sklearn trained random forest classifier.
        -“features” the list of the model features.




* **Return type**

    dict



### pvtseg.vessel_seg.MultiFilesBuildDataFrame(path_featur_dir_list, annotations_file_list, points_set_list=None, feature_list=None)
Build a dataframe with, for each point, a row with features and label. If points
sets is None, uses all annotations to build the dataframe, else, uses only points
in points sets.


* **Parameters**

    
    * **path_featur_dir_list** (*list*) – List of path to feature folder, in wich features as been computed for
    a 3d image.


    * **annotations_file_list** (*str** or **Path*) – list of path to annotation files for each image.


    * **point_set** (*list**, **optional*) – List of point sets. Result of MultiFilesDataPreparation methode.
    if None uses all annotated points as a single set for each image.
    The default is None


    * **feature_list** (*list**, **optional*) – If None, feature list is set on all the feature names in the first
    feature folder, else only files in feature list are loaded from each
    feature files.
    The default is None



* **Returns**

    **data** – dataframe with, for each point, a row with features and label.



* **Return type**

    dataframe



### pvtseg.vessel_seg.MultiFilesPointSelection(annotations_file_list, mask_file_list='auto', box_file_list_list='auto', vessel_prop=0.5, points=2000)
Performe the PointsSelection methode one multiple files.


* **Parameters**

    
    * **annotations_file_list** (*list*) – list of paths to the annotation files:  3d array where points
    from each class i have a value of i and point without class 0.
    The first class has to be the vessels annotations
    (or the object of interest to be segmented).


    * **mask_file_list** (*list**, **optional*) – list of paths to the mask files. the default is “auto” for a list of
    mask, each covering all the image. The default is “auto”.


    * **box_file_list_list** (*list**, **optional*) – list of list of path to boxe files. each box file is a mask on an area
    for one image. For one image, with a list of box files,
    the point selection will be balanced between all boxes of the list
    If auta, the point selection won’t be balanced.
    The default is “auto”.


    * **vessel_prop** (*float**, **optional*) – proportion of vessel (class 1) in the . The default is 0.5.


    * **point_num** (*int**, **optional*) – number of points to be selected in each annotation file



* **Returns**

    **points_set_list** – A list of tuple of 3 numpy 1d array, with indices of selected points.
    to be passed to the MultiFileBuildDataFrame methode.



* **Return type**

    list



### pvtseg.vessel_seg.Optimise(data, optimiser=None)
Find the best hyperparameters for a random forest classifier model through
an sklearn optimiser.


* **Parameters**

    
    * **cross_validation_dataframes** (*Dataframe*) – the cross_validation set


    * **optimiser** (*optimiser object**, **default = None*) – a RandomizedSearchCV or GridSearchCV object to find best
    hyperparameters. if None, a default
    optimiser will be constructed instead
    the default is None



* **Returns**

    **optimiser** – the optimiser fited to cross validations datas.



* **Return type**

    optimiser object



### pvtseg.vessel_seg.PointsSelection(annotations_file, mask_file='auto', box_file_list='auto', vessel_prop=0.5, points=2000)
Build a set of points that can be balanced between classes and areas
to be used for training model.
The result of this methode goes in the BuildDataFrame points_set option


* **Parameters**

    
    * **annotations_file** (*str** or **Path*) – Paths to the annotation file:  3d array where points
    from each class i have a value of i and point without class 0.
    The first class has to be the vessels annotations
    (or the object of interest to be segmented).


    * **mask_file** (*str**, **optional*) – path to the mask file. the default is “auto” for a mask covering all the image


    * **box_file_list** (*list**, **otional*) – list of path to boxe_file, boxe in wich annotations have been performed.
    the default is “auto” for a box covering all the image


    * **vessel_prop** (*int**, **optional*) – Vessel proportions.
    the default is 0.5


    * **points_set** (*int**, **optional*) – total number of selected points.



* **Returns**

    **points_set** – A tuple of 3 numpy 1d array, with indices of selected points.
    to be passed to the BuildDataFrame methode.



* **Return type**

    tuple



### pvtseg.vessel_seg.Prediction(model, mask, path_feature_dir)
A methode to product a segmentation on an entire image with a model from cross     validations results.


* **Parameters**

    
    * **model** (*dict*) – Output of BuildModel methode, a dict with:

        -“model” an sklearn trained random forest classifier.
        -“features” the list of the model features.



    * **mask** (*numpy nd_array*) – Image binary mask


    * **path_feature_dir** (*Path*) – Path to the directory where features have been pre-computed



* **Returns**

    a 3d image with the probability for each voxel to belong to the first class (to be a
    vessel)



* **Return type**

    probability_array, numpy nd_array



### pvtseg.vessel_seg.SaveModel(model, filname)
Save a model using pickle


* **Parameters**

    
    * **model** (*dict*) – Output of BuildModel methode, a dict with:

        -“model” an sklearn trained random forest classifier.
        -“features” the list of the model features.



    * **filname** (*str*) – model file name.



* **Returns**

    


* **Return type**

    None.



### pvtseg.vessel_seg.Segmentations(model, mask, path_feature_dir, threshold=0.5)
A methode to product a segmentation on an entire image with a model from cross     validations results.


* **Parameters**

    
    * **model** (*dict*) – Output of BuildModel methode, a dict with:

        -“model” an sklearn trained random forest classifier.
        -“features” the list of the model features.



    * **mask** (*numpy nd_array*) – Image binary mask


    * **path_feature_dir** (*Path*) – Path to the directory where features have been pre-computed


    * **model_index** (*int**, **optional*) – Index of the model to be used from the cross validation results.
    The default is 0



* **Returns**

    a 3d image with for each voxel 1 if it belongs to the first class (
    vessel), else 0



* **Return type**

    segmentation_array, numpy nd_array



### pvtseg.vessel_seg.ShowShap(model, data, points=None)

* **Parameters**

    
    * **model** (*dict*) – Output of BuildModel methode, a dict with:

        -“model” an sklearn trained random forest classifier.
        -“features” the list of the model features.



    * **data** (*dataframe*) – dataframe with, for each point, a row with features and label.


    * **points** (*int**, **optional*) – Number of points on wich calcule the shape_values
    if None, all points are taken (could take a lot of time).
    The default is None.



* **Returns**

    


* **Return type**

    None.



### pvtseg.vessel_seg.Summarise(result_dict, display=False)
From the results of the CrossValiation methode, build an human readable
summary


* **Parameters**

    
    * **result_dict** (*dict*) – the dict from cross_validation data.


    * **display** (*bool**, **optional*) – if true, display the ROC curv of each test
    the default is False



* **Returns**

    **nice_dataframes** – dict of human readable and interpretable dataframes, to analyse cross
    validation results



* **Return type**

    dict
