0
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:47:37 2020

@author: titou
"""
import numpy as np
import pandas as pd
from pathlib import Path
from pvtseg import features_3d as f3d
from pvtseg import data_preparation as dtp
from pvtseg import evaluation as ev
import os
import pickle
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

def BuildDirectoryTree(path="My_data"):
    """ Build the directory structure for the segmentation project.
    Parameters
    ----------
    path: string, optional
        path to the project root

    Returns
    -------
    path_s: Path 
        Path to the source directory
    path_f: Path 
        Path to the features directory
    path_r: Path 
        Path to the result directory
    """
    if not(os.path.exists(path)):
        Path(path).mkdir()
    path_s = Path(path) / Path('Sources')
    path_f = Path(path) / Path('Features')
    path_r = Path(path) / Path('Results')
    
    if not(os.path.exists(path_s)):
        path_s.mkdir()
    if not(os.path.exists(path_f)):
        path_f.mkdir()
    if not(os.path.exists(path_r)):
        path_r.mkdir()

    return path_s, path_f, path_r




def PointsSelection(annotations_file,
                     mask_file = "auto",
                     box_file_list = "auto",
                     vessel_prop = 0.5,
                     points = 2000,
                     ):
    """
    Build a set of points that can be balanced between classes and areas
    to be used for training model.
    The result of this methode goes in the BuildDataFrame points_set option 

    Parameters
    ----------
    annotations_file : str or Path
        Paths to the annotation file:  3d array where points
        from each class i have a value of i and point without class 0.
        The first class has to be the vessels annotations
        (or the object of interest to be segmented).  
    mask_file : str, optional
        path to the mask file. the default is "auto" for a mask covering all the image
    box_file_list  : list, otional
        list of path to boxe_file, boxe in wich annotations have been performed.
	the default is "auto" for a box covering all the image
    vessel_prop : int, optional
        Vessel proportions.
        the default is 0.5
    points_set : int, optional
        total number of selected points.

    Returns
    -------
    points_set: tuple
        A tuple of 3 numpy 1d array, with indices of selected points.
        to be passed to the BuildDataFrame methode.
    """
    annotations = np.load(annotations_file)
    
    if mask_file == "auto":
        mask = np.ones(annotations.shape)
    else:
        mask = np.load(mask_file)

    if box_file_list == "auto":
        box_list = [np.ones(annotations.shape)]
    else:
        box_list = list(map(np.load, box_file_list))
    
    # balancing and splitting datas for different values of vessel proportion
    splited_datas  = {}
    for i, box in enumerate(box_list):
        annot = annotations*box*mask
        seeds = dtp.Seeds(annot)
        balanced = dtp.Balance(seeds, vessel_prop, 
                                         points/len(box_list))
        splited_datas["box: " + str(i)] = balanced
            
    # merging
    
    
    points_set = dtp.Merge([
            splited_datas["box: " + str(i)]
            for i in range(len(box_list))  
        ])
    
    

    return points_set

def MultiFilesPointSelection(annotations_file_list,
                            mask_file_list = "auto",
                            box_file_list_list = "auto",
                            vessel_prop = 0.5,
                            points = 2000):
    """
    Performe the PointsSelection methode one multiple files.

    Parameters
    ----------
    annotations_file_list : list
        list of paths to the annotation files:  3d array where points
        from each class i have a value of i and point without class 0.
        The first class has to be the vessels annotations
        (or the object of interest to be segmented). 
    mask_file_list : list, optional
        list of paths to the mask files. the default is "auto" for a list of 
        mask, each covering all the image. The default is "auto".
    box_file_list_list : list, optional
        list of list of path to boxe files. each box file is a mask on an area
        for one image. For one image, with a list of box files,
        the point selection will be balanced between all boxes of the list
        If auta, the point selection won't be balanced.
        The default is "auto".
    vessel_prop : float, optional
        proportion of vessel (class 1) in the . The default is 0.5.
    point_num : int, optional
        number of points to be selected in each annotation file

    Returns
    -------
    points_set_list : list
        A list of tuple of 3 numpy 1d array, with indices of selected points.
        to be passed to the MultiFileBuildDataFrame methode.

    """
    
    if mask_file_list == "auto":
        mask_file_list=["auto" for i in range(len(annotations_file_list))]
    if box_file_list_list == "auto":
        box_file_list_list = ["auto" for i in range(len(annotations_file_list))]
    
    points_set_list = [PointsSelection(annotations_file_list[i],
                                  mask_file_list[i],
                                  box_file_list_list[i],
                                  vessel_prop = vessel_prop,
                                  points=points
                                  ) for i in range(len(annotations_file_list))]
    
    
    return points_set_list
  


def BuildDataFrame(path_featur_dir, annotations_file, points_set = None,
                   feature_list = None):
    """
    Build a dataframe with, for each point, a row with features and label. If points
    set is None, uses all annotations to build the dataframe, else, uses only points 
    in the points set. 


    Parameters
    ----------
    path_featur_dir : str or Path
        Path to featured_dir, in wich features as been computed.
    annotations_file : str or Path
        Path to annotation file.    
    points_set : list, optional
        List of point sets. Result of DataPreparation methode. if None uses all annotated
        points as a single set. The default is None
    feature_list : list, optional
        If None, all files in the featur_dir are loaded as features, else
        only files in feature list are loaded.
        The default is None

    Returns
    -------
    data: dataframe
        dataframe with, for each point, a row with features and label.

    """
    annotations = np.load(annotations_file)
    if points_set is None:
        points_set = np.nonzero(annotations)
    if feature_list is None:
        feature_list = [feat.split(".npy")[0] 
                        for feat in os.listdir(path_featur_dir)]

    data = f3d.LoadFeaturesDir(path_featur_dir, points_set, feature_list)
    data["Label"] = annotations[points_set]
    return data


def MultiFilesBuildDataFrame(path_featur_dir_list, annotations_file_list,
                            points_set_list = None, feature_list = None):
    """
    Build a dataframe with, for each point, a row with features and label. If points
    sets is None, uses all annotations to build the dataframe, else, uses only points 
    in points sets. 


    Parameters
    ----------
    path_featur_dir_list : list
        List of path to feature folder, in wich features as been computed for
        a 3d image.
    annotations_file_list : str or Path
        list of path to annotation files for each image.    
    point_set : list, optional
        List of point sets. Result of MultiFilesDataPreparation methode. 
        if None uses all annotated points as a single set for each image. 
        The default is None
    feature_list : list, optional
        If None, feature list is set on all the feature names in the first 
        feature folder, else only files in feature list are loaded from each 
        feature files. 
        The default is None

    Returns
    -------
    data: dataframe
        dataframe with, for each point, a row with features and label.

    """

    if points_set_list is None:
        points_set_list = [None for i in range(len(path_featur_dir_list))]
    if feature_list is None:
        feature_list = [feat.split(".npy")[0] 
                        for feat in os.listdir(path_featur_dir_list[0])]

    data_list = [BuildDataFrame(path_featur_dir_list[i],
                                annotations_file_list[i],
                                points_set_list[i],
                                feature_list)
                 for i in range(len(path_featur_dir_list))]
    
    data = pd.concat(data_list, axis = 0, ignore_index = True)
    return data



def DFToXY(dataframe):
    """
    Divide a dataframe producted by BuildDataFrames methode between features
    and labels

    Parameters
    ----------
    dataframe : pandas DataFrame
        dataframe with features and labels

    Returns
    -------
    x : pandas DataFrame
        dataframe of features.
    y : pandas Series
        labels

    """
    y = dataframe["Label"]
    x = dataframe.drop("Label", axis = 1)
    
    return x, y

def Optimise(data, optimiser = None):
    """
    Find the best hyperparameters for a random forest classifier model through
    an sklearn optimiser. 

    Parameters
    ----------
    cross_validation_dataframes : Dataframe
        the cross_validation set
    optimiser : optimiser object, default = None
        a RandomizedSearchCV or GridSearchCV object to find best
        hyperparameters. if None, a default
        optimiser will be constructed instead
        the default is None

    Returns
    -------
    optimiser : optimiser object
        the optimiser fited to cross validations datas.

    """
    
    x, y = DFToXY(data)
    y[y!=1] = 2 #we can only use 2 class in this optimiser, so vessel and not vessel
    if optimiser is None:
        param_grid = {
            "criterion": ["gini", "entropy"],
            'max_depth': [5, 10, 20, 50,  100],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [20, 50, 100, 250, 500]
            }
        
        classifier = RandomForestClassifier(n_jobs=6)
        optimiser = RandomizedSearchCV(estimator = classifier,
                                       param_distributions = param_grid,
                                       n_iter = 100, cv = 5, verbose=2,
                                       scoring = "roc_auc",
                                       random_state=42, n_jobs = 6)
    optimiser.fit(x, y)
    return optimiser



    
def BuildModel(data, hyperparameters = None, shuffle=False):
    """
    From a dataframes, build a segmentation model
    

    Parameters
    ----------
    data : dataframe
        Dataframe with for each point, a row with features and label.
    hyperparameters : dict
        Sklearn random forest classifier parameters. The default is None.
            If None, those parameter are going to be used:
            - "n_estimators": 100,
            - "criterion":'gini',
            - "max_depth":  None,
            - "min_samples_split": 2,
            - "min_samples_leaf": 1,
            - "max_features": 'auto'
    cv: int, optional
        Number of folds for the cross-validation. The default is 5
    shuffle: boolean, optional
        If the dataframe has to be shuffuled. The default is False 

    Returns
    -------
    model : dict
        Output of BuildModel methode.
            A dictionary with:
            - "model" an sklearn trained random forest classifier.
            - "features" the list of the model features.
    """
    data = data[sorted(data.columns)]
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)    
    
    if hyperparameters is None:
        hyperparameters = {"n_estimators": 100, "criterion":
                           'gini', "max_depth":  None,
                           "min_samples_split": 2,
                           "min_samples_leaf": 1,
                           "max_features": 'auto'}
    forest = RandomForestClassifier(
                n_estimators = hyperparameters["n_estimators"],
                max_depth = hyperparameters["max_depth"],
                max_features = hyperparameters["max_features"],
                min_samples_leaf = hyperparameters["min_samples_leaf"],
                min_samples_split = hyperparameters["min_samples_split"],
                criterion = hyperparameters["criterion"],
                n_jobs=6
                )
    x_train, y_train = DFToXY(data)
    
    forest.fit(x_train, y_train)
    
    model = {"model": forest, "features": x_train.columns}
    return model
    
def CrossValidation(data, hyperparameters = None, cv=5, shuffle=False):
    """
    From a dataframes, performe n-fold cross validation (default is 5)
    

    Parameters
    ----------
    data : dataframe
        dataframe with, for each point, a row with features and label.
    hyperparameters : dict
        best param to optimise forest.
    cv: int, optional
        number of folds for the cross-validation. The default is 5
    shuffle: boolean, optional
        if the dataframe has to be shuffuled. 

    Returns
    -------
    results_dict : dict
        a dictionary with results all cross validation experiments,
        to be treated with Summarise.

    """
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    set_size = len(data)/cv
    cross_validation_dataframes = [data.iloc[int(i*set_size):int((i+1)*set_size),:] for i 
                                   in range(cv)]
    

    results_dict = { "models": [], "test" : {}, "train" : {},
                    "features": {"names":[]}}
    set_num = len(cross_validation_dataframes)
    
    results_dict["features"]["names"] = list(cross_validation_dataframes[0].columns)
    results_dict["features"]["names"].remove('Label')
    
    if hyperparameters is None:
        hyperparameters = {"n_estimators": 100, "criterion":
                           'gini', "max_depth":  None,
                           "min_samples_split": 2,
                           "min_samples_leaf": 1,
                           "max_features": 'auto'}
    
    for i in range(set_num):
        
        forest = RandomForestClassifier(
                n_estimators = hyperparameters["n_estimators"],
                max_depth = hyperparameters["max_depth"],
                max_features = hyperparameters["max_features"],
                min_samples_leaf = hyperparameters["min_samples_leaf"],
                min_samples_split = hyperparameters["min_samples_split"],
                criterion = hyperparameters["criterion"],
                n_jobs=6
                )
        results_dict["test"][str(i)] = {}
        test = cross_validation_dataframes.pop(0)
        train = pd.concat(cross_validation_dataframes, axis = 0,
                          ignore_index = True)
        x_test, y_test = DFToXY(test)
        x_train, y_train = DFToXY(train)
        model = forest.fit(x_train, y_train)
        
        
        results_dict["models"].append({"model": model,
                                       "features": x_train.columns})
        
        
        
            
        pred_test = model.predict_proba(x_test)[:,0]
        pred_train = model.predict_proba(x_train)[:,0]
        eval_test = ev.Eval(y_test, pred_test,  n_threshold = 50)
        eval_train = ev.Eval(y_train, pred_train,  n_threshold = 50)

        
        
        metrics, curvs = ev.SummarisedMetrics(eval_test)
        metrics_train, curvs_train = ev.SummarisedMetrics(eval_train)
        results_dict["test"][str(i)] = {"metrics": metrics, 
                                              "curvs": curvs}
        results_dict["train"][str(i)] = {"metrics": metrics_train, 
                                              "curvs": curvs_train}
            
        
        cross_validation_dataframes.append(test)
        
        
    return results_dict

def Summarise(result_dict, display=False):
    """
    From the results of the CrossValiation methode, build an human readable 
    summary 

    Parameters
    ----------
    result_dict : dict
        the dict from cross_validation data.
    display : bool, optional
        if true, display the ROC curv of each test
        the default is False

    Returns
    -------
    nice_dataframes : dict
        dict of human readable and interpretable dataframes, to analyse cross
        validation results

    """
    nice_dataframes = {}
    if "features" in list(result_dict.keys()):
        
        feat_dataframe =  pd.DataFrame()
        oob_ranking = pd.DataFrame()
        
        for i in range(len(result_dict["models"])):
            oob_ranking[str(i)] = np.argsort(
                result_dict["models"][i]["model"].feature_importances_
                )
        oob_ranking.index=result_dict["features"]["names"]
        
        feat_dataframe["oob mean rank"] = oob_ranking.mean(axis = 1)
        feat_dataframe["oob rank std"] = oob_ranking.std(axis = 1)
        feat_dataframe = feat_dataframe.T
        feat_dataframe = feat_dataframe.sort_values(by = "oob mean rank"
                                                    , axis = 1)
        nice_dataframes["feat_values"] = feat_dataframe
    
    
    metric_dataframe_test = pd.DataFrame()
    metric_dataframe_train = pd.DataFrame()
    for i in range(len(result_dict["test"])):
        metric_dataframe_test["test"+str(i)] = pd.DataFrame.from_dict(
            result_dict["test"][str(i)]["metrics"], orient = "index"
            )[0]
    for i in range(len(result_dict["test"])):
        metric_dataframe_train["train"+str(i)] = pd.DataFrame.from_dict(
            result_dict["train"][str(i)]["metrics"], orient = "index"
            )[0]
    
    mean_metrics_test = metric_dataframe_test.mean(axis = 1)
    mean_metrics_train = metric_dataframe_train.mean(axis = 1)
    std_metrics_test = metric_dataframe_test.std(axis = 1)
    std_metrics_train = metric_dataframe_train.std(axis = 1)

    final_dataframe = pd.DataFrame()
    final_dataframe["test_means"] = mean_metrics_test
    final_dataframe["train_means"] = mean_metrics_train
    final_dataframe["std_test"] = std_metrics_test
    final_dataframe["std_train"] = std_metrics_train
    nice_dataframes["metrics"] = final_dataframe.T
    
        
    if display:    
        for key in list(result_dict["test"].keys()):
            sensitivity = result_dict["test"][key]["curvs"]["sensitivity"]
            specificity = result_dict["test"][key]["curvs"]["specificity"]
            plt.plot(1-specificity, sensitivity, "")
        plt.legend(["test"+str(i) for i in range(5)])
        plt.title("Models ROC curv")
        plt.ylabel('1 - specificity')
        plt.xlabel('sensitivity')
        plt.show()

        #shap_values = result_dict["features"]['shap'][0]
        #x = result_dict["features"]['shap_source'][0]
        #shap.summary_plot(shap_values = shap_values.values,features =  x)
        

        print(nice_dataframes["metrics"].round(3))
    return nice_dataframes
 
    

def FeatureSelection(data, hyperparameters=None, cv = 5,
                          shuffle=False, worst_features_set_size = 4):
    """
    A methode that evaluate  sets of features for the random
    forest classifier, by remooving worst features one by one.
    

    Parameters
    ----------
    data : dataframe
        dataframe with, for each point, a row with features and label.
    hyperparameters : dict
        Optimised parameters for random forest.
    cv : int, optional
        the nuber of folds for the cross validation. The default is 5
    shuffle: boolean, optional
        if True, rows of data are shuffuled during the cross validation
    worst_features_set_size : int, optional
        the number of worst feature to be tested before remooving
        The default is 4

    Returns
    -------
    dict
        a dictionary of metrics showing their evolution with the
        number of features and the features ranked by increassing relevance for
        the model.

    """
    if hyperparameters is None:
        hyperparameters = {"n_estimators": 100, "criterion": 'gini', "max_depth":                                       None, "min_samples_split": 2, "min_samples_leaf":1,
                           "max_features": 'auto'}
 
    n_features = len(data.columns)-1
    data_df = data.copy()
    
    best_roc = []
    best_dice = []
    best_mcc = []
    roc_05 = []
    dice_05 = []
    mcc_05 = []
    ranked_features = []


    results = CrossValidation(data=data_df, hyperparameters=hyperparameters,
                              cv=cv, shuffle=shuffle)
    summary = Summarise(results)
    
    
    for i in range(n_features):
        print("    ",str(n_features-i),
              "remaining features")
        

        
        feat_df = summary["feat_values"]
        feat_df.sort_values(by = "oob mean rank", axis = 1)
        
        best_roc.append(summary["metrics"]["min_ROC"]["test_means"])
        best_dice.append(summary["metrics"]["max_dice"]["test_means"])
        best_mcc.append(summary["metrics"]["max_MCC"]["test_means"])
        roc_05.append(summary["metrics"]["ROC_05"]["test_means"])
        dice_05.append(summary["metrics"]["dice_05"]["test_means"])
        mcc_05.append(summary["metrics"]["MCC_05"]["test_means"])
            
        worst_features = feat_df.columns[0:min([
            worst_features_set_size,
            n_features-i])]
        worst_feature = worst_features[0]
        if len(worst_features) >1 : 
            best_score = 1.0
            worst_feature = None
            
            for worst_f in worst_features :
                data_df_prov = data_df.copy().drop(worst_f, axis = 1)
                
                results_prov = CrossValidation(data=data_df_prov,
                                               hyperparameters=hyperparameters,
                                               cv=cv, shuffle=shuffle)
                summary_prov = Summarise(results_prov)
                if summary_prov["metrics"]["min_ROC"]["test_means"]<best_score:
                    best_score = summary_prov["metrics"]["min_ROC"]["test_means"]
                    worst_feature = worst_f
                    results = results_prov
                    summary = summary_prov
             
        ranked_features.append(worst_feature)
        print("        " + worst_feature + " removed")
        

        
        
        data_df = data_df.drop(worst_feature, axis = 1)

      
    result_features_opti = {"best_roc": best_roc, "best_dice": best_dice, "best_mcc":
                             best_mcc,
                            "roc_05": roc_05, "dice_05": dice_05, "mcc_05": mcc_05,
                            "ranked_features": ranked_features}


    f_metrics = list(result_features_opti.keys())
    f_metrics.remove("ranked_features")
    for key in f_metrics: 
        plt.plot(np.arange(n_features)+1, result_features_opti[key][::-1])
    for key in f_metrics:
        if "roc" in key:
             plt.plot(np.argmin(result_features_opti[key][::-1])+1,
             result_features_opti[key][::-1][np.argmin(
                 result_features_opti[key][::-1])], "r*")
        else:
            plt.plot(np.argmax(result_features_opti[key][::-1])+1,
                     result_features_opti[key][::-1][np.argmax(
                         result_features_opti[key][::-1])], "r*")
    plt.legend(f_metrics + ["best_values"])
    plt.ylabel('metrics')
    plt.xlabel('number of features')
    plt.title('Evolution of metrics values with numer of features')
    plt.show()  

      
    return result_features_opti

def Segmentations(model, mask, path_feature_dir, threshold=0.5):
    """
    A methode to product a segmentation on an entire image with a model from cross     validations results.

    Parameters
    ----------
    model : dict
        Output of BuildModel methode, a dict with:
            -"model" an sklearn trained random forest classifier.
            -"features" the list of the model features.
    mask : numpy nd_array
        Image binary mask 
    path_feature_dir : Path
        Path to the directory where features have been pre-computed
    model_index: int, optional
        Index of the model to be used from the cross validation results.
        The default is 0

    Returns
    -------
    segmentation_array, numpy nd_array
        a 3d image with for each voxel 1 if it belongs to the first class (
        vessel), else 0
    """

    mask_ = np.nonzero(mask)
    forest = model["model"]
    datas = f3d.LoadFeaturesDir(path_feature_dir, mask_, model["features"])

    probability_array = np.zeros(mask.shape)
    segmentation_array = np.zeros(mask.shape)

    proba = forest.predict_proba(datas)[:, 0]

    probability_array[mask_] = proba
    segmentation_array[probability_array>threshold] = 1.0

    return segmentation_array

def Prediction(model, mask, path_feature_dir):
    """
    A methode to product a segmentation on an entire image with a model from cross     validations results.

    Parameters
    ----------
    model : dict
        Output of BuildModel methode, a dict with:
            -"model" an sklearn trained random forest classifier.
            -"features" the list of the model features.
    mask : numpy nd_array
        Image binary mask 
    path_feature_dir : Path
        Path to the directory where features have been pre-computed


    Returns
    -------
    probability_array, numpy nd_array
        a 3d image with the probability for each voxel to belong to the first class (to be a
        vessel)
    """

    mask_ = np.nonzero(mask)
    forest = model["model"]
    datas = f3d.LoadFeaturesDir(path_feature_dir, mask_, model["features"])

    probability_array = np.zeros(mask.shape)

    proba = forest.predict_proba(datas)[:, 0]

    probability_array[mask_] = proba

    return probability_array

def ShowShap(model, data, points = None):
    """
    

    Parameters
    ----------
    model : dict
        Output of BuildModel methode, a dict with:
            -"model" an sklearn trained random forest classifier.
            -"features" the list of the model features.
    data : dataframe
        dataframe with, for each point, a row with features and label.
    points : int, optional
        Number of points on wich calcule the shape_values
        if None, all points are taken (could take a lot of time).
        The default is None.

    Returns
    -------
    None.

    """
    
    x, y = DFToXY(data)
    x = x[model["features"]]
    explainer = shap.TreeExplainer(model["model"], n_jobs=6, verbose = 0,
                                       data = x, 
                                       feature_perturbation = "interventional")
    
    if points is not None:
        x = x.sample(n=min(len(x),points))
    shap_values = explainer.shap_values(x, check_additivity=False)
    shap.summary_plot(shap_values = shap_values[0], features =  x)
    
    
def Batch_Processing(model, raw_file, mask_file=None, threshold=None,
                     feature_dir = Path("featur_provisional"),
                     clean=True ):
    """
    

    Parameters
    ----------
    model : dict
        Output of BuildModel methode, a dict with:
            -"model" an sklearn trained random forest classifier.
            -"features" the list of the model features.
    raw_file : Path or str
        path to the raw data (numpy nd array of voxel intensities).
    mask_file : Path or str, optional
        path to the mask file (numpy binary nd array of 1 and 0).
        The default is None.
    threshold : float, optional:
        A float with values between 0 and 1. If None, Batch_Processing 
        return a prediction array with for each voxel its probability to be
        a vessel. Else, return a binary (vessel: 1, non_vessel: 0)
        segmented array for the given threshold. The default is None
    feature_dir : Path or str, optional
        directory in wich feature files are stored. If the directory already
        exist, Batch Processing will load the features inside. Else, the 
        methode will build the dir ansd the feature files. At the end of the batch
        processing, this dire and all files inside are removed if clean option
        is True. The default is Path("featur_provisional").
    clean : boolean, optional
        if True, the feature_dir is removed at the end of the
        Batch_Precessing. The default is True.
     

    Returns
    -------
    results : numpy ndarray
        a prediction array or a binary segmentation array.

    """
    if type(feature_dir) is str:
        feature_dir= Path(feature_dir)
    
    raw = np.load(raw_file)
    mask = np.ones(raw.shape)
    if mask_file is not None:
        mask = np.load(mask_file)
    if not(os.path.exists(feature_dir)):
            
        feature_dir.mkdir()
        f3d.BuildFeatureFilesFromList(raw_file, model["features"],
                                       feature_dir)
    pred = Prediction(model, mask, feature_dir)
    
    if threshold is None:
        results =  pred
    else:
        results = pred > threshold
        
    #cleaning    
    if clean:
        for file in os.listdir(feature_dir):
            (feature_dir  / file).unlink()
        feature_dir.rmdir()
    
    
    return results

def SaveModel(model, filname):
    """
    Save a model using pickle

    Parameters
    ----------
    model : dict
        Output of BuildModel methode, a dict with:
            -"model" an sklearn trained random forest classifier.
            -"features" the list of the model features.
    filname : str
        model file name.

    Returns
    -------
    None.

    """
    pickle.dump(model, open(filname, 'wb'))
    
def LoadModel(filname):
    """
    Load a model using pickle

    Parameters
    ----------
    filname : str
        model file name.

    Returns
    -------
    model : dict
        Output of BuildModel methode, a dict with:
            -"model" an sklearn trained random forest classifier.
            -"features" the list of the model features.

    """
    model = pickle.load(open(filname, 'rb'))
    return model



display = False   
if __name__ == '__main__' and display:
    print("nothing")
    