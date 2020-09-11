# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:44:31 2020

@author: titou
"""

from pathlib import Path
import diplib as dip
import numpy as np
import os
import pandas as pd


def GaussianSmoothing(x, sigma, mask=None):
    
    """ Compute n-dimentional gaussian smoothing on nd array.

    Parameters
    ----------
    x : numpy nd-array
        The imput array to be smoothed
    sigma : float
        The gaussian standar deviation (smoothing coeficient)
    mask : tuple of n 1d array of size l
        coordonate of l points one wich the smoothing is wanted.

    Returns
    -------
    gaussian_s : numpy nd-array
        The nd array smoothed with a coeficient sigma
    gaussian_s[mask] : numpy 1d array
        if mask is not None, return an array of the value of the gaussian 
        smoothing for all points in mask
        

    """
    
    gaussian_s = np.asarray(dip.Gauss(x, sigma))
    
    if mask is not None:
        return gaussian_s[mask]
    else:
        return gaussian_s
    
    
def Laplacian(x, mask=None):
    
    """ Compute n-dimentional laplacian on nd array.

    Parameters
    ----------
    x : numpy nd-array
        The imput array on wich the laplacian will be computed
    mask : tuple of n 1d array of size l
        coordonate of l points one wich the laplacian is wanted.

    Returns
    -------
    laplacian : numpy nd-array
        The nd array laplacian (same shape as x)
    laplacian[mask] : numpy 1d array
        if mask is not None, return an array with the value of the laplacian
        for all points in mask
        

    """
    
    laplacian = np.asarray(dip.Laplace(x))
    
    if mask is not None:
        return laplacian[mask]
    else:
        return laplacian
    

def HessianEigenvalues(x, mask=None):
    
    """ Compute n-dimentional hessian eigenvalues on nd array.

    Parameters
    ----------
    x : numpy nd-array
        The imput array on wich the hessian eigenvalues will be computed
    mask : tuple of n 1d array of size l
        coordonate of l points one wich the hessian eigenvalues are wanted.

    Returns
    -------
    eigenvalues : numpy nd-array
        The nd array with all n eigenvalues (shape as x shape + 1)
    eigenvalues[mask] : numpy 2d array
        if mask is not None, return an array with the value of the hessian   
        eigenvalues for all points in mask
        

    """
    
    eigenvalues = np.asarray(dip.Eigenvalues(dip.Hessian(x)))
    
    if mask is not None:
        return eigenvalues[mask]
    else:
        return eigenvalues    
    

def BuildFeatureFiles(x, features=["Gau", "Lap", "HGE"],
                      sigma = 1.0, dir_path = Path("")):
    
    """ Build all feature files of an imput nd array for a given sigma
    smoothing.

    Parameters
    ----------
    x : numpy nd-array
        The imput array on wich the features will be computed
    features : list
        name of the features to be computed
    sigma : float
        The gaussian standar deviation (smoothing coeficient)
    dir_path : str
        path of the directory in wich the feature files are to be stored
    
    """
  
    feature_dic={"Gau": GaussianSmoothing,
       "Lap": Laplacian,
       "HGE": HessianEigenvalues}
    
    add_raw = False
    
    if "Raw" in features:
        add_raw = True
        features.remove("Raw")
        
    i=0
    gauss = feature_dic["Gau"](x, sigma)
    
    for feat in features:
        
        if feat == "HGE":
            eig = feature_dic[feat](gauss)
            name = Path(feat + str(0) + "_sigma" + str(sigma))
            np.save(dir_path / name, eig[:,:,:,0])
            
            i += 1
            name = Path(feat + str(1) + "_sigma" + str(sigma))
            np.save(dir_path / name, eig[:,:,:,1])
            
            i += 1
            name = Path(feat + str(2) + "_sigma" + str(sigma))
            np.save(dir_path / name, eig[:,:,:,2])
            
            del eig
            
        elif feat != "Gau": 
            name = Path(feat+"_sigma" + str(sigma))
            np.save(dir_path / name, feature_dic[feat](gauss))
            
        else:
            name = Path(feat+"_sigma" + str(sigma))
            np.save(dir_path / name, gauss)
        i+=1
    
        
    if add_raw:
        np.save(dir_path / "Raw", x)
        features.append("Raw")

def BuildFeatureFilesFromList(image_file, features_list, dir_path):
    """ Build all feature files in a targeted directory from an imput 
    features list.

    Parameters
    ----------
    image_file : Path or str
        Path to the immage 3d array on wich the features will be computed
    features_list : list
        name of the features to be computed.
        Each feature must be writen in the following format, 
        <<feature_type>>_sigma<<range>>, ex: Gau_sigma2.0 or Lap_sigma1.0.
        Only compatible features are computed, non compatible features 
        are to be separatly computed and mannualy added into the features 
        directory as numpy ndarray.
            Compatible feature type:
            - "Gau" for Gaussian filter,
            - "Lap" for Laplacian filter,
            - "HGE1", "HGE2", "HGE3" for 1st, 2nd and 3rd Hessian eigenvalues,
            
    dir_path : str
        path of the directory in wich the feature files are to be stored
    
    """
    x = np.load(image_file)
    for feature in features_list:
        if "HGE" in feature or "Gau" in feature or "Lap" in feature:
            split = feature.split("_sigma")
            sigma = float(split[1])
            feat = split[0]
            gauss = GaussianSmoothing(x, sigma)
            
            if "HGE" in feat:
                eig = int(feat.split("HGE")[1])
                eigenvalues = HessianEigenvalues(gauss)
                np.save(dir_path / Path(feature), eigenvalues[:,:,:,eig])
            elif "Gau" in feat: 
                np.save(dir_path / Path(feature), gauss)
            elif "Lap" in feat:
                np.save(dir_path / Path(feature), Laplacian(gauss))
        else:
            print("Please manualy add ", feature, "feature in " , dir_path)
            
def MultiFilesBuildFeatureFilesFromList(image_list, features_list, dir_path):
    """ For a list of image files, build in the tageted folder a list of
    sub_folder in wich for each image, all the features in the feature file 
    list are built

    Parameters
    ----------
    image_list : list
        list of image files for wich the features are to be computed
    features_list : list
        name of the features to be computed.
        Each feature must be writen in the following format, 
        <<feature_type>>_sigma<<range>>, ex: Gau_sigma2.0 or Lap_sigma1.0.
        Only compatible features are computed, non compatible features 
        are to be separatly computed and mannualy added into the features 
        directory as numpy ndarray.
            Compatible feature type:
            - "Gau" for Gaussian filter,
            - "Lap" for Laplacian filter,
            - "HGE1", "HGE2", "HGE3" for 1st, 2nd and 3rd Hessian eigenvalues,
            
    dir_path : str or Path
        path of the directory in wich the subfolder for each image feature files
        are to be stored
    
    """
    dir_path = Path(dir_path)
    if not(os.path.exists(dir_path)):
        dir_path.mkdir()
    
    for i, image_file in enumerate(image_list):
        sub_folder = dir_path / ("feature_folder" + str(i))
        if not(os.path.exists(sub_folder)):
            sub_folder.mkdir()
        BuildFeatureFilesFromList(image_file, features_list, sub_folder)
        
    feature_files_list = [dir_path / ("feature_folder" + str(i))
                          for i in range(len(image_list))]
    return feature_files_list

def LoadFeaturesDir(dir_path, mask, features_list = None):
    
    """ Assuming only feature files are in a target directory, build a 
    dataframe with the value of each feature at each points of the mask

    Parameters
    ----------
    dir_path : str
        path of the directory in wich the feature files are stored
    mask : tuple of n 1d array of size l
        coordonate of l points one wich the features are wanted.
    features_list : list, optional
        names of the features files to be loded without the ".npy". 
        Each feature must a nd numpy array the same size as the mask.
        
        The default is None   
        if None, all feature files in the directory will be loded
    
    Returns
    -------
    df : pandas DataFrame
        a n * m dataframe where n  is the number of points in mask and m the 
        number of features
    """
    
    features_file_list = os.listdir(dir_path)
    if features_list is None:
        features_list = [feat.split(".npy")[0] for feat in features_file_list]

    data = np.zeros((len(mask[0]), len(features_list)))
    
    for i, name in enumerate(features_list):
        if name + ".npy" in features_file_list:
            feat = np.load(dir_path / Path(name + ".npy"))[mask]
            data[:,i] = feat
        else:
            print("file :" + name + ".npy" +" not found in " + str(dir_path))
    df = pd.DataFrame(data=data, columns=features_list)
    df = df[sorted(df.columns)]
    return df