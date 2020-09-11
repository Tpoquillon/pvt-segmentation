# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:06:56 2020

@author: titou
"""
import numpy as np


def Eval(annotation, prediction, n_threshold = 1000):
    
    """ Construct an evaluation grid comparing predictions and annotations by
    building a segmentation for several threshold and counting True Positiv,
    False Negativ, False Positiv and True Negativ for these segmentations

    Parameters
    ----------
    annotation : numpy 1d-array
        array where points are annoted with 1 if vessels
    prediction : numpy 1d-array
        array with pevaluation_arraybabilities for each point to be a vessel
    n_threshold : int
        number of threshold to get segmentations fevaluation_arraym the
        prediction

    Returns
    -------
    evaluation_array: 2d-array
        for each threshold, the number of True Positiv, False Negativ, False
        Positiv and True Negativ
    """
    
    evaluation_array = np.zeros((n_threshold, 4))

    positiv = annotation==1       
    for i, t in enumerate(np.linspace(0, 1, n_threshold)):
        segmentation = prediction > t *1.0
        true_ = segmentation ==  positiv
        false_= segmentation !=  positiv
        
        true_positive = np.sum(true_ & positiv)
        false_negative = np.sum(false_ & positiv)
        false_positive = np.sum(false_) - false_negative
        true_negative= np.sum(true_) - true_positive
        evaluation_array[i,:] = np.asarray([true_positive, false_negative,
                              false_positive, true_negative])  
         
    return evaluation_array

def Metrics(evaluation_array):
    
    """ Compute sensitivity, speccificity, MCC and Dice from array with
    True Positiv, False Negativ, False Positiv and True Negativ for several
    threshold

    Parameters
    ----------
    evaluation_array: 2d-array
        for each threshold, the number of True Positiv, False Negativ, False
        Positiv and True Negativ
   

    Returns
    -------
    Sens: 1d array
        sensitivity for each threshold
    Spec: 1d array
        specificity for each threshold
    MCC: 1d array
        Mcc for each threshold
    Dice: 1d array
        Dice for each threshold
    
    """

    TP = evaluation_array[:,0]
    TN = evaluation_array[:,3]
    FP = evaluation_array[:,2]
    FN = evaluation_array[:,1]
    

    Sens = TP / (TP+FN)
    Spec = TN / (TN+FP)
    N = TP + TN + FP + FN
    S = (TP+FN) / N
    P = (TP+FP) / N
    D = (TP/N - S*P)
    Q = ((P * S * (1-S) * (1-P)) ** 0.5)
    MCC = np.zeros(Sens.shape)
    MCC[Q!=0] = D[Q!=0] / Q[Q!=0]
    Dice= 2 * TP / (TP+FP+TP+FN)
    
    return Sens, Spec, MCC, Dice

def SummarisedMetrics(evaluation_array):
    
    """ buils a summarising dictionary from metrics, finding best values,
    threshold for best values and values on median threshold.

    Parameters
    ----------
    evaluation_array: 2d-array
        for each threshold, the number of True Positiv, False Negativ, False
        Positiv and True Negativ

    Returns
    -------
    metrics_dic: dictionary
        dictionnary with metrics of interest
    curv_dict: dictionary
        dictionnary with curv of interest
    
    """
    
    sensitivity, specificity, MCC, Dice = Metrics(evaluation_array)
    n_threshold = len(sensitivity)
    id_ = np.argmax(Dice)
    max_dice = Dice[id_]
    dice_thresh = id_/n_threshold
    
    dice_05 = Dice[int(n_threshold/2)]
    
    id_ = np.argmax(MCC)
    max_MCC = MCC[id_]
    MCC_thresh = id_/n_threshold
    
    MCC_05 = MCC[int(n_threshold/2)]
    
    ROC = ((1-specificity)**2 + (1-sensitivity)**2)**0.5
    id_ = np.argmin(ROC)
    min_ROC = ROC[id_]
    specificity_min_ROC = specificity[id_]
    sensitivity_min_ROC = sensitivity[id_]
    ROC_thresh = id_/n_threshold
    
    ROC_05 = ROC[int(n_threshold/2)]
    
    metrics_dict = {"max_dice": max_dice, "dice_thresh": dice_thresh, 
                    "dice_05": dice_05, "min_ROC": min_ROC,
                   "ROC_thresh": ROC_thresh, "ROC_05": ROC_05,
                    "max_MCC": max_MCC, 
                  "MCC_thresh": MCC_thresh, "MCC_05": MCC_05,
                  "specificity_min_ROC": specificity_min_ROC,
                  "sensitivity_min_ROC": sensitivity_min_ROC}
    curvs_dict = {"sensitivity": sensitivity, "specificity": specificity,
                  "Dice": Dice}
    return metrics_dict, curvs_dict