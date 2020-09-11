---
date: '2020-09-11T15:54:56.292Z'
docname: evaluation
images: {}
path: /evaluation
title: pvtseg.evaluation module
---

# pvtseg.evaluation module

Created on Fri Jul 17 10:06:56 2020

@author: titou


### pvtseg.evaluation.Eval(annotation, prediction, n_threshold=1000)
Construct an evaluation grid comparing predictions and annotations by
building a segmentation for several threshold and counting True Positiv,
False Negativ, False Positiv and True Negativ for these segmentations


* **Parameters**

    
    * **annotation** (*numpy 1d-array*) – array where points are annoted with 1 if vessels


    * **prediction** (*numpy 1d-array*) – array with pevaluation_arraybabilities for each point to be a vessel


    * **n_threshold** (*int*) – number of threshold to get segmentations fevaluation_arraym the
    prediction



* **Returns**

    **evaluation_array** – for each threshold, the number of True Positiv, False Negativ, False
    Positiv and True Negativ



* **Return type**

    2d-array



### pvtseg.evaluation.Metrics(evaluation_array)
Compute sensitivity, speccificity, MCC and Dice from array with
True Positiv, False Negativ, False Positiv and True Negativ for several
threshold


* **Parameters**

    **evaluation_array** (*2d-array*) – for each threshold, the number of True Positiv, False Negativ, False
    Positiv and True Negativ



* **Returns**

    
    * **Sens** (*1d array*) – sensitivity for each threshold


    * **Spec** (*1d array*) – specificity for each threshold


    * **MCC** (*1d array*) – Mcc for each threshold


    * **Dice** (*1d array*) – Dice for each threshold




### pvtseg.evaluation.SummarisedMetrics(evaluation_array)
buils a summarising dictionary from metrics, finding best values,
threshold for best values and values on median threshold.


* **Parameters**

    **evaluation_array** (*2d-array*) – for each threshold, the number of True Positiv, False Negativ, False
    Positiv and True Negativ



* **Returns**

    
    * **metrics_dic** (*dictionary*) – dictionnary with metrics of interest


    * **curv_dict** (*dictionary*) – dictionnary with curv of interest
