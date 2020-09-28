# PVTSEG
 *Pvtseg (Pulmonary Vascular Tree SEGmentation) is a python package for pulmonary vascular tree segmentation. It allows users to build, evaluate and optimise a segmentation model based on random forest. This package was developed for covid patients with lung injuries.* (see french [internship report](/assets/text/Stage_Titouan_.pdf))

 ### THIS IS ONLY A *PREVIEW* OF THE PVTSEG PACKAGE. THE REAL PACKAGE WITH WORKING LINKS IS ON A *[gitlab.in2p3.fr](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/tree/master)* REPOSITORY. IT IS, FOR THE MOMENT, A PRIVATE REPOSITORY.

<div style="text-align: center">
<img src="/assets/images/pvt_couv.png" alt="front image" width="500"/>
</div>


## Table of Contents
1. [Instalation and Usage](#instalation-and-usage)
3. [Data Format](#data-format)
4. [Short Tutorial](#short-tutorial)


## Instalation and Usage

### Install
the pvtseg package require **python 3.7** or any later vesion.
For now, one can only install **pvtseg** package through wheel files:

*   From the [dist directory](/dist) download the .whl file with the desired version (for exemple: *pvtseg-1.0-py3-none-any.whl* ),
*   In the the downloaded file directory, install the package with [pip](https://pypi.org/project/pip/), 

        python -m pip install pvtseg-1.0-py3-none-any.whl

*   or

        python3 -m pip install pvtseg-1.0-py3-none-any.whl


The install may not work on linux without the freeglut package:

        sudo apt-get install freeglut3
   
### Usage

The **pvtseg** package is composed of 4 modules:
-  *vessel_seg* (the package main module), to build sgmentations model and performe cross-validations, optimisations and segmentations,
-  *data_preparation*, to build balanced data,
-  *evaluation* , to evaluate models segmentation compared to ground truth,
-  *features_3d*, to build and manage features needed for segmentation.
 


## Data Format

This library does not directly use a standard data format. All image file are stored as simple [numpy ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) matrix, as it is a verry basic and easy to process format. A module to convert anny kind of classical 3d immage format into nd array will be added soon. 



## Short Tutorial
Here is a quick tutorial to learn how to use the pvtseg library for blood vessel segmentation. Yon can also directly load the [jupyter notebook](Tutorial.ipynb) file for the tutorial.


### Introduction

The goal of this tutorial is to discover the functionalities of the pvtseg library. Through a blood vessel  segmentation project with random forest classifier models on a reduced data set called [**3boxes**](https://github.com/Tpoquillon/Tpoquillon.github.io/tree/master/Data/3boxes).

This dataset is completely annotated. It comes from 3 zooms on different lung area for a CT scan of a Covid-19 patient. A healthy area, a damaged area, and an area bordering the lungs. These three areas have been concatenated to form a single 3d image, on which we will work. We have chosen to do this to make this tutorial simpler and easier to view. During your projects it is likely that you will build datasets from several images, This is why the methods used to build the database from a single image that we will use have their equivalent for multiple image file. 

In addition to these images, a last sample of the Ct scan is used to serve as a test for the segmentation models that we want to build.



```python
from pvtseg import vessel_seg as ves
import os
import numpy as np
import matplotlib.pyplot as plt
```

    PyDIPjavaio unavailable:  DLL load failed while importing PyDIPjavaio: Le module spécifié est introuvable.  (libjvm not found)
    

#### Starting the segmentation project
To organize the image files with which you will be working during the tutorial, we offer a method, [**vessel_seg.BuildDirectoryTree**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segbuilddirectorytreepathmy_data), that will directly build the tree structure of the folders in which these images will be stored. This method can of course be used for other segmentation projects. but you are also free to organize your files as you wish.


```python
path_s , path_f, path_r = ves.BuildDirectoryTree("tutorial")
os.listdir("tutorial")# shows what's inside the tutorial dir
```




    ['Features', 'Results', 'Sources']



This method builds a folder containing the following 3 sub-folders:
-  path_s: path to the 'Sources' folder , for annotation file, raw file  and mask file),
-  path_f; path to the 'Features' folder, for computed feature files,
-  path_r; path to the 'Results' folder, for segmentation results.

#### Data 

Download the 4 .npy files from [3boxes](https://github.com/Tpoquillon/Tpoquillon.github.io/tree/master/Data/3boxes) and place them in the Sources folder.

Images for this tutorial are in .npy format ([numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)). Each image is a three-dimensional matrix. The pvtseg library is made to work with this image format.



```python
os.listdir(path_s)# shows what's inside the source dir. 
```




    ['annotations.npy',
     'box1.npy',
     'box2.npy',
     'box3.npy',
     'mask.npy',
     'raw.npy',
     'test.npy']



#### Visualizing data

##### raw.npy

The raw data is the results of a 3d CT scan. Each voxel of the image (each point of the matrix) has a value corresponding to its intensity.
We can see on this slice, the 3 areas contiguous to each other. From left to right:
- a healthy area, 
- a damaged area,
- an area on the lung's edge.

These 3 area diplay different structures and vessel sizes. Obtaining a model capable of detecting the blood vessels in these 3 boxes is therefore not easy.


```python

raw = np.load(path_s / "raw.npy")
plt.imshow(raw[0,:,:],"gray")
plt.show()
```


![png](/assets/images/output_11_0.png)


##### annotations.npy
The annotations show the ground truth. This image as been manually annoted. Each voxel in the annotation matrix as a value of 0, if not annoted, 1 if belonging to a vessel, 2 if belonging to the bakground (pulmonary alveoli) and 3 if belonging to the non_vessel structure (lesion or pulmonary edges) clearly distinct from the background.


```python
annotations = np.load(path_s / "annotations.npy")
plt.imshow(annotations[0,:,:],"gray")
plt.show()
```


![png](/assets/images/output_13_0.png)


##### mask.npy

The mask is covering only the areas of interest. A binary 3D numpy-array where voxels of interest have values of 1 and the others 0.
In this case, it is a bit superfluous as all the image is anotated. However, it is particularly useful when you want to work only on a portion of the image.

Its only uses in this project are to separate the areas to prevent the feature calculations from being disturbed by artificial junction between 2 boxes and to hide some non pulmonary tissues in the "edge" area.


```python
mask = np.load(path_s / "mask.npy")
plt.imshow(mask[0,:,:],"gray")
plt.show()
```


![png](/assets/images/output_15_0.png)


##### test.npy

Once a model has been trained and evaluated, we can test it on this image. this is a completely different picture from the one that was used to train and validate our dataset. It is currently not annotated, the performance evaluation on this image will therefore be done visually.


```python
test = np.load(path_s / "test.npy")
plt.imshow(test[0,:,:],"gray")
plt.show()
```


![png](/assets/images/output_17_0.png)


#### Features 

##### Building feature files
In order to be trained or to make predictions on voxels, a random forest classifier model needs features describing these voxels and theire environment. Here we build a small set of 10 features.
Each feature must be writen in the following format: *'feature_type'\_sigma'scale'*, ex: Gau_sigma2.0 or Lap_sigma1.0

The feature_type indicates the type of filter used to calculate features. The scale indicates the size of structure  the filter will detect. Pvtseg allows to compute the folowing feature types:
-  "Gau" for Gaussian filter
-  "Lap" for Laplacian filter
-  "HGE1", "HGE2", "HGE3" for 1st, 2nd and 3rd Hessian eigenvalues

The [**features_3d.BuildFeatureFilesFromList**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/features_3d.md#pvtsegfeatures_3dbuildfeaturefilesfromlistimage_file-features_list-dir_path) methode allows to build the fetures files in a target folder given a features list and raw data. 

*Fore multiple files its equivalent is the [features_3d.MultiFilesBuildFeatureFilesFromList](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/features_3d.md#pvtsegfeatures_3dmultifilesbuildfeaturefilesfromlistimage_list-features_list-dir_path)*




```python
from pvtseg import features_3d as f3d

features_list = ['HGE0_sigma0.5',
                 'Lap_sigma10.0',
                 'HGE0_sigma2.0',
                 'Gau_sigma5.0',
                 'HGE1_sigma10.0',
                 'Gau_sigma0.1',
                 'HGE1_sigma1.0',
                 'Gau_sigma10.0',
                 'Gau_sigma3.0',
                 'HGE0_sigma0.3']
f3d.BuildFeatureFilesFromList(path_s / "raw.npy", features_list, path_f)

```

##### Visualization of features
Features are 3d numpy arrays. They are the same size as the source image they were built from.
Lets visualize what some features look like:


```python
f1 = np.load(path_f / 'Gau_sigma3.0.npy')
f2 = np.load(path_f / 'HGE0_sigma0.5.npy')
plt.imshow(f1[0,:,:],"gray")
plt.show()
plt.imshow(f2[0,:,:],"gray")
plt.show()
```


![png](/assets/images/output_21_0.png)



![png](/assets/images/output_21_1.png)


The first feature is a Gaussian smoothing. You can see that it applies a kind of overall blur to the image. This effect is particularly strong and it is difficult to distinguish the details of the image. This is because the scale value of this descriptor is particularly important. It therefore makes it possible to highlight large structures.

The second filter is a Hessian filter. It highlights the texture of the image. We can see a lot more details on this image. Indeed the scale parameter is low

### First Model
#### Building Dataframe

Now that we have computed our features, it's time to build our dataset to train and validate our models.
Datasets are [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), in which each point is represented as a raw with the values of each feature for this point as well as its label.

The [**vessel_seg.BuildDataFrame**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segbuilddataframepath_featur_dir-annotations_file-points_setnone-feature_listnone) allows us to build this dataframe, given the annotations file and the feature files. With point_sets parameter set to **None**, all the annoted points will be used to build the datas.

*for multiple images its equivalent is the [vessel_seg.MultiFilesBuildDataFrame](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segmultifilesbuilddataframepath_featur_dir_list-annotations_file_list-points_set_listnone-feature_listnone) methode*


```python
data = ves.BuildDataFrame(path_featur_dir=path_f, annotations_file = path_s / "annotations.npy", points_set=None)
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gau_sigma0.1</th>
      <th>Gau_sigma10.0</th>
      <th>Gau_sigma3.0</th>
      <th>Gau_sigma5.0</th>
      <th>HGE0_sigma0.3</th>
      <th>HGE0_sigma0.5</th>
      <th>HGE0_sigma2.0</th>
      <th>HGE1_sigma1.0</th>
      <th>HGE1_sigma10.0</th>
      <th>Lap_sigma10.0</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>438.596814</td>
      <td>426.077759</td>
      <td>548.184531</td>
      <td>543.385708</td>
      <td>143.909593</td>
      <td>130.270188</td>
      <td>46.009798</td>
      <td>49.736631</td>
      <td>-0.065402</td>
      <td>-0.768993</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>232.487892</td>
      <td>425.696411</td>
      <td>538.072974</td>
      <td>539.845293</td>
      <td>127.371050</td>
      <td>116.940158</td>
      <td>45.195154</td>
      <td>38.332531</td>
      <td>-0.063775</td>
      <td>-0.756374</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>182.613623</td>
      <td>424.547451</td>
      <td>509.997781</td>
      <td>529.537154</td>
      <td>88.661816</td>
      <td>82.786219</td>
      <td>42.410576</td>
      <td>14.061333</td>
      <td>0.064576</td>
      <td>-0.711387</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>235.482843</td>
      <td>422.667602</td>
      <td>469.841668</td>
      <td>513.303809</td>
      <td>46.775770</td>
      <td>45.737847</td>
      <td>37.082034</td>
      <td>24.032579</td>
      <td>0.077133</td>
      <td>-0.625199</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>585.295337</td>
      <td>420.105780</td>
      <td>425.051972</td>
      <td>492.405584</td>
      <td>-32.849319</td>
      <td>-31.206464</td>
      <td>29.295158</td>
      <td>16.058727</td>
      <td>0.094132</td>
      <td>-0.514797</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Building  model
We have evrything we need to build our first model with [**vessel_seg.BuildModel**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segbuildmodeldata-hyperparametersnone-shufflefalse).
A model consist in a trained [random forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and a list of the model features, stored in a dictionary. 

The order of the feature is particularly important. If they are entered in the wrong order, the model predictions will be meaningless as it will use one feature instead of another during segmentation.
By convention, we have chosen to always arrange our descriptors in alphanumeric order.



```python
my_first_model = ves.BuildModel(data)
```


```python
my_first_model
```




    {'model': RandomForestClassifier(n_jobs=6),
     'features': Index(['Gau_sigma0.1', 'Gau_sigma10.0', 'Gau_sigma3.0', 'Gau_sigma5.0',
            'HGE0_sigma0.3', 'HGE0_sigma0.5', 'HGE0_sigma2.0', 'HGE1_sigma1.0',
            'HGE1_sigma10.0', 'Lap_sigma10.0'],
           dtype='object')}



#### Segmentation
##### On train image
Now that we have a model, we can start a segmentating our training image with the [**vessel_seg.Segmentations**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segsegmentationsmodel-mask-path_feature_dir-threshold05) and the [**vessel_seg.Prediction**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segpredictionmodel-mask-path_feature_dir) methodes. The difference between a segmentation and a prediction is the fact that the prediction show points probability (between 0 and 1) to be a vessel, while the segmentation is binary (1 for vessel points and 0 for non_vessel_points)


```python
segmentation_array = ves.Segmentations(my_first_model, mask, path_feature_dir = path_f )
probability_array = ves.Prediction(my_first_model, mask, path_feature_dir = path_f )


```


```python
plt.imshow(raw[0,:,:],"gray")
plt.title("raw_data")
plt.show()

plt.imshow(annotations[0,:,:]==1,"gray")
plt.title("annotations")
plt.show()

plt.imshow(probability_array[0,:,:], "gray")
plt.title("prediction")
plt.show()

plt.imshow(segmentation_array[0,:,:], "gray")
plt.title("segmentation")
plt.show()

```


![png](/assets/images/output_30_0.png)



![png](/assets/images/output_30_1.png)



![png](/assets/images/output_30_2.png)



![png](/assets/images/output_30_3.png)


This results seem pretty good, but we must not forget that we have trained this model on all points. It is therefore normal that it is good at doing exactly what he has been trained for.
However, an interesting model must be able to make good predictions on data for which it has not been trained, thus we are going to see if the results are similar one the test image.

##### On test image

To get these result we are going to use the [**vessel_seg.Batch_Processing**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segbatch_processingmodel-raw_file-mask_filenone-thresholdnone-feature_dirwindowspathfeatur_provisional-cleantrue) methode on the test image. This methode will compute the features for the new image, and store them in a provisory feature directory. Because we are going to use these features again in the tutorial, we don't want the directory to be cleaned at the en of the processing. Therefor we set the *clean* option to **False**


```python
test_pred  = ves.Batch_Processing(model=my_first_model,
                                  raw_file=path_s / "test.npy",
                                  feature_dir="test_tuto",
                                  clean = False)
```


```python
plt.imshow(test[50,:,:],"gray")
plt.title("raw_data")
plt.show()
plt.imshow(test_pred[50,:,:],"gray")
plt.title("model prediction")
plt.show()
```


![png](/assets/images/output_33_0.png)



![png](/assets/images/output_33_1.png)


On this prediction, the vessels appear to have been correctly detected. It will be noted in particular that the junction between the pulmonary lobe, visible on the raw images by a very thin line crossing the image from right to left, has not been segmented.

#### Features interpretation 
Understanding how your model works is as important as making good predictions. This is particularly true in biology where the interpretability allows a better understanding of the subjects that one deals with. For this, we use the Shapley values (from the python [Shap](https://github.com/slundberg/shap) package) which allow us to understand what impact each of the features have on the final prediction decision.

To do this, we use the hot [vessel_seg.ShowShap](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segshowshapmodel-data-pointsnone). This method requires a lot of time for this reason we will only use it on a small sample of the points which were used to train our model.


```python
ves.ShowShap(my_first_model, data, points = 200)
```

     97%|=================== | 583/600 [00:19<00:00]       


![png](/assets/images/output_36_1.png)


Unfortunately, in this case, it is impossible to interpret the Shapley values correctly. Indeed in an ideal situation one should observe for each feature 2 groups of points, one corresponding to the vessels and another corresponding to the rest. This helps determine the impact each feature has on the final decision to classify a point as a blood vessel. However in this case our dataset is completely out of balance. We mainly observe here non-vessel points.


```python
print("Number of points in the data: ",len(data["Label"]),".  Number of vessel points:  ", sum(data["Label"]==1), ", ",
     round(sum(data["Label"]==1)*100/len(data["Label"])),"%", ".  Number of non vessel structure points:  ", 
     sum(data["Label"]==3), ", ", round(sum(data["Label"]==3)*100/len(data["Label"])),"%")
```

    Number of points in the data:  76056 .  Number of vessel points:   5449 ,  7 % .  Number of non vessel structure points:   7773 ,  10 %
    

#### Cross-validation
To assess the ability of a model to make a good prediction on data for which it has not been trained, and evaluate if its not overfiting, we perform a cross validation.
In this case, it will be a cross validation at 5 folds. Our dataset will be divided into 5 partitions.
5 models will be built, trained on 4 of the 5 partitions and evaluated on both the 4 training partitions and the fifth test partition.

We will then recover the average performance of the 5 models .

To get the performances of a model on a set of points, we perform the prediction with this model for each point, and compare it to the ground truth.
A prediction is the probability (value between 0 and 1) for a point to be a blood vessel. This prediction must be thresholded to obtain a segmentation comparable to the ground truth. We use 50 different thresholds between 0 and 1. We therefore evaluate, not one, but 50 segmentations.

For each threshold value, we measure the number of true positives (TP), false positives (FP), true negatives (TN) and false negatives (FN) between the segmentation and the annotations.

These measurements are used to calculate the following metrics:

- Sensitivity: TP / (TP + FN),
- TN / (TN + FP) specificity
- Matthews correlation coefficient (MCC) (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ^ 0.5.
- Sorensen-Dice (Dice) index:
    (2TP) / (2TP + TN + FP)
- the ROC distance to optimal (ROC) wich combine sensitivity and specificity: ((1-sensitivity) ^ 2, (1-specificity) ^ 2) ^) 0.5.

High sensitivity means that a majority of vessels in the image are detected, while high specificity means that few non-vessels are detected as vessels. A high MCC is a mark of an overall classification quality. A Dice close to 1 is the mark of a good superposition between the segmented blood vessels and the annotated blood vessels. Finally, a small ROC indicate a good balance between sensitivity and specificity




- the [**vessel_seg.CrossValidation**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segcrossvalidationdata-hyperparametersnone-cv5-shufflefalse) allows us to perform the cross validation
- the [**vessel_seg.Summarise**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segsummariseresult_dict-displayfalse) methodes build human readables data showing the average performance



```python
cv_results = ves.CrossValidation(data, cv=5, shuffle = True)

```


```python
summary = ves.Summarise(cv_results, display=True)
```


![png](/assets/images/output_41_0.png)


                 max_dice  dice_thresh  dice_05  min_ROC  ROC_thresh  ROC_05  \
    test_means      0.855        0.372    0.839    0.065       0.112   0.222   
    train_means     1.000        0.460    1.000    0.000       0.460   0.000   
    std_test        0.003        0.023    0.006    0.004       0.018   0.013   
    std_train       0.000        0.020    0.000    0.000       0.020   0.000   
    
                 max_MCC  MCC_thresh  MCC_05  specificity_min_ROC  \
    test_means     0.844       0.388   0.830                0.954   
    train_means    1.000       0.460   1.000                1.000   
    std_test       0.003       0.046   0.005                0.006   
    std_train      0.000       0.020   0.000                0.000   
    
                 sensitivity_min_ROC  
    test_means                 0.954  
    train_means                1.000  
    std_test                   0.006  
    std_train                  0.000  
    

We can see a big difference in performance between the evaluation on the training data and the evaluation on the validation data. It is therefore not completely satisfactory, our model doing overfiting

### Improving our dataset

Our data includes only a small percentage of vessels. this results in imbalance when training our model. It is therefore necessary to select a more balanced dataset for better training. In addition, 80,000 points is a lot. Some processes will take a very long time with a dataset of this size. We will therefore reduce the size of our dataset to 3000 points and balance it between vessels and non vessels (both background and non vessel structure).

#### Point Selection

To buid a balanced data set, with balanced proportions between vessel points and non vessel points, we will use the [**vessel_seg.PointsSelection**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segpointsselectionannotations_file-mask_fileauto-box_file_listauto-vessel_prop05-points2000) methode and gives its result to the **BuildDataFrame** methodes.

*For multiple images its equivalent is the [vessel_seg.MultiFilesPointsSelection](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segmultifilespointselectionannotations_file_list-mask_file_listauto-box_file_list_listauto-vessel_prop05-points2000)*


```python
point_set = ves.PointsSelection(annotations_file =  path_s / "annotations.npy", vessel_prop = 0.5, points = 3000)
data = ves.BuildDataFrame(path_featur_dir=path_f, annotations_file= path_s / "annotations.npy", points_set = point_set)
```


```python
print("Number of points in the data: ",len(data["Label"]),".  Number of vessel points:  ", sum(data["Label"]==1), ", ",
     round(sum(data["Label"]==1)*100/len(data["Label"])),"%", ".  Number of non vessel structure points:  ", 
     sum(data["Label"]==3), ", ", round(sum(data["Label"]==3)*100/len(data["Label"])),"%")
```

    Number of points in the data:  3000 .  Number of vessel points:   1500 ,  50 % .  Number of non vessel structure points:   750 ,  25 %
    

We can even balanced ou dataset further. In fact we take our vessel and non vessel points from the whole picture. However, this image is made up of 3 boxes well apart. Each box has different characteristics, and not the same proportions of points. If we want to make a better data selection. We must make a balanced selection between both classes **and** boxes.


```python
box1 = np.zeros(raw.shape)
box1[:,:,0:30] = 1.0
box2 = np.zeros(raw.shape)
box2[:,:,40:70] = 1.0
box3 = np.zeros(raw.shape)
box3[:,:,80:110] = 1.0

np.save(path_s / "box1", box1)
np.save(path_s / "box2", box2)
np.save(path_s / "box3", box3)
```


```python
plt.imshow(box1[0,:,:], "gray")
plt.title("box 'healthy'")
plt.show()
plt.imshow(box2[0,:,:], "gray")
plt.title("box 'unhealthy'")
plt.show()
plt.imshow(box3[0,:,:], "gray")
plt.title("box 'edge'")
plt.show()

```


![png](/assets/images/output_51_0.png)



![png](/assets/images/output_51_1.png)



![png](/assets/images/output_51_2.png)


The [**vessel_seg.PointsSelection**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segpointsselectionannotations_file-mask_fileauto-box_file_listauto-vessel_prop05-points2000) methode also allows us to  balanced data, given a list of boxe_files. 


```python
points_set = ves.PointsSelection(annotations_file =  path_s / "annotations.npy",
                             vessel_prop = 0.5,
                             points = 3000, 
                             box_file_list = [
                                 path_s / "box1.npy",
                                 path_s / "box2.npy",
                                 path_s / "box3.npy",
                            ])

```


```python
data = ves.BuildDataFrame(path_featur_dir=path_f, annotations_file= path_s / "annotations.npy", points_set = points_set)
```


```python
print("Number of points in the data: ",len(data["Label"]),".  Number of vessel points:  ", sum(data["Label"]==1), ", ",
     round(sum(data["Label"]==1)*100/len(data["Label"])),"%", ".  Number of non vessel structure points:  ", 
     sum(data["Label"]==3), ", ", round(sum(data["Label"]==3)*100/len(data["Label"])),"%")
```

    Number of points in the data:  3000 .  Number of vessel points:   1500 ,  50 % .  Number of non vessel structure points:   500 ,  17 %
    

We observe that 17% of structural points instead of 25% previously. in fact, this is normal. The 1st box does not contain any of these points. So instead of having 3/12th of it, there are only 2/12ths, 17%.


```python
my_model = ves.BuildModel(data)

test_pred  = ves.Batch_Processing(model=my_model,
                                  raw_file=path_s / "test.npy",
                                  feature_dir="test_tuto",
                                  clean = False)

plt.imshow(test[50,:,:],"gray")
plt.title("raw_data")
plt.show()
plt.imshow(test_pred[50,:,:],"gray")
plt.title("model prediction")
plt.show()
```


![png](/assets/images/output_57_0.png)



![png](/assets/images/output_57_1.png)


We can't realy go further to improve the quality of our data set. However, it's possible to optimise the model through 2 more axes:

- the feature selectction
- the random forest hyperparameters

### Optimisation

#### Features selection
the [**vessel_seg.FeatureSelection**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segfeatureselectiondata-hyperparametersnone-cv5-shufflefalse-worst_features_set_size4)
allows us to see the impact of our features on the dataset, by removing each feature one by one and study the impact on the model prediction quality, in an internal cross validation. Finding a good feature set can allow us to improve our segmentation, or to reduce the necessary number of feature and therefore the amount of computation required.

*this method is time consuming so if you don't have time, ignore it and jump to*  **Hyperparameters optimisation**


```python
feature_selection = ves.FeatureSelection(data, worst_features_set_size = 3)
```

         10 remaining features
            Gau_sigma5.0 removed
         9 remaining features
            HGE0_sigma0.5 removed
         8 remaining features
            Lap_sigma10.0 removed
         7 remaining features
            Gau_sigma3.0 removed
         6 remaining features
            HGE1_sigma1.0 removed
         5 remaining features
            HGE0_sigma2.0 removed
         4 remaining features
            HGE0_sigma0.3 removed
         3 remaining features
            HGE1_sigma10.0 removed
         2 remaining features
            Gau_sigma10.0 removed
         1 remaining features
            Gau_sigma0.1 removed
    


![png](/assets/images/output_61_1.png)



```python
feature_selection["ranked_features"]
```




    ['Gau_sigma5.0',
     'HGE0_sigma0.5',
     'Lap_sigma10.0',
     'Gau_sigma3.0',
     'HGE1_sigma1.0',
     'HGE0_sigma2.0',
     'HGE0_sigma0.3',
     'HGE1_sigma10.0',
     'Gau_sigma10.0',
     'Gau_sigma0.1']



This can be interpreted from the previous graph and that the quality of the predictions decreases very quickly with the deletion of features. 
In fact this result would encourage us, for a more successful model, to increase the initial number of features. 
The goal of this tutorial is not that you find the perfect segmentation model, but to make you discover the package, therefore we will not go further on that side.

#### Hyperparameters optimisation

One last way to improve our segmentation, and to find a good set of hyperparameters, which characterize how our random forest works.

the [**vessel_seg.Optimise**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segoptimisedata-optimisernone) methode allows us to test a range of hyperparameters for our models and find the best hyperparameters


```python
optimiser = ves.Optimise(data)

```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits
    

    [Parallel(n_jobs=6)]: Using backend LokyBackend with 6 concurrent workers.
    [Parallel(n_jobs=6)]: Done  29 tasks      | elapsed:    6.8s
    [Parallel(n_jobs=6)]: Done 150 tasks      | elapsed:   28.3s
    [Parallel(n_jobs=6)]: Done 353 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=6)]: Done 500 out of 500 | elapsed:  1.8min finished
    


```python
hyperparameters = optimiser.best_params_
hyperparameters
```




    {'n_estimators': 50,
     'min_samples_split': 2,
     'min_samples_leaf': 2,
     'max_features': 'auto',
     'max_depth': 100,
     'criterion': 'entropy'}



#### Final model
Lets build our final model with the optimised hyperparameters.


```python
my_final_model = ves.BuildModel(data=data, hyperparameters = hyperparameters)

```


```python
test_pred  = ves.Batch_Processing(model=my_final_model,
                                  raw_file=path_s / "test.npy",
                                  feature_dir="test_tuto",
                                  clean = True)
```


```python
plt.imshow(test[50,:,:],"gray")
plt.title("raw_data")
plt.show()
plt.imshow(test_pred[50,:,:],"gray")
plt.title("model prediction")
plt.show()
```


![png](/assets/images/output_70_0.png)



![png](/assets/images/output_70_1.png)



```python
cv_results = ves.CrossValidation(data, cv=5, shuffle = True, hyperparameters = hyperparameters)
summary = ves.Summarise(cv_results, display=True)
```


![png](/assets/images/output_71_0.png)


                 max_dice  dice_thresh  dice_05  min_ROC  ROC_thresh  ROC_05  \
    test_means      0.931        0.488    0.927    0.100       0.496   0.104   
    train_means     0.994        0.500    0.993    0.009       0.500   0.009   
    std_test        0.009        0.030    0.009    0.014       0.033   0.014   
    std_train       0.001        0.020    0.001    0.002       0.020   0.002   
    
                 max_MCC  MCC_thresh  MCC_05  specificity_min_ROC  \
    test_means     0.861       0.492   0.855                0.931   
    train_means    0.988       0.512   0.987                0.994   
    std_test       0.019       0.036   0.020                0.017   
    std_train      0.002       0.023   0.003                0.002   
    
                 sensitivity_min_ROC  
    test_means                 0.930  
    train_means                0.994  
    std_test                   0.017  
    std_train                  0.002  
    


```python
ves.ShowShap(my_final_model, data, points = 200)
```


![png](/assets/images/output_72_0.png)


Finally, compared to our first model, we didn't get anything much more efficient. We can see from the result of the cross-validation that this last model does a little less overfitting than the first model, but in terms of performance or predictions, we get similar results, we even have the impression that there are false positives on the final prediction.

On the other hand, it is easier to interpret. Shapley values are more readable, especially for the most important features where we can distinguish 2 groups of points which correspond to blood vessels and non vessel points. We can read that one for a low scale Gaussian filter, a large value will tend to lead to the prediction of a vessel. On the other hand for the 1st Hesienne eigenvalue, it is the opposite. A large value will tend to denote a point that is not a blood vessel.

We can save both of these model with the [**vessel_seg.SaveModel**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segsavemodelmodel-filname) methode. so they can be loaded later with the [**vessel_seg.LoadModel**](https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation/-/blob/master/docs/build/jekyll/vessel_seg.md#pvtsegvessel_segloadmodelfilname) method


```python
ves.SaveModel(my_first_model, path_r / "first_model")
ves.SaveModel(my_final_model, path_r / "final_model")
```
