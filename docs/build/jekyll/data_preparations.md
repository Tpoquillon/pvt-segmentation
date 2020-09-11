---
date: '2020-09-11T15:54:56.292Z'
docname: data_preparations
images: {}
path: /data-preparations
title: pvtseg.data_preparation module
---

# pvtseg.data_preparation module

Created on Sun Jul 19 15:57:54 2020

@author: titou


### pvtseg.data_preparation.Annotations(classes, mask=None, boxe_list=None)
construct an annotation map with 1 for vessel and 2 for non_vessel


* **Parameters**

    
    * **classes** (*list*) – list of numpy 3d array where each point annoted  as a member of a class
    i as a value of 1.


    * **mask** (*tuple**, **optional*) – tuple of 1d array
    coordonate of points in the parenchyme mask


    * **boxe_list** (*list**, **optional*) – list of tuple, masks of boxes in wich annotation point will be chosen



* **Returns**

    **annotation** – 3d array where each point annoted  as a vessel has a value of 1 and
    each non vessel a value of 2. Only points in the intercection of
    the mask and the union of all boxes are taken into acount



* **Return type**

    3d array



### pvtseg.data_preparation.Balance(seeds, vessel_prop=0.5, size=2000)
construct a balanced dataset from vessel seed and non_vessel seed


* **Parameters**

    
    * **seeds** (*list*) – list of tuple of 1d array
    each tuple is a suffled mask of points belonging to a different
    classes. The convention is that the first class is the class of
    interest. In our case, the vessels.


    * **vessel_prop** (*float**, **otional*) – proportion of vessel points (first class points) in the balanced data
    set. The default is 0.5


    * **size** (*int**, **optional*) – size of the balanced data set. The default is 2000.



* **Returns**

    **balanced_data_set** – tuple of 1d array a shuffled balanced dataset of points from each
    classes.



* **Return type**

    tuple



### pvtseg.data_preparation.JoinMask(mask_1, mask_2)
Concatenate two masks into one


* **Parameters**

    
    * **mask_1** (*tuple*) – tuple of n 1d array of size l1
    coordonate of l1 points.


    * **mask_2** (*tuple*) – tuple of n 1d array of size l2
    coordonate of l2 points.



* **Returns**

    **mask** – conncatenation of mask_1 and mask_2.



* **Return type**

    tuple of n 1d array of size l1 + l2



### pvtseg.data_preparation.Merge(mask_list)
merge several mask in one


* **Parameters**

    **mask_list** (*list** or **tuple*) – list of mask.



* **Returns**

    **mask** – tuple of 1d array
    coordonate of points



* **Return type**

    tuple



### pvtseg.data_preparation.Seeds(annotations)
construct shuflled mask for vessel and non vessel point. These will be
seeds for the balancing of our datas


* **Parameters**

    **annotations** (*nd array*) – annotation matrix with 1 for vessel, 2 for non vessel
    and 0 for unannoted.



* **Returns**

    **seeds** – list of tuple of 1d array
    each tuple is a suffled mask for a class



* **Return type**

    list



### pvtseg.data_preparation.ShortenMask(mask, size)

* **Parameters**

    
    * **mask** (*tuple*) – tuple of n 1d array of size l
    coordonate of l points


    * **size** (*int*) – size of the shortened mask



* **Returns**

    **shortened_mask** – tuple of n 1d array of size equal to the size parameter
    coordonate of “size parameter” points



* **Return type**

    tuple



### pvtseg.data_preparation.ShuffleMask(mask)
take a mask and shuffle it


* **Parameters**

    **mask** (*tuple*) – tuple of n 1d array of size l
    coordonate of l points we want to shuffle



* **Returns**

    **shuffled_mask** – tuple of n 1d array of size l
    shuffled mask.



* **Return type**

    tuple



### pvtseg.data_preparation.Split(mask, n_split=5)
split a single data set in several smaller set of same size


* **Parameters**

    
    * **mask** (*tuple*) – tuple of 1d array
    coordonate of points


    * **n_split** (*int**, **optional*) – number of split. The default is 5



* **Returns**

    **split_set** – tuple of masks.



* **Return type**

    tuple



### pvtseg.data_preparation.SplitMask(mask, mask_1_length)

* **Parameters**

    
    * **mask** (*tuple*) – tuple of n 1d array of size s, coordonate of s points.


    * **mask_1_length** (*int*) – length of mask_1 after spliting



* **Returns**

    
    * **mask_1** (*tuple*) – tuple of n 1d array of size mask_1_length, coordonate of points


    * **mask_2** (*tuple*) – tuple of n 1d array of size s - mask_1_length, coordonate of points.
