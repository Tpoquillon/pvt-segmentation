# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:57:54 2020

@author: titou
"""
import numpy as np

def ShuffleMask(mask):
    """
    take a mask and shuffle it

    Parameters
    ----------
    mask : tuple
        tuple of n 1d array of size l
        coordonate of l points we want to shuffle

    Returns
    -------
    shuffled_mask : tuple
        tuple of n 1d array of size l
        shuffled mask.

    """
    length = len(mask[0])
    index = np.random.sample(length).argsort()
    shuffled_mask = (mask[0][index],
                     mask[1][index],
                     mask[2][index])
    
    
    return shuffled_mask

def ShortenMask(mask, size):
    """
    

    Parameters
    ----------
    mask : tuple
        tuple of n 1d array of size l
        coordonate of l points
    size : int
        size of the shortened mask
        
    Returns
    -------
    shortened_mask : tuple
        tuple of n 1d array of size equal to the size parameter
        coordonate of "size parameter" points

    """
    shortened_mask = (mask[0][:size], mask[1][:size], mask[2][:size])
    
    return shortened_mask

def JoinMask(mask_1, mask_2):
    """
    Concatenate two masks into one

    Parameters
    ----------
    mask_1 : tuple
        tuple of n 1d array of size l1
        coordonate of l1 points.
    mask_2 : tuple
        tuple of n 1d array of size l2
        coordonate of l2 points.

    Returns
    -------
    mask : tuple of n 1d array of size l1 + l2
        conncatenation of mask_1 and mask_2.

    """
    mask=[]
    for i in range(len(mask_1)):
        mask.append(np.concatenate((mask_1[i], mask_2[i])).astype(int))
    mask = tuple(mask)
    return mask


def SplitMask(mask, mask_1_length):
    """
    

    Parameters
    ----------
    mask : tuple
        tuple of n 1d array of size s, coordonate of s points.
    mask_1_length : int
        length of mask_1 after spliting

    Returns
    -------
    mask_1 : tuple
        tuple of n 1d array of size mask_1_length, coordonate of points
    mask_2 : tuple
        tuple of n 1d array of size s - mask_1_length, coordonate of points.

    """
    mask_1 = []
    mask_2 = []
    for i in range(len(mask)):
        mask_1.append(mask[i][0:mask_1_length])
        mask_2.append(mask[i][mask_1_length:])
    mask_1 = tuple(mask_1)
    mask_2 = tuple(mask_2)
    return mask_1, mask_2

def Seeds(annotations):
    """
    construct shuflled mask for vessel and non vessel point. These will be
    seeds for the balancing of our datas

    Parameters
    ----------
    annotations : nd array
        annotation matrix with 1 for vessel, 2 for non vessel
        and 0 for unannoted.

    Returns
    -------
    seeds: list
        list of tuple of 1d array
        each tuple is a suffled mask for a class

    """
    
    seeds = [ShuffleMask((annotations == i+1).nonzero())
             for i in range(int(np.max(annotations)))]
        
        
    
    return seeds

def Balance(seeds, vessel_prop = 0.5, size = 2000):
    """
    construct a balanced dataset from vessel seed and non_vessel seed

    Parameters
    ----------
    seeds : list
        list of tuple of 1d array
        each tuple is a suffled mask of points belonging to a different 
        classes. The convention is that the first class is the class of
        interest. In our case, the vessels.
    vessel_prop : float, otional
        proportion of vessel points (first class points) in the balanced data 
        set. The default is 0.5
    size : int, optional
        size of the balanced data set. The default is 2000.

    Returns
    -------
    balanced_data_set : tuple
        tuple of 1d array a shuffled balanced dataset of points from each 
        classes.

    """
    n_classes = len(seeds)
    
    n_vessel = int(vessel_prop * size)
    n_non_vessel = int((size - n_vessel)/(n_classes-1))
    
    if len(seeds[0][0]) < n_vessel:
        print("Number of annoted points in class 0 for curent boxe too low for",
              " curent size:", len(seeds[0][0]), "<", n_vessel,
              " This will change final proportions.")
        set_0 = seeds[0]
    else:
        set_0 = ShortenMask(seeds[0], n_vessel)
    
    for i, seed in enumerate(seeds[1:]):
        if len(seed[0]) < n_non_vessel:
            print("Number of annoted points in class",i," for curent boxe too",
                  " low for curent size:", len(seed[0]), "<", n_non_vessel,
                  ". This will change final data proportions.")
            set_0 = JoinMask(set_0, seed)
        else:
            set_0 = JoinMask(set_0, ShortenMask(seed, n_non_vessel))
    
    balanced_data_set = ShuffleMask(set_0)
    
    return balanced_data_set

def Split(mask, n_split = 5):
    """
    split a single data set in several smaller set of same size

    Parameters
    ----------
    mask : tuple
        tuple of 1d array
        coordonate of points
    n_split : int, optional
        number of split. The default is 5

    Returns
    -------
    split_set : tuple
        tuple of masks.

    """
    set_size = int(len(mask[0])/n_split)
    
    split_set = []
    
    for i in range(n_split):
        m, mask = SplitMask(mask, set_size)
        split_set.append(m)
    split_set = tuple(split_set)
    return split_set

def Merge(mask_list):
    """
    merge several mask in one

    Parameters
    ----------
    mask_list : list or tuple
        list of mask.

    Returns
    -------
    mask : tuple
        tuple of 1d array
        coordonate of points

    """
    mask = ([], [], [])
    for el in mask_list:
        mask = JoinMask(mask, el)
    
    return mask

def Annotations(classes, mask=None, boxe_list=None):
    """
    construct an annotation map with 1 for vessel and 2 for non_vessel

    Parameters
    ----------
    classes : list
        list of numpy 3d array where each point annoted  as a member of a class
        i as a value of 1.
    mask : tuple, optional
           tuple of 1d array
           coordonate of points in the parenchyme mask
    boxe_list : list, optional
        list of tuple, masks of boxes in wich annotation point will be chosen

    Returns
    -------
    annotation : 3d array
        3d array where each point annoted  as a vessel has a value of 1 and
        each non vessel a value of 2. Only points in the intercection of 
        the mask and the union of all boxes are taken into acount

    """
    annotation = np.zeros(classes[0].shape)
    for i in range(len(classes)):
        annotation[np.nonzero(classes[i])] = i+1.0
        
    if mask is not None:
        annotation = annotation * (mask!=0.0)
    if boxe_list is not None:
        if type(boxe_list) is not list:
            boxe_list = [boxe_list]
        boxes = np.zeros(annotation.shape)
        for boxe in boxe_list:
            boxes = boxes + boxe
        annotation = annotation * (boxes!=0.0)
    return annotation

