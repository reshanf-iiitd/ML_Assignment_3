#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Waquar Shamsi(MT20073)
"""

### CODE FOR K FOLD SPLIT ###
import random
import pandas as pd
import numpy as np

# kfold split is costly for very large datasets, for such cases simple test_train split is feasible
def kfold(dataset, folds, shuffle=True):
  '''
    arguments:
    dataset = [pandas dataframe] takes in a pandas dataframe
    fold    = [integer] number of folds to be generated
    shuffle   = [True/False] should the samples be picked at random 

    -----
    returns : a list of folds, each fold being a list of indices in that fold
  '''
  all_folds = []
  len_of_dataset = len(dataset.index)
  items_per_fold = int(len_of_dataset/folds)
  indices = list(dataset.index)
  if shuffle == True:     #if shuffled = True: get random samples in each fold
    for _ in range(folds):      #each iterations fills a fold
      one_fold = []             #create a empty fold
      for _ in range(items_per_fold):             # add items to a fold
        selected_index = random.choice(indices)   # select one random index
        one_fold.append(selected_index)           # append the selected index to the fold
        indices.remove(selected_index)            # remove that index from availbale choices
      all_folds.append(one_fold)                  # append the fold to the list of folds
  else:                         # if the dataset must not be shuffled
    for i in range(folds):                                              #each iterations fills a fold
      one_fold = indices[(i*items_per_fold):((i+1)*items_per_fold)]             #slice the list to get the fold
      all_folds.append(one_fold)
      
  return all_folds



#ask if their kfold implementation returned indices or dataframes of folds


