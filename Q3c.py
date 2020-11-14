#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:31:31 2020

@author: Waquar Shamsi
"""
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from kfold import kfold
from Q3MultiClass import MSVM
from Q3MultiClass import mySVM
from itertools import combinations 
from statistics import mode

file = 'dataset_b.mat'
mat = loadmat(file)
X= mat['samples']
y= mat['labels'][0]
df = pd.DataFrame(X,columns=list(range(X.shape[1])))
df['label']=y

def measure_accuracy(x, y):
  '''
     It is used to calculate accuracy for classification problems
  '''
  count = 0
  total = len(y)
  for i in range(len(x)):
    if x[i] == y[i]:
      count += 1
  return count / float(total) # RETURN ACCURACY
 

def grid_search(model, params, folds):
  '''plt.plot(C,C_accuracies)
  plt.title("gama vs Accuracy for gamma:"+str(gama))
  plt.xlabel("C")
  plt.ylabel("Accuracy")
  plt.show()
  Grid Search takes four arguments, model which can be 'ovr' or 'ovo',
  params is a dictionary with keys gammas and C,
  folds gives the number of folds,
  and type of kernel
  '''
  gammas = params['gamma']
  C = params['C']
  kernel = params['kernel']
  folds = kfold(df,5,True)
  max_acc = 0
  best_C = None
  best_gamma = None
  #  gama_accuracies = []
  for gama in gammas:
      C_accuracies =[]
      for cs in C:
        accuracies = []
        i=0
        for fold in folds:
          test_fold_df = df.iloc[fold,:]    # Create a dataframe with with the index values which is for the fold
          #GET X AND y FROM DATAFRAME
          train_fold_df = df.drop(fold, axis=0) # get all rows in training set for fold which are not in test set for the fold
          #GET TRAINING X AND y
          X_train = train_fold_df.drop(['label'],axis=1)
          y_train = train_fold_df.filter(['label']).to_numpy()
          #GET TESTING X AND y
          X_test = test_fold_df.drop(['label'],axis=1)
          y_test = test_fold_df.filter(['label']).to_numpy()
          sv = MSVM(model,cs,gama,kernel)
          sv.fit(X_train,y_train) #FITS THE MODEL
          pred = sv.predict(X_test) #MAKE PREDICTIONS
          acc = measure_accuracy(y_test,pred)# MEASURES ACCURACY USING USER DEFINED FUNCTION
          accuracies.append(acc)  
          print("Accuracy for Gamma:",gama," and C:",cs," and Fold: ",i+1," is:",acc)
          i+=1
        accuracies = np.array(accuracies) #NOW CALCULATE MEAN ACCURACY FOR ALL FOLDS and GIVEN GAMMA AND C
        print("MEAN ACCURACY FOR GAMMA:",gama," and C:",cs," is ",np.mean(accuracies))
        if max_acc < np.mean(accuracies):
          max_acc = np.mean(accuracies)
          best_C = cs
          best_gamma = gama
        C_accuracies.append(np.mean(accuracies)) 
      plt.plot(C,C_accuracies)
      plt.title("C vs Accuracy for gamma:"+str(gama))
      plt.xlabel("C")
      plt.ylabel("Accuracy")
      plt.show()
  print("BEST ACCURACY: ",max_acc," FOR C:",best_C," AND GAMMA:",best_gamma)
grid_search('ovo',{'gamma':[0.2,1],'C':[1,10],'kernel':'rbf'},folds=5)
