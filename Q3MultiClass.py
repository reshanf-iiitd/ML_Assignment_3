#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:35:25 2020

@author: Waquar Shamsi
"""

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from itertools import combinations 
from statistics import mode

class mySVM(object):
  kernel_uu = 'rbf'
  u_C=10
  ga=0.0
  def __init__(self,C,gamma,kernel):
    mySVM.kernel_uu=kernel
    mySVM.u_C=C
    mySVM.ga=gamma

  def simple_fit(self,X,y):
    clf=SVC(kernel=mySVM.kernel_uu,C=mySVM.u_C,gamma=mySVM.ga)
    self.clf=clf.fit(X,y)
    return self.clf 
  
  def simple_predict(self,X_test):
    y_predicted = self.clf.decision_function(X_test)
    for x in range(len(y_predicted)):
      if(y_predicted[x] > 0):
        y_predicted[x] = 1
      else :
        y_predicted[x] = 0 
    return y_predicted 

class MSVM:
  def __init__(self,ty='ovr',C=10,gamma=0.2,kernel='rbf'):
    self.ty=ty
    self.C = C
    self.gamma=gamma
    self.kernel=kernel

  def fit(self,X,y):
    if self.ty=='ovr':
      classes = np.unique(y)
      self.classes = classes
      models = [mySVM(self.C,self.gamma,self.kernel) for i in range(len(classes))]
      df = pd.DataFrame(X,columns=list(range(X.shape[1])))
      df['label']=y
      i=0
      for c in classes:
        main_class = c
        other_class = np.delete(classes,c)
        temp_df = df.copy()
        temp_df['label'] = temp_df['label'].replace(c,-1)
        temp_df['label'] = temp_df['label'].replace([other_class],0)
        temp_df['label'] = temp_df['label'].replace(-1,1)
        models[i].simple_fit(temp_df.drop(['label'],axis=1),temp_df.filter(['label']))
        i+=1
      self.models = models

    elif self.ty=='ovo':
      classes = np.unique(y)
      num_models = (classes*(classes-1))/2
      self.num_models = num_models
      combos = combinations(classes, 2) 
      combos = list(combos)
      df = pd.DataFrame(X,columns=list(range(X.shape[1])))
      df['label']=y
      models = [mySVM(self.C,self.gamma,self.kernel) for i in range(len(num_models))]
      i=0
      self.combos = combos
      for comb in combos:

        temp_df = df.copy()
        t_df = temp_df[(temp_df['label'] == comb[0]) | (temp_df['label'] == comb[1])]
        models[i].simple_fit(t_df.drop(['label'],axis=1),t_df.filter(['label']))
        i+=1
      self.models = models

  def predict(self,X_test):
    if self.ty=='ovr':
      count_classes = len(self.classes)
      predictions = np.zeros(X_test.shape[0])
      i=0
      for mod in self.models:
        predict = mod.simple_predict(X_test)
        j=0
        for p in predict:
          if p==1:
            predictions[j]=i
          j+=1
        i+=1  
      return predictions

    elif self.ty=='ovo':
      i=0
      predictions = []
      for mod in self.models:
        pred = mod.simple_predict(X_test)
        c1=self.combos[i][0]
        c2=self.combos[i][1]
        for x in range(len(pred)):
          pred[x] = c2 if pred[x]==1 else c1 
        predictions.append(pred)
        i+=1
      final_preds = []
      predictions = np.array(predictions)

      for i in range(X_test.shape[0]):
        final_preds.append(mode(predictions[:,i]))
      final_preds = np.array(final_preds)
      return final_preds
    