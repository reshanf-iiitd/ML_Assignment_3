#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:48:56 2020

@author: Waquar Shamsi
"""

import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # implement from scratch
from sklearn import preprocessing
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#LOAD DATA
num_batches=5
filename='data_batch_'
for i in range(num_batches):
  file_full = 'cifar-10-batches-mat/'+filename+str(i+1)+'.mat'
  mat = loadmat(file_full) # load mat file
  print("MAT FILE ",i+1, "LOADED")
  if i==0:  # if first batch then initialize X and y
    X = mat['data']
    y = mat['labels']
  else:     # for remaining batches, append to X and y
    X = np.append(X,mat['data'])
    y = np.append(y,mat['labels'])

df = pd.DataFrame(X.reshape(50000,3072),columns=list(range(1,3072+1)))
df['label'] = y

standardized_x = StandardScaler().fit_transform(df.drop(['label'],axis=1))
pca = PCA(.90)

pca.fit_transform(df.drop(['label'],axis=1))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
print('Number of Components such that 90% of the total variance is retained = ', pca.n_components_)


p = PCA(n_components=pca.n_components_)
pca_X = p.fit_transform(df.drop(['label'],axis=1))
pc = pd.DataFrame(pca_X,columns=list(range(pca_X.shape[1])))
pc['label'] = y

#SAVE THE FEATURE DESCRIPTOR
with open('feature_descriptor_1', 'wb') as f:
   pickle.dump(pc, f)


  #ALSO STORE LABELS WITH X
  
# pca_X_pickle = open ("feature_descriptor_1", "rb")
# pc = pickle.load(pca_X_pickle)
