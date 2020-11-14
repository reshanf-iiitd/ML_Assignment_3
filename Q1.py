#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 01:08:34 2020

@author: Waquar Shamsi
"""

import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # implement from scratch
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
pca.fit_transform(standardized_x)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
print('Number of Components such that 90% of the total variance is retained = ', pca.n_components_)

p = PCA(n_components=pca.n_components_)
pca_X = p.fit_transform(standardized_x)

with open('feature_descriptor_1', 'wb') as f:
   pickle.dump(pca_X, f)


# pca_X_pickle = open ("feature_descriptor_1", "rb")
# pca_X = pickle.load(pca_X_pickle)
