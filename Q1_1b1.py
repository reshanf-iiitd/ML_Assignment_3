#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:50:37 2020

@author: Waquar Shamsi
"""

####    HOG   ####
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2 
import pandas as pd
import math
import pickle
from hog_class import my_hog

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

X = X.reshape(50000,32,32,3)



#APPLY HOG
hog = my_hog()
hog_images = []
for im in X:
  image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  hog_img = hog.hog_fun(image)
  hog_images.append(hog_img)
hog_images_X = np.array(hog_images) 
# print(hog_images_X.shape)
df_hog = pd.DataFrame(hog_images_X,columns=list(range(hog_images_X.shape[1])))
df_hog['label'] = y

with open('df_hog2', 'wb') as f:
   pickle.dump(df_hog, f)