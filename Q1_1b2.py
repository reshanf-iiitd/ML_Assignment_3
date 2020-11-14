#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:52:35 2020

@author: triste
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2 
import pandas as pd
import pickle

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


X = X.reshape(50000,32,32,3) #reshape, 500000 images, of size 32x32=1024 rgb
bin = [0,20,40,60,80,100,120,140,160,180]
hues = []
hists = []
for i in range(0,50000):
  hsv_image = cv2.cvtColor(X[i],cv2.COLOR_BGR2HSV)
  hue =  hsv_image [:,:, 0 ]
  hues.append(hue)
  hist,bins = np.histogram(hue,bins=bin)
  hists.append(hist)

hues = np.array(hues)
hists = np.array(hists)

color_hist_df = pd.DataFrame(hists,columns=list(range(hists.shape[1])))
color_hist_df['label'] = y

with open('color_hist_df', 'wb') as f:
   pickle.dump(color_hist_df, f)