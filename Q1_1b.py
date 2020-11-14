#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:54:40 2020

@author: Waquar Shamsi
"""

from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd

#LOAD SAVED DATA
hist_f = open ("color_hist_df", "rb")
color_hist_df = pickle.load(hist_f)
hog_f = open ("df_hog2", "rb")
df_hog = pickle.load(hog_f)

y=color_hist_df['label'].to_numpy()
X_hist=color_hist_df.drop(['label'],axis=1)
X_hog=df_hog.drop(['label'],axis=1)
X_hist = X_hist.to_numpy()
X_hog=X_hog.to_numpy()
# NORMALIZE COLOR-HISTOGRAM
norm = MinMaxScaler().fit(X_hist)
X_hist = norm.transform(X_hist)

#CONCATENATE COLOR_HISTOGRAM AND HOG
X_final = np.concatenate((X_hist,X_hog),axis=1)

#ADD LABELS AND CREATE DATAFRAME
df_com = pd.DataFrame(X_final,columns=list(range(X_final.shape[1])))
df_com['label']=y

# #PRINT SHAPES
# print(X_hist.shape)
# print(X_hog.shape)
# print(X_final.shape)

#SAVE FEATURE DESCRIPTOR 2
with open('=df_com2', 'wb') as f:
   pickle.dump(df_com, f)