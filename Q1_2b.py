#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:55:09 2020

@author: Waquar Shamsi
"""
from scipy.io import loadmat
import numpy as np
from sklearn.manifold import TSNE
from sklearn import manifold
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle


#LOAD SAVED PCA
hist_hog_f = open ("df_com", "rb")
hist_hog = pickle.load(hist_hog_f)

y = hist_hog['label']

tsne = manifold.TSNE(n_components=2)
X_t = tsne.fit_transform(hist_hog.drop(['label'],axis=1))

tsne_df = pd.DataFrame(data = X_t, columns = ["X","Y"]) 
tsne_df['label'] = y

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="X", y="Y",
    hue="label",
    palette=sns.color_palette("hls", len(np.unique(tsne_df['label']))),
    data=tsne_df,
    legend="full",
    alpha=0.5
)

with open('tsne_hog_hist', 'wb') as f:
   pickle.dump(tsne_df, f)


