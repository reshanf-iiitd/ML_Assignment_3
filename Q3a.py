#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:50:47 2020

@author: Waquar Shamsi
"""
#IMPORTS
from scipy.io import loadmat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#LOAD DATA
mat = loadmat('dataset_b.mat')
X= mat['samples']   #FEATURES
y= np.array(mat['labels'])[0] #LABELS
#print(X,y) #2 features, 1 label

#DATA ANALYSIS
print('UNIQUE CLASS LABELS:\t', np.unique(y))   # Three Class Labels Found: [0 1 2]

#CREATE DATAFRAME
df = pd.DataFrame(X,columns=('A','B'))
df['T'] = y
#print(df.head(10))

#PLOT SCATTERPLOT
sns.scatterplot(
    data=df, x="A", y="B", hue="T", size="T",
    sizes=(20, 200), hue_norm=(0, 7), legend="full"
)

#Rereference: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf