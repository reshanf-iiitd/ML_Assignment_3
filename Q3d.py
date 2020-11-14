#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:38:34 2020

@author: Waquar Shamsi
"""

#Q3 d
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.io import loadmat

file = 'dataset_b.mat'
mat = loadmat(file)
X= mat['samples']
y= mat['labels'][0]
df = pd.DataFrame(X,columns=list(range(X.shape[1])))
df['label']=y

X_train, X_test, y_train, y_test = train_test_split(df.drop(['label'],axis=1),df['label'].to_numpy(),test_size=0.2)
ovr = OneVsRestClassifier(SVC(kernel='rbf', decision_function_shape='ovr',C=1,gamma=1))
ovr.fit(X_train,y_train)
pred_ovr = ovr.predict(X_test)
print("Accuracy OVR:",accuracy_score(pred_ovr,y_test))


ovo = SVC(kernel='rbf', decision_function_shape='ovo',C=1,gamma=1)
ovo.fit(X_train,y_train)
pred_ovo = ovo.predict(X_test)
print("Accuracy OVO:",accuracy_score(pred_ovo,y_test))