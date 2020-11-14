import pandas as pd
from tabulate import tabulate
import pandas as pdo
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
import plotly.express as px
from sklearn.metrics import zero_one_loss
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
import joblib
from sklearn import metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import userSVM




def test_20(X,y):
  m, n = X.shape
  X_test = X[:int(np.floor(m*0.2))]
  X_train = X[int(np.floor(m*0.2)):]

  y_test = y[:int(np.floor(m*0.2))]
  y_train = y[int(np.floor(m*0.2)):]

  return X_test,y_test,X_train,y_train





def Q1_3():
	########################### ANSWER 1-3

	mat1 = open('feature_descriptor_1',mode= 'rb')
	# print(mat1)
	data = pickle.load(mat1)
	# print(data.columns)
	# print(type(data))
	l2=np.array(data['label'])

	X=data.drop(['label'], axis = 1)
	print(l2)
	s2=np.array(X)
	print(s2.shape)
	X_test,y_test,X_train,y_train=test_20(s2,l2)
	n_ins,n_fea = X_train.shape
	xx=1/(n_fea*np.var(X_train))   ## GAMMA 
	print(xx)
	param_grid = [
	  {'C': [0.01, 1, 10,100],  
	              'gamma': [0.1, 0.01, xx,1], 
	              'kernel': ['rbf']}                            ########## xx = default gamma value , 
	                    ########### for Linear and Poly excluding gammam in grid search
	    ]


	grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
	  
	# fitting the model for grid search 
	grid.fit(X_train, y_train)

	print(grid.best_params_) 
	  
	# print how our model looks after hyper-parameter tuning 
	print(grid.best_estimator_) 
	pred = grid.predict(X_test) 
	joblib.dump(grid.best_estimator_, "gridd.pkl")

	print("Accuracy over Testing dataset",accuracy_score(y_test,pred))        ###### C = 10 gamma = 8.987756 ,kerenl = rbf
	pred1 = grid.predict(X_train)
	print("Accuracy over Training dataset",accuracy_score(y_train,pred1))

	return grid
	######################################## HOG ##################################








def Q1_4(grid):
	######################################### 1-4 ########################################

	index_supp=grid.best_estimator_.support_
	print(len(index_supp))
	print(len(X_train))
	newX_train = []
	newy_train = []
	for x in range(len(X_train)):
	  if x not in index_supp:
	    # print(x)
	    newX_train.append(X_train[x])
	    newy_train.append(y_train[x])
	newX_train=np.array(newX_train)
	newy_train=np.array(newy_train)

	print(len(newX_train))
	print(len(X_train) - len(index_supp))
	clf = svm.SVC(C=10,gamma=xx,kernel='rbf')
	clf.fit(newX_train,newy_train)
	pred14=clf.predict(X_test)
	print("Accuracy over Testing dataset",accuracy_score(y_test,pred14))        ###### C = 10 gamma = 8.987756 ,kerenl = rbf
	pred142 = clf.predict(newX_train)
	print("Accuracy over Training dataset",accuracy_score(newy_train,pred142))




def Q2_1():
	########################################### ANSWER 2-a ##################################################
	mat1 = sio.loadmat('dataset_a.mat')
	# print(mat1)
	l2=np.array(mat1['labels'][0])
	data2=[]
	s2=np.array(mat1['samples'])
	for i in range(len(s2)):
	  data2.append([s2[i][0],s2[i][1],int(l2[i])])
	rows=np.array(data2)
	columnNames=['x_value','y_value','label']
	dataframe = pd.DataFrame(data=rows, columns=columnNames)
	dataframe['label'] = dataframe['label'].astype(int)
	plt.figure(figsize=(10,6))
	sns.scatterplot(data=dataframe,x='x_value', y='y_value', hue='label',palette="deep")
	plt.legend(loc=4)
	plt.title("Scattered Plot of data",fontsize=20,color="w")
	plt.tight_layout()



def Q2_2():
	######################################################## 2-2 ##########################################

################################## GETTING THE HYPERPARAMETER
	mat1 = sio.loadmat('dataset_a.mat')
	# print(mat1)
	l2=np.array(mat1['labels'][0])
	s2=np.array(mat1['samples'])
	X_test,y_test,X_train,y_train=test_20(s2,l2)
	n_ins,n_fea = X_train.shape
	xx=1/(n_fea*np.var(X_train))   ## GAMMA 
	hyper1 = np.arange(0.01,1,.01)
	hyper2 = np.arange(0.0,1010.0,10.0)

	hyper2[0]=1.0
	hyper = np.concatenate((hyper1,hyper2))

	for x in hyper:
	  print(x)
	acc=[]
	for c in hyper:
	  clf1=userSVM("linear",c,xx)
	  clf=clf1.u_fit(X_train,y_train)
	  print(c)
	  pred1=clf1.predict_uu(X_test)
	  acc.append(accuracy_score(pred1,y_test))
	  print(accuracy_score(pred1,y_test))
	max_i = acc.index(max(acc))
	min_i = acc.index(min(acc))
	max_a = max(acc)
	min_a = min(acc)

# Third value of C
	xx=1/(n_fea*np.var(X_train))
	clf1=userSVM("linear",5,xx)
	clf=clf1.u_fit(X_train,y_train)
	pred1=clf1.predict_uu(X_test)
	rann = accuracy_score(pred1,y_test)
	print("Maximum accuracy Optimal C = ",max_a)
	print("Minimum accracy for poor C = ",min_a)
	print("Acuracy for random C = ",rann)
	print("C value for Maximum Accuracy = ",hyper[max_i])
	print("C value for Minimum Accuracy = ",hyper[min_i])
	print("Any Random C value = ",5)
	ccc=[hyper[min_i],hyper[max_i],5]
	print(ccc)

	#Plotting for these three differnt value of C. 
	for x in ccc:
	  xx=1/(n_fea*np.var(X_train))
	  clf1=userSVM("linear",x,xx)
	  clf=clf1.u_fit(X_train,y_train)
	  pred1=clf1.predict_uu(X_test)
	  plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap=plt.cm.Paired)

	# plot the decision function
	  plt.title("C = "+str(x))
	  ax = plt.gca()
	  xlim = ax.get_xlim()
	  ylim = ax.get_ylim()

	  # plot decision boundary and margins
	  ax.contour(XX, YY, Z, colors='navy', alpha=0.5,
	            linestyles=['--', '-', '--'])
	  # plot support vectors
	  ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
	            linewidth=1, facecolors='pink', edgecolors='k')
	  plt.show()





def Q2_3():
		######################################################## 2-2 ##########################################

	################################## GETTING THE HYPERPARAMETER
	mat1 = sio.loadmat('dataset_a.mat')
	# print(mat1)
	l2=np.array(mat1['labels'][0])
	s2=np.array(mat1['samples'])
	X_test,y_test,X_train,y_train=test_20(s2,l2)
	n_ins,n_fea = X_train.shape
	xx=1/(n_fea*np.var(X_train))   ## GAMMA 
	hyper1 = np.arange(0.01,1,.01)
	hyper2 = np.arange(0.0,1010.0,10.0)

	hyper2[0]=1.0                                      ## RANGE REFRENCE https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

	h_gamma=np.linspace(.01,100,200)
	hyper = np.concatenate((hyper1,hyper2))
	print(hyper[0])
	acc=np.zeros((200,200))

	for c in range(len(hyper)):
	  print(hyper[c])
	  for g in range(len(h_gamma)):
	    print(h_gamma[g])
	    clf1=userSVM("rbf",hyper[c],h_gamma[g])
	    clf=clf1.u_fit(X_train,y_train)
	    # print(c)
	    pred1=clf1.predict_uu(X_test)
	    acc[c][g] = accuracy_score(pred1,y_test)
	    print(accuracy_score(pred1,y_test))
	# max_i = acc.index(max(acc))
	# min_i = acc.index(min(acc))
	# max_a = max(acc)
	# min_a = min(acc)

	temp_max = 0
	temp_min = 2
	c_max_i=0
	c_min_i=0
	g_max_i=0
	g_min_i=0
	print(acc.shape)
	for i in range(0,200):
	  for j in range(0,200):
	    # print(acc[i][j])
	    if(temp_max <= acc[i][j]):
	      temp_max = acc[i][j]
	      c_max_i = i
	      g_max_i = j
	    if(temp_min >= acc[i][j]):
	      temp_min = acc[i][j]
	      c_min_i = i
	      g_min_i = j

	zz=1/(n_fea*np.var(X_train))
	cc1=[hyper[c_min_i],hyper[c_max_i],5]   #################### here # differnt hyper parmete baesed on ,1- poor 2 -optimal 3- random
	gg1=[h_gamma[g_min_i],h_gamma[g_max_i],5]
	print(cc)
	print(gg)
	print("Maximum accuracy =",temp_max)
	print("Minimum accuracy =",temp_min)
	print("C = {0} and Gamma = {1} value for Maximum accuracy ".format(hyper[c_max_i],h_gamma[g_max_i]))
	print("C = {0} and Gamma = {1} value for Minimum accuracy ".format(hyper[c_min_i],h_gamma[g_min_i]))

	# print("G value for Maximum accuracy  =",g_max_i)
	# print("G value for Minimum accuracy =",g_min_i)
	clf1=userSVM("rbf",hyper[c],h_gamma[g])
	clf=clf1.u_fit(X_train,y_train)
	# print(c)
	pred1=clf1.predict_uu(X_test)
	nnn = accuracy_score(pred1,y_test)
	print("Accuracy = {0} for random C = 5 and Gamma = 5 ".format(nnn))


	#Plotting for these three differnt value of C. 

	for x in cc:
	  for y in gg:

	    xx=1/(n_fea*np.var(X_train))
	    clf1=userSVM("rbf",x,y)
	    clf=clf1.u_fit(X_train,y_train)
	    pred1=clf1.predict_uu(X_test)
	    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap=plt.cm.Paired)
	    
	  # plot the decision function
	    ax = plt.gca()
	    xlim = ax.get_xlim()
	    ylim = ax.get_ylim()
	    plt.title("C = "+str(x)+ " and gamma = "+str(y))
	    # create grid to evaluate model
	    xx = np.linspace(xlim[0], xlim[1], 30)
	    yy = np.linspace(ylim[0], ylim[1], 30)
	    YY, XX = np.meshgrid(yy, xx)
	    xy = np.vstack([XX.ravel(), YY.ravel()]).T
	    Z = clf.decision_function(xy).reshape(XX.shape)

	    # plot decision boundary and margins
	    ax.contour(XX, YY, Z, colors='navy',levels=[-1,0,1] ,alpha=0.5,
	              linestyles=['--', '-', '--'])
	    # plot support vectors
	    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
	              linewidth=1, facecolors='pink', edgecolors='k')
	    print("Accuracy =", accuracy_score(pred1,y_test))
	    plt.show()
	u_clff2=userSVM("linear",.0000001,1)
	u_clf = u_clff2.u_fit(X_train,y_train)
	u_pred22=u_clff2.predict_uu(X_test)

	print("Accuracy for RBF kernel SVM with C = 1000 and gamma = 100 with user Defined SVM" ,accuracy_score(u_pred22,y_test))

def Q2_4():
	################################################# 2-4 #################################################

	mat1 = sio.loadmat('dataset_a.mat')
	# print(mat1)
	l2=np.array(mat1['labels'][0])
	s2=np.array(mat1['samples'])
	X_test,y_test,X_train,y_train=test_20(s2,l2)
	n_ins,n_fea = X_train.shape
	xx=1/(n_fea*np.var(X_train))   ## GAMMA 

	clf = svm.SVC(kernel = 'linear' , C =.14)  ############# From 2-2
	clf.fit(X_train,y_train)
	pred1 = clf.predict(X_test)
	print("Accuracy for Linear SVM with C = 0.14 with sklearn" ,accuracy_score(pred1,y_test))

	u_clff1=userSVM('linear',.14,xx)
	u_clf=u_clff1.u_fit(X_train,y_train)
	u_pred1=u_clff1.predict_uu(X_test)

	print("Accuracy for Linear SVM with C = 0.14 with user Defined SVM" ,accuracy_score(u_pred1,y_test))

	clf = svm.SVC(kernel = 'rbf' , C =1000,gamma=100)  ############# From 2-3
	clf.fit(X_train,y_train)
	pred2 = clf.predict(X_test)
	print("Accuracy for RBF kernel SVM with C = 1000 and gamma =100 with sklearn" ,accuracy_score(pred2,y_test))

	u_clff2=userSVM('rbf',1000,100)
	clf = u_clff2.u_fit(X_train,y_train)
	u_pred2=u_clff2.predict_uu(X_test)

	print("Accuracy for RBF kernel SVM with C = 1000 and gamma = 100 with user Defined SVM" ,accuracy_score(u_pred2,y_test))
















if __name__ == "__main__":
    print("PhD19006")
    grid = Q1_3()
    # Q1_4(grid)
    # Q2_1()
    # Q2_2()
    # Q2_3()
    # Q2_4()
    
