
########################### ANSWER 1-3
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

mat1 = open('/content/drive/My Drive/Colab Notebooks/df_com',mode= 'rb')
# print(mat1)
data = pickle.load(mat1)
# print(data.columns)
# print(type(data))
l2=np.array(data['label'])

X=data.drop(['label'], axis = 1)
print(l2)

s2=np.array(X)
print(s2.shape)
X_train,X_test,y_train,y_test=train_test_split(s2,l2,test_size=0.2)
n_ins,n_fea = X_train.shape
xx=1/(n_fea*np.var(X_train))   ## GAMMA 
print(xx)




param_grid = [
  {'C': [0.01, 1, 10,100],  
	              'gamma': [0.1, 0.01, xx,1], 
	              'kernel': ['rbf']}                            
                  
    ]


grid = GridSearchCV(svm.SVC(), param_grid) 

# fitting the model for grid search 
grid.fit(X_train, y_train)

print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
pred = grid.predict(X_test) 
print(accuracy_score(pred,y_test))
pred = grid.predict(X_train) 
print(accuracy_score(pred,y_train))
# with open('/content/drive/My Drive/Colab Notebooks/grid_cv_hoghsv', 'wb') as f:
#    pickle.dump(grid, f)

# filename = 'griddd.sav'
# pickle.dump(grid, open(griddd, 'wb'))