import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import datasets
data_set=datasets.load_breast_cancer()
X,Y=data_set.data,data_set.target
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1234)

from logistic_regression import LogisticRegression
logistic_reg=LogisticRegression(lr=0.001,n_iters=1000)
logistic_reg.fit(X_train,Y_train)
predictions=logistic_reg.predict(X_test)

def accuracy(y_true,y_pred):
    accuracy=np.sum(y_true==y_pred)/len(y_true)
    return accuracy
print ('The accuracy of the model is:',accuracy(Y_test,predictions))   
    
