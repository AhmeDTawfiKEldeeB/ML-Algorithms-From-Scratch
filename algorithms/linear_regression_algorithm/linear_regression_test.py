import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
X,Y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1234)

from linear_regression import LinearRegression
regressor=LinearRegression(lr=0.1,n_iterations=1000)
regressor.fit(X_train,Y_train)
predictions=regressor.predict(X_test)

def MSE(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

mse_value=MSE(Y_test,predictions)
print("MSE Value is : ", mse_value)