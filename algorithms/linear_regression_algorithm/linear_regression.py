import numpy as np

class LinearRegression:
    def __init__(self,lr,n_iterations):
        self.lr=lr
        self.n_iterations=n_iterations
        self.weights=None
        self.bias=None

    def fit(self,X,Y):
<<<<<<< HEAD
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
=======

        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        
>>>>>>> 6ff06f079b3a5e08a5ca4f5beb3101d3218fdb3a
        for i in range(self.n_iterations):
            y_predict=np.dot(X,self.weights)+self.bias
            dw=(1/n_samples)*np.dot(X.T,(y_predict-Y))
            db=(1/n_samples)*np.sum(y_predict-Y)    
            self.weights-=self.lr*dw
            self.bias-=self.lr*db

    def predict(self,X):
            y_predict=np.dot(X,self.weights)+self.bias
            return y_predict
