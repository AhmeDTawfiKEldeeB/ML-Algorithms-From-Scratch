import numpy as np

class LogisticRegression:

    def __init__(self,lr,n_iters):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

    def fit(self,X,Y):
        #initialize parameters
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        #gradient descent
        for i in range(self.n_iters):
            linear_model=np.dot(X,self.weights)+self.bias
            y_predict=self._segmoid(linear_model)
            dw=(1/n_samples)*np.dot(X.T,(y_predict-Y))
            db=(1/n_samples)*np.sum(y_predict-Y)
            self.weights-=self.lr*dw
            self.bias-=self.lr*db

    def predict(self,X):
        linear_model=np.dot(X,self.weights)+self.bias
        y_predict=self._segmoid(linear_model)
        y_predict_class=[1 if i>0.5 else 0 for i in y_predict]
        return y_predict_class
        
    def _segmoid(self,s):
        return (1/(1+np.exp(-s)))