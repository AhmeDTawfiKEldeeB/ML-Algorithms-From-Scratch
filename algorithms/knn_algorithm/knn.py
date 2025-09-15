
import numpy as np

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    
    def __init__(self,k) -> None:
        self.k=k

    def fit(self,X,Y):
        self.X_train=X
        self.Y_train=Y

    def predict(self,X):
        predictions=[self._predict(x) for x in X]
        return predictions
    def _predict(self,x):
        distance=[euclidean_distance(x,x_train)for x_train in self.X_train]  
<<<<<<< HEAD
        k_indices=np.argsort(distance)[:self.k]
        k_nearest_labels=[self.Y_train[i] for i in k_indices]
=======
        
        k_indices=np.argsort(distance)[:self.k]
        k_nearest_labels=[self.Y_train[i] for i in k_indices]
        
>>>>>>> 6ff06f079b3a5e08a5ca4f5beb3101d3218fdb3a
        most_common=np.bincount(k_nearest_labels).argmax()
        return most_common





