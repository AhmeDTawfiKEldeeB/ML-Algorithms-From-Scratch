import numpy as np
from collections import Counter
from decision_tree import DecisionTree

def bootstrap_sample(X,Y):
    n_samples=X.shape[0]
    idxs=np.random.choice(n_samples,size=n_samples,replace=True)
    return X[idxs],Y[idxs]
def most_common(Y):
    counter=Counter(Y)
    most_common=counter.most_common(1)[0][0]
    return most_common 
class RandomForest:
    def __init__(self,n_trees=100,max_depth=10,min_samples_split=2,n_feats=None):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_feats=n_feats
        self.trees=[]
        
    def fit(self,X,Y):
        self.trees=[]
        for _ in range(self.n_trees):
            tree=DecisionTree(max_depth=self.max_depth,min_samples_split=self.min_samples_split,n_feats=self.n_feats)
            X_sample,Y_sample=bootstrap_sample(X,Y)
            tree.fit(X_sample,Y_sample)
            self.trees.append(tree)
    def predict(self,X):
        tree_preds=[tree.predict(X) for tree in self.trees]
        tree_preds=np.swapaxes(tree_preds,1,0)  
        model_pred=np.array([most_common(tree_pred) for tree_pred in tree_preds])
        return model_pred      

        