from collections import Counter
import numpy as np

def entropy(Y):
    
    hist=np.bincount(Y)
    ps=hist/len(Y)
    return -np.sum([p*np.log2(p) for p in ps if p>0 ])

class Node:

    def __init__(self,features=None,threshold=None,left=None,right=None,*,value=None) :

        self.features=features
        self.threshold=threshold
        self.left=left
        self.right=right 
        self.value=value

    def is_leaf_node(self):

        return self.value is not None

class DecisionTree:

    def __init__(self,min_samples_split=2,max_depth=100,n_feats=None):

        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_feats=n_feats
        self.root=None

    def fit(self,X,Y):

        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, Y) 

    def _grow_tree(self,X,Y,depth=0):

        n_samples,n_features=X.shape
        n_labels=len(np.unique(Y))

        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value=self._most_common_label(Y)
            return Node(value=leaf_value)

        feats_idxs=np.random.choice(n_features,self.n_feats,replace=False)

        best_feat,best_thresh=self._best_criteria(X,Y,feats_idxs) 
        left_idxs,right_idxs=self._split(X[:,best_feat],best_thresh)
        left=self._grow_tree(X[left_idxs,:],Y[left_idxs],depth+1)
        right=self._grow_tree(X[right_idxs,:],Y[right_idxs],depth+1)
        return Node(best_feat,best_thresh,left,right)

    def _best_criteria(self,X,Y,feats_idxs):

        best_gain=-1
        split_idx,split_thresh= None,None
        for feat_idx in feats_idxs:
            X_columns=X[:,feat_idx]
            thresholds=np.unique(X_columns)
            for threshold in thresholds:
                gain=self._information_gain(Y,X_columns,threshold)
                if gain>best_gain:
                    best_gain=gain
                    split_idx=feat_idx
                    split_thresh=threshold
        return split_idx,split_thresh

    def _information_gain(self,Y,X_columns,split_thresh):

        parent_entropy=entropy(Y)               
        left_idxs,right_idxs=self._split(X_columns,split_thresh)
        if len(left_idxs) ==0 or len(right_idxs) ==0 :
            return 0

        n=len(Y)
        n_l,n_r=len(left_idxs),len(right_idxs)
        e_l,e_r=entropy(Y[left_idxs]),entropy(Y[right_idxs])
        child_entropy=(n_l/n)*e_l+(n_r/n)*e_r
        return parent_entropy - child_entropy    

    def _split(self,X_columns,split_thresh):
        left_idxs=np.argwhere(X_columns<=split_thresh).flatten()
        right_idxs=np.argwhere(X_columns>split_thresh).flatten()
        return left_idxs,right_idxs    

    def _most_common_label(self,Y):

        if len(Y) == 0:
            return 0  # Default value for empty arrays
        count=Counter(Y)
        most_common=count.most_common(1)[0][0]
        return most_common

    def predict(self,X):

        return np.array([self._traverse_tree(x,self.root) for x in X])

    def _traverse_tree(self,x,node):

            if node.is_leaf_node():
                return node.value
            if x[node.features]<=node.threshold:
                return self._traverse_tree(x,node.left)
            return self._traverse_tree(x,node.right)        

