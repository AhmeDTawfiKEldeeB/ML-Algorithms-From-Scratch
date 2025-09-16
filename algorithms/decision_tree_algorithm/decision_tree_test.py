import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

data = datasets.load_breast_cancer()
X, Y = data.data, data.target

X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1234
    )
from decision_tree import DecisionTree
clf = DecisionTree(max_depth=10)
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)
acc = accuracy(Y_test, y_pred)

print("Accuracy:", acc)