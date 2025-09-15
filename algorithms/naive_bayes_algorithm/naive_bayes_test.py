import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, Y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=123
    )
from naive_bayes import NaiveBayes
nb = NaiveBayes()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracy(Y_test, predictions))