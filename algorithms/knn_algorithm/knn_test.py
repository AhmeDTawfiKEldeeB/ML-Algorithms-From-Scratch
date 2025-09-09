from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
X_train,X_test,Y_train,Y_test=train_test_split(iris.data ,iris.target ,random_state=0,test_size=0.2)
from knn import KNN
clf=KNN(k=5)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print(accuracy_score(Y_test,y_pred))