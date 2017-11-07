from sklearn import tree
from sklearn.metrics import accuracy_score

def dectree(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    dec = tree.DecisionTreeClassifier()
    dec.fit(iris_x_train, iris_y_train)
    predicted =  dec.predict(iris_x_test)
    return accuracy_score(iris_y_test, predicted)