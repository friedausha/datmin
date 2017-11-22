from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from operator import truediv
from sklearn.neural_network import MLPClassifier
import numpy as np

def measure(y_actual, y_predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    clazz = len(np.unique(y_predicted))
    for j in range(clazz):
        for i in range(len(y_predicted)):
            if y_actual[i] == y_predicted[i] == j:
                TP += 1
        for i in range(len(y_predicted)):
            if y_predicted[i] == j and y_actual[i] == y_predicted[i]:
                FP += 1
        for i in range(len(y_predicted)):
            if y_actual[i] == y_predicted[i] != j:
                TN += 1
        for i in range(len(y_predicted)):
            if y_predicted[i] != j and y_actual[i] == y_predicted[i]:
                FN += 1
    TP = truediv(TP, len(y_predicted)) / clazz * 100
    FN = truediv(FN, len(y_predicted)) / clazz * 100
    return TP, FN
    # print 'The sensitivity percentage is : ' + str(TP)
    # print 'The specificity percentage is : ' + str(FN)

def dectree(iris_x_train, iris_x_test, iris_y_train, iris_y_test):
    dec = tree.DecisionTreeClassifier()
    dec.fit(iris_x_train, iris_y_train)
    predicted = dec.predict(iris_x_test)
    return accuracy_score(iris_y_test, predicted) * 100, measure(iris_y_test, predicted)

def KNN(iris_x_train, iris_x_test, iris_y_train, iris_y_test):
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(iris_x_train, iris_y_train)
    predicted = knn.predict(iris_x_test)
    return knn.score(iris_x_test, iris_y_test) * 100 , measure(iris_y_test, predicted)


def NeuralNetwork(iris_x_train, iris_x_test, iris_y_train, iris_y_test):
    nn = MLPClassifier()
    nn.fit(iris_x_train, iris_y_train)
    predicted = nn.predict(iris_x_test)
    return nn.score(iris_x_test, iris_y_test) * 100 , measure(iris_y_test, predicted)
