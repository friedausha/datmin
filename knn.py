import numpy as np
import pylab as pl
from sklearn import neighbors

def KNN(iris):
    np.random.seed(0)
    iris_X = iris.data
    iris_Y = iris.target
    indices = np.random.permutation(len(iris_X))

    iris_x_train = iris_X[indices[:-10]]
    iris_y_train = iris_Y[indices[:-10]]
    iris_x_test = iris_X[indices[-10:]]
    iris_y_test = iris_Y[indices[-10:]]
    np.unique(iris_Y)

    knn = neighbors.KNeighborsClassifier()
    knn.fit(iris_x_train, iris_y_train)
    return knn.predict(iris_x_test)
