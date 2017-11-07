import numpy as np
from sklearn import linear_model

def line_reg(iris):
    np.random.seed(0)
    iris_X = iris.data
    iris_Y = iris.target
    indices = np.random.permutation(len(iris_X))

    iris_x_train = iris_X[indices[:-10]]
    iris_y_train = iris_Y[indices[:-10]]
    iris_x_test = iris_X[indices[-10:]]
    iris_y_test = iris_Y[indices[-10:]]
    np.unique(iris_Y)

    regr = linear_model.LinearRegression()
    regr.fit(iris_x_train, iris_y_train)
    print regr.score(iris_x_test, iris_y_test)
    return regr.predict(iris_x_test)