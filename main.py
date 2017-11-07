import numpy as np
from sklearn import datasets
import random
from knn import KNN
from linereg import line_reg
from dectree import dectree

iris = datasets.load_iris()

np.random.seed(0)
iris_X = iris.data
iris_Y = iris.target
indices = np.random.permutation(len(iris_X))
rnd = random.randint(20, 100)

iris_x_train = iris_X[indices[:-rnd]]
iris_y_train = iris_Y[indices[:-rnd]]
iris_x_test = iris_X[indices[-rnd:]]
iris_y_test = iris_Y[indices[-rnd:]]
np.unique(iris_Y)

print ('Choose Method :')
print ('1. K-Nearest Neighbour')
print ('2. Decision Tree')
print ('3. Linear Regression')
method = raw_input()

if method == '1':
    print str(KNN(iris_x_train, iris_x_test, iris_y_train, iris_y_test))
if method == '2':
    print str(dectree(iris_x_train, iris_x_test, iris_y_train, iris_y_test))
if method == '3':
    print str(line_reg(iris_x_train, iris_x_test, iris_y_train, iris_y_test))


