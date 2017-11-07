import numpy as np
from sklearn import datasets, neighbors
from knn import KNN
from linereg import line_reg

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
np.unique(iris_Y)

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_Y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_Y[indices[-10:]]

print ('Choose Method :')
print ('1. K-Nearest Neighbour')
print ('2. Decision Tree')
print ('3. Linear Regression')
method = raw_input()

if method == '1':
    print str(KNN(iris))
if method == '2':
    print str(line_reg(iris))



