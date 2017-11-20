import numpy as np
from sklearn import datasets
import random
from classifier import line_reg, dectree, KNN

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

print ('KNN')
print ('Accuracy score of knn :') + str(KNN(iris_x_train, iris_x_test, iris_y_train, iris_y_test))
print ('')
print ('Decision Tree')
print ('Accuracy score of decision tree :') + str(dectree(iris_x_train, iris_x_test, iris_y_train, iris_y_test))
print ('')
print ('Linear Regression')
print ('Accuracy score of linear regression :') + str(line_reg(iris_x_train, iris_x_test, iris_y_train, iris_y_test))
print ('')
