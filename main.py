import numpy as np
from sklearn import datasets, neighbors
from knn import KNN
from linereg import line_reg

iris = datasets.load_iris()

print ('Choose Method :')
print ('1. K-Nearest Neighbour')
print ('2. Decision Tree')
print ('3. Linear Regression')
method = raw_input()

if method == '1':
    print str(KNN(iris))
if method == '2':
    print str(line_reg(iris))



