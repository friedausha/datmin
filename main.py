import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import datasets
from classifier import dectree, KNN, NeuralNetwork

iris = datasets.load_breast_cancer()
iris_X = iris.data
iris_Y = iris.target
sum_KNN = 0.0
sum_dectree = 0.0
sum_NN = 0.0
sum_recall_knn = 0.0
sum_spec_knn = 0.0
sum_recall_dec = 0.0
sum_spec_dec = 0.0
sum_recall_nn = 0.0
sum_spec_nn = 0.0

sizecv = 5
kf = StratifiedKFold(n_splits=sizecv, shuffle=True, random_state=123)
for train, test in kf.split(iris_X, iris_Y):
    iris_x_train = iris_X[train]
    iris_y_train = iris_Y[train]
    iris_x_test = iris_X[test]
    iris_y_test = iris_Y[test]
    #print ('KNN')
    sumtemp = []
    val_KNN , sumtemp = KNN(iris_x_train, iris_x_test, iris_y_train, iris_y_test)
    sum_KNN += val_KNN
    sum_recall_knn += sumtemp[0]
    sum_spec_knn += sumtemp[1]
    # print ('Accuracy score of knn :') + str(val_KNN)
    # print ('')

    #print ('Decision Tree')
    sumtemp = []
    val_dectree, sumtemp = dectree(iris_x_train, iris_x_test, iris_y_train, iris_y_test)
    sum_dectree += val_dectree
    sum_recall_dec += sumtemp[0]
    sum_spec_dec += sumtemp[1]
    # print ('Accuracy score of decision tree :') + str(val_dectree)
    # print ('')

    # print ('Neural Network')
    sumtemp = []
    val_NN, sumtemp = NeuralNetwork(iris_x_train, iris_x_test, iris_y_train, iris_y_test)
    sum_NN += val_NN
    sum_recall_nn += sumtemp[0]
    sum_spec_nn += sumtemp[1]
    # print ('Accuracy score of decision tree :') + str(val_NN)
    # print ('')
print ('KNN :')
print ('Average sensitivity : ') + str(sum_recall_knn/sizecv)
print ('Average specificity : ') + str(sum_spec_knn/sizecv)
print ('Average score of    : ') + str(sum_KNN / sizecv)
print (' ')
print ('Decision Tree :')
print ('Average sensitivity : ') + str(sum_recall_dec/sizecv)
print ('Average specificity : ') + str(sum_spec_dec/sizecv)
print ('Average score of decision tree : ') + str(sum_dectree / sizecv)
print ('')
print ('Neural Network :')
print ('Average sensitivity : ') + str(sum_recall_nn/sizecv)
print ('Average specificity : ') + str(sum_spec_nn/sizecv)
print ('Average score of Neural Network: ') + str(sum_NN / sizecv)
