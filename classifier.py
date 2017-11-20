from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import neighbors
from operator import truediv
from sklearn.metrics import recall_score, average_precision_score

def measure(y_actual, y_predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # print y_actual
    # print y_predicted
    #print str(len(y_actual))
    print str(len(y_predicted))
    for j in range(3):
        for i in range(len(y_predicted)):
            if y_actual[i]==y_predicted[i]==j:
               TP += 1
        for i in range(len(y_predicted)):
            if y_predicted[i]==j and y_actual[i]==y_predicted[i]:
               FP += 1
        for i in range(len(y_predicted)):
            if y_actual[i]==y_predicted[i]!=j:
               TN += 1
        for i in range(len(y_predicted)):
            if y_predicted[i]!=j and y_actual[i]==y_predicted[i]:
               FN += 1
    TP = truediv(TP, len(y_predicted))/3*100
    FN = truediv(FN, len(y_predicted))/3*100
    print 'The sensitivity percentage is : ' + str(TP)
    print 'The specificity percentage is : ' + str(FN)

def dectree(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    dec = tree.DecisionTreeClassifier()
    dec.fit(iris_x_train, iris_y_train)
    predicted =  dec.predict(iris_x_test)
    measure(iris_y_test, predicted)
    #print str(sum(recall_score(iris_y_test, predicted, average=None))/3)
    return accuracy_score(iris_y_test, predicted)*100

def line_reg(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    regr = linear_model.LinearRegression()
    regr.fit(iris_x_train, iris_y_train)
    predicted = regr.predict(iris_x_test)
    return regr.score(iris_x_test, iris_y_test)*100

def KNN(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    knn = neighbors.KNeighborsClassifier()
    knn.fit(iris_x_train, iris_y_train)
    predicted = knn.predict(iris_x_test)
    measure(iris_y_test, predicted)
    return knn.score(iris_x_test, iris_y_test)*100