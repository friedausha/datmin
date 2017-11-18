from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import recall_score, average_precision_score

def dectree(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    dec = tree.DecisionTreeClassifier()
    dec.fit(iris_x_train, iris_y_train)
    predicted =  dec.predict(iris_x_test)
    print str(sum(recall_score(iris_y_test, predicted, average=None))/3)
    return accuracy_score(iris_y_test, predicted)

def line_reg(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    regr = linear_model.LinearRegression()
    regr.fit(iris_x_train, iris_y_train)
#    sensitivity(iris_y_test, predicted)
    return regr.score(iris_x_test, iris_y_test)

def KNN(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    knn = neighbors.KNeighborsClassifier()
    knn.fit(iris_x_train, iris_y_train)
    return knn.score(iris_x_test, iris_y_test)
