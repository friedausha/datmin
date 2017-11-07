from sklearn import neighbors

def KNN(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    knn = neighbors.KNeighborsClassifier()
    knn.fit(iris_x_train, iris_y_train)
    #return knn.predict(iris_x_test)
    return knn.score(iris_x_test, iris_y_test)