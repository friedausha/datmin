from sklearn import linear_model

def line_reg(iris_x_train, iris_x_test, iris_y_train, iris_y_test):

    regr = linear_model.LinearRegression()
    regr.fit(iris_x_train, iris_y_train)
    return regr.score(iris_x_test, iris_y_test)

