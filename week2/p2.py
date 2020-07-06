import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics.regression import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    return plt

def answer_one():
    results = np.zeros((4, 100))
    X_train_aux = X_train.reshape(-1, 1)
    x_predict = np.linspace(0, 10, 100).reshape(-1, 1)
    poly = PolynomialFeatures(degree = 9)
    x_predict_poly = poly.fit_transform(x_predict)
    x_train_poly = poly.fit_transform(X_train_aux)
    indice = 0
    for i in [1, 3, 6, 9]:
        x_predict_aux = x_predict_poly[:,:1+i]
        row1 = x_train_poly[:,:1+i]
        linreg = LinearRegression().fit(row1, y_train)
        temp = linreg.predict(x_predict_aux)
        results[indice,:] = temp
        indice = indice + 1    
    return results
    

def answer_two():
    results_train = np.zeros((10,1))
    results_test = np.zeros((10,1))
    X_train_aux = X_train.reshape(-1, 1)
    X_test_aux = X_test.reshape(-1, 1)
    # Polynomial
    poly = PolynomialFeatures(degree = 9)
    x_train_poly = poly.fit_transform(X_train_aux)
    x_test_poly = poly.fit_transform(X_test_aux)
    for i in range(0,10):
        x_train_poly_i = x_train_poly[:,:1+i]
        x_test_poly_i = x_test_poly[:,:1+i]
        linreg = LinearRegression().fit(x_train_poly_i, y_train)
        results_train[i] = r2_score(y_train, linreg.predict(x_train_poly_i))
        results_test[i] = r2_score(y_test, linreg.predict(x_test_poly_i))
    return (results_train, results_test)
    

def answer_three():
    plt.figure()
    r2_train, r2_test = answer_two()
    return (0, 9, 7)


def answer_four():
    results_train = np.zeros((10,1))
    results_test = np.zeros((10,1))
    X_train_aux = X_train.reshape(-1, 1)
    X_test_aux = X_test.reshape(-1, 1)
    poly = PolynomialFeatures(degree = 12)
    x_train_poly = poly.fit_transform(X_train_aux)
    x_test_poly = poly.fit_transform(X_test_aux)
    linreg = LinearRegression().fit(x_train_poly, y_train)
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(x_train_poly, y_train)
    r2_linear_score = r2_score(y_test, linreg.predict(x_test_poly))
    r2_lasso_score = r2_score(y_test, lasso.predict(x_test_poly))
    return (r2_linear_score, r2_lasso_score)
    

def answer_five():
    dectree = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    vary = pd.Series(data=dectree.feature_importances_, index=X_train2.columns.values)
    lista = vary.sort_values(axis=0, ascending=False).index.tolist()[:5]
    return lista


def answer_six():
    svc = SVC(kernel='rbf', C=1, random_state=0).fit(X_train2, y_train2)
    train_scores, test_scores = validation_curve(svc, X_subset, y_subset, param_name='gamma', param_range=np.logspace(-4,1,6))
    training_scores = np.mean(train_scores, axis=1)
    testing_scores = np.mean(test_scores, axis=1)
    return (training_scores, testing_scores)       


def answer_seven():
    return (0.0001, 10, 0.1)
 
    

np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)
X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)
X_subset = X_test2
y_subset = y_test2
answer_one()

