import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)


def answer_one():
    X_train_aux = X_train.reshape(-1, 1)
    linreg = LinearRegression().fit(X_train_aux, y_train)
    poly = PolynomialFeatures(1)
    row1 = poly.fit_transform(X_train_aux)
    poly = PolynomialFeatures(3)
    row2 = poly.fit_transform(X_train_aux)
    poly = PolynomialFeatures(6)
    row3 = poly.fit_transform(X_train_aux)
    poly = PolynomialFeatures(9)
    row4 = poly.fit_transform(X_train_aux)
    print(row2)
    A = [[row1[1].reshape(1, -1)],
         [row2.reshape(1, -1)],
         [row3.reshape(1, -1)],
         [row4.reshape(1, -1)]]
    #print(A)

np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
answer_one()
