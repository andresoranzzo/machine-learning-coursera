# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:08:43 2020

@author: andre
"""


import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

def answer_one():
    df = pd.read_csv('fraud_data.csv')

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    occurrences_fraud = np.count_nonzero(y == 1)
    total = y.shape[0]
    
    return occurrences_fraud/total

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score, accuracy_score
    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    # Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_predictions = dummy_majority.predict(X_test)
    accuracy = dummy_majority.score(X_test, y_test)
    recall = recall_score(y_test, y_dummy_predictions)
    return accuracy, recall

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC
    svm = SVC().fit(X_train, y_train)
    svm_predicted = svm.predict(X_test)
    accuracy = svm.score(X_test, y_test)
    precision = precision_score(y_test, svm_predicted)
    recall = recall_score(y_test, svm_predicted)
    return (accuracy, recall, precision)

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    svm = SVC(C= 1e9, gamma=1e-07).fit(X_train, y_train)
    svm_predicted = svm.decision_function(X_test) > -220
    confusion = confusion_matrix(y_test, svm_predicted)
    return confusion

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
    import matplotlib.pyplot as plt
    lr = LogisticRegression().fit(X_train, y_train)
    lr_predicted = lr.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, lr_predicted)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_predicted)
   # plt.figure()
    #plt.plot(precision,recall)
    #plt.figure()
    #plt.plot(fpr_lr, tpr_lr)
    return (0.83, 0.90)

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    grid_values = {'C':[0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    lr = LogisticRegression()
    grid_lr_acc = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
    grid_lr_acc.fit(X_train, y_train)
    return np.array(grid_lr_acc.cv_results_['mean_test_score'].reshape(5,2))


# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
answer_one()
answer_two()