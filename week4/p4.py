# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:37:22 2020

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#def blight_model():
features = ['fine_amount', 'admin_fee', 'state_fee', 'late_fee']
datatest = pd.read_csv('train.csv', encoding="ISO-8859-1")
datatest = datatest.head(20)
datatest.index = datatest.ticket_id
test = pd.read_csv('test.csv', encoding="ISO-8859-1")
test = test[features]
test.head(20)
test.index = test.ticket_id
datatest.compliance = datatest.compliance.fillna(-1)
datatest = datatest[datatest.compliance != -1]
X = datatest[features]
X.fillna(0)
y = datatest.compliance
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = RandomForestClassifier(n_estimators = 10,
                        random_state=0).fit(X_train, y_train)
clf.predict_proba(test)