import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from adspy_shared_utilities import plot_fruit_knn
import cv2

#Read data
fruits = pd.read_table('fruit_data_with_colors.txt')
print(fruits.head())
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
#Create a train-test-split
X = fruits[['mass','width','height']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#Creata a classifier object
knn = KNeighborsClassifier(n_neighbors=5)
#Train the data
knn.fit(X_train, y_train)
#Estimate accuracy of the classifier on future data, using the test data
print(knn.score(X_test, y_test))
#Individual prediction test
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
print(lookup_fruit_name[fruit_prediction[0]])
fruit_prediction = knn.predict([[100, 6.3, 8.5]])
print(lookup_fruit_name[fruit_prediction[0]])
#Plot the decision boundaries of the k-NN classifier
plot_fruit_knn(X_train, y_train, 5, 'uniform')