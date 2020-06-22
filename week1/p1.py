import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

#Question 0
#How many features does the breast cancer dataset have?
def answer_0():
    return len(cancer['feature_names'])


#Question 1
#Convert the sklearn.dataset cancer to a DataFrame.
def answer_1():
    df = pd.DataFrame(data=cancer['data'],
                      index=range(0, 569, 1),
                      columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                               'mean smoothness', 'mean compactness', 'mean concavity',
                               'mean concave points', 'mean symmetry', 'mean fractal dimension',
                               'radius error', 'texture error', 'perimeter error', 'area error',
                               'smoothness error', 'compactness error', 'concavity error',
                               'concave points error', 'symmetry error', 'fractal dimension error',
                               'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                               'worst smoothness', 'worst compactness', 'worst concavity',
                               'worst concave points', 'worst symmetry', 'worst fractal dimension'])
    df['frame'] = cancer['target']
    return df


def answer_2():
    cancer_df = answer_1()
    print(cancer_df)


cancer = load_breast_cancer()
answer_2()
