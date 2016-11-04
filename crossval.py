# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:39:17 2016

@author: ubu
"""

from time import time
import pandas as pd
from matplotlib import pyplot as plt
#import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neural_network
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
#from sklearn.svm import SVC
#from sklearn.linear_model import SGDClassifier
from sklearn import naive_bayes
from sklearn import metrics
import scipy as sc
import numpy as np

#gimd = pd.read_csv('~/ownCloud/docs/papers/paper_greedy/code/gimd.csv')
#smr = pd.read_csv('~/ownCloud/docs/papers/paper_greedy/code/smr.csv')
pima = pd.read_csv('~/ownCloud/docs/papers/paper_greedy/code/pima-indians-diabetes.csv', header=None)
y = pima.iloc[:,8]
pima = pima.drop(pima.columns[8], axis=1)

np.random.seed(0)

# logistic
start = time()
log_cv = model_selection.cross_val_score(linear_model.LogisticRegression(), pima, y, cv=5, scoring='roc_auc') 
print("logistic:", time()-start)
print(log_cv.mean())

# rf
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 7],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
start2 = time()
est1 = GridSearchCV(ensemble.RandomForestClassifier(), param_grid=param_grid, scoring='roc_auc')
rf_cv = model_selection.cross_val_score(est1, pima, y, cv=5, scoring='roc_auc')
print("rf:", time()-start2)
print(rf_cv.mean())

# nnet
parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'alpha': [0.001, 0.0001, 0.00001, 0.000001],
              'max_iter': [500]}

est2 = GridSearchCV(estimator=neural_network.MLPClassifier(random_state=1), param_grid=parameters,
     scoring='roc_auc', cv=5)
start3 = time()
p_cv = model_selection.cross_val_score(est2, pima, y, cv=5, scoring='roc_auc') 
print("perceptron:", time()-start3)
print(p_cv.mean())

# nb
start3 = time()
nb_cv = model_selection.cross_val_score(naive_bayes.GaussianNB(), pima, y, cv=5, scoring='roc_auc') 
print("nb:", time()-start3)
print(nb_cv.mean())









