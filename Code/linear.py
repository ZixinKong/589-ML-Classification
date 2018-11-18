#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


# create decision tree
def linear_model(a, l, train_x, train_y, test_x):
    linear = SGDClassifier(loss=l, alpha=a)
    linear.fit(train_x, train_y)
    return linear.predict(test_x)

# implement 5-fold-cross-validation with hinge loss
def hinge_cross_validation(alphas, train_x, train_y):
    scores = [] # record avg MAE with differnt num_neighbor(k) 
    for a in alphas:
        hinge = SGDClassifier(loss='hinge', alpha=a)
        score = cross_val_score(hinge, train_x, train_y, cv=5) #list
        scores.append(1-np.mean(score)) # put avg in scores list  #"{0:.4f}".format(10.1234567890)
        
# create a dictionary with key=num_neighbor, value=corresponding Avg of MAE when k=key
    return dict(zip(alphas, scores))

# implement 5-fold-cross-validation with logistic regression loss
def log_cross_validation(alphas, train_x, train_y):
    scores = [] # record avg MAE with differnt num_neighbor(k) 
    for a in alphas:
        log = SGDClassifier(loss='log', alpha=a)
        score = cross_val_score(log, train_x, train_y, cv=5) #list
        scores.append(1-np.mean(score)) # put avg in scores list  #"{0:.4f}".format(10.1234567890)
        
# create a dictionary with key=num_neighbor, value=corresponding Avg of MAE when k=key
    return dict(zip(alphas, scores))
#########################################################################