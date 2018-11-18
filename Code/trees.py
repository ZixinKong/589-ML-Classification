#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import modules needed
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# create decision tree
def decision_tree(d, train_x, train_y, test_x):
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(train_x, train_y)
    return dt.predict(test_x)

# implement 5-fold-cross-validation
def cross_validation(max_depths, train_x, train_y):
    scores = [] # record avg MAE with differnt num_neighbor(k) 
    for d in max_depths:
        dt = DecisionTreeClassifier(max_depth=d) 
        score = cross_val_score(dt, train_x, train_y, cv=5) #list
        scores.append(1-np.mean(score)) # put avg in scores list  #"{0:.4f}".format(10.1234567890)
        
# create a dictionary with key=num_neighbor, value=corresponding Avg of MAE when k=key
    return dict(zip(max_depths, scores))

###########################################################################3
