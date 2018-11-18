#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import modules needed
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# create K nearest neighbors model
def knn_model(num_neighbor, train_x, train_y, test_x):
    knn = KNeighborsClassifier(n_neighbors=num_neighbor)
    knn.fit(train_x, train_y)
    return knn.predict(test_x)


# implement 5-fold-cross-validation
def cross_validation(num_neighbors, train_x, train_y):
    scores = [] # record avg MAE with differnt num_neighbor(k) 
    for k in num_neighbors:
        print(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, train_x, train_y, cv=5) #list
        scores.append(1-np.mean(score)) # put avg in scores list  #"{0:.4f}".format(10.1234567890)
        
# create a dictionary with key=num_neighbor, value=corresponding Avg of MAE when k=key
    return dict(zip(num_neighbors, scores))


##########################################################################

