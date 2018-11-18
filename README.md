# 589-ML-Classification
Train and evaluate different classification models on one dataset, based on Python 3.6 via Anaconda.

Given a set of gray scale images (32  32 pixels) with one (and only one) of the following objects: horse, truck, frog, ship (labels 0, 1, 2 and 3, respectively). The goal is to train a model to recognize which of the objects is present in an image. 

Train different models (Decision Trees, Nearest Neighbors, Linear models, Neural Networks), compare their performances and training time, and perform model selection using cross-validation.

The dataset is already partitioned into train and test blocks. Each partition is saved in a numpy binary format(.npy) file.
 Size of training set: 20; 000  1024
 Size of testing set: 3; 346  1024
