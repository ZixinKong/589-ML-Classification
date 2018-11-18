# Import python modules

import numpy as np
import kaggle
import trees
import KNN
import linear
from sklearn.neighbors import KNeighborsClassifier

# Read in train and test data
def read_image_data():
	print('Reading image data ...')
	temp = np.load('../../Data/data_train.npz')
	train_x = temp['data_train']
	temp = np.load('../../Data/labels_train.npz')
	train_y = temp['labels_train']
	temp = np.load('../../Data/data_test.npz')
	test_x = temp['data_test']
	return (train_x, train_y, test_x)

############################################################################

train_x, train_y, test_x = read_image_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)


# Questions 1: Decision Trees
# (b)

print ("Now is for question 1(b): decision tree model ...")
max_depths = [3,6,9,12,14]
print ("Find best max depth using 5-fold cross validation...")
dt_output = trees.cross_validation(max_depths, train_x, train_y)
print(dt_output)
best_parameter = min(dt_output, key = dt_output.get)

print ("Now training full dataset with max depth =", best_parameter)
predicted_y = trees.decision_tree(best_parameter, train_x, train_y, test_x)


# Output file location
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name, "\n")
kaggle.kaggleize(predicted_y, file_name)


# Question 2: Nearest neighbors ################################
# (b) 

print ("Now is for question 2(b): Nearest neighbors model ...")

num_neighbors = [3,5,7,9,11] 

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x, train_y)
print(knn.predict(test_x))
#print(cross_val_score(knn, train_x, train_y, cv=5))


print ("Find best k using 5-fold cross validation...")
knn_output = KNN.cross_validation(num_neighbors, train_x, train_y)
print ("k - Average out-of-sample error: ", knn_output)

best_parameter = min(knn_output, key = knn_output.get)
print ("The best parameter for KNN model is: ", best_parameter)

print ("Now training full dataset with k =", best_parameter)
predicted_y = KNN.knn_model(best_parameter, train_x, train_y, test_x)

# Output file location
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name, "\n")
kaggle.kaggleize(predicted_y, file_name)


# Question 3: Linear Model #######################################
# (a)

print ("Now is for question 3(a): Linear model ...")

alphas = [10**-6, 10**-4, 10**-2, 1, 10]

print ("Find best alpha of linear model with hinge loss using 5-fold cross validation ...")
hinge_output = linear.hinge_cross_validation(alphas, train_x, train_y)
print ("Alpha - Average MAE of each alpha (hinge): ", hinge_output)

para1 = min(hinge_output, key = hinge_output.get)
print ("The best parameter for linear model with hinge loss is: ", para1)


print ("Find best alpha of linear model with log-regrssion loss using 5-fold cross validation ...")
log_output = linear.log_cross_validation(alphas, train_x, train_y)
print ("Alpha - Average MAE of each alpha (log): ", log_output)

para2 = min(log_output, key = log_output.get)
print ("The best parameter for linear model with log-regression loss is: ", para2)

if hinge_output[para1] < log_output[para2]:
    best_parameter = para1
    print ("The best linear model is: hinge loss with alpha=", para1)
    print ("Now training full dataset using linear model with hinge loss and aplha=", best_parameter)
    predicted_y = linear.linear_model(best_parameter, 'hinge', train_x, train_y, test_x)
else:
    best_parameter = para2
    print ("The best linear model is: log-regrssion loss with aplha=", para2)
    print ("Now training full dataset using linear model with log-regression loss and alpha=", best_parameter)
    predicted_y = linear.linear_model(best_parameter, 'log', train_x, train_y, test_x)

#predicted_y = linear.linear_model(0.01, 'log', train_x, train_y, test_x)
# Output file location
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name, "\n")
kaggle.kaggleize(predicted_y, file_name)         










