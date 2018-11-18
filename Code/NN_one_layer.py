import autograd.numpy as np
import autograd
from autograd.util import flatten
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import kaggle

# Function to compute classification accuracy
def mean_zero_one_loss(weights, x, y_integers, unflatten):
	(W, b, V, c) = unflatten(weights)
	out = feedForward(W, b, V, c, x)
	pred = np.argmax(out, axis=1)
	return(np.mean(pred != y_integers))

# Feed forward output i.e. L = -O[y] + log(sum(exp(O[j])))
def feedForward(W, b, V, c, train_x):
        hid = np.tanh(np.dot(train_x, W) + b)
        out = np.dot(hid, V) + c
        return out

# Logistic Loss function
def logistic_loss_batch(weights, x, y, unflatten):
	# regularization penalty
        lambda_pen = 10

        # unflatten weights into W, b, V and c respectively 
        (W, b, V, c) = unflatten(weights)

        # Predict output for the entire train data
        out  = feedForward(W, b, V, c, x)
        pred = np.argmax(out, axis=1)

	# True labels
        true = np.argmax(y, axis=1)
        # Mean accuracy
        class_err = np.mean(pred != true)

        # Computing logistic loss with l2 penalization
        logistic_loss = np.sum(-np.sum(out * y, axis=1) + np.log(np.sum(np.exp(out),axis=1))) + lambda_pen * np.sum(weights**2)
        
        # returning loss. Note that outputs can only be returned in the below format
        return (logistic_loss, [autograd.util.getval(logistic_loss), autograd.util.getval(class_err)])

# Loading the dataset
print('Reading image data ...')
temp = np.load('../../Data/data_train.npz')
train_x = temp['data_train']
temp = np.load('../../Data/labels_train.npz')
train_y_integers = temp['labels_train']
temp = np.load('../../Data/data_test.npz')
test_x = temp['data_test']




# train-validation split 
X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y_integers, test_size=0.2, stratify=train_y_integers)


# Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
X_train -= .5
X_test -= .5

# Number of output dimensions
dims_out = 4
# Number of hidden units
dims_hids = [5, 40, 70]
# Learning rate
epsilon = 0.0001
# Momentum of gradients update
momentum = 0.1
# Number of epochs
nEpochs = 1000
# Number of train examples
nTrainSamples = X_train.shape[0]
# Number of input dimensions
dims_in = X_train.shape[1]

# Convert integer labels to one-hot vectors
# i.e. convert label 2 to 0, 0, 1, 0
train_y = np.zeros((nTrainSamples, dims_out))
train_y[np.arange(nTrainSamples), Y_train] = 1


assert momentum <= 1
assert epsilon <= 1

# Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
grad_fun = autograd.grad_and_aux(logistic_loss_batch)
"""
"""
plotsum = [] # [[5-yvalue], [40-yvalue], [70-yvalue]]
times = []
sample_errors = []

for dims_hid in dims_hids:
    
    print("unit: ", dims_hid)
    start = time.time()
    mean_loss = []
    
    # Initializing weights
    W = np.random.randn(dims_in, dims_hid)
    b = np.random.randn(dims_hid)
    V = np.random.randn(dims_hid, dims_out)
    c = np.random.randn(dims_out)
    smooth_grad = 0
    
    
    # Compress all weights into one weight vector using autograd's flatten
    all_weights = (W, b, V, c)
    weights, unflatten = flatten(all_weights)


    for i in range(nEpochs):
        # Compute gradients (partial derivatives) using autograd toolbox
        weight_gradients, returned_values = grad_fun(weights, X_train, train_y, unflatten)
        #print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])
        mean = returned_values[0] / nTrainSamples
        #print('logistic loss: ',mean)
        mean_loss.append(mean)
    
        # Update weight vector
        smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
        weights = weights - epsilon * smooth_grad
        
    plotsum.append(mean_loss)
    end = time.time()
    times.append((end - start) * 1000)
    
    error = mean_zero_one_loss(weights, X_test, Y_test, unflatten)
    print("error ", error)
    sample_errors.append(error)
    


print("Training time for each NN is:(in ms) ", times)


#Create values and labels for line graphs
labels =["Hidden unit=5","Hidden unit=40", "Hidden unit=70"]
#Plot a line graph
plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(range(1, 1001), plotsum[0],'or-', linewidth=1) #Plot the first series in red with circle marker
plt.plot(range(1, 1001), plotsum[1],'sb-', linewidth=1) #Plot the first series in blue with square marker
plt.plot(range(1, 1001), plotsum[2],'^g-', linewidth=1) 

#This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Mean Log-loss") #Y-axis label
plt.xlabel("Number of epochs") #X-axis label
plt.title("Mean Log-loss vs Number of epochs") #Plot title
#plt.xlim(1,) #set x axis range
#plt.ylim(0,1) #Set yaxis range
plt.legend(labels,loc="best")

#Save the chart
plt.savefig("../Figures/loss_line_plot.pdf")

#Displays the plots.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()



   
result = dict(zip(dims_hids, sample_errors))
print('Validation error = ', result)
best_unit = min(result, key = result.get)
print ("Now training full dataset with hidden unit =", best_unit)

# Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
train_x -= .5
test_x  -= .5

# Number of output dimensions
dims_out = 4
# Number of hidden units
dims_hid = 40
# Learning rate
epsilon = 0.0001
# Momentum of gradients update
momentum = 0.1
# Number of epochs
nEpochs = 1000
# Number of train examples
nTrainSamples = train_x.shape[0]
# Number of input dimensions
dims_in = train_x.shape[1]

# Convert integer labels to one-hot vectors
# i.e. convert label 2 to 0, 0, 1, 0
train_y = np.zeros((nTrainSamples, dims_out))
train_y[np.arange(nTrainSamples), train_y_integers] = 1

assert momentum <= 1
assert epsilon <= 1

# Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
grad_fun = autograd.grad_and_aux(logistic_loss_batch)

# Initializing weights
W = np.random.randn(dims_in, best_unit)
b = np.random.randn(best_unit)
V = np.random.randn(best_unit, dims_out)
c = np.random.randn(dims_out)
smooth_grad = 0

# Compress all weights into one weight vector using autograd's flatten
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)

for i in range(nEpochs):
    # Compute gradients (partial derivatives) using autograd toolbox
    weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
    #print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])

    # Update weight vector
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad

(W, b, V, c) = unflatten(weights)
out = feedForward(W, b, V, c, test_x)
predicted_y = np.argmax(out, axis=1)

# Output file location
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name, "\n")
kaggle.kaggleize(predicted_y, file_name)        

#print('Train accuracy =', 1-mean_zero_one_loss(weights, train_x, train_y_integers, unflatten))

