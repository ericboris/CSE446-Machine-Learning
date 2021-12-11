# Define Performance functions for Problems 5-7.

import numpy as np

# Let the following be the threshold, below which a value
# is considered to be zero.
ZERO_THRESHOLD = 1E-14

def true_positive(actual, predicted, threshold=ZERO_THRESHOLD):
    ''' Return the count of true positives. '''
    return np.sum(np.logical_and(abs(actual)>threshold, abs(predicted)>threshold))

def true_negative(actual, predicted, threshold=ZERO_THRESHOLD):
    ''' Return the count of true negative. '''
    return np.sum(np.logical_and(abs(actual)<=threshold, abs(predicted)<=threshold))

def false_positive(actual, predicted, threshold=ZERO_THRESHOLD):
    ''' Return the count of false positives. '''
    return np.sum(np.logical_and(abs(actual)<=threshold, abs(predicted)>threshold))

def false_negative(actual, predicted, threshold=ZERO_THRESHOLD):
    ''' Return the count of false negatives. '''
    return np.sum(np.logical_and(abs(actual)>threshold, abs(predicted)<=threshold))

def fdr(actual, predicted):
	''' Return the False Discovery Rate of the given data. '''
	tp = true_positive(actual, predicted)
	fp = false_positive(actual, predicted)
	return (fp / (tp + fp))

def tpr(actual, predicted):
	''' Return the True Positive Rate of the given data. '''
	tp = true_positive(actual, predicted)
	fn = false_negative(actual, predicted)
	return (tp / (tp + fn))

def non_zeros(array, threshold=ZERO_THRESHOLD):
	''' Return the count of non_zero values in the given np array. '''
	return np.sum(abs(array) > threshold)

def mse(actual, predicted):
	''' Return the Mean Squared Error of the given data. '''
	return np.mean((actual - predicted) ** 2)

def misclass_error(actual, predicted):
	''' Return the misclassification error. '''
	return 1 - np.mean(actual == predicted)
