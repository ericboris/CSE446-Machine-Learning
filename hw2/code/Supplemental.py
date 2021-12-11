# Define Supplemental functions for Problems 5-7.

import numpy as np

def max_lambda(X, y):
    ''' Return the smallest value of lambda for which the solution w_hat is entirely zero. '''
    return np.max(np.sum(2 * X * (y - np.mean(y))[:, None], axis=0))

def regularized_lambdas(max_lambda, n=20, constant=1.5):
	''' Return a list of n regularized lambdas starting from max_lambda and 
		decreasing by a constant ratio. '''
	return [(max_lambda / (constant ** i)) for i in range(n)]

def _mu(X, y, w, b):
	''' Compute the inverse exponential. '''
	return 1 / (1 + np.exp(-y * (b + X.dot(w))))
	
def gradient_w(X, y, w, b, regularized_lambda=1E-1):
	''' Return the gradient J loss function with respect to w. '''
	return np.mean(((_mu(X, y, w, b) - 1) * y)[:, None] * X, axis=0) + (2 * regularized_lambda * w)

def gradient_b(X, y, w, b):
	''' Return the gradient J loss function with respect to b. '''
	return np.mean(((_mu(X, y, w, b) - 1) * y), axis=0)

def gradient_loss(X, y, w, b, regularized_lambda=1E-1):
	''' Return the loss function J. '''
	return np.mean(np.log(1 + np.exp(-y * (b + X.dot(w))))) + (regularized_lambda * w.dot(w))

def gradient_predict(X, w, b):
	''' Perform a prediction using the gradient weights. '''
	return np.sign(X.dot(w) + b)
	
