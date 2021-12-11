'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
from numpy.linalg import pinv
from copy import deepcopy

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

	def __init__(self, degree=1, reg_lambda=1E-8):
		"""
		Constructor
		"""
		self.degree = degree
		self.reg_lambda = reg_lambda

	def polyfeatures(self, X, degree):
		"""
		Expands the given X into an n * d array of polynomial features of
		degree d.
		Returns:
			A n-by-d numpy array, with each row comprising of
			X, X * X, X ** 3, ... up to the dth power of X.
			Note that the returned matrix will not include the zero-th power.
		Arguments:
			X is an n-by-1 column numpy array
			degree is a positive integer
		"""
		# Prevent side-effects.
		X_copy = deepcopy(X)

		# Expand X to degree d.	
		for d in range(1, degree):
			X_copy = np.concatenate((X_copy, X ** d), axis=1)

		return X_copy

	def fit(self, X, y):
		"""
		Trains the model
		Arguments:
			X is a n-by-1 array
			y is an n-by-1 array
		Returns:
			No return value
		Note:
			You need to apply polynomial expansion and scaling
			at first
		"""
		# Validate input.
		assert (X.shape == y.shape)

		X = self.preprocess(X, train=True)

		# Compute Theta using closed form solution with regularized matrix M = \lambda * I.
		# \Theta = (X^T X + M)^{-1} X^T y
		M = self.reg_lambda * np.eye(X.shape[1])
		M[0, 0] = 0 	# Ignore offset.
		self.theta = np.dot(np.dot(pinv(np.dot(X.T, X) + M), X.T), y)

	def predict(self, X):
		"""
		Use the trained model to predict values for each instance in X
		Arguments:
			X is a n-by-1 numpy array
		Returns:
			an n-by-1 numpy array of the predictions
	
		"""
		X = self.preprocess(X)

		# Return y_hat.
		return np.dot(X, self.theta)
		
	def preprocess(self, X, train=False):
		"""
		Prepare the given array for training and predicting.
		Arguments:
			X is a n-by-1 numpy array.
			train is a flag representing whether model is training or predicting.
		Returns:
			An n-by-d numpy array, standardized with an offset column.
		"""
		# Prevent side-effects.
		X = deepcopy(X)

		# Let n be the number of features in X.
		n = X.shape[0]

		# Expand to polynomial degree d.
		X = self.polyfeatures(X, self.degree)

		# Only compute mean and std while training.
		# Otherwise, use the computed mean and std during prediction.
		if train:
			self.mean = np.mean(X, axis=0) if n > 1 else 0
			self.std = np.std(X, axis=0) if n > 1 else 1

		# Standardize using Z-score.
		X = (X - self.mean) / self.std

		# Add offset column.
		X = np.concatenate((np.ones([n, 1]), X), axis=1)

		return X

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------

def error(y_hat, y):
	"""
	Return the square error.
	Arguments:
		y_hat -- the predicted output.
		y	  -- the actual output.
	Returns:
		The square error.
	""" 
	return np.mean((y_hat - y) ** 2)

def learningCurve(X_train, Y_train, X_test, Y_test, reg_lambda, degree):
	"""
	Compute learning curve
	Arguments:
		X_train -- Training X, n-by-1 matrix
		Y_train -- Training y, n-by-1 matrix
		X_test -- Testing X, m-by-1 matrix
		Y_test -- Testing Y, m-by-1 matrix
		regLambda -- regularization factor
		degree -- polynomial degree
	Returns:
		error_train -- error_train[i] is the training accuracy using
		model trained by X_train[0:(i+1)]
		error_test -- error_train[i] is the testing accuracy using
		model trained by X_train[0:(i+1)]
	Note:
		error_train[0:1] and error_test[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
	"""
	n = len(X_train)

	# Let the following store model errors
	# s.t. error_t..[i] = error on model_n where n = i+1.
	error_train = np.zeros(n)
	error_test = np.zeros(n)

	# Compute model performance on differently sized datasets
	# starting at n=2 and increasing.
	for i in range(1, n):
		# Get the training sets.
		X = X_train[0 : i + 1]
		y = Y_train[0 : i + 1]

		# Get and train the model.
		model = PolynomialRegression(degree, reg_lambda)
		model.fit(X, y)

		# Get the model's predictions on training and test sets.
		y_hat_train = model.predict(X)
		y_hat_test = model.predict(X_test)

		# Compute and store model performance.
		error_train[i] = error(y_hat_train, y)
		error_test[i] = error(y_hat_test, Y_test)

	return error_train, error_test
