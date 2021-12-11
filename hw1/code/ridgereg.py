import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def load_dataset():
	'''
	Load the mnist handwritten digits dataset.
	Returns:
		X_train 	-- The training dataset inputs.
		labels_train 	-- The training dataset labels.
		X_test		-- The testing dataset inputs.
		labels_test		-- The testing dataset labels.
	'''
	mndata = MNIST('./data/python-mnist/data/')
	X_train, labels_train = map(np.array, mndata.load_training())
	X_test, labels_test = map(np.array, mndata.load_testing())
	X_train = X_train / 255.0 
	X_test = X_test / 255.0
	return X_train, labels_train, X_test, labels_test

def one_hot(array):
	'''
	Returns a one-hot encoding of the given array.
	Arguments:
		array	-- (n, ) numpy array.
	Returns:
		array' 	-- (n, k) numpy array one hot encoding.
	'''
	n = len(set(array))
	array_prime = np.eye(n)[array]
	return array_prime

def error(y, y_hat):
	'''
	Compute the error between the two sets of labels.
	Arguments:
		y		-- (n, ) numpy array of acutal labels.
		y_hat 	-- (n, ) numpy array of predicted labels.
	Returns:
		error 	-- The measure of error between the two sets.
	'''
	error = 1 - np.mean(y == y_hat)
	return error

class RidgeRegression:
	def train(self, X, y, reg_lambda=1E-4):
		'''
		Trains the ridge regression model using a closed-form solution.
		Arguments:
			X			-- (n, m) numpy array of training features.
			y 			-- (n, k) numpy array of training labels.
			reg_lambda 	-- The regularization parameter.
		Returns:
			A ridge regression trained model.
		'''
		# Let M be the regularized matrix s.t. M = \lambda * I
		M = reg_lambda * np.eye(X.shape[1])
		
		# Let W_hat be the trained weights s.t. W_hat = (X^T X + M)^{-1} X^T y	
		W_hat = np.linalg.solve(np.dot(X.T, X) + M, np.dot(X.T, y))

		return W_hat
	
	def predict(self, X, W):
		'''
		Predicts the labels of the given dataset using the given model.
		Arguments:
			X		-- (n, m) numpy array of features.
			W  		-- (m, k) numpy array of weights.
		Returns:
			y_hat	-- (n, ) numpy array of predicted labels.
		'''
		y_hat = np.argmax(np.dot(W.T, X.T), axis=0)
		return y_hat
		

if __name__ == '__main__':
	# Load the data.
	print('Loading data')
	X_train, labels_train, X_test, labels_test = load_dataset()

	# Use one-hot encoding for training the model.
	y_train = one_hot(labels_train)

 	# Train the ridge regression model.
	print('Training model')
	model = RidgeRegression()
	W_hat = model.train(X_train, y_train)	

	# Predict using the model.
	print('Predicting')
	y_hat_train = model.predict(X_train, W_hat)
	y_hat_test = model.predict(X_test, W_hat)	

	# Compute the model's error.
	print('Computing error')
	train_error = error(labels_train, y_hat_train)
	test_error = error(labels_test, y_hat_test)

	# Output the results.
	print(f'Training error: {train_error}')
	print(f'Testing error: {test_error}')
