# Define Data Management functions for Problems 5-7.

import numpy as np
from mnist import MNIST

def synthetic_data(n, d, k, sigma):
    ''' Generate synthetic training data. '''
    # Let X be an (n, d) matrix with values drawn from a normal distribution.
    X = np.random.normal(0.0, 1.0, (n, d))

    # Let w be a (d, ) array with values d/k if d in range {1, ..., k} else 0.
    w = np.arange(d) / k
    w[k + 1: ] = 0

    # Let y be a (n, ) array with values y_i = w^T x_i + epsilon_i
    # where epsilon is a (n, ) array with values drawn from a normal distribution.
    epsilon = np.random.normal(0.0, sigma, (n, ))
    y = X.dot(w) + epsilon

    return X, y, w

def split_on_column(df, column):
	''' Return split the given data frame df on column s.t.
		X consists of all data except column and
		y consists of only the data in column. '''
	return df.drop(column, axis=1), df[column]

def load_mnist():
	''' Load and return the mnist training and testing datasets. '''
	data = MNIST('data/python-mnist/data/')

	# Load the data.
	X_train, y_train = map(np.array, data.load_training())
	X_test, y_test = map(np.array, data.load_testing())

	# Reduce the data.
	X_train = X_train / 255.0
	X_test = X_test / 255.0

	return X_train, y_train, X_test, y_test

def indices(data, cols):
	indices = 0
	for c in cols:
		indices += (data == c).astype('int')
	return indices

def strip_cols(data, indices):
	''' Return data with all columns not in cols removed. '''
	return data[indices.astype('bool')].astype('float')
