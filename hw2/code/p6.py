# Lasso Part 2 - Problem 6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Lasso import Lasso
import DataManagement as dm
import Supplemental as sp
import Performance as pf

def part_a(regularized_lambdas, non_zeros):
	''' Plot the graph defined in part a. '''
	plt.figure(figsize=(8, 5))
	plt.plot(regularized_lambdas, non_zeros)
	plt.title('Problem 6.a: Nonzeros of each solution versus lambda')
	plt.xlabel('Regularized lambdas')
	plt.ylabel('Count of nonzero elements of each solution')
	plt.xscale('log')
	plt.savefig('P6_a.pdf')
	plt.show()

def part_b(regularization_paths, regularized_lambdas, coefficient_names, coefficient_indices):
	''' Plot the graph defined in part b. '''
	plt.figure(figsize=(8, 5))
	for coeff, label in zip(np.array(regularization_paths)[:, coefficient_indices].T, coefficient_names):
		plt.plot(regularized_lambdas, coeff, label=label)
	plt.title('Problem 6.b: regularization paths for the coefficients for input variables')
	plt.xlabel('Regularized lambdas')
	plt.ylabel('Coefficients')
	plt.xscale('log')	
	plt.legend()
	plt.savefig('P6_b.pdf')
	plt.show()

def part_c(regularized_lambdas, train_mse, test_mse):
	''' Plot the graph defined in part c. '''
	plt.figure(figsize=(8, 5))
	plt.plot(regularized_lambdas, train_mse, label='train_mse')
	plt.plot(regularized_lambdas, test_mse, label='test_mse')
	plt.title('Problem 6.c: Squared error on the training and test data versus lambda')
	plt.xlabel('Regularized lambdas')
	plt.ylabel('Training and test squared error')
	plt.xscale('log')
	plt.savefig('P6_c.pdf')
	plt.show()

def part_d(weights):
	''' Plot the graph defined in pard d. '''
	plt.figure(figsize=(8, 5))
	plt.plot(weights)
	plt.title('Problem 6.d: Weights versus features')
	plt.xlabel('Features')
	plt.ylabel('Weights')
	plt.savefig('P6_d.pdf')
	plt.show()

def main():
	# Load the data frames.
	df_train = pd.read_table('data/crime-train.txt')
	df_test = pd.read_table('data/crime-test.txt')

	# Split the data and labels.
	# Let column be the column to split the data frames on.
	column = 'ViolentCrimesPerPop'
	X_train, y_train = dm.split_on_column(df_train, column)
	X_test, y_test = dm.split_on_column(df_test, column)

	# Let the following be the list of regularized lambdas to train over.
	max_lambda = sp.max_lambda(X_train, y_train)
	regularized_lambdas = sp.regularized_lambdas(max_lambda, n=20, constant=2)

	# Let the following hold data relevant to each training iteration for plotting graphs.
	non_zeros = []
	train_mse = []
	test_mse = []
	paths = []

	# Passing in the values of w_pred into the model is faster 
	# than initializing with 0 weights each time.	
	w_pred = None

	for rl in regularized_lambdas: 
		print(f'Lambda: {rl}')

		# Train the model.
		model = Lasso(rl)
		model.train(X_train.values, y_train.values, w_pred, delta=1E-4, verbose=True)

		# Use as the initialization weights on the next iteration.
		# Must copy to prevent side effects.
		w_pred = np.copy(model.w)
		paths.append(w_pred)

		# Run the model.
		y_hat_train = model.predict(X_train)
		y_hat_test = model.predict(X_test)

		# Record model performance.
		non_zeros.append(pf.non_zeros(w_pred))
		train_mse.append(pf.mse(y_train, y_hat_train))
		test_mse.append(pf.mse(y_test, y_hat_test))

	# Plot part a.
	part_a(regularized_lambdas, non_zeros)

	# Plot part b.
	coefficient_names = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
	coefficient_indices = [X_train.columns.get_loc(name) for name in coefficient_names]
	part_b(paths, regularized_lambdas, coefficient_names, coefficient_indices)

	# Plot part c.
	part_c(regularized_lambdas, train_mse, test_mse)

	# Plot part d.
	model_d = Lasso(30)
	model_d.train(X_train.values, y_train.values, w_pred, delta=1E-4, verbose=True)
	part_d(model_d.w)

	min_w_val = float('inf')	
	min_w_idx = None
	max_w_val = float('-inf')
	max_w_idx = None	
	for i, w in enumerate(model_d.w):
		if w > max_w_val:
			max_w_val = w
			max_w_idx = i
		if w < min_w_val:
			min_w_val = w
			min_w_idx = i
	print(f'Min Weight: index={min_w_idx} val={min_w_val}')
	print(f'Max Weight: index={max_w_idx} val={max_w_val}')

	for i, weight in enumerate(model_d.w):
		if weight != 0:
			print(f'{i}: {weight}')
	
	for i, label in enumerate(X_train):
		print(f'{i}: {label}')

if __name__ == '__main__':
	main()
