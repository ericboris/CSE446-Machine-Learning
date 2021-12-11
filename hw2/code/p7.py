# Logistic Regression - Problem 7

import matplotlib.pyplot as plt
import DataManagement as dm
import Supplemental as sp
import Performance as pf
from GradientDescent import gradient_descent
from StochasticGradientDescent import stochastic_gradient_descent as sgd

def two_part_plot(x1, x1_label, x2, x2_label, title, x_label, y_label, file_name):
	plt.figure(figsize=(8, 5))
	plt.plot(x1, label=x1_label)
	plt.plot(x2, label=x2_label)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend()
	plt.savefig(file_name)
	plt.show()

def main():
	# Load the mnist data with all columns.
	print(f'Loading data')
	X_train_all, y_train_all, X_test_all, y_test_all = dm.load_mnist()

	# Strip from each data set all columns except columns 2 and 7.
	print(f'Stripping columns')
	keep_cols = (2, 7)

	i_train = dm.indices(y_train_all, keep_cols)
	X_train = dm.strip_cols(X_train_all, i_train)
	y_train = dm.strip_cols(y_train_all, i_train)

	i_test = dm.indices(y_test_all, keep_cols)
	X_test = dm.strip_cols(X_test_all, i_test)
	y_test = dm.strip_cols(y_test_all, i_test)

	# Set the y label values. 
	print('Setting label values')
	pos, neg = 1, -1
	y_train[y_train == 7] = pos
	y_train[y_train == 2] = neg
	y_test[y_test == 7] = pos
	y_test[y_test == 2] = neg

	# Run gradient descent.
	print('Running gradient descent')
	w, b, w_history, b_history, train_loss = gradient_descent(X_train, y_train, alpha=0.1, verbose=True)

	print('Plotting part b')
	# Plot measured loss versus iteration.
	test_loss = [sp.gradient_loss(X_test, y_test, w, b) for w, b in zip(w_history, b_history)]

	#part_b_i(train_loss, test_loss)
	two_part_plot(x1 = train_loss, 
				x1_label = 'train loss', 
				x2 = test_loss, 
				x2_label = 'test loss',
				title = 'Problem 7.b.i: Measured loss versus iteration using gradient descent',
				x_label = 'Iteration',
				y_label = 'Loss',
				file_name = 'P7_b_i.pdf')

	# Plot misclassification error versus iteration.
	y_hat_train = [sp.gradient_predict(X_train, w, b) for w, b in zip(w_history, b_history)]
	y_train_error = [pf.misclass_error(y_train, y_hat) for y_hat in y_hat_train]

	y_hat_test = [sp.gradient_predict(X_test, w, b) for w, b in zip(w_history, b_history)]
	y_test_error = [pf.misclass_error(y_test, y_hat) for y_hat in y_hat_test]

	two_part_plot(x1 = y_train_error,
				x1_label = 'train misclass error', 
				x2 = y_test_error, 
				x2_label = 'test misclass error',
				title = 'Problem 7.b.ii: Misclassification error versus iteration using gradient descent',
				x_label = 'Iteration',
				y_label = 'Misclassification error',
				file_name = 'P7_b_ii.pdf')

	# Run stochastic gradient descent with batch = 1.
	print('Running Stochastic Gradient Descent with batch=1')
	w, b, w_history, b_history, train_loss = sgd(X_train, y_train, alpha=0.01, batch_size=1, max_iterations=500, verbose=True)
	
	print('Plotting part c')
	# Plot measured loss versus iteration using sgd and batch = 1.
	test_loss = [sp.gradient_loss(X_test, y_test, w, b) for w, b in zip(w_history, b_history)]

	two_part_plot(x1 = train_loss, 
				x1_label = 'train loss', 
				x2 = test_loss, 
				x2_label = 'test loss',
				title = 'Problem 7.c.i: Measured loss versus iteration using sgd and batch=1',
				x_label = 'Iteration',
				y_label = 'Loss',
				file_name = 'P7_c_i.pdf')

	# Plot misclassification error versus iteration using sgd and batch = 1.
	y_hat_train = [sp.gradient_predict(X_train, w, b) for w, b in zip(w_history, b_history)]
	y_train_error = [pf.misclass_error(y_train, y_hat) for y_hat in y_hat_train]

	y_hat_test = [sp.gradient_predict(X_test, w, b) for w, b in zip(w_history, b_history)]
	y_test_error = [pf.misclass_error(y_test, y_hat) for y_hat in y_hat_test]

	two_part_plot(x1 = y_train_error,
				x1_label = 'train misclass error', 
				x2 = y_test_error, 
				x2_label = 'test misclass error',
				title = 'Problem 7.c.ii: Misclassification error versus iteration using sgd and batch=1.',
				x_label = 'Iteration',
				y_label = 'Misclassification error',
				file_name = 'P7_c_ii.pdf')

	# Run stochastic gradient descent with batch = 1.
	print('Running Stochastic Gradient Descent with batch=100')
	w, b, w_history, b_history, train_loss = sgd(X_train, y_train, alpha=0.01, batch_size=100, max_iterations=500, verbose=True)
	
	print('Plotting part c')
	# Plot measured loss versus iteration using sgd and batch = 100.
	test_loss = [sp.gradient_loss(X_test, y_test, w, b) for w, b in zip(w_history, b_history)]

	two_part_plot(x1 = train_loss, 
				x1_label = 'train loss', 
				x2 = test_loss, 
				x2_label = 'test loss',
				title = 'Problem 7.d.i: Measured loss versus iteration using sgd and batch=100',
				x_label = 'Iteration',
				y_label = 'Loss',
				file_name = 'P7_d_i.pdf')

	# Plot misclassification error versus iteration using sgd and batch = 100.
	y_hat_train = [sp.gradient_predict(X_train, w, b) for w, b in zip(w_history, b_history)]
	y_train_error = [pf.misclass_error(y_train, y_hat) for y_hat in y_hat_train]

	y_hat_test = [sp.gradient_predict(X_test, w, b) for w, b in zip(w_history, b_history)]
	y_test_error = [pf.misclass_error(y_test, y_hat) for y_hat in y_hat_test]

	two_part_plot(x1 = y_train_error,
				x1_label = 'train misclass error', 
				x2 = y_test_error, 
				x2_label = 'test misclass error',
				title = 'Problem 7.d.ii: Misclassification error versus iteration using sgd and batch=100.',
				x_label = 'Iteration',
				y_label = 'Misclassification error',
				file_name = 'P7_d_ii.pdf')

if __name__ == '__main__':
	main()
