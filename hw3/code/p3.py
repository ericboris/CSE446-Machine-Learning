# Problem 3: Kernel Ridge Regression

import numpy as np
import matplotlib.pyplot as plt

def generate_data(f, n):
	''' Generate n random samples of data and actual output using function f. '''
	# Let x_i be uniformly random on [0, 1].
	x = np.random.rand(n)
	# Let epsilon_i ~ N(0, 1).
	epsilon = np.random.randn(n)
	# Let y_i = f(x_i) + epsilon_i.
	y = f(x) + epsilon
	# Return x as a column vector.	
	return x.reshape(-1, 1), y
	
def f_star(x):
	''' Compute the f star function given in the spec. '''
	return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)

def polynomial_kernel(x, z, d):
	''' Define the polynomial kernel given in the spec where d is a hyperparameter. '''
	return (1 + x.dot(z.T)) ** d

def rbf_kernel(x, z, gamma):
	''' Define the RBF kernel given in the spec where gamma is a hyperparameter. '''
	return np.exp(-gamma * squared_difference(x, z))

def squared_difference(x, z):
	''' Return the squared difference between x and z. '''
	return np.sum((x[:, :, None] - z[:, :, None].T) ** 2, axis=1)

def leave_one_out_cv(model, X, y, regularized_lambdas, hyperparameters):
	''' Compute the error of lambda and hypermeter combinations on the model using LOOCV. '''

	def error(model, X, y, regularized_lambda, hyperparameter):
		''' Perform LOOCV on one lambda / hyperparameter pair and return the model mean error. '''
		# Set the model to use the parameters
		model.regularized_lambda = regularized_lambda
		model.hyperparameter = hyperparameter

		# Perform LOOCV.
		error = []
		for i in range(len(X)):
			# Use this to determine which indices to include in the "in" subsets.
			indices = np.full(len(X), True)
			indices[i] = False

			# Define the cross validation subsets.
			X_in, y_in = X[indices], y[indices]
			X_out, y_out = X[i].reshape(1, -1), y[i].reshape(1, )

			# Interpolate with the model.
			model.train(X_in, y_in)
			y_hat = model.predict(X_out)

			# Compute and store the mean squared error.
			error.append(np.mean((y_hat - y_out) ** 2))
		
		return np.mean(error)
	
	# Perform LOOCV.
	results = [(error(model, X, y, rl, hp), rl, hp) for rl in regularized_lambdas for hp in hyperparameters]
	return np.array(results)

def plot(title, subtitle, x_label, y_label, file_path, original_x, original_y, original_label, true_x, true_y, true_label, pred_y, pred_label, argsort, dim=(8, 5)):
	''' Plot the graphs for Problem 3 Part b. '''
	plt.figure(figsize=dim)
	plt.title(f'{title}\n{subtitle}')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.plot(original_x[argsort, 0], original_y[argsort], 'o', label=original_label)
	plt.plot(true_x[:, 0], true_y, label=true_label)
	plt.plot(true_x[:, 0], pred_y, label=pred_label)
	plt.legend()
	plt.ylim([-10, 10])
	plt.savefig(file_path)
	plt.show()

class KernelRidgeRegression:
	def __init__(self, kernel, regularized_lambda=1E-8, hyperparameter=None):
		self.kernel = kernel
		self.regularized_lambda = regularized_lambda
		self.hyperparameter = hyperparameter
		
	def train(self, X_train, y_train):
		''' Train the model. '''
		self.mean = np.mean(X_train, axis=0)
		self.std = np.std(X_train, axis=0)
	
		X_train = (X_train - self.mean) / self.std	
		self.X_train = X_train
		
		K = self.kernel(X_train, X_train, self.hyperparameter)
		self.alpha = np.linalg.solve(K + self.regularized_lambda * np.eye(K.shape[0]), y_train)

		return self.alpha
	
	def predict(self, X):
		''' Predict using the model. '''
		X = (X - self.mean) / self.std
		K = self.kernel(self.X_train, X, self.hyperparameter)
		return K.T.dot(self.alpha)

def main():
	# Generate training data.
	X_train, y_train = generate_data(f=f_star, n=30)

	# Set the hyperparameter bounds.
	regularized_lambdas = 10.0 ** (-np.arange(2, 10))
	ds = np.arange(4, 20)
	gammas = (1 / np.median(squared_difference(X_train, X_train))) * np.linspace(0, 2, 10)

	# Build and score both models.
	poly_model = KernelRidgeRegression(polynomial_kernel)
	poly_results = leave_one_out_cv(poly_model, X_train, y_train, regularized_lambdas, ds)
	
	rbf_model = KernelRidgeRegression(rbf_kernel)
	rbf_results = leave_one_out_cv(rbf_model, X_train, y_train, regularized_lambdas, gammas)

	# Part a. Find and assign the best lambdas and hyperparameters for and to the models.
	poly_model.regularized_lambda = poly_results[np.argmin(poly_results[:, 0])][1] 
	poly_model.hyperparameter = poly_results[np.argmin(poly_results[:, 0])][2]

	rbf_model.regularized_lambda = rbf_results[np.argmin(rbf_results[:, 0])][1]
	rbf_model.hyperparameter = rbf_results[np.argmin(rbf_results[:, 0])][2]

	print(f'Problem 3.a.i. Best Lambda and d values with polynomial kernel: Lambda={poly_model.regularized_lambda}\td={poly_model.hyperparameter}')
	print(f'Problem 3.a.ii. Best Lambda and gamma values with RBF kernel: Lambda={rbf_model.regularized_lambda}\tgamma={rbf_model.hyperparameter}')
	
	# Part b. Plot the learned functions using the best lambdas and hyperparameters from part a.
	X_evenly_spaced = np.linspace(0, 1, 100).reshape(-1, 1)
	y_evenly_spaced = f_star(X_evenly_spaced[:, 0])
	argsort = np.argsort(X_train[:, 0])

	poly_model.train(X_train, y_train)
	y_hat_evenly_spaced_poly = poly_model.predict(X_evenly_spaced)

	plot(title='Problem 3.b.i. Plot of Polynomial Kernel on Ridge Regression model',
		subtitle=f'Lambda={poly_model.regularized_lambda} d={poly_model.hyperparameter}',
		x_label='x',
		y_label='f(x)',
		file_path='../plots/3bi.pdf',
		original_x=X_train,
		original_y=y_train,
		original_label='original data',
		true_x=X_evenly_spaced,
		true_y=y_evenly_spaced,
		true_label='true f(x)',
		pred_y=y_hat_evenly_spaced_poly,
		pred_label='predicted f(x)',
		argsort=argsort)
	
	rbf_model.train(X_train, y_train)
	y_hat_evenly_spaced_rbf = rbf_model.predict(X_evenly_spaced)

	plot(title='Problem 3.b.ii. Plot of RBF Kernel on Ridge Regression model',
		subtitle=f'Lambda={rbf_model.regularized_lambda} gamma={rbf_model.hyperparameter}',
		x_label='x',
		y_label='f(x)',
		file_path='../plots/3bii.pdf',
		original_x=X_train,
		original_y=y_train,
		original_label='original data',
		true_x=X_evenly_spaced,
		true_y=y_evenly_spaced,
		true_label='true f(x)',
		pred_y=y_hat_evenly_spaced_rbf,
		pred_label='predicted f(x)',
		argsort=argsort)

if __name__ == '__main__':
	main()
