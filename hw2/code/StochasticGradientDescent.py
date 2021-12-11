# Implementation of Stochastic Gradient Descent for Problem 7.

import numpy as np
import Supplemental as sp

def stochastic_gradient_descent(X, y, alpha, batch_size, regularized_lambda=1E-1, w_init=None, b_init=0, delta=1E-4, max_iterations=1E4, verbose=False):
	# Initialize the given weights if none given.
	n, d  = X.shape
	if w_init is None:
		w_init = np.zeros(d)

	# Initialize the working weights and bias.
	w_curr, b_curr = w_init, b_init
	w_prev = w_init + float('inf')

	# Store the current iteration of the function.
	i = 0

	# Store the states of each iteration.
	w_history = [] 
	b_history = [] 
	loss_history = [] 

	# While not converged.
	dw = float('inf')
	while dw >= delta and i <= max_iterations:
		# Batch the descent.
		batch_index = np.random.choice(n, batch_size)
		X_batch = X[batch_index]
		y_batch = y[batch_index]

		# Store the previous iteration for alpha comparison.
		# Copy to prevent side effects.
		w_prev = np.copy(w_curr)

		# Step down the gradient.
		w_curr = w_curr - alpha * sp.gradient_w(X_batch, y_batch, w_curr, b_curr, regularized_lambda)
		b_curr = b_curr - alpha * sp.gradient_b(X_batch, y_batch, w_curr, b_curr) 

		# Compute the loss.
		loss = sp.gradient_loss(X, y, w_curr, b_curr, regularized_lambda)

		# Store results of current iterations.  
		w_history.append(w_curr)
		b_history.append(b_curr)
		loss_history.append(loss)

		# Output results of this iteration.
		if verbose:
			print(f'{i}\tLoss: {loss}')

		# Update for next iteration.
		dw = np.linalg.norm(w_curr - w_prev, np.inf)
		i += 1

	return w_curr, b_curr, w_history, b_history, loss_history
