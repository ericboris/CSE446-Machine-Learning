# Implementation of Gradient Descent for Problem 7.

import numpy as np
import Supplemental as sp
def gradient_descent(X, y, alpha, regularized_lambda=1E-1, w_init=None, b_init=0, delta=1E-4, max_iterations=1E4, verbose=False):
	# Initialize the given weights if none given.
	d  = X.shape[1]
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
		# Store the previous iteration for step comparison.
		# Copy to prevent side effects.
		w_prev = np.copy(w_curr)

		# Step down the gradient.
		w_curr = w_curr - alpha * sp.gradient_w(X, y, w_curr, b_curr, regularized_lambda)
		b_curr = b_curr - alpha * sp.gradient_b(X, y, w_curr, b_curr)

		# Compute the loss.
		loss = sp.gradient_loss(X, y, w_curr, b_curr, regularized_lambda)
	
		# Store results of current iterations.	
		w_history.append(w_curr)
		b_history.append(b_curr)
		loss_history.append(loss)

		# Output results of this iteration.
		if verbose and i % 10 == 0:
			print(f'{i}\tLoss: {loss}')

		# Update for next iteration.
		dw = np.linalg.norm(w_curr - w_prev, np.inf)
		i += 1

	return w_curr, b_curr, w_history, b_history, loss_history
