# Lasso Part 1 - Problem 5

import sys
import numpy as np
import matplotlib.pyplot as plt
from Lasso import Lasso
import Performance as pf
import Supplemental as sp
import DataManagement as dm

def part_a(lambdas, non_zeros):
	plt.figure(figsize=(8, 5))
	plt.plot(lambdas, non_zeros)
	plt.title('Problem 5.a: Plot of non_zero weights against regularized lambda')
	plt.xscale('log')
	plt.ylabel('Non-zero weights')
	plt.xlabel('Regularized Lambda')
	plt.savefig('P5_a.pdf')
	plt.show()

def part_b(FDR, TPR):
	plt.figure(figsize=(8, 5))
	plt.plot(FDR, TPR)
	plt.title('Problem 5.b: Plot of True Positive Rate against False Discovery Rate')
	plt.ylabel('True Postive Rate')
	plt.xlabel('False Discovery Rate')
	plt.savefig('P5_b.pdf')
	plt.show()

def main(args):
	# Training data values.
	if len(args) == 5:
		n = args[1]
		d = args[2]
		k = args[3]
		sigma = args[4]	
	else:
		n = 500
		d = 1000
		k = 100
		sigma = 1

	# Get the synthetic data.
	X, y, w_actual = dm.synthetic_data(n, d, k, sigma)

	# False Discovery Rate (FDR) and True Positive Rate (TPR).
	FDR = []
	TPR = []

	# Count of nonzero weights.
	non_zeros = []

	# Let begin as the max lambda value
	# and be reduced by constant ratio 1.5.
	max_lam = sp.max_lambda(X, y) 

	# Let lambdas hold the precomputed regularized lambdas.
	regularized_lambdas = sp.regularized_lambdas(max_lam)
	
	# Passing in the values of w_pred into the model is faster 
	# than initializing with 0 weights each time. 
	w_pred = None

	for rl in regularized_lambdas:
		print(f'Lambda: {rl}')

		# Train the model.
		model = Lasso(rl)
		model.train(X, y, w_pred, delta=1E-3, verbose=True)
		w_pred = model.w

		# Count non-zeros for part a.	
		non_zeros.append(np.sum(abs(w_pred) > 1E-14))
	
		FDR.append(pf.fdr(w_actual, w_pred))
		TPR.append(pf.tpr(w_actual, w_pred))
				
	# Plot the graphs for part a and part b.
	part_a(regularized_lambdas, non_zeros)
	part_b(FDR, TPR)

if __name__ == '__main__':
	main(sys.argv)
