# HW4 Problem 3 - K-Means Clustering

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_dataset(path):
	''' Load the mnist handwritten digits dataset. '''
	mndata = MNIST(path)
	X_train, y_train = map(np.array, mndata.load_training())
	X_test, y_test = map(np.array, mndata.load_testing())
	X_train = X_train / 255.0 
	X_test = X_test / 255.0
	return X_train, y_train, X_test, y_test

def save(output, path):
	''' Save the output to the file path. '''
	with open(path, 'wb') as f:
		pickle.dump(output, f)

def plot(*series, title, x_label, y_label, file_path, x_ticks=None):
	''' Plot the data. '''
	plt.title(title)
	for x, y, label in series:
		if not x:
			plt.plot(y, label=label, marker='o')
		else:
			plt.plot(x, y, label=label, marker='o')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	if x_ticks:
		plt.xticks(x_ticks)
	plt.legend()
	plt.savefig(file_path)
	plt.show()

def visualize(title, centers, file_path, fig_size=(8, 5), dim=(28, 28), font_size=16):
	''' Visualize the cluster centers. '''
	fig, ax = plt.subplots(2, 5, figsize=fig_size)
	for j in range(len(centers)):
		ax.ravel()[j].imshow(centers[j, :].reshape(dim))
		ax.ravel()[j].axis('off')
	fig.suptitle(title, fontsize=font_size)
	plt.savefig(file_path)
	plt.show()

class K_Means:
	def train(self, X, k, conv_distance=0.01, verbose=False):
		'''
		Use Lloyds algorithm to compute K-Means clusters over the data X and 
		return the resultant centers and each iteration's measured objective values.

		Arguments:
			X -- Numpy array of data
			k -- Integer of clusters to compute
			conv_distance -- Float minimum distance for determining convergence
			verbose -- Boolean display progress output if true
		Returns:
			centers -- k length array of computed center values
			objectives -- Array of objective values per iteration
		'''
		if verbose:
			print(f'Training with {k} clusters')

		self.centers = X[np.random.choice(np.arange(len(X)), size=k, replace=False)]

		# Only used for plotting results.
		objectives = []

		converged = False
		while not converged:
			# Compute the point distribution 
			point_dist, objective = self._get_point_distribution(X)

			# Move each of the k centers closer to the center of mass. 
			prev_centers = self.centers
			self.centers = np.array([np.average(X[i], axis=0) for i in point_dist])

			# Check whether the centers have converged.
			max_distance = np.max(np.sum((prev_centers - self.centers) ** 2, axis=1))
			converged = max_distance < conv_distance

			if verbose:
				print(max_distance)

			objectives.append(objective)

		return self.centers, objectives

	def _get_point_distribution(self, X):
		'''
		Return a distribution of points from X such that for each point i in X
		i is assigned to the closest of the k centers.

		Arguments:
			X -- Numpy array of data
		Returns:
			point_dist -- Nested list; distribution of points from X onto each of k points
			objective -- Float; the measured objective for this iteration
		'''
		objective = 0
		k = len(self.centers)

		point_dist = [[] for _ in range(k)]

		# Compute the sum of euclidean distances between two arrays.
		distance = lambda x, y: np.sum((x - y) ** 2)

		# Find which center j each point i is closest to. 
		for i in range(len(X)):
			distances = [distance(X[i], self.centers[j]) for j in range(k)]
			j = np.argmin(distances)
			point_dist[j].append(i)
			objective += distances[j]

		return point_dist, objective
    
	def error(self, X):
		''' Compute the error of the trained model on the dataset X. '''
		objective = 0

		for i in X:
			min_distance = float('inf')
			for j in self.centers:
				distance = np.linalg.norm((i - j) ** 2)
				min_distance = min(distance, min_distance)

			objective += min_distance

		return objective / len(X)

if __name__ == '__main__':
	print('Loading data')
	X_train, y_train, X_test, y_test = load_dataset('../data/python-mnist/data/')

	# Run the algorithm on the MNIST training dataset with k = 10.
	model = K_Means()
	centers, objectives = model.train(X_train, k=10, verbose=True)

	# Save the results.
	save(centers, path='../data/a3_part_b_centers.pickle')
	save(objectives, path='../data/a3_part_b_objectives.pickle')

	# Plot the objective as a function of the iteration number.
	plot((None, objectives, 'objective'),
		title='Objective value per iteration number',
		x_label='iteration number',
		y_label='objective value',
		file_path='../figures/a3_plot_b.pdf')

	# Visualize the cluster centers as a 28 x 28 image. 
	visualize(title='Visualization of cluster centers', 
		centers=centers,
		file_path='../figures/a3_clusters.pdf')

	# For k = {2, 4, 8, 16, 32, 64} run the algorithm on the training dataset to obtain centers.
	k_vals = [2, 4, 8, 16, 32, 64]

	train_errors = []
	test_errors = []

	for k in k_vals:
		model = K_Means()
		centers, objectives = model.train(X_train, k, verbose=True)
		train_errors.append(model.error(X_train))
		test_errors.append(model.error(X_test))

	# Save the results.
	save(train_errors, path='../data/a3_part_c_train_errors.pickle')
	save(test_errors, path='../data/a3_part_c_test_errors.pickle')

	# Plot the training and test error as a function of k.
	plot((k_vals, train_errors, 'train'),
		(k_vals, test_errors, 'test'),
		title='Training vs test errors as a function of k',
		x_label='k',
		y_label='error',
		x_ticks=k_vals,
		file_path='../figures/a3_plot_c.pdf')
