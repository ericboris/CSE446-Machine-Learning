# HW4 Problem 4 - PCA

from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

TQDM_FORM = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

def load_dataset(path):
	''' Load the mnist handwritten digits dataset. '''
	mndata = MNIST(path)
	X_train, y_train = map(np.array, mndata.load_training())
	X_test, y_test = map(np.array, mndata.load_testing())
	X_train = X_train / 255.0 
	X_test = X_test / 255.0
	return X_train, y_train, X_test, y_test

def split(arr, index):
	''' Split numpy array into separate sets on the given index. '''
	return arr[:index, :], arr[index:, :]

def reconstruct(X, V, mu, k):
	''' 
	Return the PCA reconstruction from the principle components. 

	Arguments: 
		X -- Numpy array of data
		V -- Numpy array of components
		mu -- Float of the data mean
		k -- Integer of top components to use
	Returns:
		Numpy array of reconstructed data
	'''
	# Include only top k components.
	V = V[:k + 1].T
	pc_scores = np.matmul(X - mu, V)
	eigenvectors = np.matmul(pc_scores, V.T)
	return eigenvectors + mu

def error(X, V, mu, k):
	''' 
	Return the mean-squared reconstruction errors of the data over range k. 

	Arguments: 
		X -- Numpy array of data
		V -- Numpy array of components
		mu -- Float of the data mean
		k -- Integer of top components to use
	'''
	error = []
	for i in tqdm(range(k), bar_format=TQDM_FORM):
		X_hat = reconstruct(X, V, mu, i)
		error.append(np.square(X - X_hat).mean())
	return error

def fractional_reconstruction_error(eigenvalues, k):
	''' Return a k length array of fractional reconstruction error of the eigenvalues. '''
	fre = []
	for i in tqdm(range(k), bar_format=TQDM_FORM):
		fre.append(1. - np.sum(eigenvalues[:(i + 1)]) / np.sum(eigenvalues))
	return fre

def matching_index(X, digit):
	''' Find the first index in the data where the image matches the digit. '''
	for i in tqdm(range(len(X)), bar_format=TQDM_FORM):
		if X[i] == digit:
			break
	return i

def plot(*series, title, x_label, y_label, save_path):
	''' Plot the data. '''
	plt.title(title)
	for x, y, label in series:
		plt.plot(x, y, label=label)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend()
	plt.savefig(save_path)
	plt.show()

def visualize(data, iterable, save_path=None, n_rows=2, n_cols=5, dim=(28, 28)):
    ''' Visualize the data. '''
    # Minimize the margins 
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    for i, j in enumerate(iterable):
        plt.subplot(n_rows, n_cols, i + 1)
        imgplot = plt.imshow(data[j].reshape(dim))
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':
	print('Load data')
	X_train, y_train, X_test, y_test = load_dataset('../data/python-mnist/data/')
	X_train, X_test = split(X_train, index=50000)

	mu = np.mean(X_train, axis=0)

	print('Get eigenvalues')
	U, S, V = np.linalg.svd(X_train - mu, False)
	n = X_train.shape[0]
	eigenvalues = S ** 2 / n

	print('\nPart a')
	print('What are the eigenvalues 1, 2, 10, 30, and 50?')
	for i in [0, 1, 9, 29, 49]:
		print(f'lambda{i + 1}: {eigenvalues[i]}')

	print('What is the sum of eigenvalues?')
	print(f'Sum of eigenvalues: {np.sum(eigenvalues)}')

	print('\nPart c')
	print('Compute reconstruction error')
	train_error = error(X_train, V, mu, k=100)
	test_error = error(X_test, V, mu, k=100)

	print('Plot reconstruction error')
	plot((range(1, 101), train_error, 'train error'),
		(range(1, 101), test_error, 'test error'),
		title='Reconstruction error vs PCA directions',
		x_label='PCA directions',
		y_label='Reconstruction error',
		save_path='../figures/a4_re.pdf')

	print('Compute fractional reconstruction error')
	fre = fractional_reconstruction_error(eigenvalues, k=100)

	print('Plot fractional reconstruction error')
	plot((range(1, 101), fre, 'FRE'),
		title='Fractional reconstrution error vs PCA directions',
		x_label='PCA directions',
		y_label='Fractional reconstruction error',
		save_path='../figures/a4_fre.pdf')

	print('\nPart d')
	print('Display the first 10 eigenvectors')
	visualize(V, range(0, 10), 
		save_path='../figures/a4_eigenvectors.pdf')

	print('\nPart e')
	print('Show reconstructions for digits 2, 6, 7 with values k = 5, 15, 40, 100')
	indices = []
	for digit in [2, 6, 7]:
		indices.append(matching_index(y_train, digit))

	# Display the actual digits.
	visualize(X_train, indices,
		save_path=f'../figures/a4_actual.pdf',
		n_rows=1,
		n_cols=3)

	# Display the reconstructions for different k.
	for k in [5, 15, 40, 100]:
		reconstruction = reconstruct(X_train, V, mu, k)
		visualize(reconstruction, indices, 
			save_path=f'../figures/a4_recon_{k}.pdf', 
			n_rows=1, 
			n_cols=3)

	# Problem 5 Part d.
	# Re-run PCA with different k to compare results against AutoEncoder.

	# Indices for digits 0-9.
	digit_indices = [1, 14, 16, 12, 9, 11, 13, 15, 17, 4]

	# Display the actual digits.
	visualize(X_train, digit_indices,
		save_path=f'../figures/a5_d_actual.pdf',
		n_rows=1,
		n_cols=10)

	# Display the reconstructions for different k.
	for k in [32, 64, 128]:
		reconstruction = reconstruct(X_train, V, mu, k)
		visualize(reconstruction, digit_indices, 
			save_path=f'../figures/a5_d_recon_{k}.pdf', 
			n_rows=1, 
			n_cols=10)
