# HW4 Problem 5 - AutoEncoders

import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_dataset():
	''' Load the mnist handwritten digits dataset. '''
	train = datasets.MNIST(root='./', train=True, download=True, transform=None)
	test = datasets.MNIST(root='./', train=False, download=True, transform=None)
	X_train = train.data.view(-1, 784).float()
	y_train = train.targets
	X_test = test.data.view(-1, 784).float()
	y_test = test.targets
	return X_train, y_train, X_test, y_test

def visualize(data, iterable, save_path=None, n_rows=1, n_cols=10, dim=(28, 28)):
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

def save(output, path):
	''' Save the output to the file path. '''
	with open(path, 'w') as f:
		f.write(output)

class AutoEncoder:
	def __init__(self, d, h, is_linear=True):
		self.h = h
		self.model = self._get_model(d, h, is_linear)
		self.loss_fn = nn.MSELoss()

	def _get_model(self, d, h, is_linear):
		''' Return a linear or non-linear model with the dimensions d and h. '''
		if is_linear:
			return torch.nn.Sequential(nn.Linear(d, h), nn.Linear(h, d))
		else:
			return torch.nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, d), nn.ReLU())

	def train(self, X, n_epochs, learning_rate=1E-3, verbose=False):
		''' Train the model and return a list of training losses. '''
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
		losses = []
		for i in tqdm(range(n_epochs)):
			loss = self.get_loss(X)
			losses.append(loss.item())
			if verbose:
				print(f'iter: {i+1}\tloss: {loss.item()}')
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		return losses

	def get_loss(self, X):
		''' Return the model loss against the given data. '''
		X_hat = self.model(X)
		return self.loss_fn(X_hat, X)

if __name__ == '__main__':
	# Model parameters
	d = 784
	h_vals = [32, 64, 128]
	n_epochs = 2000

	# Indices for digits 0-9.
	digit_indices = [1, 14, 16, 12, 9, 11, 13, 15, 17, 4]

	# Determine training device; GPU or CPU.
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Load the datasets. 
	X_train, y_train, X_test, y_test_target = load_dataset()

	# Save the losses. 
	train_losses = ''
	test_losses = ''

	# Build linear and non-linear models.
	for is_linear in (True, False):
		# Build models with d x h dimensions.
		models = [AutoEncoder(d, h, is_linear) for h in h_vals]

		# Display the actual digits.
		visualize(X_train, digit_indices, save_path=f'../figures/a5_{is_linear}_actual.pdf')

		for i, m in enumerate(models):
			print(f'h = {h_vals[i]}')

			# Train the model.
			losses = m.train(X_train, n_epochs, verbose=False)

			# For parts a and b.
			# Display each reconstructed digit.
			with torch.no_grad():
				visualize(m.model(X_train), digit_indices, save_path=f'../figures/a5_{is_linear}_{h_vals[i]}.pdf')

			# For part c.
			# Measure the model loss over the test data.
			test_loss = m.get_loss(X_test)

			# Display the losses.
			print(f'Train loss: {losses[-1]}')
			print(f'Test loss: {test_loss.item()}')

			# Update the loss measures.
			train_losses += f'{is_linear} {h_vals[i]} {losses[-1]}\n'
			test_losses += f'{is_linear} {h_vals[i]} {test_loss.item()}\n'

	save(train_losses, path=f'../data/a5_train_losses.txt')
	save(test_losses, path=f'../data/a5_test_losses.txt')
