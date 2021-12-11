# Problem 4: Neural Networks for MNIST

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_loaders(root_path):
	''' Return MNIST train and test DataLoaders. '''
	train = torch.utils.data.DataLoader(
		datasets.MNIST(
			root=root_path,
			train=True,
			download=True,
			transform=transforms.ToTensor()),
		batch_size=128,
		shuffle=True)
	test = torch.utils.data.DataLoader(
		datasets.MNIST(
			root=root_path,
			train=False,
			download=True,
			transform=transforms.ToTensor()),
		batch_size=128,
		shuffle=True)
	return train, test

def count_parameters(weights, biases):
	''' Return the total number of parameters used in the given weights and biases. '''
	parameters = sum([np.prod(w.shape) for w in weights])
	parameters += sum([np.prod(b.shape) for b in biases])
	return parameters

def plot(title, x_label, y_label, file_name, x_1, y_1, label_1, x_2, y_2, label_2, dim=(8, 5)):
	''' Plot two lines on single chart. '''
	plt.figure(figsize=dim)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.plot(x_1, y_1, '-o', label=label_1)
	plt.plot(x_2, y_2, '-o', label=label_2)
	plt.xticks(np.arange(0, max(x_1)+1, 2.0))
	plt.legend()
	plt.savefig(file_name)
	plt.show()

class NeuralNetwork:
	def __init__(self, data_loader, learning_rate, n_neurons, n_layers, input_dim, output_dim):
		self.data_loader = data_loader
		self.learning_rate = learning_rate

		alpha = 1 / (np.sqrt(input_dim))
		self._weights(alpha, n_neurons, n_layers, input_dim, output_dim)
		self._biases(alpha, n_neurons, n_layers, output_dim)
	
	def _weights(self, alpha, n_neurons, n_layers, input_dim, output_dim):
		''' Create the model's weights. '''
		weights = []

		# Define the input layer weights.
		weights.append(-2 * alpha * torch.rand(n_neurons, input_dim) + alpha)
		weights[-1].requires_grad = True
		
		# Define the hidden layer weights.
		# Don't add the input and output layer weights.
		for _ in range(n_layers - 2):
			weights.append(-2 * alpha * torch.rand(n_neurons, n_neurons) + alpha)
			weights[-1].requires_grad = True
		
		# Define the output layer weights.
		weights.append(-2 * alpha * torch.rand(output_dim, n_neurons) + alpha)
		weights[-1].requires_grad = True
		
		self.weights = weights
		
	def _biases(self, alpha, n_neurons, n_layers, output_dim):
		''' Create the model's biases. '''
		biases = []

		# Define the input and hidden layer biases.
		# Don't add the output layer biases.
		for _ in range(n_layers - 1):
			biases.append(-2 * alpha * torch.rand(n_neurons) + alpha)
			biases[-1].requires_grad = True
		
		# Define the output layer biases.
		biases.append(-2 * alpha * torch.rand(output_dim) + alpha)
		biases[-1].requires_grad = True
	
		self.biases = biases
	
	def train(self, n_epochs, accuracy_threshold=0.99, verbose=False):
		''' Train the model. '''
		# Use the weights and biases as the parameter.
		optimizer = torch.optim.Adam(self.weights + self.biases, lr=self.learning_rate)

		accuracies = []
		losses = []
		for epoch in range(n_epochs):
			# Perform forward and backwards passes through the data and return the model performance.
			accuracy, loss = self.measure_performance(self.data_loader, optimizer=optimizer, train=True)

			accuracies.append(accuracy)
			losses.append(loss)

			if verbose:
				print(f'Epoch={epoch}\tTraining Loss={losses[-1]}\tAccuracy={accuracy}')
	
			# End training if minimium accuracy threshold is met.	
			if accuracy > accuracy_threshold:
				break

		return accuracies, losses, self.weights, self.biases

	def measure_performance(self, data_loader, optimizer=None, train=False):
		''' Perform forwards and backwards passes on the model and return the performance. '''
		accuracy = 0
		loss = 0

		for X, y in tqdm(iter(data_loader)):
			# Change the dimensions of X.
			X = torch.flatten(X, start_dim=1, end_dim=3)
		
			# Perform the forward pass and get the predictions.	
			logits = self._forward(X)
			y_hat = torch.argmax(logits, 1)
		
			# Compute the accuracy and loss.	
			accuracy += torch.sum(y == y_hat)
			loss_tmp = torch.nn.functional.cross_entropy(logits, y, size_average=False)
		
			# Gradient descent backward pass.
			if train:
				optimizer.zero_grad()
				loss_tmp.backward()
				optimizer.step()

			loss += loss_tmp
	
		# Normalize the performance measures.
		loss /= len(data_loader.dataset)
		accuracy = accuracy.to(dtype=torch.float) / len(data_loader.dataset)

		return accuracy, loss

	def _forward(self, x):
		''' Perform a pass through the network using the given input x and ReLU nonlinearities. '''
		y = torch.matmul(x, self.weights[0].T) + self.biases[0]
		for i in range(1, len(self.weights)):
			y = torch.matmul(nn.functional.relu(y), self.weights[i].T) + self.biases[i]
		return y
	
def main():
	# Load the training and test data.
	train_loader, test_loader = get_loaders('../data/python_mnist/')

	# Part a: Build, train, and test a wide neural network.
	wide_net = NeuralNetwork(data_loader=train_loader, learning_rate=1E-3, n_neurons=64, n_layers=2, input_dim=784, output_dim=10)

	wide_train_accuracies, wide_train_losses, wide_weights, wide_biases = wide_net.train(n_epochs=500, verbose=True)
	wide_parameters = count_parameters(wide_weights, wide_biases)
	print(f'Wide net training results:\nAccuracy={wide_train_accuracies[-1]}\tLoss={wide_train_losses[-1]}\tN Parameters={wide_parameters}\n')

	wide_test_accuracy, wide_test_loss = wide_net.measure_performance(test_loader)
	print(f'Wide net test results:\nAccuracy={wide_test_accuracy}\tLoss={wide_test_loss}\n')

	# Part b: Build, train, and test a deep neural network.
	deep_net = NeuralNetwork(data_loader=train_loader, learning_rate=1E-3, n_neurons=32, n_layers=3, input_dim=784, output_dim=10)

	deep_train_accuracies, deep_train_losses, deep_weights, deep_biases = wide_net.train(n_epochs=500, verbose=True)
	deep_parameters = count_parameters(deep_weights, deep_biases)
	print(f'Deep net training results:\nAccuracy={deep_train_accuracies[-1]}\tLoss={deep_train_losses[-1]}\tN Parameters={deep_parameters}\n')

	deep_test_accuracy, deep_test_loss = wide_net.measure_performance(test_loader)
	print(f'Deep Net Test Results:\nAccuracy={deep_test_accuracy}\tLoss={deep_test_loss}\n')

	# Plot the performance of both models.
	x_wide_evenly_spaced = range(len(wide_train_losses))
	x_deep_evenly_spaced = range(len(deep_train_losses))

	plot(title='Training loss per epoch on wide and deep neural networks',
		x_label='epoch',
		y_label='error',
		file_name='../plots/4_losses.pdf',
		x_1=x_wide_evenly_spaced,
		y_1=[float(loss) for loss in wide_train_losses],
		label_1='wide network',
		x_2=x_deep_evenly_spaced,
		y_2=[float(loss) for loss in deep_train_losses],
		label_2='deep network')
		
	plot(title='Training accuracy per epoch on wide and deep neural networks',
		x_label='epoch',
		y_label='accuracy',
		file_name='../plots/4_accuracies.pdf',
		x_1=x_wide_evenly_spaced,
		y_1=wide_train_accuracies,
		label_1='wide network',
		x_2=x_deep_evenly_spaced,
		y_2=deep_train_accuracies,
		label_2='deep network')
	
	
if __name__ == '__main__':
	main()
