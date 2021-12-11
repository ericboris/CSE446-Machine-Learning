# HW4 Problem 7 - Text Classification

import torch
import torch.nn as nn

def collate_fn(batch):
	"""
	Create a batch of data given a list of N sequences and labels. 
	Sequences are stacked into a single tensor of shape (N, max_sequence_length), 
	where max_sequence_length is the maximum length of any sequence in the batch. 
	Sequences shorter than this length should be filled up with 0's. 
	Also returns a tensor of shape (N, 1)containing the label of each sequence.

	:param batch: A list of size N, where each element is a tuple containing a sequence tensor 
	and a single item tensor containing the true label of the sequence.

	:return: A tuple containing two tensors. The first tensor has shape 
	(N, max_sequence_length) and contains all sequences. 
	Sequences shorter than max_sequence_length are padded with 0s at the end. 
	The second tensor has shape (N, 1) and contains all labels.
	"""
	sentences, labels = zip(*batch)
	sentences, labels = list(sentences), torch.stack(list(labels))

	max_sequence_length = max(len(s) for s in sentences)

	for i, sentence in enumerate(sentences):
		length_diff = max_sequence_length - len(sentence)
		sentences[i] = torch.nn.functional.pad(sentence, [0, length_diff])

	sentences_tensor = torch.stack(sentences)
	return sentences_tensor, labels


class RNNBinaryClassificationModel(nn.Module):
	def __init__(self, embedding_matrix, hidden_size=64):
		super().__init__()
		embedding_dim = embedding_matrix.shape[1]
		self.num_layers = 6

		# Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
		self.embedding = nn.Embedding(num_embeddings=embedding_matrix.shape[0],
									embedding_dim=embedding_dim,
									padding_idx=0)
		self.embedding.weight.data = embedding_matrix

		# Construct 3 different types of RNN for comparison.
		self.RNN = nn.RNN(input_size=embedding_dim,
						hidden_size=hidden_size,
						num_layers=self.num_layers,
						batch_first=True)

		self.GRU = nn.GRU(input_size=embedding_dim, 
						hidden_size=hidden_size,
						num_layers=self.num_layers, 
						batch_first=True)

		self.LSTM = nn.LSTM(input_size=embedding_dim,
						hidden_size=hidden_size,
						num_layers=self.num_layers,
						batch_first=True,
						bidirectional=True)

		self.linear = nn.Linear(in_features=hidden_size, out_features=1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, inputs):
		"""
		Takes in a batch of data of shape (N, max_sequence_length). 
		Returns a tensor of shape (N, 1), where each
		element corresponds to the prediction for the corresponding sequence.
		:param inputs: Tensor of shape (N, max_sequence_length) containing N
			 sequences to make predictions for.
		:return: Tensor of predictions for each sequence of shape (N, 1).
		"""
		# Un-comment for training other models.
		#return self.sigmoid(self.linear(self.RNN(self.embedding(inputs))[1][-1].squeeze(0)))
		#return self.sigmoid(self.linear(self.GRU(self.embedding(inputs))[1][-1].squeeze(0)))
		return self.sigmoid(self.linear(self.LSTM(self.embedding(inputs))[1][0][-1].squeeze(0)))

	def loss(self, logits, targets):
		"""
		Computes the binary cross-entropy loss.
		:param logits: Raw predictions from the model of shape (N, 1)
		:param targets: True labels of shape (N, 1)
		:return: Binary cross entropy loss between logits and targets as a scalar tensor.
		"""
		return nn.BCELoss()(logits, targets.float())

	def accuracy(self, logits, targets):
		"""
		Computes the accuracy, i.e number of correct predictions / N.
		:param logits: Raw predictions from the model of shape (N, 1)
		:param targets: True labels of shape (N, 1)
		:return: Accuracy as a scalar tensor.
		"""
		correct = 0

		for i in range(len(logits)):
			prediction = torch.round(logits[i])
			correct += (prediction == targets[i]).sum().item()

		accuracy = correct / len(logits)
		return torch.tensor(accuracy)

# Training parameters
TRAINING_BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.0001

# Batch size for validation, this only affects performance.
VAL_BATCH_SIZE = 128
