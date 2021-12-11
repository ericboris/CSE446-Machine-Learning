import torch
from torch.utils.data import Dataset

import os
import pickle
import csv

from collections import Counter


def load_embedding_matrix(vocab, glove_file_path="glove.6B.50d.txt"):
    embedding_dim = -1

    embeddings = {}

    with open(glove_file_path, "r") as f:
        for token_embedding in f.readlines():
            token, *embedding = token_embedding.strip().split(" ")

            if token not in vocab:
                continue

            embedding = torch.tensor([float(e) for e in embedding], dtype=torch.float32)

            assert token not in embeddings
            assert embedding_dim < 0 or embedding_dim == len(embedding)

            embeddings[token] = embedding
            embedding_dim = len(embedding)

    all_embeddings = torch.stack(list(embeddings.values()), dim=0)

    embedding_mean = all_embeddings.mean()
    embedding_std = all_embeddings.std()

    # Randomly initialize embeddings
    embedding_matrix = torch.normal(embedding_mean, embedding_std, (len(vocab), embedding_dim))

    # Overwrite the embeddings we get from GloVe. The ones we don't find are left randomly initialized.
    for token, embedding in embeddings.items():
        embedding_matrix[vocab[token], :] = embedding

    # The padding token is explicitly initialized to 0.
    embedding_matrix[vocab["[pad]"]] = 0.

    return embedding_matrix


class SST2Dataset(Dataset):
    def __init__(self, path, vocab=None, reverse_vocab=None):
        super().__init__()

        sentences = []
        labels = []

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)  # Ignore header
            for row in reader:
                # Each row contains a sentence and label (either 0 or 1)
                sentence, label = row
                sentences.append(sentence.strip().split(" "))
                labels.append([int(label)])

        # Vocab maps tokens to indices
        if vocab is None:
            vocab = self._build_vocab(sentences)
            reverse_vocab = None

        # Reverse vocab maps indices to tokens
        if reverse_vocab is None:
            reverse_vocab = {index: token for token, index in vocab.items()}

        self.vocab = vocab
        self.reverse_vocab = reverse_vocab

        indexed_sentences = [torch.tensor(self.tokens_to_indices(sentence)) for sentence in sentences]
        labels = torch.tensor(labels)

        self.sentences = indexed_sentences
        self.labels = labels

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def _build_vocab(sentences, unk_cutoff=1, vocab_file_path="vocab.pkl"):
        # Load cached vocab if existent
        if os.path.exists(vocab_file_path):
            with open(vocab_file_path, "rb") as f:
                return pickle.load(f)

        word_counts = Counter()

        # Count unique words (lower case)
        for sentence in sentences:
            for token in sentence:
                word_counts[token.lower()] += 1

        # Special tokens: padding, beginning of sentence, end of sentence, and unknown word
        vocab = {"[pad]": 0, "[unk]": 1}
        token_id = 2

        # Assign a unique id to each word that occurs at least unk_cutoff number of times
        for token, count in word_counts.items():
            if count >= unk_cutoff:
                vocab[token] = token_id
                token_id += 1

        # Cache vocab
        with open(vocab_file_path, "wb") as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        return vocab

    def tokens_to_indices(self, tokens):
        """
        Converts tokens to indices.
        :param tokens: A list of tokens (strings)
        :return: A tensor of shape (n, 1) containing the token indices
        """
        indices = []

        unk_token = self.vocab["[unk]"]

        for token in tokens:
            indices.append(self.vocab.get(token.lower(), unk_token))

        return torch.tensor(indices)

    def indices_to_tokens(self, indices):
        """
        Converts indices to tokens and concatenates them as a string.
        :param indices: A tensor of indices of shape (n, 1), a list of (1, 1) tensors or a list of indices (ints)
        :return: The string containing tokens, concatenated by a space.
        """
        tokens = []

        for index in indices:
            if torch.is_tensor(index):
                index = index.item()
            token = self.reverse_vocab.get(index, "[unk]")
            if token == "[pad]":
                continue
            tokens.append(token)

        return " ".join(tokens)
