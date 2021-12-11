import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from util import SST2Dataset, load_embedding_matrix
from hw4_p7 import RNNBinaryClassificationModel, collate_fn, TRAINING_BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,\
                VAL_BATCH_SIZE


def train():
    # Load datasets
    train_dataset = SST2Dataset("./SST-2/train.tsv")
    val_dataset = SST2Dataset("./SST-2/dev.tsv", train_dataset.vocab, train_dataset.reverse_vocab)

    # Create data loaders for creating and iterating over batches
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, collate_fn=collate_fn)

    # Print out some random examples from the data
    print("Data examples:")
    random_indices = torch.randperm(len(train_dataset))[:8].tolist()
    for index in random_indices:
        sequence_indices, label = train_dataset.sentences[index], train_dataset.labels[index]
        sentiment = "Positive" if label == 1 else "Negative"
        sequence = train_dataset.indices_to_tokens(sequence_indices)
        print(f"Sentiment: {sentiment}. Sentence: {sequence}")
    print()

    embedding_matrix = load_embedding_matrix(train_dataset.vocab)

    model = RNNBinaryClassificationModel(embedding_matrix)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        # Total loss across train data
        train_loss = 0.
        # Total number of correctly predicted training labels
        train_correct = 0
        # Total number of training sequences processed
        train_seqs = 0

        tqdm_train_loader = tqdm(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        model.train()
        for batch_idx, batch in enumerate(tqdm_train_loader):
            sentences_batch, labels_batch = batch

            # Make predictions
            logits = model(sentences_batch)

            # Compute loss and number of correct predictions
            loss = model.loss(logits, labels_batch)
            correct = model.accuracy(logits, labels_batch).item() * len(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics and update status
            train_loss += loss.item()
            train_correct += correct
            train_seqs += len(sentences_batch)
            tqdm_train_loader.set_description_str(
                f"[Loss]: {train_loss / (batch_idx + 1):.4f} [Acc]: {train_correct / train_seqs:.4f}")
        print()

        avg_train_loss = train_loss / len(tqdm_train_loader)
        train_accuracy = train_correct / train_seqs
        print(f"[Training Loss]: {avg_train_loss:.4f} [Training Accuracy]: {train_accuracy:.4f}")

        print("Validating")
        # Total loss across validation data
        val_loss = 0.
        # Total number of correctly predicted validation labels
        val_correct = 0
        # Total number of validation sequences processed
        val_seqs = 0

        tqdm_val_loader = tqdm(val_loader)

        model.eval()
        for batch_idx, batch in enumerate(tqdm_val_loader):
            sentences_batch, labels_batch = batch

            with torch.no_grad():
                # Make predictions
                logits = model(sentences_batch)

                # Compute loss and number of correct predictions and accumulate metrics and update status
                val_loss += model.loss(logits, labels_batch).item()
                val_correct += model.accuracy(logits, labels_batch).item() * len(logits)
                val_seqs += len(sentences_batch)
                tqdm_val_loader.set_description_str(
                    f"[Loss]: {val_loss / (batch_idx + 1):.4f} [Acc]: {val_correct / val_seqs:.4f}")
        print()

        avg_val_loss = val_loss / len(tqdm_val_loader)
        val_accuracy = val_correct / val_seqs
        print(f"[Validation Loss]: {avg_val_loss:.4f} [Validation Accuracy]: {val_accuracy:.4f}")


if __name__ == "__main__":
    train()
