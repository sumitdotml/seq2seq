#!/usr/bin/env python3
"""
Step 4: PyTorch Dataset & DataLoader
Convert tokenized text pairs to padded tensor batches for efficient training
"""

import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TranslationDataset(Dataset):
    """PyTorch Dataset for German-English translation pairs"""

    def __init__(self, pairs_file, german_tokenizer_file, english_tokenizer_file):
        """
        Initialize dataset with tokenized sentence pairs

        Args:
            pairs_file: JSON file with [{'src': german, 'tgt': english}, ...]
            german_tokenizer_file: Pickle file with German tokenizer
            english_tokenizer_file: Pickle file with English tokenizer
        """

        # Load sentence pairs
        with open(pairs_file, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

        # Load tokenizers
        self.german_tokenizer = self._load_tokenizer(german_tokenizer_file)
        self.english_tokenizer = self._load_tokenizer(english_tokenizer_file)

        print(f"Loaded {len(self.pairs)} sentence pairs")
        print(f"German vocab size: {self.german_tokenizer.vocab_size}")
        print(f"English vocab size: {self.english_tokenizer.vocab_size}")

    def _load_tokenizer(self, filepath):
        """Load tokenizer from pickle file"""
        from .tokenization import SimpleTokenizer  # Import our tokenizer class

        tokenizer = SimpleTokenizer()
        tokenizer.load(filepath)
        return tokenizer

    def __len__(self):
        """Return number of sentence pairs"""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Get a single training example

        Returns:
            src_tokens: German sentence as token IDs (without special tokens)
            tgt_tokens: English sentence as token IDs (with <START> and <END>)
        """
        pair = self.pairs[idx]
        german_sentence = pair["src"]
        english_sentence = pair["tgt"]

        # Tokenize German (source) - no special tokens needed for encoder input
        src_tokens = self.german_tokenizer.encode(german_sentence)

        # Tokenize English (target) - add <START> and <END> for decoder
        tgt_tokens = self.english_tokenizer.encode_with_special_tokens(
            english_sentence, add_start=True, add_end=True
        )

        # Convert to tensors
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)

        return src_tensor, tgt_tensor

    def get_vocab_sizes(self):
        """Return vocabulary sizes for model initialization"""
        return self.german_tokenizer.vocab_size, self.english_tokenizer.vocab_size

    def get_tokenizers(self):
        """Return tokenizers for inference/evaluation"""
        return self.german_tokenizer, self.english_tokenizer


def collate_batch(batch):
    """
    Custom collate function to pad sequences in a batch

    Args:
        batch: List of (src_tensor, tgt_tensor) tuples

    Returns:
        src_batch: Padded source sequences [batch_size, max_src_len]
        tgt_batch: Padded target sequences [batch_size, max_tgt_len]
    """

    # Separate source and target sequences
    src_sequences = [item[0] for item in batch]
    tgt_sequences = [item[1] for item in batch]

    # Pad sequences to same length within batch
    # We need (batch_size, seq_len), so doing batch_first=True
    src_batch = pad_sequence(
        src_sequences, batch_first=True, padding_value=0
    )  # 0 = <PAD>
    tgt_batch = pad_sequence(
        tgt_sequences, batch_first=True, padding_value=0
    )  # 0 = <PAD>

    return src_batch, tgt_batch


def create_dataloaders(
    train_pairs_file,
    val_pairs_file,
    german_tokenizer_file,
    english_tokenizer_file,
    batch_size=32,
    num_workers=0,
):
    """
    Create training and validation DataLoaders

    Args:
        train_pairs_file: Training sentence pairs JSON
        val_pairs_file: Validation sentence pairs JSON
        german_tokenizer_file: German tokenizer pickle file
        english_tokenizer_file: English tokenizer pickle file
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        vocab_sizes: Tuple of (german_vocab_size, english_vocab_size)
        tokenizers: Tuple of (german_tokenizer, english_tokenizer)
    """

    print("Creating datasets...")

    # Create datasets
    train_dataset = TranslationDataset(
        train_pairs_file, german_tokenizer_file, english_tokenizer_file
    )

    val_dataset = TranslationDataset(
        val_pairs_file, german_tokenizer_file, english_tokenizer_file
    )

    # Get vocab sizes and tokenizers from training dataset
    vocab_sizes = train_dataset.get_vocab_sizes()
    tokenizers = train_dataset.get_tokenizers()

    print(f"Training dataset: {len(train_dataset)} pairs")
    print(f"Validation dataset: {len(val_dataset)} pairs")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader, vocab_sizes, tokenizers


def test_dataloader():
    """Test the DataLoader with a small batch"""

    print("Testing DataLoader...")

    # Create DataLoaders
    train_loader, val_loader, vocab_sizes, tokenizers = create_dataloaders(
        "data/train_pairs.json",
        "data/val_pairs.json",
        "data/german_tokenizer.pkl",
        "data/english_tokenizer.pkl",
        batch_size=4,  # Small batch for testing
        num_workers=0,
    )

    german_tokenizer, english_tokenizer = tokenizers

    # Get first batch
    src_batch, tgt_batch = next(iter(train_loader))

    print("\nBatch shapes:")
    print(f"Source batch: {src_batch.shape}")  # [batch_size, max_src_len]
    print(f"Target batch: {tgt_batch.shape}")  # [batch_size, max_tgt_len]

    print("\nFirst example in batch:")
    print(f"German tokens: {src_batch[0].tolist()}")
    print(f"English tokens: {tgt_batch[0].tolist()}")

    # Decode back to text
    german_text = german_tokenizer.decode(src_batch[0].tolist())
    english_text = english_tokenizer.decode(tgt_batch[0].tolist())

    print("\nDecoded text:")
    print(f"German: {german_text}")
    print(f"English: {english_text}")

    return train_loader, val_loader, vocab_sizes, tokenizers


if __name__ == "__main__":
    # Test the DataLoader
    train_loader, val_loader, vocab_sizes, tokenizers = test_dataloader()

    print("\nDataLoader creation successful!")
    print(f"German vocab size: {vocab_sizes[0]}")
    print(f"English vocab size: {vocab_sizes[1]}")
    print("\nNext Steps:")
    print("1. DataLoader created and tested")
    print("2. Setup model hyperparameters")
    print("3. Initialize seq2seq model")
    print("4. Start training!")
