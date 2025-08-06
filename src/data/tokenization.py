#!/usr/bin/env python3
"""
Step 3: Tokenization & Vocabulary Building
Create word-to-ID mappings and convert text to sequences
"""

import json
import os
import pickle
from collections import Counter


class SimpleTokenizer:
    """Simple word-level tokenizer for seq2seq"""

    def __init__(self):
        # Special tokens (these MUST be first for consistent IDs)
        self.special_tokens = {
            "<PAD>": 0,  # Padding token
            "<UNK>": 1,  # Unknown word
            "<START>": 2,  # Sequence start
            "<END>": 3,  # Sequence end
        }
        self.word2id = self.special_tokens.copy()
        self.id2word = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)

    def build_vocabulary(self, sentences, max_vocab_size=15000):
        """Build vocabulary from list of sentences"""
        print(f"Building vocabulary (max size: {max_vocab_size})...")

        # Count word frequencies
        word_counter = Counter()
        for sentence in sentences:
            words = sentence.lower().split()  # Simple word tokenization
            word_counter.update(words)

        print(f"Found {len(word_counter)} unique words")

        # Add most frequent words to vocabulary (excluding special tokens)
        most_common = word_counter.most_common(
            max_vocab_size - len(self.special_tokens)
        )

        current_id = len(self.special_tokens)
        for word, count in most_common:
            if word not in self.word2id:  # Skip if already exists
                self.word2id[word] = current_id
                self.id2word[current_id] = word
                current_id += 1

        self.vocab_size = len(self.word2id)
        print(f"Final vocabulary size: {self.vocab_size}")
        print(f"Most common words: {[word for word, _ in most_common[:10]]}")

        return self.vocab_size

    def encode(self, sentence):
        """Convert sentence to list of token IDs"""
        words = sentence.lower().split()
        token_ids = []

        for word in words:
            if word in self.word2id:
                token_ids.append(self.word2id[word])
            else:
                token_ids.append(self.word2id["<UNK>"])  # Unknown word

        return token_ids

    def decode(self, token_ids):
        """Convert list of token IDs back to sentence"""
        words = []
        for token_id in token_ids:
            if token_id in self.id2word:
                word = self.id2word[token_id]
                if word not in [
                    "<PAD>",
                    "<START>",
                    "<END>",
                ]:  # Skip special tokens in output
                    words.append(word)

        return " ".join(words)

    def encode_with_special_tokens(self, sentence, add_start=True, add_end=True):
        """Encode sentence with <START> and <END> tokens"""
        token_ids = self.encode(sentence)

        if add_start:
            token_ids = [self.word2id["<START>"]] + token_ids
        if add_end:
            token_ids = token_ids + [self.word2id["<END>"]]

        return token_ids

    def save(self, filepath):
        """Save tokenizer to file"""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "word2id": self.word2id,
                    "id2word": self.id2word,
                    "vocab_size": self.vocab_size,
                },
                f,
            )
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.word2id = data["word2id"]
        self.id2word = data["id2word"]
        self.vocab_size = data["vocab_size"]
        print(f"Tokenizer loaded from {filepath}")


def create_tokenizers():
    """Create tokenizers for German and English"""

    # Load our prepared data
    with open("data/train_pairs.json", "r", encoding="utf-8") as f:
        train_pairs = json.load(f)

    print(f"Loaded {len(train_pairs)} training pairs")

    # Extract all sentences
    german_sentences = [pair["src"] for pair in train_pairs]
    english_sentences = [pair["tgt"] for pair in train_pairs]

    # Creating tokenizers with larger vocabularies for full dataset
    print("\nCreating German tokenizer...")
    german_tokenizer = SimpleTokenizer()
    german_vocab_size = german_tokenizer.build_vocabulary(
        german_sentences, max_vocab_size=30000
    )

    print("\nCreating English tokenizer...")
    english_tokenizer = SimpleTokenizer()
    english_vocab_size = english_tokenizer.build_vocabulary(
        english_sentences, max_vocab_size=25000
    )

    # Save tokenizers
    os.makedirs("data", exist_ok=True)
    german_tokenizer.save("data/german_tokenizer.pkl")
    english_tokenizer.save("data/english_tokenizer.pkl")

    # Test tokenization
    print("\nTesting tokenization:")
    test_german = train_pairs[0]["src"]
    test_english = train_pairs[0]["tgt"]

    print(f"\nOriginal German: {test_german}")
    german_ids = german_tokenizer.encode_with_special_tokens(test_german)
    print(f"Token IDs: {german_ids}")
    print(f"Decoded: {german_tokenizer.decode(german_ids)}")

    print(f"\nOriginal English: {test_english}")
    english_ids = english_tokenizer.encode_with_special_tokens(test_english)
    print(f"Token IDs: {english_ids}")
    print(f"Decoded: {english_tokenizer.decode(english_ids)}")

    return german_tokenizer, english_tokenizer, german_vocab_size, english_vocab_size


if __name__ == "__main__":
    german_tokenizer, english_tokenizer, de_vocab_size, en_vocab_size = (
        create_tokenizers()
    )

    print("\nTokenization Complete!")
    print(f"German vocab size: {de_vocab_size}")
    print(f"English vocab size: {en_vocab_size}")
    print("\nNext Steps:")
    print("1. Tokenizers created and saved")
    print("2. Convert dataset to tensors")
    print("3. Create PyTorch DataLoader")
    print("4. Setup model training!")
