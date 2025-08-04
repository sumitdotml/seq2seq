#!/usr/bin/env python3
"""
Step 1: Dataset Loading & Exploration
Understanding the WMT19 DE-EN dataset structure before preprocessing

This is unrelated to the model training; it's just a way to understand the
dataset structure and get a feel for the data.
"""

from datasets import load_dataset


def explore_wmt_dataset():
    """Load and explore the WMT19 German-English dataset"""

    print("Loading WMT19 DE-EN dataset...")
    # Load only a small subset first (1000 examples for exploration)
    dataset = load_dataset(
        "wmt19", "de-en", split="train[:1000]"  # Only first 1000 examples
    )

    print(f"Loaded {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")
    print()

    # Looking at the first few examples
    print("First 5 examples:")
    for i in range(5):
        example = dataset[i]
        german = example["translation"]["de"]
        english = example["translation"]["en"]

        print(f"\n--- Example {i+1} ---")
        print(f"German:  {german}")
        print(f"English: {english}")
        print(f"DE length: {len(german.split())} words")
        print(f"EN length: {len(english.split())} words")

    # Basic statistics of the dataset
    print("\nDataset Statistics:")
    german_lengths = [len(ex["translation"]["de"].split()) for ex in dataset]
    english_lengths = [len(ex["translation"]["en"].split()) for ex in dataset]

    print(
        f"German sentences - Avg: {sum(german_lengths)/len(german_lengths):.1f} words"
    )
    print(
        f"English sentences - Avg: {sum(english_lengths)/len(english_lengths):.1f} words"
    )
    print(f"Max German length: {max(german_lengths)} words")
    print(f"Max English length: {max(english_lengths)} words")

    return dataset


if __name__ == "__main__":
    explore_wmt_dataset()
    print("\nNext Steps:")
    print("1. Dataset loaded and explored")
    print("2. Create small training subset")
    print("3. Build tokenization")
    print("4. Create vocabularies")
