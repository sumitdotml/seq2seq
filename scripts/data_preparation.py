#!/usr/bin/env python3
"""
Step 1 & 2: Dataset Loading and Small Training Subset Creation
Load WMT19 and create manageable subset for seq2seq learning
"""

import json
import os
from collections import Counter

from datasets import load_dataset


def create_training_dataset(use_full_dataset=True, subset_size=50000):
    """Create training dataset from WMT19 DE-EN"""

    if use_full_dataset:
        print("Loading FULL WMT19 DE-EN dataset...")
        train_dataset = load_dataset("wmt19", "de-en", split="train")
        val_dataset = load_dataset("wmt19", "de-en", split="validation")
    else:
        print(f"Loading {subset_size} examples from WMT19 DE-EN...")
        # Load subset for testing
        train_dataset = load_dataset("wmt19", "de-en", split=f"train[:{subset_size}]")
        val_dataset = load_dataset(
            # More validation examples
            "wmt19",
            "de-en",
            split="validation[:2000]",
        )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    # Converting to simple format with improved filtering for full dataset
    def extract_pairs(dataset, max_length=80, min_length=3):
        pairs = []
        processed = 0

        for example in dataset:
            german = example["translation"]["de"].strip()
            english = example["translation"]["en"].strip()

            # More sophisticated filtering for production dataset
            german_words = len(german.split())
            english_words = len(english.split())

            # Filter criteria:
            # 1. Length constraints (not too short or too long)
            # 2. Skip empty sentences
            # 3. Skip sentences with weird characters (basic cleanup)
            if (
                min_length <= german_words <= max_length
                and min_length <= english_words <= max_length
                and german
                and english
                and not any(char in german + english for char in ["<", ">", "{", "}"])
            ):

                pairs.append(
                    {
                        "src": german,  # Source (German)
                        "tgt": english,  # Target (English)
                    }
                )

            processed += 1
            if processed % 100000 == 0:
                print(f"Processed {processed} examples, kept {len(pairs)}")

        return pairs

    train_pairs = extract_pairs(train_dataset)
    val_pairs = extract_pairs(val_dataset)

    print(
        f"After filtering (3-80 words): {len(train_pairs)
                                         } train, {len(val_pairs)} val"
    )

    # Saving to JSON files for easy loading
    os.makedirs("data", exist_ok=True)

    with open("data/train_pairs.json", "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2, ensure_ascii=False)

    with open("data/val_pairs.json", "w", encoding="utf-8") as f:
        json.dump(val_pairs, f, indent=2, ensure_ascii=False)

    print("Saved to data/train_pairs.json and data/val_pairs.json")

    #  some examples
    print("\nSample training pairs:")
    for i, pair in enumerate(train_pairs[:3]):
        print(f"\n--- Pair {i+1} ---")
        print(f"SRC: {pair['src']}")
        print(f"TGT: {pair['tgt']}")

    # Analyzing vocabulary
    print("\nVocabulary Analysis:")

    # Collect all words
    src_words = set()
    tgt_words = set()
    src_counter = Counter()
    tgt_counter = Counter()

    for pair in train_pairs:
        # Simple word tokenization (split by space)
        src_tokens = pair["src"].lower().split()
        tgt_tokens = pair["tgt"].lower().split()

        src_words.update(src_tokens)
        tgt_words.update(tgt_tokens)
        src_counter.update(src_tokens)
        tgt_counter.update(tgt_tokens)

    print(f"German vocabulary size: {len(src_words)} unique words")
    print(f"English vocabulary size: {len(tgt_words)} unique words")
    print(f"\nMost common German words: {src_counter.most_common(10)}")
    print(f"Most common English words: {tgt_counter.most_common(10)}")

    return train_pairs, val_pairs


if __name__ == "__main__":
    # Creating training dataset - full dataset will be used by default but
    # this might kill your RAM - just like it killed my mac mini lol

    # I had to set use_full_dataset=False for testing with smaller subset
    # If you have confidence in your computer, you can use the full dataset
    train_pairs, val_pairs = create_training_dataset(
        use_full_dataset=False,  # Using subset for manageable memory usage
        subset_size=2000000,  # 2M examples
    )

    print("\nNext Steps:")
    print("1. Full WMT19 dataset processed and saved")
    print("2. Vocabulary analyzed")
    print("3. Build tokenizer with larger vocabulary")
    print("4. Update training configuration for larger dataset")
