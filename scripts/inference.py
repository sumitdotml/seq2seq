#!/usr/bin/env python3
"""
Inference script for seq2seq German-English translation
Load trained model and generate translations for input German sentences
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.tokenization import SimpleTokenizer
from src.models import Decoder, Encoder, Seq2Seq


class Config:
    """Training configuration and hyperparameters - needed for checkpoint loading"""

    # Model hyperparameters
    EMBED_SIZE = 256  # Embedding dimension
    HIDDEN_SIZE = 512  # LSTM hidden size
    NUM_LAYERS = 2  # Number of LSTM layers
    DROPOUT = 0.3  # Dropout rate


class Seq2SeqTranslator:
    """Wrapper class for loading and using trained seq2seq model"""

    def __init__(self, model_path="checkpoints/best_model.pt", device=None):
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # loading tokenizers
        self.german_tokenizer = SimpleTokenizer()
        self.english_tokenizer = SimpleTokenizer()

        print("Loading tokenizers...")
        self.german_tokenizer.load("data/german_tokenizer.pkl")
        self.english_tokenizer.load("data/english_tokenizer.pkl")

        # loading model
        print(f"Loading model from {model_path}...")
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # extracting model configuration from checkpoint
        config = checkpoint.get("config", Config())
        embed_size = getattr(config, "EMBED_SIZE", 256)
        hidden_size = getattr(config, "HIDDEN_SIZE", 512)
        num_layers = getattr(config, "NUM_LAYERS", 2)
        dropout = getattr(config, "DROPOUT", 0.3)

        src_vocab_size = self.german_tokenizer.vocab_size
        tgt_vocab_size = self.english_tokenizer.vocab_size

        print(
            f"Model config: embed={embed_size}, hidden={
                hidden_size}, layers={num_layers}"
        )
        print(
            f"Vocabularies: German={
              src_vocab_size}, English={tgt_vocab_size}"
        )

        # initializing model architecture
        encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        decoder = Decoder(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.model = Seq2Seq(encoder, decoder, self.device)

        # loading trained weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def translate(self, german_sentence, max_length=50, verbose=False):
        """
        Translate a German sentence to English

        Args:
            german_sentence: Input German text
            max_length: Maximum length of generated translation
            verbose: Print tokenization details

        Returns:
            Translated English sentence
        """
        if verbose:
            print(f"\nInput: {german_sentence}")

        # tokenizing input
        src_tokens = self.german_tokenizer.encode_with_special_tokens(
            german_sentence, add_start=False, add_end=True
        )

        if verbose:
            print(f"German tokens: {src_tokens}")
            print(
                f"German decoded: {
                  self.german_tokenizer.decode(src_tokens)}"
            )

        # converting to tensor
        src_tensor = (
            torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        )

        # generating translation
        with torch.no_grad():
            generated_ids = self.model.generate(
                src_tensor,
                start_token=self.english_tokenizer.word2id["<START>"],
                end_token=self.english_tokenizer.word2id["<END>"],
                max_length=max_length,
            )

        # decoding generated tokens
        generated_tokens = generated_ids[0].cpu().tolist()  # removing batch dimension

        if verbose:
            print(f"Generated tokens: {generated_tokens}")

        translation = self.english_tokenizer.decode(generated_tokens)

        if verbose:
            print(f"Translation: {translation}")

        return translation

    def translate_batch(self, german_sentences, max_length=50):
        """Translate multiple German sentences"""
        translations = []
        for sentence in german_sentences:
            translation = self.translate(sentence, max_length)
            translations.append(translation)
        return translations


def demo_translation():
    """Demo function with sample German sentences"""
    print("=" * 60)
    print("Seq2Seq German-English Translation Demo")
    print("=" * 60)

    # initializing translator
    translator = Seq2SeqTranslator()

    # sample german sentences for testing
    test_sentences = [
        "Hallo, wie geht es dir?",
        "Ich liebe Deutschland.",
        "Das Wetter ist heute schön.",
        "Können Sie mir helfen?",
        "Wo ist der Bahnhof?",
        "Ich bin Student.",
        "Das ist ein gutes Buch.",
        "Wir gehen ins Kino.",
    ]

    print("\nTranslating sample sentences:")
    print("-" * 40)

    for i, german_text in enumerate(test_sentences, 1):
        print(f"{i}. German: {german_text}")
        translation = translator.translate(german_text, verbose=False)
        print(f"   English: {translation}")
        print()

    return translator


def interactive_mode():
    """Interactive translation mode"""
    print("\n" + "=" * 60)
    print("Interactive Translation Mode")
    print("Enter German sentences to translate (type 'quit' to exit)")
    print("=" * 60)

    translator = Seq2SeqTranslator()

    while True:
        try:
            german_input = input("\nGerman: ").strip()

            if german_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not german_input:
                continue

            translation = translator.translate(german_input, verbose=True)
            print(f"English: {translation}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seq2Seq German-English Translation")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--sentence", "-s", type=str, help="Translate a single German sentence"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show tokenization details"
    )

    args = parser.parse_args()

    if args.sentence:
        # translating single sentence
        translator = Seq2SeqTranslator()
        translation = translator.translate(args.sentence, verbose=args.verbose)
        print(f"German: {args.sentence}")
        print(f"English: {translation}")
    elif args.interactive:
        # interactive mode
        interactive_mode()
    else:
        # demo mode (default)
        demo_translation()
