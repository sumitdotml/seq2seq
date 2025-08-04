#!/usr/bin/env python3
"""
Step 5: Model Training Setup
Configure hyperparameters, initialize model, and start training loop
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Encoder, Decoder, Seq2Seq
from src.data import create_dataloaders
from src.utils import LossVisualizer


class Config:
    """Training configuration and hyperparameters"""

    # Data paths
    TRAIN_PAIRS = "data/train_pairs.json"
    VAL_PAIRS = "data/val_pairs.json"
    GERMAN_TOKENIZER = "data/german_tokenizer.pkl"
    ENGLISH_TOKENIZER = "data/english_tokenizer.pkl"

    # Model hyperparameters
    EMBED_SIZE = 256  # Embedding dimension
    HIDDEN_SIZE = 512  # LSTM hidden size
    NUM_LAYERS = 2  # Number of LSTM layers
    DROPOUT = 0.3  # Dropout rate

    # Training hyperparameters - adjusted for larger dataset
    BATCH_SIZE = 64  # Increased batch size for efficiency
    NUM_EPOCHS = 5   # Reduced epochs since we have much more data
    LEARNING_RATE = 0.0003  # Lower learning rate for stability with more data

    # Training settings
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    CLIP_GRAD = 1.0  # Gradient clipping threshold
    SAVE_EVERY = 2  # Save model every N epochs

    # Model save path
    MODEL_SAVE_DIR = "checkpoints"


def initialize_model(config, vocab_sizes):
    """Initialize the seq2seq model"""

    german_vocab_size, english_vocab_size = vocab_sizes

    print("Initializing model components...")
    print(f"German vocabulary: {german_vocab_size}")
    print(f"English vocabulary: {english_vocab_size}")
    print(f"Embed size: {config.EMBED_SIZE}")
    print(f"Hidden size: {config.HIDDEN_SIZE}")
    print(f"Num layers: {config.NUM_LAYERS}")
    print(f"Dropout: {config.DROPOUT}")

    # Create encoder
    encoder = Encoder(
        src_vocab_size=german_vocab_size,
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )

    # Create decoder
    decoder = Decoder(
        tgt_vocab_size=english_vocab_size,
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )

    # Create seq2seq model
    model = Seq2Seq(encoder, decoder, config.DEVICE)

    # Move to device
    model = model.to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model moved to: {config.DEVICE}")

    return model


def initialize_training(model, config):
    """Initialize loss function, optimizer, and scheduler"""

    # Loss function (ignore padding tokens in loss calculation)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = <PAD> token

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler (reduce LR every few epochs)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    print(f"Loss function: CrossEntropyLoss (ignoring padding)")
    print(f"Optimizer: Adam (lr={config.LEARNING_RATE})")
    print(f"Scheduler: StepLR (step_size=3, gamma=0.5)")

    return criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, config, epoch):
    """Train for one epoch"""

    model.train()  # Set to training mode
    total_loss = 0
    num_batches = len(train_loader)

    print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
    print("-" * 50)

    for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
        # Move data to device
        src_batch = src_batch.to(config.DEVICE)  # [batch_size, src_seq_len]
        tgt_batch = tgt_batch.to(config.DEVICE)  # [batch_size, tgt_seq_len]

        # Forward pass
        predictions, expected_output = model(src_batch, tgt_batch)

        # predictions: [batch_size, tgt_seq_len-1, english_vocab_size]
        # expected_output: [batch_size, tgt_seq_len-1]

        # Reshape for loss calculation
        predictions_flat = predictions.reshape(
            -1, predictions.size(-1)
        )  # [batch*seq, vocab]
        expected_flat = expected_output.reshape(-1)  # [batch*seq]

        # Calculate loss
        loss = criterion(predictions_flat, expected_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item()
                                                                      :.4f}, Avg Loss: {avg_loss:.4f}"
            )

    avg_epoch_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss


def validate(model, val_loader, criterion, config):
    """Validate the model"""

    model.eval()  # Set to evaluation mode
    total_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():  # No gradient computation during validation
        for src_batch, tgt_batch in val_loader:
            # Move data to device
            src_batch = src_batch.to(config.DEVICE)
            tgt_batch = tgt_batch.to(config.DEVICE)

            # Forward pass
            predictions, expected_output = model(src_batch, tgt_batch)

            # Reshape for loss calculation
            predictions_flat = predictions.reshape(-1, predictions.size(-1))
            expected_flat = expected_output.reshape(-1)

            # Calculate loss
            loss = criterion(predictions_flat, expected_flat)
            total_loss += loss.item()

    avg_val_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")

    return avg_val_loss


def main():
    """Main training function"""

    config = Config()

    # Create output directory
    import os

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("SEQ2SEQ TRANSLATION MODEL TRAINING")
    print("=" * 60)

    # Create data loaders
    print("\n1. Loading data...")
    train_loader, val_loader, vocab_sizes, tokenizers = create_dataloaders(
        config.TRAIN_PAIRS,
        config.VAL_PAIRS,
        config.GERMAN_TOKENIZER,
        config.ENGLISH_TOKENIZER,
        batch_size=config.BATCH_SIZE,
        num_workers=0,
    )

    # Initialize model
    print("\n2. Initializing model...")
    model = initialize_model(config, vocab_sizes)

    # Initialize training components
    print("\n3. Setting up training...")
    criterion, optimizer, scheduler = initialize_training(model, config)

    # Initialize visualizer
    print("\n4. Initializing visualization...")
    visualizer = LossVisualizer(save_dir="training_plots")

    # Training loop
    print("\n5. Starting training...")
    best_val_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, config)

        # Update visualization
        visualizer.update(epoch, train_loss, val_loss)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                f"{config.MODEL_SAVE_DIR}/best_model.pt",
            )
            print(f"New best model saved! (val_loss: {val_loss:.4f})")

        # Save checkpoint every few epochs
        if (epoch + 1) % config.SAVE_EVERY == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                f"{config.MODEL_SAVE_DIR}/checkpoint_epoch_{epoch+1}.pt",
            )
            print(f"Checkpoint saved for epoch {epoch+1}")

        print("-" * 50)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.MODEL_SAVE_DIR}/")

    # Save final visualization and cleanup
    visualizer.save_final_plot(best_val_loss)
    visualizer.close()


if __name__ == "__main__":
    main()
