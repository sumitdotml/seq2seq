#!/usr/bin/env python3
"""
Training Visualization
Real-time plotting of training and validation losses
"""

import os

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np


class LossVisualizer:
    """Real-time loss visualization for seq2seq training"""

    def __init__(self, save_dir: str = "plots"):
        """
        Initialize the visualizer

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Set up matplotlib for better-looking plots
        style.use("seaborn-v0_8" if "seaborn-v0_8" in style.available else "default")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12

        # Initialize the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle("Seq2Seq Training Progress", fontsize=16, fontweight="bold")

        # Enable interactive mode for real-time updates
        plt.ion()

        print(
            f"Loss visualization initialized. Plots will be saved to: {
                save_dir}/"
        )

    def update(self, epoch: int, train_loss: float, val_loss: float):
        """
        Update the plots with new loss values

        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
        """
        # Store the data
        self.epochs.append(epoch + 1)  # 1-indexed for display
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Plot 1: Loss curves
        self.ax1.plot(
            self.epochs,
            self.train_losses,
            "b-",
            linewidth=2,
            label="Training Loss",
            marker="o",
        )
        self.ax1.plot(
            self.epochs,
            self.val_losses,
            "r-",
            linewidth=2,
            label="Validation Loss",
            marker="s",
        )
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss")
        self.ax1.set_title("Training & Validation Loss")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Add loss values as text
        if len(self.epochs) > 0:
            latest_train = self.train_losses[-1]
            latest_val = self.val_losses[-1]
            self.ax1.text(
                0.02,
                0.98,
                f"Latest - Train: {latest_train:.4f}, Val: {latest_val:.4f}",
                transform=self.ax1.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Plot 2: Loss improvement (difference from first epoch)
        if len(self.train_losses) > 1:
            train_improvement = [
                (self.train_losses[0] - loss) for loss in self.train_losses
            ]
            val_improvement = [(self.val_losses[0] - loss) for loss in self.val_losses]

            self.ax2.plot(
                self.epochs,
                train_improvement,
                "b-",
                linewidth=2,
                label="Training Improvement",
                marker="o",
            )
            self.ax2.plot(
                self.epochs,
                val_improvement,
                "r-",
                linewidth=2,
                label="Validation Improvement",
                marker="s",
            )
            self.ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            self.ax2.set_xlabel("Epoch")
            self.ax2.set_ylabel("Loss Improvement (from start)")
            self.ax2.set_title("Training Progress (Improvement from Start)")
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)

        # Refresh the plot
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Brief pause for real-time update

        # Save the plot
        plot_path = os.path.join(self.save_dir, "training_progress.png")
        self.fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        print(
            f"📊 Epoch {
                epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

    def save_final_plot(self, best_val_loss: float):
        """Save final summary plot"""

        # Add best validation loss marker
        if self.val_losses:
            best_epoch = np.argmin(self.val_losses) + 1
            self.ax1.axvline(
                x=best_epoch,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"Best Val Loss: {
                    best_val_loss:.4f} (Epoch {best_epoch})",
            )
            self.ax1.legend()

        # Save final plot
        final_path = os.path.join(self.save_dir, "final_training_results.png")
        self.fig.savefig(final_path, dpi=300, bbox_inches="tight")

        print(f"Final training plot saved to: {final_path}")

        # Show summary statistics
        if len(self.train_losses) > 0:
            print("\nTraining Summary:")
            print(f"   Initial Train Loss: {self.train_losses[0]:.4f}")
            print(f"   Final Train Loss: {self.train_losses[-1]:.4f}")
            print(
                f"   Train Improvement: {
                    self.train_losses[0] - self.train_losses[-1]:.4f}"
            )
            print(f"   Initial Val Loss: {self.val_losses[0]:.4f}")
            print(f"   Best Val Loss: {best_val_loss:.4f}")
            print(
                f"   Val Improvement: {
                    self.val_losses[0] - best_val_loss:.4f}"
            )

    def close(self):
        """Close the visualizer"""
        plt.ioff()  # Turn off interactive mode
        plt.close(self.fig)


# Quick test function
def test_visualizer():
    """Test the visualizer with fake data"""
    visualizer = LossVisualizer("test_plots")

    # Simulate training with decreasing loss
    np.random.seed(42)
    for epoch in range(10):
        # Simulate realistic loss curves
        train_loss = 5.0 * np.exp(-epoch * 0.3) + np.random.normal(0, 0.1)
        val_loss = 5.2 * np.exp(-epoch * 0.25) + np.random.normal(0, 0.15)

        visualizer.update(epoch, train_loss, val_loss)

        # Simulate training time
        import time

        time.sleep(0.5)

    visualizer.save_final_plot(min(visualizer.val_losses))
    visualizer.close()

    print("Visualizer test completed!")


if __name__ == "__main__":
    test_visualizer()
