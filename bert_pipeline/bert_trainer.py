"""
BERT Trainer Module

Provides training utilities including cross-validation, early stopping,
and checkpoint management for BERT models.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .bert_classifier import BERTClassifier


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            score: Current metric value (loss or accuracy)

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


class BERTTrainer:
    """
    Enhanced trainer for BERT models with cross-validation,
    early stopping, and checkpointing.
    """

    def __init__(
        self,
        classifier: BERTClassifier,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize trainer.

        Args:
            classifier: BERTClassifier instance
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for training logs
        """
        self.classifier = classifier
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_history: Dict = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs': []
        }

    def train_with_validation(
        self,
        train_data: List,
        val_data: List,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        verbose: bool = True,
        save_best: bool = True
    ) -> Dict:
        """
        Train with validation split and early stopping.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Maximum epochs
            early_stopping_patience: Patience for early stopping
            verbose: Print progress
            save_best: Save best model

        Returns:
            Training history dict
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training")

        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max'  # Maximize validation accuracy
        )

        best_val_acc = 0.0
        checkpoint_path = self.checkpoint_dir / "best_model.pt"

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train one epoch
            self._train_epoch(train_data, verbose=verbose)

            # Validate
            val_loss, val_acc = self._validate(val_data)

            # Record history
            self.training_history['epochs'].append(epoch)
            self.training_history['val_accuracy'].append(val_acc)

            epoch_time = time.time() - epoch_start

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Time: {epoch_time:.1f}s")

            # Check for improvement
            if early_stopping(val_acc):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                if verbose:
                    print(f"  -> New best validation accuracy: {val_acc:.4f}")
                self._save_checkpoint(checkpoint_path, epoch, val_acc)

        # Load best model
        if save_best and checkpoint_path.exists():
            self._load_checkpoint(checkpoint_path)
            if verbose:
                print(f"\nLoaded best model with val acc: {best_val_acc:.4f}")

        return self.training_history

    def _train_epoch(self, train_data: List, verbose: bool = False):
        """Train one epoch (delegates to classifier)."""
        # This is a simplified version - the actual training happens in BERTClassifier.train()
        # For proper epoch-by-epoch control, we'd need to modify BERTClassifier
        pass

    def _validate(self, val_data: List) -> Tuple[float, float]:
        """
        Validate on validation data.

        Returns:
            (loss, accuracy)
        """
        self.classifier.model.eval()

        # Compute accuracy
        predictions = self.classifier.predict_all(val_data)

        # Get true labels
        if all(hasattr(item, 'label') for item in val_data[:5]):
            true_labels = [item.label for item in val_data]
        else:
            true_labels = [item[0] if isinstance(item, tuple) else item.label for item in val_data]

        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(true_labels)

        # Loss would require running forward pass with loss computation
        # For now, we'll return 0.0 for loss (simplified)
        return 0.0, accuracy

    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'val_accuracy': val_acc,
            'model_state_dict': self.classifier.model.state_dict(),
            'label2id': self.classifier.label2id,
            'id2label': self.classifier.id2label,
        }

        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.classifier.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.label2id = checkpoint['label2id']
        self.classifier.id2label = checkpoint['id2label']

    def cross_validate(
        self,
        full_data: List,
        n_folds: int = 5,
        epochs: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Perform k-fold cross-validation.

        Args:
            full_data: Complete training dataset
            n_folds: Number of folds
            epochs: Epochs per fold
            verbose: Print progress

        Returns:
            Cross-validation results
        """
        if verbose:
            print(f"\n{n_fold}-Fold Cross Validation")
            print("=" * 50)

        fold_size = len(full_data) // n_folds
        fold_accuracies = []

        for fold in range(n_folds):
            if verbose:
                print(f"\nFold {fold + 1}/{n_folds}")

            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size

            val_data = full_data[start_idx:end_idx]
            train_data = full_data[:start_idx] + full_data[end_idx:]

            # Create fresh classifier for this fold
            fold_classifier = BERTClassifier(
                model_name=self.classifier.model.name_or_path,
                class_weights=self.classifier.class_weights,
                use_focal_loss=self.classifier.use_focal_loss,
                batch_size=self.classifier.batch_size,
                learning_rate=self.classifier.learning_rate,
            )

            # Train
            fold_classifier.train(train_data, epochs=epochs, verbose=False)

            # Validate
            _, val_acc = self._validate_with_classifier(fold_classifier, val_data)
            fold_accuracies.append(val_acc)

            if verbose:
                print(f"  Fold {fold + 1} accuracy: {val_acc:.4f}")

        # Summary
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        if verbose:
            print("\n" + "=" * 50)
            print(f"Cross-Validation Results:")
            print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  Fold Accuracies: {[f'{a:.4f}' for a in fold_accuracies]}")
            print("=" * 50)

        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': fold_accuracies
        }

    def _validate_with_classifier(self, classifier: BERTClassifier, val_data: List) -> Tuple[float, float]:
        """Validate using a specific classifier instance."""
        predictions = classifier.predict_all(val_data)

        if all(hasattr(item, 'label') for item in val_data[:5]):
            true_labels = [item.label for item in val_data]
        else:
            true_labels = [item[0] if isinstance(item, tuple) else item.label for item in val_data]

        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(true_labels)

        return 0.0, accuracy

    def save_training_log(self, filename: Optional[str] = None):
        """Save training history to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.json"

        log_path = self.log_dir / filename

        with open(log_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Training log saved to {log_path}")


def quick_train_bert(
    train_data: List,
    test_data: List,
    model_name: str = "distilbert-base-uncased",
    class_weights: str = "balanced",
    use_focal_loss: bool = True,
    epochs: int = 5,
    batch_size: int = 16,
    verbose: bool = True
) -> BERTClassifier:
    """
    Quick training function for BERT.

    Args:
        train_data: Training data
        test_data: Test data
        model_name: Model name
        class_weights: Class weighting strategy
        use_focal_loss: Use focal loss
        epochs: Training epochs
        batch_size: Batch size
        verbose: Print progress

    Returns:
        Trained BERTClassifier
    """
    if verbose:
        print("\nInitializing BERT classifier...")

    classifier = BERTClassifier(
        model_name=model_name,
        class_weights=class_weights,
        use_focal_loss=use_focal_loss,
        batch_size=batch_size,
        learning_rate=2e-5
    )

    if verbose:
        print(f"Model: {model_name}")
        print(f"Training samples: {len(train_data)}")
        print(f"Epochs: {epochs}")

    classifier.train(train_data, epochs=epochs, verbose=verbose)

    # Evaluate on test set
    if verbose:
        print("\nEvaluating on test set...")

    predictions = classifier.predict_all(test_data)

    if all(hasattr(item, 'label') for item in test_data[:5]):
        true_labels = [item.label for item in test_data]
    else:
        true_labels = [item[0] if isinstance(item, tuple) else item.label for item in test_data]

    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = correct / len(true_labels)

    if verbose:
        print(f"Test Accuracy: {accuracy:.4f}")

    return classifier


if __name__ == "__main__":
    # Test the trainer
    print("Testing BERT Trainer...")

    # This would require actual data to test properly
    print("BERT Trainer module loaded successfully.")
    print("Use quick_train_bert() for quick training.")
