"""
BERT Classifier for TREC Question Classification

Optimized BERT implementation using DistilBERT with class-balanced loss
for small, imbalanced datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from collections import Counter

try:
    from data_reader import Text
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False


class TRECQuestionDataset(Dataset):
    """PyTorch Dataset for TREC questions."""

    def __init__(self, texts: List[str], labels: List[str], tokenizer, label2id: Dict[str, int]):
        """
        Initialize dataset.

        Args:
            texts: List of question texts
            labels: List of labels (e.g., "HUM:ind")
            tokenizer: DistilBERT tokenizer
            label2id: Mapping from label names to IDs
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=64, # reduced
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label2id[label], dtype=torch.long)
        }


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where p_t is the model's estimated probability for the correct class.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Class weights tensor (shape: [num_classes])
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (shape: [batch_size, num_classes])
            targets: Ground truth labels (shape: [batch_size])

        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BERTClassifier:
    """
    BERT-based classifier optimized for small, imbalanced datasets.

    Uses DistilBERT with class-balanced or focal loss.

    Compatible with the existing training interface:
    - train(training_data, epochs, verbose)
    - predict_all(test_data)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 50,
        class_weights: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0, # 1.5= less aggressive, 3= more emphasis on hard examples
        max_length: int = 128, # questions short—64 better than 128?
        learning_rate: float = 3e-5, # changeable from 2e-5
        batch_size: int = 16, # can it be 8 instead of 16?
        gradient_accumulation_steps: int = 2,
        warmup_steps: int = 500, # compare to total_steps
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize BERT classifier.

        Args:
            model_name: HuggingFace model name (default: distilbert-base-uncased)
            num_labels: Number of classification labels
            class_weights: 'balanced', 'inverse', or custom array
            use_focal_loss: Use focal loss instead of cross-entropy
            focal_gamma: Focal loss gamma parameter
            max_length: Maximum sequence length
            learning_rate: Learning rate for AdamW
            batch_size: Batch size for training
            gradient_accumulation_steps: Steps to accumulate gradients
            warmup_steps: Warmup steps for learning rate scheduler
            weight_decay: Weight decay for AdamW
            device: 'cuda', 'cpu', or None (auto-detect)
            random_seed: Random seed for reproducibility
        """
        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.device)

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # Loss configuration
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        self.class_weights_tensor = None

        # State
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.is_trained = False

    def _compute_class_weights(self, labels: List[str]) -> torch.Tensor:
        """
        Compute class weights based on label frequency.

        Args:
            labels: List of training labels

        Returns:
            Tensor of shape [num_classes] with class weights
        """
        label_counts = Counter(labels)

        # Get unique labels and create mappings
        unique_labels = sorted(set(labels))
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        num_classes = len(unique_labels)

        if isinstance(self.class_weights, np.ndarray):
            # Use provided weights directly
            return torch.tensor(self.class_weights, dtype=torch.float).to(self.device)

        if self.class_weights == 'balanced':
            # Inverse frequency weighting
            total = len(labels)
            weights = []
            for label in unique_labels:
                count = label_counts[label]
                weight = total / (num_classes * count)
                weights.append(weight)
            return torch.tensor(weights, dtype=torch.float).to(self.device)

        elif self.class_weights == 'inverse':
            # Simpler inverse frequency
            max_count = max(label_counts.values())
            weights = []
            for label in unique_labels:
                count = label_counts[label]
                weight = max_count / count
                weights.append(weight)
            return torch.tensor(weights, dtype=torch.float).to(self.device)

        else:
            # No weighting
            return torch.ones(num_classes, dtype=torch.float).to(self.device)

    def train(
        self,
        training_data: List,
        epochs: int = 3,
        verbose: bool = True,
        val_data: Optional[List] = None
    ):
        """
        Train the BERT classifier.

        Args:
            training_data: List of items with .label and .text attributes (compatible with existing code)
            epochs: Number of training epochs
            verbose: Whether to print progress
            val_data: Optional validation data for early stopping
        """
        if verbose:
            print(f"\nTraining BERT classifier for {epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Training samples: {len(training_data)}")

        # Extract texts and labels
        if DATAREADER_AVAILABLE and all(hasattr(item, 'label') and hasattr(item, 'text') for item in training_data[:5]):
            # Standard TREC format with Text objects
            texts = [item.text for item in training_data]
            labels = [item.label for item in training_data]
        else:
            # Assume tuple format (label, text)
            texts = [item[1] if isinstance(item, tuple) else item.text for item in training_data]
            labels = [item[0] if isinstance(item, tuple) else item.label for item in training_data]

        # Create label mappings and compute class weights
        self.class_weights_tensor = self._compute_class_weights(labels)

        if verbose:
            print(f"Unique labels: {len(self.label2id)}")
            print(f"Using focal loss: {self.use_focal_loss}")
            if self.class_weights_tensor is not None and not torch.all(self.class_weights_tensor == 1.0):
                print(f"Class weights range: {self.class_weights_tensor.min():.2f} - {self.class_weights_tensor.max():.2f}")

        # Create dataset and dataloader
        dataset = TRECQuestionDataset(texts, labels, self.tokenizer, self.label2id)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )

        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        total_steps = len(dataloader) * epochs // self.gradient_accumulation_steps
        # does total_steps go below 500 (cf. warmup_steps)?
        print('TOTAL STEPS: ', total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # Loss function
        if self.use_focal_loss:
            criterion = FocalLoss(
                alpha=self.class_weights_tensor,
                gamma=self.focal_gamma,
                reduction='mean'
            )
        else:
            # Use weighted cross-entropy
            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights_tensor if self.class_weights_tensor is not None else None
            )

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            total_loss = 0
            accumulation_steps = 0

            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}:")

            for step, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

                # Compute loss
                loss = criterion(logits, labels_batch)
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                accumulation_steps += 1

                # Update weights every gradient_accumulation_steps
                if accumulation_steps % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    total_loss += loss.item() * self.gradient_accumulation_steps

                    if verbose and step % 10 == 0:
                        avg_loss = total_loss / (step + 1)
                        print(f"  Step {step}/{len(dataloader)} | Loss: {avg_loss:.4f}")

            # Epoch summary
            avg_loss = total_loss / len(dataloader)
            if verbose:
                print(f"  Epoch {epoch + 1} average loss: {avg_loss:.4f}")

            # Validation if provided
            if val_data is not None:
                val_acc = self._evaluate_accuracy(val_data)
                if verbose:
                    print(f"  Validation accuracy: {val_acc:.4f}")

        self.is_trained = True

        if verbose:
            print("\nTraining complete!")

    def _evaluate_accuracy(self, data: List) -> float:
        """Compute accuracy on given data."""
        predictions = self.predict_all(data)

        if all(hasattr(item, 'label') for item in data[:5]):
            true_labels = [item.label for item in data]
        else:
            true_labels = [item[0] if isinstance(item, tuple) else item.label for item in data]

        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        return correct / len(true_labels)

    def predict_all(self, test_data: List) -> List[str]:
        """
        Predict labels for test data.

        Args:
            test_data: List of items with .text attribute or tuples

        Returns:
            List of predicted labels
        """
        if not self.is_trained and not self.label2id:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()

        # Extract texts
        if all(hasattr(item, 'text') for item in test_data[:5]):
            texts = [item.text for item in test_data]
        else:
            texts = [item[1] if isinstance(item, tuple) else item.text for item in test_data]

        predictions = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Get predictions
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()

                for pred_id in batch_predictions:
                    predictions.append(self.id2label[int(pred_id)])

        return predictions

    def predict_top_k_all(self, test_data: List, k: int = 2) -> List[List[Tuple[str, float]]]:
        """
        Return top K predictions with scores for each test item.

        Args:
            test_data: List of items with .text attribute
            k: Number of top predictions to return

        Returns:
            List of [(label, score), ...] for each test item
        """
        if not self.is_trained and not self.label2id:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()

        # Extract texts
        if all(hasattr(item, 'text') for item in test_data[:5]):
            texts = [item.text for item in test_data]
        else:
            texts = [item[1] if isinstance(item, tuple) else item.text for item in test_data]

        results = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Get softmax probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                # Get top k for each sample
                top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]

                for j, top_indices in enumerate(top_k_indices):
                    sample_results = []
                    for idx in top_indices:
                        label = self.id2label[int(idx)]
                        score = float(probs[j, idx])
                        sample_results.append((label, score))
                    results.append(sample_results)

        return results

    def save(self, save_path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    @classmethod
    def load(cls, load_path: str, **kwargs):
        """Load a saved model."""
        model = cls(**kwargs)
        model.model = DistilBertForSequenceClassification.from_pretrained(load_path)
        model.tokenizer = DistilBertTokenizer.from_pretrained(load_path)
        model.model.to(model.device)
        model.is_trained = True

        # Try to load label mappings if saved
        import json
        try:
            with open(f"{load_path}/label_mappings.json", 'r') as f:
                mappings = json.load(f)
                model.label2id = {k: int(v) for k, v in mappings['label2id'].items()}
                model.id2label = {int(k): v for k, v in mappings['id2label'].items()}
        except FileNotFoundError:
            print("Warning: Label mappings not found. You'll need to train before predicting.")

        return model

    def save_label_mappings(self, save_path: str):
        """Save label mappings separately."""
        import json
        mappings = {
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        with open(f"{save_path}/label_mappings.json", 'w') as f:
            json.dump(mappings, f)


if __name__ == "__main__":
    # Test the classifier
    print("Testing BERT Classifier...")

    # Create dummy data
    dummy_texts = [
        "What is the capital of France?",
        "Who is the president of the United States?",
        "How many states are in the USA?",
    ]
    dummy_labels = ["LOC:city", "HUM:ind", "NUM:count"]

    # Create classifier
    classifier = BERTClassifier(
        model_name="distilbert-base-uncased",
        class_weights='balanced',
        use_focal_loss=True,
        batch_size=2,
        epochs=1
    )

    # Create dummy training data in the expected format
    class DummyItem:
        def __init__(self, label, text):
            self.label = label
            self.text = text

    train_data = [DummyItem(l, t) for l, t in zip(dummy_labels, dummy_texts)]

    print("\nTraining on dummy data...")
    classifier.train(train_data, epochs=1, verbose=True)

    print("\nTesting predictions...")
    predictions = classifier.predict_all(train_data)
    for item, pred in zip(train_data, predictions):
        print(f"True: {item.label:15s} | Pred: {pred:15s} | Text: {item.text[:50]}")
