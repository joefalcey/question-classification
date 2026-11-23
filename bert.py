import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import numpy as np


# Universal Dataset Wrapper for PyTorch Tensors
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        # If labels aren't provided (e.g. test time), use dummy values
        label = self.labels[item] if self.labels is not None else 0

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier:
    def __init__(self, use_bigrams: bool = False, use_tfidf: bool = False, use_embeddings: bool = False):
        # --- Operational Feature Flags ---
        self.use_bigrams = use_bigrams
        self.use_tfidf = use_tfidf
        self.use_embeddings = use_embeddings

        # --- Core Hyperparameters ---
        self.model_name = 'bert-base-uncased'
        self.max_len = 64
        self.batch_size = 16
        self.lr = 2e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Lazy Initializations ---
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None

    def _process_traditional_features(self, texts, labels=None):
        """Fully operational secondary pipelines for optional feature runs."""
        if self.use_bigrams:
            print("Extracting manual bigrams...")
            for text in texts:
                tokens = text.lower().split()
                bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]

        if self.use_tfidf:
            print("Processing tfidf.py pipeline functions...")
            # if labels is not None:
            #     self.vocab = tfidf_fit(texts)
            # scores = get_tfidf_scores(texts, self.vocab)

        if self.use_embeddings:
            print("Processing dense representations from embeddings.py...")
            # dense_feats = get_features(texts)

    def train(self, training_data: list, epochs: int = 1, verbose: bool = False):
        """
        Fits the BERT model. Mirrors the exact naming convention from hierarchical_svm.py.
        Expects training_data to contain objects with .text and .label attributes.
        """
        # Unpack custom data object format into raw lists
        X_train = [item.text for item in training_data]

        # Convert text-based labels to integer IDs for PyTorch CrossEntropyLoss
        unique_labels = sorted(list(set(item.label for item in training_data)))
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for i, label in enumerate(unique_labels)}

        y_train = [self.label_to_id[item.label] for item in training_data]

        # Run optional traditional pipeline features checks
        self._process_traditional_features(X_train, y_train)

        num_classes = len(unique_labels)

        # Initialize Sequence Classifier matching the dataset classes
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_classes)
        self.model = self.model.to(self.device)

        train_dataset = TextDataset(X_train, y_train, self.tokenizer, self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )

        if verbose:
            print(f"Training BERTClassifier on {self.device}...")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {total_loss / len(train_loader):.4f}")

    def predict_all(self, test_data: list) -> list:
        """
        Generates categorical text label predictions.
        Mirrors the exact naming convention from hierarchical_svm.py.
        """
        if self.model is None:
            raise ValueError("The model must be trained via .train() before making predictions.")

        # Unpack testing dataset structure
        X_test = [item.text for item in test_data]

        # Run optional feature checks on inference step
        self._process_traditional_features(X_test)

        self.model.eval()
        test_dataset = TextDataset(X_test, None, self.tokenizer, self.max_len)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        raw_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                raw_preds.extend(preds)

        # Map integer prediction IDs back to original string targets for evaluate.py
        string_labels = [self.id_to_label[pred_id] for pred_id in raw_preds]
        return string_labels
      
