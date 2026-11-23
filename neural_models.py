from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import numpy as np
from embeddings import DenseVectorizer


class MLPClassifier:
    """
    Multi-Layer Perceptron (Neural Network) for text classification.

    Uses feedforward neural networks with optional TF-IDF, bigram, or embedding features.
    Maintains compatibility with existing sklearn model interface.
    """

    def __init__(self, hidden_layer_sizes=(256, 128), activation='relu',
                 solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate_init=0.001, max_iter=500, early_stopping=True,
                 use_bigrams=False, use_tfidf=False, use_embeddings=False):
        self.use_bigrams = use_bigrams
        self.use_tfidf = use_tfidf
        self.use_embeddings = use_embeddings

        # Feature extractors
        self.dict_vec = DictVectorizer(sparse=not use_embeddings)
        self.vectorizer = None
        self.embedder = None

        # Neural network model
        self.model = SklearnMLP(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=42
        )

    def _extract_tokens(self, text: str) -> set:
        """Extract tokens from text."""
        tokens = text.lower().split()
        if self.use_bigrams:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            return set(bigrams)
        return set(tokens)

    def get_features(self, text: str) -> dict:
        """Convert text to feature dictionary."""
        if self.use_embeddings and self.embedder is not None:
            return self.embedder.get_features(text)
        else:
            tokens = set(self._extract_tokens(text))
            return {token: 1.0 for token in tokens}

    def _fix_sparse_indices(self, matrix):
        """Fix sparse matrix indices for compatibility."""
        if sp.issparse(matrix):
            matrix.indices = matrix.indices.astype(np.int32)
            matrix.indptr = matrix.indptr.astype(np.int32)
        return matrix

    def train(self, training_data: list, epochs: int = 1, verbose: bool = False):
        """
        Train the neural network.

        Parameters
        ----------
        training_data : list
            List of training examples with .label and .text attributes
        epochs : int
            Number of training epochs (kept for API compatibility)
        verbose : bool
            Whether to print training progress
        """
        y = [item.label for item in training_data]

        # Extract features
        if self.use_tfidf:
            ngram_setting = (1, 2) if self.use_bigrams else (1, 1)
            self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=ngram_setting)
            texts = [item.text for item in training_data]
            X_matrix = self.vectorizer.fit_transform(texts)
        else:
            if self.use_embeddings:
                self.embedder = DenseVectorizer()

            X_dicts = [self.get_features(item.text) for item in training_data]
            X_matrix = self.dict_vec.fit_transform(X_dicts)

        X_matrix = self._fix_sparse_indices(X_matrix)

        # For sklearn MLP, we need to handle sparse matrices
        # Convert sparse to dense if needed
        if sp.issparse(X_matrix):
            if verbose:
                print("Converting sparse matrix to dense for neural network...")
            X_matrix = X_matrix.toarray()

        # Train the model
        if verbose:
            print(f"Training MLPClassifier on {X_matrix.shape[0]} samples with {X_matrix.shape[1]} features...")

        self.model.fit(X_matrix, y)

        if verbose:
            print("Training complete.")

    def predict_all(self, test_data: list) -> list:
        """
        Predict labels for test data.

        Parameters
        ----------
        test_data : list
            List of test examples with .text attribute

        Returns
        -------
        list
            Predicted labels for each test example
        """
        # Extract features
        if self.use_tfidf:
            texts = [item.text for item in test_data]
            X_matrix = self.vectorizer.transform(texts)
        else:
            X_dicts = [self.get_features(item.text) for item in test_data]
            X_matrix = self.dict_vec.transform(X_dicts)

        X_matrix = self._fix_sparse_indices(X_matrix)

        # Convert sparse to dense if needed
        if sp.issparse(X_matrix):
            X_matrix = X_matrix.toarray()

        return self.model.predict(X_matrix).tolist()

    def predict_top_k_all(self, test_data: list, k: int = 2) -> list:
        """
        Return top K predictions with their scores for each test item.

        Parameters
        ----------
        test_data : list
            List of test examples
        k : int
            Number of top predictions to return

        Returns
        -------
        list
            List of [(label, score), ...] for each test item
        """
        # Extract features
        if self.use_tfidf:
            texts = [item.text for item in test_data]
            X_matrix = self.vectorizer.transform(texts)
        else:
            X_dicts = [self.get_features(item.text) for item in test_data]
            X_matrix = self.dict_vec.transform(X_dicts)

        X_matrix = self._fix_sparse_indices(X_matrix)

        # Convert sparse to dense
        if sp.issparse(X_matrix):
            X_matrix = X_matrix.toarray()

        results = []

        # MLP doesn't have decision_function, but has predict_proba
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_matrix)

            for i in range(len(test_data)):
                sample_probas = probas[i]
                top_k_indices = np.argsort(sample_probas)[-k:][::-1]
                top_k_labels = [(self.model.classes_[idx], sample_probas[idx])
                                for idx in top_k_indices]
                results.append(top_k_labels)
        else:
            # Fallback to single prediction
            predictions = self.model.predict(X_matrix)
            for pred in predictions:
                results.append([(pred, 1.0)])

        return results
