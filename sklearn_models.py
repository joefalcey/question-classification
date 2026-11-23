from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import numpy as np
from embeddings import DenseVectorizer

class SklearnClassifier:
    def __init__(self, model_type="SVM", use_bigrams=False, use_tfidf=False, use_embeddings=False):
        self.use_bigrams = use_bigrams
        self.use_tfidf = use_tfidf
        self.use_embeddings = use_embeddings

        if model_type == "SVM":
            # Kernel
            #self.model = SVC(kernel='linear', C=1.0, random_state=42)
            self.model = LinearSVC(dual='auto', C=1.0, random_state=42, max_iter=5000)
        elif model_type == "RandomForest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # DictVectorizer
        self.dict_vec = DictVectorizer(sparse=not use_embeddings)
        self.vectorizer = None
        self.embedder = None

    def _extract_tokens(self, text: str) -> set:
        tokens = text.lower().split()
        if self.use_bigrams:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            return set(bigrams) 
        return set(tokens)

    def get_features(self, text: str) -> dict:
        if self.use_embeddings and self.embedder is not None:
            return self.embedder.get_features(text)
        else:
            tokens = set(self._extract_tokens(text))
            return {token: 1.0 for token in tokens}

    def _fix_sparse_indices(self, matrix):
        if sp.issparse(matrix):
            matrix.indices = matrix.indices.astype(np.int32)
            matrix.indptr = matrix.indptr.astype(np.int32)
        return matrix

    def train(self, training_data: list, epochs: int = 1, verbose: bool = False):
        y = [item.label for item in training_data]

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

        # Train
        self.model.fit(X_matrix, y)

    def predict_all(self, test_data: list) -> list:
        if self.use_tfidf:
            texts = [item.text for item in test_data]
            X_matrix = self.vectorizer.transform(texts)
        else:
            X_dicts = [self.get_features(item.text) for item in test_data]
            X_matrix = self.dict_vec.transform(X_dicts)

        X_matrix = self._fix_sparse_indices(X_matrix)

        return self.model.predict(X_matrix).tolist()

    def predict_top_k_all(self, test_data: list, k: int = 2) -> list:
        """
        Return top K predictions with their scores for each test item.
        Returns: [[(label, score), ...], ...] - one list per test item
        """
        # Transform test data
        if self.use_tfidf:
            texts = [item.text for item in test_data]
            X_matrix = self.vectorizer.transform(texts)
        else:
            X_dicts = [self.get_features(item.text) for item in test_data]
            X_matrix = self.dict_vec.transform(X_dicts)

        X_matrix = self._fix_sparse_indices(X_matrix)

        results = []

        # Get scores based on model type
        if hasattr(self.model, 'decision_function'):
            raw_scores = self.model.decision_function(X_matrix)

            if len(raw_scores.shape) == 1:
                raw_scores = np.column_stack([-raw_scores, raw_scores])

            for i in range(len(test_data)):
                sample_scores = raw_scores[i]
                top_k_indices = np.argsort(sample_scores)[-k:][::-1] 
                top_k_labels = [(self.model.classes_[idx], sample_scores[idx])
                                for idx in top_k_indices]
                results.append(top_k_labels)

        elif hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_matrix)

            for i in range(len(test_data)):
                sample_probas = probas[i]
                top_k_indices = np.argsort(sample_probas)[-k:][::-1]
                top_k_labels = [(self.model.classes_[idx], sample_probas[idx])
                                for idx in top_k_indices]
                results.append(top_k_labels)
        else:
            predictions = self.model.predict(X_matrix)
            for pred in predictions:
                results.append([(pred, 1.0)])

        return results