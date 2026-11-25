from collections import defaultdict
from tfidf import TFIDFVectorizer
from embeddings import DenseVectorizer

class Perceptron:
    def __init__(self, use_bigrams: bool = False, use_tfidf: bool = False, use_embeddings: bool = False):
        self.weights = defaultdict(lambda: defaultdict(float))
        self.classes = []
        self.use_bigrams = use_bigrams
        self.use_tfidf = use_tfidf
        self.use_embeddings = use_embeddings
        self.vectorizer = None
        self.embedder = None

    def extract_features(self, text: str) -> set:
        tokens = text.lower().split()
        
        if self.use_bigrams:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            return set(bigrams) 
            
        else:
            return set(tokens)
        
    def get_features(self, text: str) -> dict:
        if self.use_embeddings and self.embedder is not None:
            return self.embedder.get_features(text)
        elif self.use_tfidf and self.vectorizer is not None:
            return self.vectorizer.get_tfidf_scores(text)
        else:
            # Standard Bag-of-Words
            tokens = set(self.extract_features(text))
            return {token: 1.0 for token in tokens}
        
    def predict_with_features(self, features: dict) -> str:
        scores = {c: 0.0 for c in self.classes}
        for c in self.classes:
            for f, val in features.items():
                scores[c] += self.weights[c][f] * val
        return max(scores, key=scores.get)

    def predict(self, text: str) -> str:
        features = self.get_features(text)
        return self.predict_with_features(features)

    def train(self, training_data: list, epochs: int = 10, verbose: bool = True):
        self.classes = sorted(list(set(item.label for item in training_data)))
        
        if self.use_embeddings:
            if verbose: print(f"Training Dense Embeddings Perceptron for {epochs} epochs...")
            self.embedder = DenseVectorizer()
        elif self.use_tfidf:
            mode = "TF-IDF"
            feature_type = "Bigram" if self.use_bigrams else "Unigram"
            if verbose: print(f"Training {mode} {feature_type} Perceptron for {epochs} epochs...")
            self.vectorizer = TFIDFVectorizer(use_bigrams=self.use_bigrams)
            self.vectorizer.fit(training_data)
        else:
            mode = "Bag-of-Words"
            feature_type = "Bigram" if self.use_bigrams else "Unigram"
            if verbose: print(f"Training {mode} {feature_type} Perceptron for {epochs} epochs...")

        # Pre-compute features
        cached_features = [self.get_features(item.text) for item in training_data]
        
        # The training loop
        for epoch in range(epochs):
            mistakes = 0
            for item, features in zip(training_data, cached_features):
                # Make a guess based on current weights
                guess = self.predict_with_features(features)
                
                if guess != item.label:
                    mistakes += 1
                    for f, val in features.items():
                        self.weights[item.label][f] += val  # Reward true label
                        self.weights[guess][f] -= val       # Punish wrong guess
            if verbose:            
                print(f"  Epoch {epoch + 1}/{epochs} completed. Total mistakes: {mistakes}")

    def predict_all(self, test_data: list) -> list:
        return [self.predict(item.text) for item in test_data]
