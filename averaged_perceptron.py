from collections import defaultdict
from tfidf import TFIDFVectorizer
from embeddings import DenseVectorizer

class AveragedPerceptron:
    def __init__(self, use_bigrams: bool = False, use_tfidf: bool = False, use_embeddings: bool = False):
        self.weights = defaultdict(lambda: defaultdict(float))
        self.avg_weights = defaultdict(lambda: defaultdict(float))
        self.timestamps = defaultdict(lambda: defaultdict(int))
        self.step = 0
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
            tokens = set(self.extract_features(text))
            return {token: 1.0 for token in tokens}
        
    def predict_with_features(self, features: dict, use_avg: bool = True) -> str:
        scores = self.score_features(features, use_avg=use_avg)
        return max(scores, key=scores.get)

    def score_features(self, features: dict, use_avg: bool = True) -> dict:
        """Return a dict mapping each class to its raw score."""
        scores = {c: 0.0 for c in self.classes}
        weight_dict = self.avg_weights if use_avg else self.weights

        for c in self.classes:
            for f, val in features.items():
                scores[c] += weight_dict[c][f] * val
        return scores

    def predict_top_k(self, text: str, k: int = 2, use_avg: bool = True) -> list:
        """
        Return top K predictions with their scores.
        Returns: [(label, score), ...] sorted by score descending
        """
        features = self.get_features(text)
        scores = self.score_features(features, use_avg=use_avg)

        # Get top k by sorting
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return top_k

    def predict(self, text: str, use_avg: bool = True) -> str:
        features = self.get_features(text)
        return self.predict_with_features(features, use_avg=use_avg)

    def train(self, training_data: list, epochs: int = 10, verbose: bool = True):
        self.classes = sorted(list(set(item.label for item in training_data)))

        if self.use_embeddings:
            if verbose: print(f"Training Dense Embeddings Averaged Perceptron for {epochs} epochs...")
            self.embedder = DenseVectorizer()
        elif self.use_tfidf:
            feature_type = "Bigram" if self.use_bigrams else "Unigram"
            if verbose: print(f"Training TF-IDF {feature_type} Averaged Perceptron for {epochs} epochs...")
            self.vectorizer = TFIDFVectorizer(use_bigrams=self.use_bigrams)
            self.vectorizer.fit(training_data)
        else:
            feature_type = "Bigram" if self.use_bigrams else "Unigram"
            if verbose: print(f"Training Bag-of-Words {feature_type} Averaged Perceptron for {epochs} epochs...")

        cached_features = [self.get_features(item.text) for item in training_data]
        
        for epoch in range(epochs):
            mistakes = 0
            for item, features in zip(training_data, cached_features):
                self.step += 1 
                
                guess = self.predict_with_features(features, use_avg=False)
                
                if guess != item.label:
                    mistakes += 1
                    for f, val in features.items():
                        survival_time_true = self.step - self.timestamps[item.label][f]
                        self.avg_weights[item.label][f] += survival_time_true * self.weights[item.label][f]
                        
                        survival_time_guess = self.step - self.timestamps[guess][f]
                        self.avg_weights[guess][f] += survival_time_guess * self.weights[guess][f]
                        
                        self.weights[item.label][f] += val  
                        self.weights[guess][f] -= val       
                        
                        self.timestamps[item.label][f] = self.step
                        self.timestamps[guess][f] = self.step
            if verbose:            
                print(f"  Epoch {epoch + 1}/{epochs} completed. Total mistakes: {mistakes}")

        for c in self.classes:
            for f in self.weights[c]:
                survival_time = self.step - self.timestamps[c][f]
                self.avg_weights[c][f] += survival_time * self.weights[c][f]

    def predict_all(self, test_data: list) -> list:
        return [self.predict(item.text, use_avg=True) for item in test_data]
      
