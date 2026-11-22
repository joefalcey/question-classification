from collections import defaultdict
import argparse

import data_reader
from data_reader import Text
from tfidf import TFIDFVectorizer
from embedder import DenseVectorizer

main_parser = data_reader.parser
parser = argparse.ArgumentParser(parents=[main_parser], add_help=False)
parser.add_argument('-bg', '--bigrams', action='store_true',
                    help='Uses bigrams from the data')
parser.add_argument('-em', '--use-embeddings', action='store_true',
                    help='Uses embeddings from the data')
parser.add_argument('-tf', '--use-tfidf', action='store_true',
                    help='Applies TFIDF to the data')
args = parser.parse_args()

if args.bigrams: use_bigrams = True
if args.use_embeddings: use_embeddings = True
if args.use_tfidf: use_tfidf = True
if args.verbose: use_verbose = True

class Perceptron:
    def __init__(self, use_bigrams: bool = False, use_tfidf: bool = False,
                 use_embeddings: bool = False, epochs = 10, verbose = False):
        """Multiclass perceptron classifier for labeled text data.

        Attributes:
            use_bigrams: Bigrams as features instead of single tokens.
            use_tfidf: Implementation with TF-IDF.
            use_embeddings = Embeddings as features.
            epochs: Number of full passes over the training dataset.
        """
        self.weights = defaultdict(lambda: defaultdict(float))
        self.classes = []
        self.use_bigrams = use_bigrams
        self.use_tfidf = use_tfidf
        self.use_embeddings = use_embeddings
        self.vectorizer = None
        self.embedder = None

    def _extract_features(self, text: Text) -> set:
        tokens = ' '.join(text.tokens).split()
        if self.use_bigrams:
            bigrams = [f'{tokens[i]}_{tokens[i + 1]}' for i in range(len(tokens) - 1)]
            return set(bigrams)
        else:
            return set(tokens)

    def _get_features(self, text: Text) -> dict[str, float]:
        if self.use_embeddings and self.embedder is not None:
            return self.embedder.get_features(text)
        elif self.use_tfidf and self.vectorizer is not None:
            return self.vectorizer.get_tfidf_scores(text)
        else:
            # Standard Bag-of-Words
            tokens = set(self._extract_features(text))
            return {token: 1.0 for token in tokens}

    def _get_max_score(self, features: dict) -> str:
        scores = {c: 0.0 for c in self.classes}
        for c in self.classes:
            for f, val in features.items():
                scores[c] = self.weights[c][f] * val
        return max(scores.keys(), key=lambda k: scores[k])

    def train(self, training_data: list[Text], epochs: int = 10, verbose: bool = True) -> None:
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

        self.classes = sorted(set(([item.label[0] for item in training_data])))
        cached_features = [self._get_features(item) for item in training_data] # Pre-computed features
        for epoch in range(epochs): # The training loop
            mistakes = 0
            for item, features in zip(training_data, cached_features):
                coarse_label = item.label[0]
                guess = self._get_max_score(features) # Make a guess based on current weights
                if guess != coarse_label:
                    mistakes += 1
                    for f, val in features.items():
                        self.weights[item.label][f] += val # Reward correct guess
                        self.weights[guess][f] -= val # Punish wrong guess
            if verbose:
                print(f"  Epoch {epoch + 1}/{epochs} completed. Total mistakes: {mistakes}")

    def predict(self, test_data: list[Text]) -> list:
        extract = self._get_features
        prediction = self._get_max_score
        return [prediction(extract(item)) for item in test_data]

    @property
    def name(self) -> str:
        return "Perceptron"

if __name__ == "__main__":
    import pathlib
    import os
    from data_reader import read_data
    path = pathlib.Path(os.getcwd()) / 'data'
    train_file = read_data(path / 'train_5500.label')
    test_file = read_data(path / 'TREC_10.label')
    training_data = [Text(line) for line in train_file]
    test_data = [Text(line) for line in test_file]
    perceptron = Perceptron(verbose=True)
    perceptron.train(training_data)
    predictions = perceptron.predict(test_data)
    true_labels = [line.label[0] for line in training_data]
    print('\nLoading sample set...')
  
    for true, predicted in zip(true_labels[:10], predictions[:10]):
        print(f'true label={true}, predicted label={predicted}')
