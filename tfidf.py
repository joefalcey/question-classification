import math
from collections import defaultdict

class TFIDFVectorizer:
    def __init__(self, use_bigrams: bool = False):
        self.use_bigrams = use_bigrams
        self.idf = defaultdict(float)
        self.num_docs = 0

    def _extract_tokens(self, text: str) -> list:
        tokens = text.lower().split()
        if self.use_bigrams:
            return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return tokens

    def fit(self, training_data: list):
        self.num_docs = len(training_data)
        doc_counts = defaultdict(int)

        # Count how many documents contain each feature
        for item in training_data:
            unique_features = set(self._extract_tokens(item.text))
            for f in unique_features:
                doc_counts[f] += 1

        # Add 1 to avoid division by zero for completely unseen features
        for f, df in doc_counts.items():
            self.idf[f] = math.log(self.num_docs / float(df))

    def get_tfidf_scores(self, text: str) -> dict:
        tokens = self._extract_tokens(text)
        if not tokens:
            return {}

        tf_counts = defaultdict(int)
        for t in tokens:
            tf_counts[t] += 1

        tfidf_dict = {}
        for feature, count in tf_counts.items():
            tf = count / len(tokens)
            
            idf_value = self.idf.get(feature, math.log(self.num_docs / 1.0) if self.num_docs > 0 else 1.0)
            
            tfidf_dict[feature] = tf * idf_value

        return tfidf_dict
