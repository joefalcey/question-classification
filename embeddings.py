import gensim.downloader as api
import numpy as np

class DenseVectorizer:
    def __init__(self, model_name='glove-wiki-gigaword-300'):

        # Using a lightweight 25-D model
        self.model = api.load(model_name)
        self.dim = self.model.vector_size

    def get_features(self, text: str) -> dict:
        tokens = text.lower().split()
        valid_vectors = []
        
        for token in tokens:
            if token in self.model:
                valid_vectors.append(self.model[token])
                
        # Fallback for completely unknown sentences
        if not valid_vectors:
            return {f"dim_{i}": 0.0 for i in range(self.dim)}
            
        # Mean Pooling
        sentence_vector = np.mean(valid_vectors, axis=0)
        
        return {f"dim_{i}": float(val) for i, val in enumerate(sentence_vector)}
