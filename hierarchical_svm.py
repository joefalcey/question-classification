from collections import defaultdict
from sklearn_models import SklearnClassifier

class _TempText:
    def __init__(self, label, text):
        self.label = label
        self.text = text

class HierarchicalSVM:
    def __init__(self, master_kwargs=None, sub_kwargs=None, top_k=2):
        self.master_kwargs = master_kwargs if master_kwargs else {}
        self.sub_kwargs = sub_kwargs if sub_kwargs else {}
        self.top_k = top_k
        
        self.master_model = SklearnClassifier(**self.master_kwargs)
        self.sub_models = {}

    def train(self, training_data: list, epochs: int = 1, verbose: bool = False):
        coarse_training_data = []
        sub_training_data = defaultdict(list)

        for item in training_data:
            coarse_label = item.label.split(':')[0]
            
            coarse_training_data.append(_TempText(coarse_label, item.text))
 
            sub_training_data[coarse_label].append(item)

        # Master Model
        if verbose: print("Training Master SVM (Coarse Classes)...")
        self.master_model.train(coarse_training_data, epochs=epochs, verbose=verbose)

        # Sub-Models
        for coarse_label, data_bucket in sub_training_data.items():
            if verbose: print(f"Training Expert SVM for {coarse_label}...")
            sub_model = SklearnClassifier(**self.sub_kwargs)
            sub_model.train(data_bucket, epochs=epochs, verbose=verbose)
            self.sub_models[coarse_label] = sub_model

    def predict_all(self, test_data: list) -> list:
        master_preds = self.master_model.predict_top_k_all(test_data, k=self.top_k)
       
        expert_queries = defaultdict(list)
        for idx, (item, m_preds) in enumerate(zip(test_data, master_preds)):
            for coarse_label, coarse_score in m_preds:
                expert_queries[coarse_label].append((idx, coarse_score, item))
                
        best_scores = [-float('inf')] * len(test_data)
        best_labels = [None] * len(test_data)
        
        for coarse_label, queries in expert_queries.items():
            indices = [q[0] for q in queries]
            coarse_scores = [q[1] for q in queries]
            items = [q[2] for q in queries]
            
            if coarse_label in self.sub_models:
                expert = self.sub_models[coarse_label]
                fine_preds = expert.predict_top_k_all(items, k=1)
                
                for idx, c_score, f_pred in zip(indices, coarse_scores, fine_preds):
                    fine_label, f_score = f_pred[0]
                    combined_score = c_score + f_score
                    if combined_score > best_scores[idx]:
                        best_scores[idx] = combined_score
                        best_labels[idx] = fine_label
            else:
                for idx, c_score in zip(indices, coarse_scores):
                    fallback_label = coarse_label + ":other"
                    if c_score > best_scores[idx]:
                        best_scores[idx] = c_score
                        best_labels[idx] = fallback_label
                        
        return best_labels