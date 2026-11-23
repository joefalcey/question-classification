from collections import defaultdict
from averaged_perceptron import AveragedPerceptron

class _TempText:
    def __init__(self, label, text):
        self.label = label
        self.text = text

class HierarchicalPerceptron:
    def __init__(self, master_kwargs=None, sub_kwargs=None, top_k=2):
        self.master_kwargs = master_kwargs if master_kwargs else {}
        self.sub_kwargs = sub_kwargs if sub_kwargs else {}
        self.top_k = top_k
        
        self.master_model = AveragedPerceptron(**self.master_kwargs)
        self.sub_models = {}

    def train(self, training_data: list, epochs: int = 10, verbose: bool = True):
        coarse_training_data = []
        sub_training_data = defaultdict(list)

        for item in training_data:
            coarse_label = item.label.split(':')[0]
            coarse_training_data.append(_TempText(coarse_label, item.text))
            sub_training_data[coarse_label].append(item)

        # Train the Master Model
        if verbose: 
            print("\n" + "="*50)
            print("TRAINING MASTER MODEL (COARSE CLASSES)")
            print("="*50)
        self.master_model.train(coarse_training_data, epochs=epochs, verbose=verbose)

        # Train the Sub-Models
        for coarse_label, data_bucket in sub_training_data.items():
            if verbose: 
                print("\n" + "-"*50)
                print(f"TRAINING SUB-MODEL: {coarse_label} (Expert for {len(data_bucket)} questions)")
                print("-"*50)
                
            sub_model = AveragedPerceptron(**self.sub_kwargs)
            sub_model.train(data_bucket, epochs=epochs, verbose=verbose)
            self.sub_models[coarse_label] = sub_model

    def predict(self, text: str, use_avg: bool = True) -> str:
        master_preds = self.master_model.predict_top_k(text, k=self.top_k, use_avg=use_avg)
        
        best_fine_label = None
        best_combined_score = -float('inf')
        
        for coarse_label, coarse_score in master_preds:
            if coarse_label in self.sub_models:
                fine_preds = self.sub_models[coarse_label].predict_top_k(text, k=1, use_avg=use_avg)
                fine_label, fine_score = fine_preds[0]
            else:
                fine_label = coarse_label + ":other"
                fine_score = 0.0
                
            combined_score = coarse_score + fine_score
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_fine_label = fine_label
                
        return best_fine_label

    def predict_all(self, test_data: list) -> list:
        return [self.predict(item.text, use_avg=True) for item in test_data]
