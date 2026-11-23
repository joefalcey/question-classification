from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class KMeansHierarchical:
    def __init__(self, master_model_class, expert_class, k_per_coarse=2, master_kwargs=None, sub_kwargs=None):
        self.master_model_class = master_model_class
        self.expert_class = expert_class
        self.k_per_coarse = k_per_coarse
        self.master_kwargs = master_kwargs if master_kwargs else {}
        self.sub_kwargs = sub_kwargs if sub_kwargs else {}

        # Master Model
        self.master_model = self.master_model_class(**self.master_kwargs)

        # Vectorizers, K-means, and Experts per coarse category
        self.vectorizers = {}
        self.kmeans_models = {}
        self.sub_models = {}

    def train(self, training_data: list, epochs: int = 1, verbose: bool = False):
        class _TempText:
            def __init__(self, label, text):
                self.label = label
                self.text = text

        coarse_training_data = []
        coarse_buckets = defaultdict(list)

        for item in training_data:
            coarse_label = item.label.split(':')[0]
            coarse_training_data.append(_TempText(coarse_label, item.text))
            coarse_buckets[coarse_label].append(item)

        # Train Master Model
        if verbose: print("Training Master Model (Coarse Classes)...")
        self.master_model.train(coarse_training_data, epochs=epochs, verbose=verbose)

        for coarse_label, bucket_data in coarse_buckets.items():
            if verbose: print(f"\nProcessing Coarse Category: {coarse_label} ({len(bucket_data)} items)")

            vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
            texts = [item.text for item in bucket_data]
            X_matrix = vectorizer.fit_transform(texts)

            kmeans = KMeans(n_clusters=self.k_per_coarse, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(X_matrix)

            self.vectorizers[coarse_label] = vectorizer
            self.kmeans_models[coarse_label] = kmeans

            sub_clusters = defaultdict(list)
            for item, c_id in zip(bucket_data, cluster_labels):
                sub_clusters[c_id].append(item)

            for c_id, sub_data in sub_clusters.items():
                expert_key = f"{coarse_label}_{c_id}"

                unique_labels = set(item.label for item in sub_data)
                if len(unique_labels) == 1:
                    single_label = list(unique_labels)[0]
                    class DummyExpert:
                        def __init__(self, label):
                            self.label = label
                        def predict_top_k_all(self, items, k=1):
                            return [[(self.label, 1.0)] for _ in items]
                        def predict_top_k(self, text, k=1):
                            return [(self.label, 1.0)]
                    self.sub_models[expert_key] = DummyExpert(single_label)
                    continue

                expert = self.expert_class(**self.sub_kwargs)
                expert.train(sub_data, epochs=epochs, verbose=False)
                self.sub_models[expert_key] = expert

    def predict_all(self, test_data: list) -> list:
        if hasattr(self.master_model, 'predict_top_k_all'):
            master_preds = self.master_model.predict_top_k_all(test_data, k=1)
            coarse_preds = [p[0][0] for p in master_preds]
        else:
            coarse_preds = [self.master_model.predict_top_k(item.text, k=1)[0][0] for item in test_data]

        best_labels = [None] * len(test_data)

        # Group items by predicted coarse label
        coarse_groups = defaultdict(list)
        for idx, (item, coarse_label) in enumerate(zip(test_data, coarse_preds)):
            coarse_groups[coarse_label].append((idx, item))

        for coarse_label, items_with_idx in coarse_groups.items():
            if coarse_label not in self.kmeans_models:
                for idx, _ in items_with_idx:
                    best_labels[idx] = f"{coarse_label}:other"
                continue

            indices = [x[0] for x in items_with_idx]
            items = [x[1] for x in items_with_idx]
            texts = [item.text for item in items]

            vectorizer = self.vectorizers[coarse_label]
            kmeans = self.kmeans_models[coarse_label]

            X_matrix = vectorizer.transform(texts)
            test_clusters = kmeans.predict(X_matrix)

            expert_queries = defaultdict(list)
            for local_i, cluster_id in enumerate(test_clusters):
                global_idx = indices[local_i]
                item_obj = items[local_i]
                expert_queries[cluster_id].append((global_idx, item_obj))

            for cluster_id, query_list in expert_queries.items():
                expert_key = f"{coarse_label}_{cluster_id}"
                exp_indices = [q[0] for q in query_list]
                exp_items = [q[1] for q in query_list]

                if expert_key in self.sub_models:
                    expert = self.sub_models[expert_key]
                    if hasattr(expert, 'predict_top_k_all'):
                        fine_preds = expert.predict_top_k_all(exp_items, k=1)
                    else:
                        fine_preds = [expert.predict_top_k(it.text, k=1) for it in exp_items]

                    for global_idx, f_pred in zip(exp_indices, fine_preds):
                        best_labels[global_idx] = f_pred[0][0]
                else:
                    for global_idx in exp_indices:
                        best_labels[global_idx] = f"{coarse_label}:other"

        return best_labels