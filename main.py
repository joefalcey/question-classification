import data_reader
import numpy as np
from evaluate import evaluate, print_detailed_report
from majority_baseline import MajorityVote
from random_baseline import RandomChoice
from perceptron import Perceptron
from averaged_perceptron import AveragedPerceptron
from sklearn_models import SklearnClassifier
from hierarchical_perceptron import HierarchicalPerceptron
from hierarchical_svm import HierarchicalSVM
from hybrid_kmeans import KMeansHierarchical
from neural_models import MLPClassifier as NeuralMLP

# BERT integration (optional - requires torch/transformers)
try:
    from bert_pipeline import BERTClassifier
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Note: BERT not available. Install dependencies with: uv sync")

def test_models():
    np.random.seed(42)

    print("Loading data sets...")
    # Lemmatized Data
    full_lem = data_reader.read("data/train_5500_augmented.label", lemmatize=True)
    test_lem = data_reader.read("data/TREC_10.label", lemmatize=True)

    # Shuffle to ensure all classes are represented in both splits
    import random
    random.shuffle(full_lem)
    split_idx = int(len(full_lem) * 0.8)
    train_lem, dev_lem = full_lem[:split_idx], full_lem[split_idx:]

    # Raw Data
    full_raw = data_reader.read("data/train_5500_augmented.label", lemmatize=False)
    test_raw = data_reader.read("data/TREC_10.label", lemmatize=False)

    random.shuffle(full_raw)
    split_idx = int(len(full_raw) * 0.8)
    train_raw, dev_raw = full_raw[:split_idx], full_raw[split_idx:]
    
    dev_true = [item.label for item in dev_lem]
    test_true = [item.label for item in test_lem]

    # Baselines
    majority_model = MajorityVote()
    majority_model.train(train_lem)
    maj_dev = evaluate(dev_true, majority_model.predict_all(dev_lem), verbose=False)
    maj_test = evaluate(test_true, majority_model.predict_all(test_lem), verbose=False)
    
    rand_dev_scores, rand_test_scores = [], []
    for _ in range(5):
        rand_model = RandomChoice()
        rand_model.train(train_lem)
        rand_dev_scores.append(evaluate(dev_true, rand_model.predict_all(dev_lem), verbose=False))
        rand_test_scores.append(evaluate(test_true, rand_model.predict_all(test_lem), verbose=False))

    # Evaluation Loop
    models_to_test = [
        # PERCEPTRONS
        ("Std. Perceptron (BoW Unigram)", Perceptron, {"use_bigrams": False, "use_tfidf": False, "use_embeddings": False}),
        ("Std. Perceptron (BoW Bigram)", Perceptron, {"use_bigrams": True, "use_tfidf": False, "use_embeddings": False}),
        ("Std. Perceptron (TF-IDF Unigram)", Perceptron, {"use_bigrams": False, "use_tfidf": True, "use_embeddings": False}),
        ("Std. Perceptron (TF-IDF Bigram)", Perceptron, {"use_bigrams": True, "use_tfidf": True, "use_embeddings": False}),
        #("Std. Perceptron (Dense Embeddings)", Perceptron, {"use_bigrams": False, "use_tfidf": False, "use_embeddings": True}),
        
        ("Avg. Perceptron (BoW Unigram)", AveragedPerceptron, {"use_bigrams": False, "use_tfidf": False, "use_embeddings": False}),
        ("Avg. Perceptron (BoW Bigram)", AveragedPerceptron, {"use_bigrams": True, "use_tfidf": False, "use_embeddings": False}),
        ("Avg. Perceptron (TF-IDF Unigram)", AveragedPerceptron, {"use_bigrams": False, "use_tfidf": True, "use_embeddings": False}),
        ("Avg. Perceptron (TF-IDF Bigram)", AveragedPerceptron, {"use_bigrams": True, "use_tfidf": True, "use_embeddings": False}),
        #("Avg. Perceptron (Dense Embeddings)", AveragedPerceptron, {"use_bigrams": False, "use_tfidf": False, "use_embeddings": True}),

        # Scikit-Learn Non-Linear Models
        ("SVM (BoW Unigram)", SklearnClassifier, {"model_type": "SVM", "use_bigrams": False, "use_tfidf": False, "use_embeddings": False}),
        ("MLP (BoW Unigram)", NeuralMLP, {"use_bigrams": False, "use_tfidf": False, "use_embeddings": False}),
        ("SVM (TF-IDF Unigram)", SklearnClassifier, {"model_type": "SVM", "use_bigrams": False, "use_tfidf": True, "use_embeddings": False}),
        #("SVM (Dense Embeddings)", SklearnClassifier, {"model_type": "SVM", "use_bigrams": False, "use_tfidf": False, "use_embeddings": True}),

        #("Random Forest (Dense Embed)", SklearnClassifier, {"model_type": "RandomForest", "use_bigrams": False, "use_tfidf": False, "use_embeddings": True})

        ("Hierarchical Perceptron (BoW Unigram)", HierarchicalPerceptron, {
            "master_kwargs": {"use_bigrams": False, "use_tfidf": False, "use_embeddings": False},
            "sub_kwargs": {"use_bigrams": False, "use_tfidf": False, "use_embeddings": False}
        }),
        ("Hierarchical SVM (BoW Unigram)", HierarchicalSVM, {
            "master_kwargs": {"model_type": "SVM", "use_bigrams": False, "use_tfidf": False, "use_embeddings": False},
            "sub_kwargs": {"model_type": "SVM", "use_bigrams": False, "use_tfidf": False, "use_embeddings": False}
        }),
        ("K-Means Perceptron (k=4)", KMeansHierarchical, {
        "master_model_class": AveragedPerceptron,
        "expert_class": AveragedPerceptron,
        "k_per_coarse": 2,
        "master_kwargs": {"use_bigrams": False, "use_tfidf": False, "use_embeddings": False},
        "sub_kwargs": {"use_bigrams": False, "use_tfidf": False, "use_embeddings": False}
        }),

        ("K-Means SVM (k=4)", KMeansHierarchical, {
        "master_model_class": SklearnClassifier,
        "expert_class": SklearnClassifier,
        "k_per_coarse": 2,
        "master_kwargs": {"model_type": "SVM", "use_bigrams": False, "use_tfidf": False, "use_embeddings": False},
        "sub_kwargs": {"model_type": "SVM", "use_bigrams": False, "use_tfidf": False, "use_embeddings": False}
        }),
    ]

    # Add BERT models if available
    if BERT_AVAILABLE:
        bert_models = [
            ("BERT (DistilBERT, Balanced)", BERTClassifier, {
                "model_name": "distilbert-base-uncased",
                "class_weights": "balanced",
                "use_focal_loss": True,
                "batch_size": 16,
                "learning_rate": 2e-5,
            }),
        ]
        models_to_test.extend(bert_models)
        print("Note: BERT models added to evaluation. This will significantly increase runtime.")

    # Tracking Dictionary
    results = {name: {"Lemma": {"dev": [], "test": []}, "Raw": {"dev": [], "test": []}} for name, _, _ in models_to_test}

    print("\nStarting 5-run evaluation loop (Testing Lemma vs Raw simultaneously)...")
    for i in range(5):
        print(f"  --- Run {i+1}/5 ---")

        indices = np.arange(len(train_lem))
        np.random.shuffle(indices)
        shuf_train_lem = [train_lem[idx] for idx in indices]
        shuf_train_raw = [train_raw[idx] for idx in indices]

        for name, ModelClass, kwargs in models_to_test:

            # Determine epochs for this model type
            # BERT needs fewer epochs due to longer training time
            if "BERT" in name:
                model_epochs = 2  # BERT trains slower
            else:
                model_epochs = 10  # Standard models

            # Train and Test Lemmatized
            model_lem = ModelClass(**kwargs)
            model_lem.train(shuf_train_lem, epochs=model_epochs, verbose=False)
            results[name]["Lemma"]["dev"].append(evaluate(dev_true, model_lem.predict_all(dev_lem), verbose=False))
            results[name]["Lemma"]["test"].append(evaluate(test_true, model_lem.predict_all(test_lem), verbose=False))

            # Train and Test Raw
            model_raw = ModelClass(**kwargs)
            model_raw.train(shuf_train_raw, epochs=model_epochs, verbose=False)
            results[name]["Raw"]["dev"].append(evaluate(dev_true, model_raw.predict_all(dev_raw), verbose=False))
            results[name]["Raw"]["test"].append(evaluate(test_true, model_raw.predict_all(test_raw), verbose=False))

    # Comparison Matrix
    print("\n")
    print(f"{'Method':<35} | {'Lemmatized (Dev / Test)':<35} | {'Raw Text (Dev / Test)':<35}")
    print("-" * 115)
    
    # Baselines
    print(f"{'Majority Baseline':<35} | {maj_dev:.4f} ± 0.0000 / {maj_test:.4f} ± 0.0000 | {maj_dev:.4f} ± 0.0000 / {maj_test:.4f} ± 0.0000")
    rand_dev_m, rand_dev_s = np.mean(rand_dev_scores), np.std(rand_dev_scores)
    rand_test_m, rand_test_s = np.mean(rand_test_scores), np.std(rand_test_scores)
    print(f"{'Random Baseline':<35} | {rand_dev_m:.4f} ± {rand_dev_s:.4f} / {rand_test_m:.4f} ± {rand_test_s:.4f} | {rand_dev_m:.4f} ± {rand_dev_s:.4f} / {rand_test_m:.4f} ± {rand_test_s:.4f}")
    print("-" * 115)

    # Models
    for name, _, _ in models_to_test:
        lem_dev_m, lem_dev_s = np.mean(results[name]["Lemma"]["dev"]), np.std(results[name]["Lemma"]["dev"])
        lem_test_m, lem_test_s = np.mean(results[name]["Lemma"]["test"]), np.std(results[name]["Lemma"]["test"])
        raw_dev_m, raw_dev_s = np.mean(results[name]["Raw"]["dev"]), np.std(results[name]["Raw"]["dev"])
        raw_test_m, raw_test_s = np.mean(results[name]["Raw"]["test"]), np.std(results[name]["Raw"]["test"])

        lem_str = f"{lem_dev_m:.4f} ± {lem_dev_s:.4f} / {lem_test_m:.4f} ± {lem_test_s:.4f}"
        raw_str = f"{raw_dev_m:.4f} ± {raw_dev_s:.4f} / {raw_test_m:.4f} ± {raw_test_s:.4f}"

        print(f"{name:<35} | {lem_str:<35} | {raw_str:<35}")

        if name == "Std. Perceptron (Dense Embeddings)":
            print("-" * 115)

    print("\n" + "="*115)
    print("DETAILED EVALUATION: Best Model on Test Set (Lemmatized)")
    print("="*115)

    # Find best model based on test performance
    best_model_name = max(results.keys(), key=lambda k: np.mean(results[k]["Lemma"]["test"]))
    best_score = np.mean(results[best_model_name]["Lemma"]["test"])

    print(f"\nBest Model: {best_model_name}")
    print(f"Macro F1 Score: {best_score:.4f}\n")

    # Find the model class and kwargs for the best model
    best_model_info = None
    for name, ModelClass, kwargs in models_to_test:
        if name == best_model_name:
            best_model_info = (ModelClass, kwargs)
            break

    if best_model_info:
        ModelClass, kwargs = best_model_info
        best_model = ModelClass(**kwargs)
        best_model.train(shuf_train_lem, epochs=10, verbose=False)
        test_predictions = best_model.predict_all(test_lem)

        print_detailed_report(test_true, test_predictions, best_model_name)

if __name__ == "__main__":
    test_models()
