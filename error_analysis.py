"""
Error Analysis for TREC Question Classification

Analyzes model errors to understand:
1. Which classes are most frequently confused
2. Why they are confused (feature overlap)
3. Examples of misclassified questions
4. Rare class performance analysis
"""

import data_reader
from sklearn_models import SklearnClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import defaultdict, Counter
import itertools


def get_coarse_label(fine_label):
    """Extract coarse label from fine-grained label."""
    return fine_label.split(':')[0] if ':' in fine_label else fine_label


def get_confusion_pairs(y_true, y_pred):
    """
    Find which class pairs are most frequently confused.
    Returns dictionary mapping (true, pred) -> count.
    """
    confusion_pairs = defaultdict(int)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            confusion_pairs[(true_label, pred_label)] += 1
    return confusion_pairs


def analyze_errors():
    """
    Perform comprehensive error analysis on the best model.
    """
    print("=" * 80)
    print("ERROR ANALYSIS - Linear SVM (BoW Unigram, Raw Text)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    full_raw = data_reader.read("data/train_5500.label", lemmatize=False)
    train_raw = full_raw[:4500]
    test_raw = data_reader.read("data/TREC_10.label", lemmatize=False)

    # Train model
    print("Training model...")
    model = SklearnClassifier(
        model_type="SVM",
        use_bigrams=False,
        use_tfidf=False,
        use_embeddings=False
    )
    model.train(train_raw, epochs=1, verbose=False)

    # Get predictions
    y_true = [item.label for item in test_raw]
    y_pred = model.predict_all(test_raw)

    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Get confusion pairs
    confusion_pairs = get_confusion_pairs(y_true, y_pred)

    # Sort by frequency
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTotal misclassifications: {sum(confusion_pairs.values())}")
    print(f"Unique confusion pairs: {len(confusion_pairs)}")

    # Analyze top confusion pairs
    print("\n" + "=" * 80)
    print("TOP 20 MOST FREQUENT CONFUSION PAIRS")
    print("=" * 80)

    print(f"\n{'True Class':<20} {'Predicted Class':<20} {'Count':<8} {'Example'}")
    print("-" * 80)

    for (true_label, pred_label), count in sorted_pairs[:20]:
        # Find an example of this confusion
        example = None
        for item, t, p in zip(test_raw, y_true, y_pred):
            if t == true_label and p == pred_label:
                example = item.text
                break

        print(f"{true_label:<20} {pred_label:<20} {count:<8} {example[:50] if example else 'N/A'}")

    # Coarse-level confusion analysis
    print("\n" + "=" * 80)
    print("COARSE-LEVEL CONFUSION ANALYSIS")
    print("=" * 80)

    coarse_confusion = defaultdict(int)
    for (true_label, pred_label), count in confusion_pairs.items():
        true_coarse = get_coarse_label(true_label)
        pred_coarse = get_coarse_label(pred_label)
        coarse_confusion[(true_coarse, pred_coarse)] += count

    sorted_coarse = sorted(coarse_confusion.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'True Coarse':<15} {'Pred Coarse':<15} {'Count':<8}")
    print("-" * 50)

    for (true_coarse, pred_coarse), count in sorted_coarse:
        if true_coarse != pred_coarse:  # Only show cross-coarse confusions
            print(f"{true_coarse:<15} {pred_coarse:<15} {count:<8}")

    # Analyze rare class performance
    print("\n" + "=" * 80)
    print("RARE CLASS PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Get class frequencies in training set
    train_label_counts = Counter([item.label for item in train_raw])
    test_label_counts = Counter([item.label for item in test_raw])

    # Calculate per-class metrics
    class_metrics = {}
    for cls in set(y_true):
        true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        false_positives = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        false_negatives = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[cls] = {
            'train_count': train_label_counts[cls],
            'test_count': test_label_counts[cls],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_negatives': false_negatives
        }

    # Find classes with worst performance
    print("\nClasses with ZERO RECALL (completely missed):")
    print(f"{'Class':<20} {'Train Count':<12} {'Test Count':<12}")
    print("-" * 50)

    zero_recall = [(cls, metrics) for cls, metrics in class_metrics.items() if metrics['recall'] == 0]
    zero_recall_sorted = sorted(zero_recall, key=lambda x: x[1]['test_count'], reverse=True)

    for cls, metrics in zero_recall_sorted:
        print(f"{cls:<20} {metrics['train_count']:<12} {metrics['test_count']:<12}")

    print(f"\nTotal classes with zero recall: {len(zero_recall)}")

    print("\nClasses with LOWEST F1 (< 0.3):")
    print(f"{'Class':<20} {'Train':<8} {'Test':<8} {'F1':<8}")
    print("-" * 50)

    low_f1 = [(cls, metrics) for cls, metrics in class_metrics.items() if metrics['f1'] < 0.3 and metrics['f1'] > 0]
    low_f1_sorted = sorted(low_f1, key=lambda x: x[1]['f1'])

    for cls, metrics in low_f1_sorted[:15]:
        print(f"{cls:<20} {metrics['train_count']:<8} {metrics['test_count']:<8} {metrics['f1']:.4f}")

    # Detailed error examples for top confusion pairs
    print("\n" + "=" * 80)
    print("DETAILED ERROR EXAMPLES")
    print("=" * 80)

    for i, ((true_label, pred_label), count) in enumerate(sorted_pairs[:5]):
        print(f"\n--- Confusion {i+1}: {true_label} → {pred_label} ({count} instances) ---")

        # Find 3 examples of this confusion
        examples = []
        for item, t, p in zip(test_raw, y_true, y_pred):
            if t == true_label and p == pred_label:
                examples.append(item.text)
                if len(examples) >= 3:
                    break

        for j, example in enumerate(examples):
            print(f"  Example {j+1}: {example}")

    # Save results to file
    save_error_analysis_results(sorted_pairs, sorted_coarse, class_metrics, test_raw, y_true, y_pred)

    print("\n" + "=" * 80)
    print("Error analysis complete!")
    print("Results saved to: error_analysis_results.txt")
    print("=" * 80)


def save_error_analysis_results(sorted_pairs, sorted_coarse, class_metrics, test_data, y_true, y_pred):
    """Save error analysis results to a text file."""

    with open('error_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ERROR ANALYSIS - TREC Question Classification\n")
        f.write("Model: Linear SVM (BoW Unigram, Raw Text)\n")
        f.write("=" * 80 + "\n\n")

        # Top confusion pairs
        f.write("TOP 30 MOST FREQUENT CONFUSION PAIRS\n\n")
        f.write(f"{'True Class':<20} {'Predicted Class':<20} {'Count':<8}\n")
        f.write("-" * 80 + "\n")

        for (true_label, pred_label), count in sorted_pairs[:30]:
            f.write(f"{true_label:<20} {pred_label:<20} {count:<8}\n")

        # Coarse-level confusion
        f.write("\n" + "=" * 80 + "\n")
        f.write("COARSE-LEVEL CONFUSION\n\n")
        f.write(f"{'True Coarse':<15} {'Pred Coarse':<15} {'Count':<8}\n")
        f.write("-" * 50 + "\n")

        for (true_coarse, pred_coarse), count in sorted_coarse:
            if true_coarse != pred_coarse:
                f.write(f"{true_coarse:<15} {pred_coarse:<15} {count:<8}\n")

        # Zero recall classes
        f.write("\n" + "=" * 80 + "\n")
        f.write("CLASSES WITH ZERO RECALL\n\n")
        f.write(f"{'Class':<20} {'Train Count':<12} {'Test Count':<12}\n")
        f.write("-" * 50 + "\n")

        zero_recall = [(cls, metrics) for cls, metrics in class_metrics.items() if metrics['recall'] == 0]
        zero_recall_sorted = sorted(zero_recall, key=lambda x: x[1]['test_count'], reverse=True)

        for cls, metrics in zero_recall_sorted:
            f.write(f"{cls:<20} {metrics['train_count']:<12} {metrics['test_count']:<12}\n")

        f.write(f"\nTotal: {len(zero_recall)} classes completely missed\n")

        # Per-class performance summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("PER-CLASS PERFORMANCE SUMMARY (Sorted by F1)\n\n")
        f.write(f"{'Class':<20} {'Train':<8} {'Test':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}\n")
        f.write("-" * 80 + "\n")

        sorted_by_f1 = sorted(class_metrics.items(), key=lambda x: x[1]['f1'])
        for cls, metrics in sorted_by_f1:
            f.write(f"{cls:<20} {metrics['train_count']:<8} {metrics['test_count']:<8} "
                   f"{metrics['precision']:.4f}   {metrics['recall']:.4f}   {metrics['f1']:.4f}\n")


if __name__ == "__main__":
    analyze_errors()
