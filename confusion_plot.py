import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn_models import SklearnClassifier
from hierarchical_svm import HierarchicalSVM
import numpy as np
import data_reader

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generate and save a confusion matrix plot."""
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    np.fill_diagonal(cm, 0)

    plt.figure(figsize=(14, 12))

    sns.heatmap(cm, annot=False, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels)

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('Actual True Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)

    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def main():
    full_lem = data_reader.read("data/train_5500.label", lemmatize=True)
    test_data = data_reader.read("data/TREC_10.label", lemmatize=True)
    train_data = full_lem[:4500]

    y_true = [item.label for item in test_data]

    # ========== Flat SVM ==========
    print("Training Flat SVM...")
    flat_svm = SklearnClassifier(model_type="SVM", use_bigrams=False, use_tfidf=False, use_embeddings=False)
    flat_svm.train(train_data)
    y_pred_flat = flat_svm.predict_all(test_data)

    plot_confusion_matrix(
        y_true, y_pred_flat,
        "Flat SVM - Pure Error Analysis",
        "confusion_matrix_flat_svm.png"
    )

    # ========== Hierarchical SVM ==========
    print("Training Hierarchical SVM...")
    hier_svm = HierarchicalSVM(
        master_kwargs={"model_type": "SVM", "use_bigrams": False, "use_tfidf": False, "use_embeddings": False},
        sub_kwargs={"model_type": "SVM", "use_bigrams": False, "use_tfidf": False, "use_embeddings": False},
        top_k=2
    )
    hier_svm.train(train_data, epochs=10, verbose=False)
    y_pred_hier = hier_svm.predict_all(test_data)

    plot_confusion_matrix(
        y_true, y_pred_hier,
        "Hierarchical SVM - Pure Error Analysis",
        "confusion_matrix_hierarchical_svm.png"
    )

    # ========== Side-by-Side Comparison ==========
    print("Generating side-by-side comparison...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    labels = sorted(list(set(y_true)))

    # Flat SVM
    cm_flat = confusion_matrix(y_true, y_pred_flat, labels=labels)
    np.fill_diagonal(cm_flat, 0)
    sns.heatmap(cm_flat, annot=False, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title("Flat SVM - Error Analysis", fontsize=16, pad=20)
    ax1.set_ylabel('Actual True Category', fontsize=12)
    ax1.set_xlabel('Predicted Category', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax1.set_xticklabels(labels, rotation=90)

    # Hierarchical SVM
    cm_hier = confusion_matrix(y_true, y_pred_hier, labels=labels)
    np.fill_diagonal(cm_hier, 0)
    sns.heatmap(cm_hier, annot=False, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title("Hierarchical SVM - Error Analysis", fontsize=16, pad=20)
    ax2.set_ylabel('Actual True Category', fontsize=12)
    ax2.set_xlabel('Predicted Category', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    ax2.set_xticklabels(labels, rotation=90)

    plt.tight_layout()
    plt.savefig("confusion_matrix_comparison.png", dpi=300)
    plt.close()
    print("Saved: confusion_matrix_comparison.png")

    print("\nAll confusion matrices generated!")

if __name__ == "__main__":
    main()
  
