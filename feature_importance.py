"""
Feature Importance Extraction for TREC Question Classification

Extracts and analyzes the most important features (words) for each class
from the Linear SVM model to prove the "trigger word" hypothesis.
"""

import data_reader
from sklearn_models import SklearnClassifier
import numpy as np
from collections import defaultdict

def get_coarse_label(fine_label):
    """Extract coarse label from fine-grained label."""
    return fine_label.split(':')[0] if ':' in fine_label else fine_label

def extract_feature_importance():
    """
    Train the best model and extract feature importance for each class.
    """
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS - Linear SVM (BoW Unigram, Raw Text)")
    print("=" * 80)

    # Load data - Raw text (best performing according to results)
    print("\nLoading data...")
    full_raw = data_reader.read("data/train_5500.label", lemmatize=False)
    train_raw = full_raw[:4500]

    print(f"Training set size: {len(train_raw)}")

    # Train the best model
    print("\nTraining Linear SVM (BoW Unigram)...")
    model = SklearnClassifier(
        model_type="SVM",
        use_bigrams=False,
        use_tfidf=False,
        use_embeddings=False
    )
    model.train(train_raw, epochs=1, verbose=True)

    # Get feature names and weights
    feature_names = model.dict_vec.get_feature_names_out()
    classes = model.model.classes_
    coef_matrix = model.model.coef_

    print(f"\nTotal features (unique words): {len(feature_names)}")
    print(f"Total classes: {len(classes)}")

    # Group by coarse classes
    coarse_to_fine = defaultdict(list)
    for cls in classes:
        coarse = get_coarse_label(cls)
        coarse_to_fine[coarse].append(cls)

    # Output feature importance for each class
    print("\n" + "=" * 80)
    print("TOP 15 TRIGGER WORDS PER CLASS")
    print("=" * 80)

    results = {}

    for coarse_label in sorted(coarse_to_fine.keys()):
        print(f"\n{'=' * 80}")
        print(f"COARSE CLASS: {coarse_label}")
        print(f"{'=' * 80}\n")

        fine_classes = sorted(coarse_to_fine[coarse_label])
        results[coarse_label] = {}

        for fine_cls in fine_classes:
            # Get index of this class
            cls_idx = list(classes).index(fine_cls)

            # Get weights for this class
            weights = coef_matrix[cls_idx]

            # Top 15 highest-weighted features (most predictive)
            top_indices = np.argsort(weights)[-15:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            top_weights = [weights[idx] for idx in top_indices]

            # Bottom 10 lowest-weighted features (most negative)
            bottom_indices = np.argsort(weights)[:10]
            bottom_words = [feature_names[idx] for idx in bottom_indices]
            bottom_weights = [weights[idx] for idx in bottom_indices]

            results[coarse_label][fine_cls] = {
                'top': list(zip(top_words, top_weights)),
                'bottom': list(zip(bottom_words, bottom_weights))
            }

            # Print results
            print(f"  {fine_cls}")
            print(f"    Top 15 trigger words: {', '.join(top_words)}")
            # print(f"    (weights: {[f'{w:.3f}' for w in top_weights]})")
            # print(f"    Bottom 10 (negative): {', '.join(bottom_words)}")
            print()

    # Save summary to file
    save_summary(results, feature_names, classes, coef_matrix)

    print("\n" + "=" * 80)
    print("Feature importance analysis complete!")
    print("Results saved to: feature_importance_results.txt")
    print("=" * 80)

    return results, model


def save_summary(results, feature_names, classes, coef_matrix):
    """Save feature importance summary to a text file."""

    with open('feature_importance_results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS - TREC Question Classification\n")
        f.write("Model: Linear SVM (BoW Unigram, Raw Text)\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY BY COARSE CLASS\n\n")

        for coarse_label in sorted(results.keys()):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"COARSE CLASS: {coarse_label}\n")
            f.write(f"{'=' * 80}\n\n")

            for fine_cls in sorted(results[coarse_label].keys()):
                f.write(f"{fine_cls}\n")
                f.write(f"  Top 15 trigger words:\n")
                for word, weight in results[coarse_label][fine_cls]['top']:
                    f.write(f"    {word:<20} (weight: {weight:.4f})\n")
                f.write("\n")

        # Global analysis - most distinctive words across all classes
        f.write("\n" + "=" * 80 + "\n")
        f.write("GLOBAL ANALYSIS - Most Distinctive Words per Coarse Class\n")
        f.write("=" * 80 + "\n\n")

        # For each coarse class, find words with highest average weight
        coarse_analysis = analyze_coarse_level_words(results)
        for coarse, words in coarse_analysis.items():
            f.write(f"\n{coarse}:\n")
            for word, score in words[:10]:
                f.write(f"  {word:<20} (avg distinctiveness: {score:.4f})\n")


def analyze_coarse_level_words(results):
    """
    Analyze which words are most distinctive for each coarse class.
    Returns dictionary mapping coarse class to list of (word, score) tuples.
    """
    # This is a simplified version - we could make this more sophisticated
    coarse_words = defaultdict(list)

    for coarse_label, fine_classes in results.items():
        all_words = defaultdict(list)

        for fine_cls, data in fine_classes.items():
            for word, weight in data['top']:
                all_words[word].append(weight)

        # Average weight per word
        avg_words = {word: np.mean(weights) for word, weights in all_words.items()}
        coarse_words[coarse_label] = sorted(avg_words.items(), key=lambda x: x[1], reverse=True)

    return coarse_words


def analyze_trigger_word_hypothesis(results):
    """
    Analyze the results to validate the trigger word hypothesis.
    """
    print("\n" + "=" * 80)
    print("TRIGGER WORD HYPOTHESIS ANALYSIS")
    print("=" * 80)

    # Expected trigger words based on domain knowledge
    expected_triggers = {
        'ABBR': ['abbreviation', 'abbreviate', 'stand', 'stands', 'means', 'meaning'],
        'DESC': ['describe', 'definition', 'what', 'define', 'explanation'],
        'ENTY': ['what', 'type', 'kind', 'name', 'category', 'class'],
        'HUM': ['who', 'person', 'someone', 'name', 'character'],
        'LOC': ['where', 'location', 'place', 'city', 'country', 'state'],
        'NUM': ['how', 'many', 'much', 'count', 'number', 'often']
    }

    print("\nChecking for expected trigger words in top features:\n")

    for coarse_label, triggers in expected_triggers.items():
        if coarse_label not in results:
            continue

        print(f"\n{coarse_label}:")
        print(f"  Expected: {', '.join(triggers[:5])}")

        # Check if these appear in top words for any fine-grained class
        found_triggers = set()
        for fine_cls, data in results[coarse_label].items():
            top_words = [word for word, _ in data['top']]
            for trigger in triggers:
                if trigger in top_words:
                    found_triggers.add(trigger)

        if found_triggers:
            print(f"  ✓ Found in top features: {', '.join(sorted(found_triggers))}")
        else:
            print(f"  ✗ None found in top features")


if __name__ == "__main__":
    results, model = extract_feature_importance()
    analyze_trigger_word_hypothesis(results)
