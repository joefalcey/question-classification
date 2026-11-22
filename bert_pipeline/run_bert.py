"""
BERT Training Runner for TREC Question Classification

This is a standalone script for training and evaluating BERT models
on the TREC dataset, separate from the main.py evaluation loop.

Usage:
    # Run with default settings (augmented dataset)
    python -m bert_pipeline.run_bert

    # Run with specific settings
    python -m bert_pipeline.run_bert --epochs 5 --batch-size 32

    # Run from parent directory
    uv run python -m bert_pipeline.run_bert
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_reader import Text
from evaluate import evaluate, print_detailed_report
from .bert_classifier import BERTClassifier


def load_data(
    train_file: str = "train_5500_augmented.label",
    test_file: str = "TREC_10.label",
    lemmatize: bool = True
):
    """
    Load training and test data.

    Args:
        train_file: Training filename
        test_file: Test filename
        lemmatize: Whether to use lemmatized text

    Returns:
        (train_data, test_data, dev_data, dev_true, test_true)
    """
    from data_reader import read
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data"

    print(f"Loading data from {data_path}...")

    # Load training data
    train_path = data_path / train_file
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    full_train = read(str(train_path), lemmatize=lemmatize)

    # Split into train/dev (80/20 split of training data)
    # Shuffle first to ensure class distribution is preserved
    import random
    random.shuffle(full_train)

    # Use 80% for train, 20% for dev
    split_idx = int(len(full_train) * 0.8)
    train_data = full_train[:split_idx]
    dev_data = full_train[split_idx:]

    # Load test data
    test_path = data_path / test_file
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test_data = read(str(test_path), lemmatize=lemmatize)

    # Extract true labels for evaluation
    dev_true = [item.label for item in dev_data]
    test_true = [item.label for item in test_data]

    # Verify class coverage
    train_labels = set(item.label for item in train_data)
    dev_labels = set(item.label for item in dev_data)
    test_labels = set(item.label for item in test_data)

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    print(f"Train classes: {len(train_labels)}, Dev classes: {len(dev_labels)}, Test classes: {len(test_labels)}")
    print(f"Dev labels missing from train: {len(dev_labels - train_labels)}")
    print(f"Test labels missing from train: {len(test_labels - train_labels)}")

    return train_data, dev_data, test_data, dev_true, test_true


def run_single_experiment(
    train_data,
    dev_data,
    test_data,
    dev_true,
    test_true,
    model_name: str = "distilbert-base-uncased",
    class_weights: str = "balanced",
    use_focal_loss: bool = True,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    verbose: bool = True,
    experiment_name: str = "BERT"
):
    """
    Run a single BERT training experiment.

    Args:
        train_data: Training data
        dev_data: Development data
        test_data: Test data
        dev_true: True dev labels
        test_true: True test labels
        model_name: HuggingFace model name
        class_weights: Class weighting strategy
        use_focal_loss: Whether to use focal loss
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        verbose: Print progress
        experiment_name: Name for this experiment

    Returns:
        Dict with results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Class weights: {class_weights}")
        print(f"Focal loss: {use_focal_loss}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*60}\n")

    # Create classifier
    classifier = BERTClassifier(
        model_name=model_name,
        class_weights=class_weights,
        use_focal_loss=use_focal_loss,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Train
    classifier.train(train_data, epochs=epochs, verbose=verbose)

    # Evaluate on dev set
    if verbose:
        print("\nEvaluating on development set...")
    dev_predictions = classifier.predict_all(dev_data)
    dev_f1 = evaluate(dev_true, dev_predictions, verbose=verbose)

    # Evaluate on test set
    if verbose:
        print("\nEvaluating on test set...")
    test_predictions = classifier.predict_all(test_data)
    test_f1 = evaluate(test_true, test_predictions, verbose=verbose)

    # Detailed report
    if verbose:
        print_detailed_report(test_true, test_predictions, experiment_name)

    return {
        "experiment_name": experiment_name,
        "dev_f1": dev_f1,
        "test_f1": test_f1,
        "classifier": classifier
    }


def run_multiple_runs(
    train_data,
    dev_data,
    test_data,
    dev_true,
    test_true,
    num_runs: int = 3,
    model_name: str = "distilbert-base-uncased",
    class_weights: str = "balanced",
    use_focal_loss: bool = True,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    verbose: bool = True
):
    """
    Run multiple experiments with different random seeds.

    Args:
        Same as run_single_experiment, plus:
        num_runs: Number of runs to perform

    Returns:
        Dict with aggregated results
    """
    dev_f1_scores = []
    test_f1_scores = []

    for run in range(num_runs):
        if verbose:
            print(f"\n\n{'#'*60}")
            print(f"# RUN {run + 1}/{num_runs}")
            print(f"{'#'*60}")

        # Set different seed for each run
        seed = 42 + run
        np.random.seed(seed)

        result = run_single_experiment(
            train_data, dev_data, test_data,
            dev_true, test_true,
            model_name=model_name,
            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
            experiment_name=f"BERT_Run_{run+1}"
        )

        dev_f1_scores.append(result["dev_f1"])
        test_f1_scores.append(result["test_f1"])

    # Summary
    if verbose:
        print(f"\n\n{'='*60}")
        print("MULTIPLE RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Runs: {num_runs}")
        print(f"Dev F1:    {np.mean(dev_f1_scores):.4f} ± {np.std(dev_f1_scores):.4f}")
        print(f"Test F1:   {np.mean(test_f1_scores):.4f} ± {np.std(test_f1_scores):.4f}")
        print(f"{'='*60}")

    return {
        "num_runs": num_runs,
        "dev_mean": np.mean(dev_f1_scores),
        "dev_std": np.std(dev_f1_scores),
        "test_mean": np.mean(test_f1_scores),
        "test_std": np.std(test_f1_scores),
        "all_dev_scores": dev_f1_scores,
        "all_test_scores": test_f1_scores
    }


def compare_configurations(
    train_data,
    dev_data,
    test_data,
    dev_true,
    test_true,
    configurations: list,
    verbose: bool = True
):
    """
    Compare multiple BERT configurations.

    Args:
        configurations: List of dicts with configuration parameters

    Returns:
        Comparison results
    """
    results = []

    for i, config in enumerate(configurations):
        if verbose:
            print(f"\n\n{'#'*60}")
            print(f"# CONFIGURATION {i + 1}/{len(configurations)}")
            print(f"{'#'*60}")

        result = run_single_experiment(
            train_data, dev_data, test_data,
            dev_true, test_true,
            experiment_name=config.get("name", f"Config_{i+1}"),
            model_name=config.get("model_name", "distilbert-base-uncased"),
            class_weights=config.get("class_weights", "balanced"),
            use_focal_loss=config.get("use_focal_loss", True),
            epochs=config.get("epochs", 3),
            batch_size=config.get("batch_size", 16),
            learning_rate=config.get("learning_rate", 2e-5),
            verbose=verbose
        )

        results.append({
            "config": config,
            "dev_f1": result["dev_f1"],
            "test_f1": result["test_f1"]
        })

    # Summary comparison
    if verbose:
        print(f"\n\n{'='*60}")
        print("CONFIGURATION COMPARISON")
        print(f"{'='*60}")
        print(f"{'Configuration':<30} | {'Dev F1':<10} | {'Test F1':<10}")
        print("-" * 60)

        for r in results:
            name = r["config"].get("name", "Unknown")
            print(f"{name:<30} | {r['dev_f1']:.4f}     | {r['test_f1']:.4f}")

        print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate BERT on TREC question classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with augmented dataset, default settings
  python -m bert_pipeline.run_bert

  # Run with more epochs
  python -m bert_pipeline.run_bert --epochs 5

  # Run multiple runs for statistical significance
  python -m bert_pipeline.run_bert --runs 3 --epochs 5

  # Compare multiple configurations (loss functions, class weights)
  python -m bert_pipeline.run_bert --compare

  # Try different BERT models
  python -m bert_pipeline.run_bert --model-name bert-base-uncased
  python -m bert_pipeline.run_bert --model-name roberta-base

  # Try different learning rates
  python -m bert_pipeline.run_bert --learning-rate 3e-5 --epochs 5

  # Use raw text instead of lemmatized
  python -m bert_pipeline.run_bert --no-lemmatize

  # Use original dataset instead of augmented
  python -m bert_pipeline.run_bert --train-file train_5500.label

  # Save the trained model
  python -m bert_pipeline.run_bert --save-model
        """
    )

    # Data arguments
    parser.add_argument(
        "--train-file",
        default="train_5500_augmented.label",
        help="Training file name (default: train_5500_augmented.label)"
    )
    parser.add_argument(
        "--test-file",
        default="TREC_10.label",
        help="Test file name (default: TREC_10.label)"
    )
    parser.add_argument(
        "--no-lemmatize",
        action="store_true",
        help="Use raw text instead of lemmatized"
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="HuggingFace model name (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--class-weights",
        default="balanced",
        choices=["balanced", "inverse", "none"],
        help="Class weighting strategy (default: balanced)"
    )
    parser.add_argument(
        "--no-focal-loss",
        action="store_true",
        help="Use standard cross-entropy instead of focal loss"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )

    # Experiment arguments
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs with different seeds (default: 1)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple predefined configurations"
    )

    # Output arguments
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model"
    )
    parser.add_argument(
        "--output-dir",
        default="bert_pipeline/models",
        help="Directory to save models (default: bert_pipeline/models)"
    )

    args = parser.parse_args()

    # Load data
    lemmatize = not args.no_lemmatize
    train_data, dev_data, test_data, dev_true, test_true = load_data(
        train_file=args.train_file,
        test_file=args.test_file,
        lemmatize=lemmatize
    )

    use_focal_loss = not args.no_focal_loss

    # Run experiments
    if args.compare:
        # Compare multiple configurations
        configurations = [
            {
                "name": "DistilBERT_Balanced_Focal",
                "model_name": "distilbert-base-uncased",
                "class_weights": "balanced",
                "use_focal_loss": True,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate
            },
            {
                "name": "DistilBERT_Balanced_CEL",
                "model_name": "distilbert-base-uncased",
                "class_weights": "balanced",
                "use_focal_loss": False,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate
            },
            {
                "name": "DistilBERT_None_Focal",
                "model_name": "distilbert-base-uncased",
                "class_weights": "none",
                "use_focal_loss": True,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate
            },
        ]

        results = compare_configurations(
            train_data, dev_data, test_data,
            dev_true, test_true,
            configurations=configurations
        )

    elif args.runs > 1:
        # Multiple runs
        results = run_multiple_runs(
            train_data, dev_data, test_data,
            dev_true, test_true,
            num_runs=args.runs,
            model_name=args.model_name,
            class_weights=args.class_weights,
            use_focal_loss=use_focal_loss,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

    else:
        # Single run
        result = run_single_experiment(
            train_data, dev_data, test_data,
            dev_true, test_true,
            model_name=args.model_name,
            class_weights=args.class_weights,
            use_focal_loss=use_focal_loss,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            experiment_name=f"BERT_{args.model_name}"
        )

        if args.save_model:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            model_path = output_path / f"{args.model_name}_augmented"
            result["classifier"].save(str(model_path))
            result["classifier"].save_label_mappings(str(model_path))

            print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
