"""
Data Analysis Module for TREC Question Classification

Analyzes class distribution and identifies targets for data augmentation.
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

from .utils import (
    get_data_path,
    load_trec_file,
    write_trec_file,
    extract_coarse_label,
    ClassDistribution,
    print_progress
)


def analyze_class_distribution(
    train_file: str = "train_5500.label",
    output_dir: Optional[Path] = None
) -> ClassDistribution:
    """
    Analyze the class distribution of the training dataset.

    Args:
        train_file: Name of the training file
        output_dir: Optional directory to save analysis CSV

    Returns:
        ClassDistribution object with statistics
    """
    data_path = get_data_path() / train_file
    print(f"Loading data from {data_path}...")
    data = load_trec_file(data_path)

    dist = ClassDistribution(data)
    dist.print_summary()

    # Print coarse distribution
    print("\nCOARSE CLASS DISTRIBUTION:")
    print("-" * 40)
    for coarse, count in sorted(dist.coarse_counts.items(), key=lambda x: -x[1]):
        print(f"{coarse:10s}: {count:4d} samples")

    # Print rare classes
    rare_classes = dist.get_rare_classes(threshold=20)
    if rare_classes:
        print(f"\n\nRARE CLASSES (< 20 samples):")
        print("-" * 40)
        for label, count in rare_classes:
            print(f"{label:20s}: {count:2d} samples")

    # Print majority classes
    majority_classes = dist.get_majority_classes(threshold=300)
    if majority_classes:
        print(f"\n\nMAJORITY CLASSES (> 300 samples):")
        print("-" * 40)
        for label, count in majority_classes:
            print(f"{label:20s}: {count:4d} samples")

    # Print middle tail (for back-translation)
    middle_classes = dist.get_middle_tail_classes(min_threshold=20, max_threshold=100)
    if middle_classes:
        print(f"\n\nMIDDLE TAIL CLASSES (20-100 samples, candidates for back-translation):")
        print("-" * 40)
        for label, count in middle_classes:
            print(f"{label:20s}: {count:3d} samples")

    # Save to CSV if output directory provided
    if output_dir:
        save_distribution_to_csv(dist, output_dir / "class_distribution.csv")
        save_augmentation_targets(dist, output_dir / "augmentation_targets.csv")

    return dist


def save_distribution_to_csv(dist: ClassDistribution, output_path: Path):
    """Save detailed class distribution to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Fine_Label", "Coarse_Label", "Count", "Percentage"])

        total = len(dist.data)
        for label in sorted(dist.fine_counts.keys()):
            count = dist.fine_counts[label]
            coarse = extract_coarse_label(label)
            percentage = (count / total) * 100
            writer.writerow([label, coarse, count, f"{percentage:.2f}"])

    print(f"\nDistribution saved to {output_path}")


def save_augmentation_targets(dist: ClassDistribution, output_path: Path):
    """Save augmentation targets to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "Label", "Current_Count", "Target_Count", "Needed"])

        # Extreme tail (synthetic generation)
        extreme_tail = dist.get_rare_classes(threshold=20)
        for label, count in extreme_tail:
            target = 100
            needed = max(0, target - count)
            writer.writerow(["synthetic", label, count, target, needed])

        # Middle tail (back-translation)
        middle_tail = dist.get_middle_tail_classes(min_threshold=20, max_threshold=100)
        for label, count in middle_tail:
            target = 150
            needed = max(0, target - count)
            writer.writerow(["back_translation", label, count, target, needed])

        # Head (downsampling)
        majority = dist.get_majority_classes(threshold=300)
        for label, count in majority:
            target = 300
            writer.writerow(["downsample", label, count, target, count - target])

    print(f"Augmentation targets saved to {output_path}")


def load_augmentation_targets(csv_path: Path) -> Dict[str, List[Dict]]:
    """
    Load augmentation targets from CSV.

    Returns:
        Dict with keys: 'synthetic', 'back_translation', 'downsample'
        Each containing list of dicts with target info
    """
    targets = {"synthetic": [], "back_translation": [], "downsample": []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy = row["Strategy"]
            if strategy in targets:
                targets[strategy].append({
                    "label": row["Label"],
                    "current_count": int(row["Current_Count"]),
                    "target_count": int(row["Target_Count"]),
                    "needed": int(row["Needed"])
                })

    return targets


def print_augmentation_plan(targets: Dict[str, List[Dict]]):
    """Print a summary of the augmentation plan."""
    print("\n" + "=" * 60)
    print("AUGMENTATION PLAN")
    print("=" * 60)

    print("\n1. SYNTHETIC GENERATION (Extreme Tail: < 20 samples)")
    print("-" * 60)
    synthetic_total_needed = sum(t["needed"] for t in targets["synthetic"])
    print(f"Classes affected: {len(targets['synthetic'])}")
    print(f"Total samples to generate: {synthetic_total_needed}")

    if targets["synthetic"]:
        print("\nTop 10 rarest classes:")
        for t in sorted(targets["synthetic"], key=lambda x: x["current_count"])[:10]:
            print(f"  {t['label']:20s}: {t['current_count']:2d} → {t['target_count']} (need {t['needed']})")

    print("\n2. BACK-TRANSLATION (Middle Tail: 20-100 samples)")
    print("-" * 60)
    bt_total_needed = sum(t["needed"] for t in targets["back_translation"])
    print(f"Classes affected: {len(targets['back_translation'])}")
    print(f"Total samples to generate: {bt_total_needed}")

    print("\n3. DOWNSAMPLING (Majority Classes: > 300 samples)")
    print("-" * 60)
    downsample_total = sum(t["needed"] for t in targets["downsample"])
    print(f"Classes affected: {len(targets['downsample'])}")
    print(f"Total samples to remove: {downsample_total}")

    if targets["downsample"]:
        print("\nClasses to downsample:")
        for t in targets["downsample"]:
            print(f"  {t['label']:20s}: {t['current_count']:4d} → {t['target_count']} (remove {t['needed']})")

    print("\n" + "=" * 60)
    print(f"NET EFFECT: {synthetic_total_needed + bt_total_needed - downsample_total:+d} samples")
    print("=" * 60)


if __name__ == "__main__":
    # Run analysis when script is executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TREC dataset class distribution")
    parser.add_argument("--train-file", default="train_5500.label", help="Training file name")
    parser.add_argument("--output-dir", default=None, help="Output directory for analysis files")

    args = parser.parse_args()

    output_path = Path(args.output_dir) if args.output_dir else None
    dist = analyze_class_distribution(args.train_file, output_path)

    if output_path:
        targets = load_augmentation_targets(output_path / "augmentation_targets.csv")
        print_augmentation_plan(targets)
