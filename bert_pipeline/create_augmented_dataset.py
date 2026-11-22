"""
Augmented Dataset Creation Pipeline

Orchestrates the complete data augmentation process:
1. Analyze class distribution
2. Generate synthetic data for rare classes (1-20 samples)
3. Apply back-translation for middle tail (20-100 samples)
4. Downsample majority classes (>300 samples)
5. Create final augmented dataset
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random

from .utils import (
    get_data_path,
    load_trec_file,
    write_trec_file,
    print_progress
)
from .data_analysis import (
    analyze_class_distribution,
    load_augmentation_targets,
    print_augmentation_plan
)
from .data_balancer import downsample_majority_classes
from .synthetic_generator import SyntheticQuestionGenerator
from .back_translation_augment import (
    BackTranslationAugmenter,
    DEEP_TRANSLATOR_AVAILABLE
)


def create_augmented_dataset(
    input_file: str = "train_5500.label",
    output_file: str = "train_5500_augmented.label",
    output_dir: Optional[Path] = None,
    skip_synthetic: bool = False,
    skip_back_translation: bool = False,
    manual_synthetic_dir: Optional[Path] = None
) -> Path:
    """
    Create an augmented TREC dataset.

    Args:
        input_file: Original training file
        output_file: Output file name
        output_dir: Optional directory for analysis and prompts
        skip_synthetic: Skip synthetic generation (use manual only)
        skip_back_translation: Skip back-translation
        manual_synthetic_dir: Directory with manually generated questions

    Returns:
        Path to the output file
    """
    print("\n" + "=" * 60)
    print("TREC DATASET AUGMENTATION PIPELINE")
    print("=" * 60)

    # Setup paths
    data_path = get_data_path()
    input_path = data_path / input_file
    output_path = data_path / output_file

    if output_dir is None:
        output_dir = data_path.parent / "bert_pipeline" / "augmentation_output"
    output_dir = Path(output_dir)

    # Step 1: Analyze distribution
    print("\n[Step 1/5] Analyzing class distribution...")
    dist = analyze_class_distribution(input_file, output_dir)
    original_data = load_trec_file(input_path)

    # Load or create targets
    targets_file = output_dir / "augmentation_targets.csv"
    if targets_file.exists():
        targets = load_augmentation_targets(targets_file)
    else:
        # This would create targets, but we need the analysis to have saved them
        # For now, we'll compute them directly
        from .data_analysis import save_augmentation_targets
        save_augmentation_targets(dist, targets_file)
        targets = load_augmentation_targets(targets_file)

    print_augmentation_plan(targets)

    # Step 2: Synthetic generation for extreme tail
    print("\n[Step 2/5] Synthetic generation for rare classes...")

    if skip_synthetic:
        print("Skipping synthetic generation (will use manual data only)")
    else:
        generator = SyntheticQuestionGenerator()
        synthetic_dir = output_dir / "synthetic_prompts"
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        # Generate prompts for each rare class
        for target in targets["synthetic"]:
            fine_label = target["label"]
            existing = dist.get_samples_for_class(fine_label)
            existing_questions = [text for _, text in existing]

            generator.save_prompts_for_class(
                fine_label,
                existing_questions,
                target["needed"],
                synthetic_dir
            )

        print(f"\nPrompts saved to {synthetic_dir}")
        print("Please generate questions manually using the prompts, then re-run with")
        print("--manual-synthetic-dir to load the generated questions.")
        print("\nContinuing with manual data only for now...")

    # Load manually generated synthetic data if available
    synthetic_data: List[Tuple[str, str]] = []
    if manual_synthetic_dir:
        manual_dir = Path(manual_synthetic_dir)
        print(f"\nLoading manually generated synthetic data from {manual_dir}...")

        generator = SyntheticQuestionGenerator()
        for target in targets["synthetic"]:
            fine_label = target["label"]
            generated = generator.load_generated_questions(fine_label, manual_dir)

            for q in generated:
                synthetic_data.append((fine_label, q))

            if generated:
                print(f"  Loaded {len(generated)} synthetic questions for {fine_label}")

    # Step 3: Back-translation for middle tail
    print("\n[Step 3/5] Back-translation for middle tail classes...")

    if skip_back_translation or not DEEP_TRANSLATOR_AVAILABLE:
        print("Skipping back-translation")
        bt_data: List[Tuple[str, str]] = []
    else:
        try:
            augmenter = BackTranslationAugmenter(pivot_languages=['german', 'french', 'spanish'])

            # Get middle tail targets
            middle_targets = [
                {"label": t["label"], "target_count": t["target_count"]}
                for t in targets["back_translation"]
            ]

            bt_data = augmenter.augment_dataset(
                original_data,
                middle_targets,
                min_samples=20,
                max_paraphrases_per_sample=2
            )
        except Exception as e:
            print(f"Back-translation failed: {e}")
            print("Continuing without back-translation...")
            bt_data = []

    # Step 4: Combine original + synthetic + back-translated
    print("\n[Step 4/5] Combining augmented data...")

    combined_data = list(original_data)  # Start with original

    # Add synthetic data
    for label, text in synthetic_data:
        combined_data.append((label, text))

    # Add back-translated data (only the augmented samples)
    for label, text in bt_data:
        combined_data.append((label, text))

    print(f"Original: {len(original_data)}")
    print(f"Synthetic: {len(synthetic_data)}")
    print(f"Back-translated: {len(bt_data)}")
    print(f"Combined: {len(combined_data)}")

    # Step 5: Downsample majority classes
    print("\n[Step 5/5] Downsampling majority classes...")

    if targets["downsample"]:
        final_data = downsample_majority_classes(
            combined_data,
            targets["downsample"],
            strategy="diverse"
        )
    else:
        final_data = combined_data

    print(f"\nFinal dataset size: {len(final_data)}")
    print(f"Net change: {len(final_data) - len(original_data):+d} samples")

    # Write output
    write_trec_file(output_path, final_data)
    print(f"\nAugmented dataset saved to: {output_path}")

    # Final summary
    from .utils import ClassDistribution
    final_dist = ClassDistribution(final_data)
    final_dist.print_summary()

    return output_path


def run(
    input_file: str = "train_5500.label",
    output_file: str = "train_5500_augmented.label",
    output_dir: Optional[str] = None,
    skip_synthetic: bool = False,
    skip_back_translation: bool = False,
    manual_synthetic_dir: Optional[str] = None
):
    """
    Run the augmentation pipeline with command-line style arguments.

    This function can be called directly or used as an entry point.
    """
    output_path = create_augmented_dataset(
        input_file=input_file,
        output_file=output_file,
        output_dir=Path(output_dir) if output_dir else None,
        skip_synthetic=skip_synthetic,
        skip_back_translation=skip_back_translation,
        manual_synthetic_dir=Path(manual_synthetic_dir) if manual_synthetic_dir else None
    )

    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")
    print("\nNext steps:")
    print("1. Review the augmented dataset")
    print("2. Update main.py to use the augmented file")
    print("3. Train BERT model on the balanced dataset")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create augmented TREC dataset for BERT training"
    )
    parser.add_argument(
        "--input-file",
        default="train_5500.label",
        help="Input training file"
    )
    parser.add_argument(
        "--output-file",
        default="train_5500_augmented.label",
        help="Output augmented file"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis and prompts"
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic generation prompts"
    )
    parser.add_argument(
        "--skip-back-translation",
        action="store_true",
        help="Skip back-translation"
    )
    parser.add_argument(
        "--manual-synthetic-dir",
        default=None,
        help="Directory with manually generated synthetic questions"
    )

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        output_file=args.output_file,
        output_dir=args.output_dir,
        skip_synthetic=args.skip_synthetic,
        skip_back_translation=args.skip_back_translation,
        manual_synthetic_dir=args.manual_synthetic_dir
    )
