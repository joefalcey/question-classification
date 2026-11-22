"""
Data Balancer Module for TREC Question Classification

Handles downsampling of majority classes while preserving diversity.
"""

import random
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import numpy as np

from .utils import extract_coarse_label, diversity_score, compute_text_variance


def downsample_majority_classes(
    data: List[Tuple[str, str]],
    targets: List[Dict],
    strategy: str = "diverse"
) -> List[Tuple[str, str]]:
    """
    Downsample majority classes according to targets.

    Args:
        data: Original training data as [(label, text), ...]
        targets: List of dicts with 'label' and 'target_count'
        strategy: 'diverse' (keep diverse samples) or 'random'

    Returns:
        Downsampled data
    """
    target_map = {t["label"]: t["target_count"] for t in targets}

    # Group data by class
    by_class: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for label, text in data:
        by_class[label].append((label, text))

    result = []
    total_removed = 0

    for label, samples in by_class.items():
        target_count = target_map.get(label)

        if target_count is None or len(samples) <= target_count:
            # No downsampling needed
            result.extend(samples)
        else:
            # Downsample this class
            if strategy == "diverse":
                selected = select_diverse_samples(samples, target_count)
            else:
                selected = random.sample(samples, target_count)

            result.extend(selected)
            total_removed += len(samples) - target_count

    print(f"Downsampling complete: removed {total_removed} samples")
    return result


def select_diverse_samples(
    samples: List[Tuple[str, str]],
    n: int,
    corpus_vocabulary: Set[str] = None
) -> List[Tuple[str, str]]:
    """
    Select n most diverse samples from a list.
    Uses vocabulary diversity and text variance as metrics.

    Args:
        samples: List of (label, text) tuples
        n: Number of samples to select
        corpus_vocabulary: Optional set of vocabulary for rarity scoring

    Returns:
        Selected samples
    """
    if len(samples) <= n:
        return samples

    # Compute diversity scores
    scored_samples = []
    for label, text in samples:
        score = diversity_score(text, corpus_vocabulary)
        variance = compute_text_variance([text])
        # Combine: diversity score + variance factor
        combined_score = score + (variance / 100.0)  # Normalize variance
        scored_samples.append((combined_score, label, text))

    # Sort by score (highest first) and select top n
    scored_samples.sort(key=lambda x: x[0], reverse=True)
    return [(label, text) for _, label, text in scored_samples[:n]]


def compute_class_vocabulary(data: List[Tuple[str, str]], top_n: int = 1000) -> Set[str]:
    """
    Compute the vocabulary of the dataset for rarity scoring.

    Returns:
        Set of top N most common words (for rarity computation)
    """
    word_counts = defaultdict(int)
    for _, text in data:
        for word in text.lower().split():
            word_counts[word] += 1

    # Return top N words as "common" vocabulary
    # (rare words are those NOT in this set)
    top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:top_n]
    return set(word for word, _ in top_words)


def balanced_sample(data: List[Tuple[str, str]], target_per_class: int = 300) -> List[Tuple[str, str]]:
    """
    Create a balanced dataset by either upsampling or downsampling.

    Args:
        data: Original data
        target_per_class: Target samples per class

    Returns:
        Balanced data
    """
    by_class: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for label, text in data:
        by_class[label].append((label, text))

    result = []
    for label, samples in by_class.items():
        if len(samples) >= target_per_class:
            # Downsample
            result.extend(random.sample(samples, target_per_class))
        else:
            # Upsample with repetition (simple approach)
            # For better results, use synthetic generation instead
            result.extend(samples)
            while len(result) % target_per_class != len(samples):
                result.extend(random.sample(samples, min(len(samples), target_per_class - len(result))))

    return result


def split_train_val(
    data: List[Tuple[str, str]],
    val_ratio: float = 0.15,
    stratify: bool = True
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split data into train and validation sets.

    Args:
        data: Input data
        val_ratio: Fraction for validation set
        stratify: Whether to maintain class proportions

    Returns:
        (train_data, val_data)
    """
    if not stratify:
        random.shuffle(data)
        split_idx = int(len(data) * (1 - val_ratio))
        return data[:split_idx], data[split_idx:]

    # Stratified split
    by_class: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for item in data:
        by_class[item[0]].append(item)

    train_data = []
    val_data = []

    for label, samples in by_class.items():
        random.shuffle(samples)
        split_idx = max(1, int(len(samples) * (1 - val_ratio)))
        train_data.extend(samples[:split_idx])
        val_data.extend(samples[split_idx:])

    random.shuffle(train_data)
    random.shuffle(val_data)

    return train_data, val_data


if __name__ == "__main__":
    # Test downsampling logic
    from .utils import load_trec_file, get_data_path

    print("Testing data balancer...")

    # Load sample data
    data_path = get_data_path() / "train_5500.label"
    data = load_trec_file(data_path)

    # Test downsampling
    targets = [
        {"label": "HUM:ind", "target_count": 300},
        {"label": "LOC:other", "target_count": 300},
    ]

    downsampled = downsample_majority_classes(data, targets, strategy="diverse")
    print(f"Original: {len(data)} samples")
    print(f"After downsampling: {len(downsampled)} samples")
