"""
Shared utilities for the BERT pipeline.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import random
import numpy as np


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent


def get_data_path() -> Path:
    """Get the data directory path."""
    return get_project_root() / "data"


def setup_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def extract_coarse_label(fine_label: str) -> str:
    """Extract coarse label from fine-grained label (e.g., 'HUM:ind' -> 'HUM')."""
    return fine_label.split(":")[0] if ":" in fine_label else fine_label


def count_vocabulary_diversity(texts: List[str]) -> float:
    """
    Measure vocabulary diversity using unique token count.
    Returns the ratio of unique tokens to total tokens.
    """
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())

    if not all_tokens:
        return 0.0

    unique_tokens = set(all_tokens)
    return len(unique_tokens) / len(all_tokens)


def compute_text_variance(texts: List[str]) -> float:
    """
    Compute text length variance as a measure of diversity.
    Higher variance = more diverse lengths.
    """
    lengths = [len(text.split()) for text in texts]
    if not lengths:
        return 0.0
    return float(np.var(lengths))


def diversity_score(text: str, corpus_vocabulary: Optional[set] = None) -> float:
    """
    Compute a diversity score for a single text.
    Higher score = more unique/rare vocabulary.
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    if corpus_vocabulary:
        # Score based on rare words in corpus
        unique_rare = sum(1 for t in tokens if t in corpus_vocabulary)
        return unique_rare / len(tokens)
    else:
        # Score based on type-token ratio
        return len(set(tokens)) / len(tokens)


def load_trec_file(filepath: Path) -> List[Tuple[str, str]]:
    """
    Load a TREC format label file.
    Returns list of (label, text) tuples.
    """
    data = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                label, text = parts
                data.append((label, text))
    return data


def write_trec_file(filepath: Path, data: List[Tuple[str, str]]):
    """
    Write data to a TREC format label file.
    Uses UTF-8 encoding to support Unicode characters from back-translation.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for label, text in data:
            f.write(f"{label} {text}\n")


def print_progress(current: int, total: int, prefix: str = ""):
    """Print a simple progress bar."""
    percent = 100 * current / total if total > 0 else 100
    bar_length = 50
    filled = int(bar_length * current / total) if total > 0 else bar_length
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  # New line when complete


class ClassDistribution:
    """Helper class to store and analyze class distribution statistics."""

    def __init__(self, data: List[Tuple[str, str]]):
        """
        Initialize with TREC format data [(label, text), ...].

        Args:
            data: List of (label, text) tuples
        """
        self.data = data
        self.fine_counts = Counter([label for label, _ in data])
        self.coarse_counts = Counter([extract_coarse_label(label) for label, _ in data])

        # Group by coarse label
        self.by_coarse = defaultdict(list)
        for label, text in data:
            coarse = extract_coarse_label(label)
            self.by_coarse[coarse].append((label, text))

    def get_rare_classes(self, threshold: int = 20) -> List[Tuple[str, int]]:
        """Get fine-grained classes with fewer than threshold samples."""
        return sorted([(label, count) for label, count in self.fine_counts.items() if count < threshold],
                      key=lambda x: x[1])

    def get_majority_classes(self, threshold: int = 300) -> List[Tuple[str, int]]:
        """Get fine-grained classes with more than threshold samples."""
        return sorted([(label, count) for label, count in self.fine_counts.items() if count > threshold],
                      key=lambda x: -x[1])

    def get_middle_tail_classes(self, min_threshold: int = 20, max_threshold: int = 100) -> List[Tuple[str, int]]:
        """Get classes in the middle range for back-translation."""
        return sorted([(label, count) for label, count in self.fine_counts.items()
                       if min_threshold <= count <= max_threshold],
                      key=lambda x: x[1])

    def get_samples_for_class(self, label: str) -> List[Tuple[str, str]]:
        """Get all samples for a specific class."""
        return [(l, t) for l, t in self.data if l == label]

    def get_samples_for_coarse(self, coarse: str) -> List[Tuple[str, str]]:
        """Get all samples for a coarse class."""
        return self.by_coarse.get(coarse, [])

    def summary(self) -> Dict:
        """Return summary statistics."""
        return {
            "total_samples": len(self.data),
            "num_fine_classes": len(self.fine_counts),
            "num_coarse_classes": len(self.coarse_counts),
            "max_count": max(self.fine_counts.values()) if self.fine_counts else 0,
            "min_count": min(self.fine_counts.values()) if self.fine_counts else 0,
            "imbalance_ratio": max(self.fine_counts.values()) / min(self.fine_counts.values())
            if self.fine_counts and min(self.fine_counts.values()) > 0 else float('inf'),
        }

    def print_summary(self):
        """Print a formatted summary of the distribution."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {s['total_samples']}")
        print(f"Fine-grained classes: {s['num_fine_classes']}")
        print(f"Coarse classes: {s['num_coarse_classes']}")
        print(f"Imbalance ratio: {s['imbalance_ratio']:.1f}:1")
        print(f"Range: {s['min_count']} - {s['max_count']} samples")
        print("=" * 60)
