"""
Back-Translation Augmenter for TREC Question Classification

Uses back-translation to generate paraphrased questions for classes
in the middle range (20-100 samples).
"""

import random
import time
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    print("Warning: deep-translator not installed. Back-translation will not work.")
    print("Install with: pip install deep-translator")


class BackTranslationAugmenter:
    """
    Generate paraphrased questions using back-translation.

    Process: English → Pivot Language → English
    This creates diverse paraphrases while preserving semantic meaning.
    """

    # Pivot languages for back-translation
    PIVOT_LANGUAGES = {
        'german': 'de',
        'french': 'fr',
        'spanish': 'es',
        'italian': 'it',
        'portuguese': 'pt',
        'dutch': 'nl',
    }

    def __init__(self, pivot_languages: Optional[List[str]] = None):
        """
        Initialize the back-translation augmenter.

        Args:
            pivot_languages: List of pivot languages to use.
                           If None, uses a random selection of 3.
        """
        if not DEEP_TRANSLATOR_AVAILABLE:
            raise ImportError("deep-translator is required for back-translation. "
                            "Install with: pip install deep-translator")

        if pivot_languages is None:
            # Use a diverse set of 3 languages by default
            pivot_languages = ['german', 'french', 'spanish']

        self.pivot_languages = [
            (name, self.PIVOT_LANGUAGES[name])
            for name in pivot_languages
            if name in self.PIVOT_LANGUAGES
        ]

        print(f"Initialized back-translation with pivot languages: "
              f"{[name for name, _ in self.pivot_languages]}")

    def translate(self, text: str, target_lang: str, source_lang: str = 'en') -> str:
        """
        Translate text to target language.

        Args:
            text: Source text
            target_lang: Target language code
            source_lang: Source language code

        Returns:
            Translated text
        """
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            result = translator.translate(text)
            return result
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original on error

    def back_translate(
        self,
        text: str,
        pivot_lang: Optional[str] = None
    ) -> Optional[str]:
        """
        Perform back-translation on a single text.

        Args:
            text: Source text in English
            pivot_lang: Optional specific pivot language to use

        Returns:
            Back-translated text or None if failed
        """
        if pivot_lang:
            lang_name, lang_code = next(
                ((n, c) for n, c in self.pivot_languages if c == pivot_lang),
                (None, None)
            )
            if lang_code is None:
                raise ValueError(f"Unknown pivot language: {pivot_lang}")
        else:
            lang_name, lang_code = random.choice(self.pivot_languages)

        try:
            # English → Pivot
            pivot_text = self.translate(text, lang_code, source_lang='en')

            # Small delay to avoid rate limiting
            time.sleep(0.1)

            # Pivot → English
            back_text = self.translate(pivot_text, 'en', source_lang=lang_code)

            return back_text

        except Exception as e:
            print(f"Back-translation error for '{text[:50]}...': {e}")
            return None

    def augment_text(
        self,
        text: str,
        num_paraphrases: int = 2,
        require_diversity: bool = True
    ) -> List[str]:
        """
        Generate multiple paraphrases of a text using different pivot languages.

        Args:
            text: Source text
            num_paraphrases: Number of paraphrases to generate
            require_diversity: If True, filter out paraphrases too similar to original

        Returns:
            List of unique paraphrases
        """
        paraphrases = []
        seen_texts: Set[str] = {text.lower()}

        # Use different pivot languages for each paraphrase
        available_langs = list(self.pivot_languages)
        random.shuffle(available_langs)

        for i in range(min(num_paraphrases, len(available_langs))):
            lang_name, lang_code = available_langs[i % len(available_langs)]

            result = self.back_translate(text, lang_code)

            if result and result.lower() not in seen_texts:
                # Check diversity if required
                if require_diversity:
                    similarity = self._compute_similarity(text, result)
                    if similarity > 0.8:  # Too similar
                        continue

                paraphrases.append(result)
                seen_texts.add(result.lower())

        return paraphrases

    def augment_dataset(
        self,
        data: List[Tuple[str, str]],
        targets: List[Dict],
        min_samples: int = 100,
        max_paraphrases_per_sample: int = 2
    ) -> List[Tuple[str, str]]:
        """
        Augment dataset using back-translation for middle-tail classes.

        Args:
            data: Original training data
            targets: List of augmentation targets with 'label' and 'target_count'
            min_samples: Minimum threshold for augmentation (20-100 range)
            max_paraphrases_per_sample: Max paraphrases to generate per original

        Returns:
            Augmented data (original + generated)
        """
        # Build target map
        target_map = {t["label"]: t["target_count"] for t in targets}

        # Group data by class
        by_class: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for label, text in data:
            by_class[label].append((label, text))

        result = []
        total_generated = 0

        for label, samples in by_class.items():
            current_count = len(samples)
            target_count = target_map.get(label)

            # Skip if no target or already enough samples
            if target_count is None or current_count >= target_count:
                result.extend(samples)
                continue

            # Skip if class is too rare (should use synthetic generation instead)
            if current_count < 20:
                result.extend(samples)
                continue

            # Calculate how many samples we need
            needed = target_count - current_count

            # Add original samples
            result.extend(samples)

            # Generate paraphrases
            paraphrases = []
            for label, text in samples:
                if len(paraphrases) >= needed:
                    break

                # Generate 1-2 paraphrases per sample
                n_paraphrases = min(
                    max_paraphrases_per_sample,
                    needed - len(paraphrases)
                )

                new_paraphrases = self.augment_text(
                    text,
                    num_paraphrases=n_paraphrases,
                    require_diversity=True
                )

                for p in new_paraphrases:
                    if len(paraphrases) < needed:
                        paraphrases.append((label, p))

            # Add generated samples
            result.extend(paraphrases[:needed])
            total_generated += len(paraphrases[:needed])

            print(f"  {label}: {current_count} → {current_count + len(paraphrases[:needed])} "
                  f"(generated {len(paraphrases[:needed])})")

        print(f"\nBack-translation complete: generated {total_generated} samples")
        return result

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple word overlap similarity.
        Returns 1.0 for identical, 0.0 for completely different.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


def augment_with_simple_rules(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Fallback: Simple rule-based augmentation when deep-translator is unavailable.

    Uses basic transformations:
    - Synonym replacement
    - Word order changes (for some patterns)
    - Adding/removing filler words
    """
    SYNONYMS = {
        "what": ["what", "which", "which thing"],
        "who": ["who", "which person", "what person"],
        "where": ["where", "in what place", "at what location"],
        "when": ["when", "at what time", "on what date"],
        "how": ["how", "in what way", "by what method"],
        "is": ["is", "represents", "means", "signifies"],
        "are": ["are", "represent", "constitute"],
    }

    def transform(text: str) -> List[str]:
        """Generate simple transformations of a question."""
        variations = [text]
        words = text.lower().split()

        # Simple synonym replacements
        for i, word in enumerate(words):
            if word in SYNONYMS:
                for synonym in SYNONYMS[word][1:]:  # Skip the word itself
                    new_words = words.copy()
                    new_words[i] = synonym
                    variations.append(" ".join(new_words))

        return list(set(variations))

    result = []
    for label, text in data:
        variations = transform(text)
        for v in variations[:2]:  # Limit to 2 variations
            result.append((label, v))

    return result


if __name__ == "__main__":
    # Test the augmenter
    print("Testing back-translation augmenter...")

    if DEEP_TRANSLATOR_AVAILABLE:
        augmenter = BackTranslationAugmenter(pivot_languages=['german', 'french'])

        test_text = "What is the capital of France?"
        print(f"\nOriginal: {test_text}")

        paraphrases = augmenter.augment_text(test_text, num_paraphrases=2)
        for i, p in enumerate(paraphrases, 1):
            print(f"Paraphrase {i}: {p}")
    else:
        print("deep-translator not available, testing simple rules...")
        test_data = [("DESC:def", "What is the capital of France?")]
        augmented = augment_with_simple_rules(test_data)
        for label, text in augmented:
            print(f"{label}: {text}")
