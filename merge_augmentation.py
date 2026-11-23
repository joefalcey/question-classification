"""
Merge partner's generated_{COARSE}_{fine}.txt augmentation files into a
single TREC-format .label file that data_reader.read() / run_bert.py can
consume with zero code changes.

Each input file (e.g. generated_ENTY_body.txt) contains one plain question
per line. The label is derived entirely from the filename: coarse label
uppercase, fine label lowercase, joined with ':' -> "ENTY:body".

Lines starting with '#' (like the placeholder header in the sample file)
and blank lines are skipped automatically.

Usage (run from your tl_project root):

    # Just the new augmented questions, as their own file
    python merge_augmentation.py --augmentation-dir bert_pipeline/augmentation_output \
        --output data/train_partner_augmented_only.label

    # Combine with your existing base training file into one file you can
    # point --train-file at directly
    python merge_augmentation.py --augmentation-dir bert_pipeline/augmentation_output \
        --base-file data/train_5500.label \
        --output data/train_5500_augmented.label

Then:
    python -m bert_pipeline.run_bert --train-file train_5500_augmented.label
"""

import argparse
import re
from pathlib import Path

FILENAME_PATTERN = re.compile(r"generated_([A-Za-z]+)_([A-Za-z]+)\.txt$", re.IGNORECASE)


def parse_augmentation_file(path: Path):
    """Return list of 'COARSE:fine question text' lines for one file."""
    match = FILENAME_PATTERN.search(path.name)
    if not match:
        print(f"  Skipping (name doesn't match generated_COARSE_fine.txt): {path.name}")
        return []

    coarse, fine = match.group(1).upper(), match.group(2).lower()
    label = f"{coarse}:{fine}"

    lines_out = []
    skipped_comments = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                skipped_comments += 1
                continue
            lines_out.append(f"{label} {line}")

    print(f"  {path.name}: label={label}, {len(lines_out)} questions"
          + (f" ({skipped_comments} comment/blank lines skipped)" if skipped_comments else ""))
    return lines_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--augmentation-dir", required=True,
                         help="Folder containing generated_COARSE_fine.txt files")
    parser.add_argument("--base-file", default=None,
                         help="Optional existing .label file to prepend (e.g. data/train_5500.label)")
    parser.add_argument("--output", required=True,
                         help="Path to write the combined .label file")
    args = parser.parse_args()

    aug_dir = Path(args.augmentation_dir)
    if not aug_dir.exists():
        raise FileNotFoundError(f"Augmentation directory not found: {aug_dir}")

    files = sorted(aug_dir.glob("generated_*.txt"))
    if not files:
        raise FileNotFoundError(f"No generated_*.txt files found in {aug_dir}")

    print(f"Found {len(files)} augmentation files in {aug_dir}\n")

    all_new_lines = []
    label_counts = {}
    for path in files:
        new_lines = parse_augmentation_file(path)
        all_new_lines.extend(new_lines)
        if new_lines:
            label = new_lines[0].split(" ", 1)[0]
            label_counts[label] = label_counts.get(label, 0) + len(new_lines)

    base_lines = []
    if args.base_file:
        base_path = Path(args.base_file)
        if not base_path.exists():
            raise FileNotFoundError(f"Base file not found: {base_path}")
        with open(base_path, "r", encoding="utf-8", errors="replace") as f:
            base_lines = [line.rstrip("\n") for line in f if line.strip()]
        print(f"\nBase file: {base_path} ({len(base_lines)} existing questions)")

    combined = base_lines + all_new_lines

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined) + "\n")

    print(f"\nWrote {len(combined)} total lines to {output_path}")
    print(f"  Base:      {len(base_lines)}")
    print(f"  Augmented: {len(all_new_lines)} across {len(label_counts)} fine labels")
    print("\nPer-label augmented counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:20s} {count}")


if __name__ == "__main__":
    main()
