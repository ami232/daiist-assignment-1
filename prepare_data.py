"""
One-off: shrink the full Kaggle CSV to a sample small enough to commit.

Run ONCE, locally, after downloading the dataset from Kaggle:

    uv run python prepare_data.py path/to/heart_2022_with_nans.csv

It writes data/heart_2022_sample.csv, which you commit. main.py never runs
this file; train.py only reads the committed sample. Keep this script in the
repo anyway so anyone can see exactly how the sample was made.
"""

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_PATH, SEED, TARGET

SAMPLE_SIZE = 20_000


def main(src: str) -> None:
    full = pd.read_csv(src)
    print(f"Full file: {full.shape[0]:,} rows x {full.shape[1]} columns")
    print("Target distribution (full):")
    print(full[TARGET].value_counts(dropna=False, normalize=True).round(4))

    # A row without a label can't be used to train or to evaluate.
    full = full.dropna(subset=[TARGET])

    # Stratified sample: keeps the real-world share of "Yes" answers, so the
    # probabilities the model learns stay on the same scale as the full data.
    sample, _ = train_test_split(
        full, train_size=SAMPLE_SIZE, stratify=full[TARGET], random_state=SEED
    )

    DATA_PATH.parent.mkdir(exist_ok=True)
    sample.to_csv(DATA_PATH, index=False)
    size_mb = DATA_PATH.stat().st_size / 1e6
    print(f"\nWrote {DATA_PATH.relative_to(DATA_PATH.parent.parent)}: "
          f"{len(sample):,} rows, {size_mb:.1f} MB")
    print("Target distribution (sample):")
    print(sample[TARGET].value_counts(normalize=True).round(4))
    print("\nMissing values per column (sample, top 10):")
    print(sample.isna().mean().sort_values(ascending=False).head(10).round(3))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python prepare_data.py path/to/heart_2022_*.csv")
    main(sys.argv[1])
