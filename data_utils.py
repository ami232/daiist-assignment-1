"""Shared data loading, used by both train.py and app.py so they read the data the same way."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_wine() -> pd.DataFrame:
    """Load the UCI Wine Quality red + white files into one DataFrame.

    The raw CSVs use ';' as separator. We add an `is_red` column (1 = red,
    0 = white) so the two can be combined into a single table.
    """
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red["is_red"] = 1
    white["is_red"] = 0
    df = pd.concat([red, white], ignore_index=True)
    # snake_case column names are easier to use in code
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df
