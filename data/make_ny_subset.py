"""
Builds data/brfss_2022_new_york.csv from the full Kaggle download.

Source: CDC BRFSS 2022, as cleaned by Kamil Pytlak on Kaggle
("Indicators of Heart Disease (2022 UPDATE)", file heart_2022_with_nans.csv,
445,132 rows x 40 columns, ~140 MB — too large to commit).

We keep only respondents from New York State (the business framing is a
New York State health department), and keep the missing values (NaN)
untouched so preprocessing decisions stay inside train.py.

Run once, manually:
    uv run python data/make_ny_subset.py path/to/heart_2022_with_nans.csv
"""

import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
out = Path(__file__).resolve().parent / "brfss_2022_new_york.csv"

df = pd.read_csv(src)
ny = df[df["State"] == "New York"].drop(columns="State")
ny.to_csv(out, index=False)
print(f"{len(ny):,} rows x {ny.shape[1]} columns -> {out}")
