"""Feature engineering and time-based splitting for the Walmart weekly sales project.

Shared by train.py and app.py so both build features in exactly the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "walmart.csv"

TEST_WEEKS = 52  # last year = test set (contains one full holiday season)
VAL_WEEKS = 13   # the quarter before it = validation set, used only for tuning

TARGET = "log_sales"
# Continuous drivers: standardised, so each coefficient means "effect of +1 standard deviation".
# CPI is deliberately excluded: 99.9% of its variation is between stores (already captured by the
# Store columns), so it was nearly collinear with them and its coefficient was unstable.
NUMERIC = ["Temperature", "Fuel_Price", "Unemployment"]
# 0/1 flags: kept as 0/1, so each coefficient means "% difference vs. a normal week"
FLAGS = ["is_superbowl", "is_labor_day", "is_thanksgiving", "is_pre_christmas", "is_post_christmas"]
# Labels, not quantities: one-hot encoded so each store / month gets its own level
CATEGORICAL = ["Store", "Month"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")  # day-month-year
    return df.sort_values(["Date", "Store"]).reset_index(drop=True)


def add_features(df):
    df = df.copy()
    month = df["Date"].dt.month
    day = df["Date"].dt.day
    flagged = df["Holiday_Flag"] == 1

    # The original flag lumps four holidays together; split it so each gets its own coefficient.
    df["is_superbowl"] = (flagged & (month == 2)).astype(int)
    df["is_labor_day"] = (flagged & (month == 9)).astype(int)
    df["is_thanksgiving"] = (flagged & (month == 11)).astype(int)
    # The week flagged as "Christmas" ends after Dec 25, so it is really the post-Christmas dip.
    df["is_post_christmas"] = (flagged & (month == 12)).astype(int)
    # The real peak, the last shopping week before Christmas, is not flagged at all.
    df["is_pre_christmas"] = ((month == 12) & day.between(18, 24)).astype(int)

    df["Month"] = month
    df[TARGET] = np.log(df["Weekly_Sales"])
    return df


def split_by_time(df):
    """Chronological split: train on the past, validate on the next quarter, test on the last year."""
    dates = df["Date"].drop_duplicates().sort_values().reset_index(drop=True)
    test_start = dates.iloc[-TEST_WEEKS]
    val_start = dates.iloc[-(TEST_WEEKS + VAL_WEEKS)]
    train = df[df["Date"] < val_start]
    val = df[(df["Date"] >= val_start) & (df["Date"] < test_start)]
    test = df[df["Date"] >= test_start]
    return train, val, test


def build_preprocessor():
    """Unfitted transformer. It must be fitted on training rows only, then reused unchanged."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("flags", "passthrough", FLAGS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
