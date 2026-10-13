"""
Data loading and feature engineering shared by train.ipynb and app.ipynb.

Keeping this in one place guarantees the dashboard builds features for its
"what-if" predictions exactly the way the models were trained on them.
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/SeoulBikeData.csv"

# Original column names contain "°" and units; rename to short identifiers.
COLUMNS = [
    "date", "count", "hour", "temp", "humidity", "wind", "visibility",
    "dew_point", "solar", "rain", "snow", "season", "holiday", "functioning",
]
SEASONS = ["Autumn", "Spring", "Summer", "Winter"]  # Autumn = reference level
WEATHER = ["temp", "solar", "rain", "snow"]
HOUR_TERMS = ["hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2"]


def load_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, drop non-functioning days, and add a weekend flag."""
    raw = pd.read_csv(path, encoding="latin1")
    raw.columns = COLUMNS
    raw["date"] = pd.to_datetime(raw["date"], dayfirst=True)

    # Bikes could not be rented on non-functioning days, so those rows say
    # nothing about demand. Afterwards the column is constant, so drop it.
    df = raw[raw["functioning"] == "Yes"].drop(columns="functioning")
    df = df.reset_index(drop=True)
    df["weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn cleaned rows into the model's feature matrix.

    Expects columns: hour, weekend, holiday, season, and the WEATHER columns.
    """
    X = pd.DataFrame(index=df.index)

    # Cyclical hour: 23:00 lands next to 00:00. The first pair allows one
    # peak per day; the second pair (twice the frequency) allows two.
    angle = 2 * np.pi * df["hour"] / 24
    X["hour_sin1"] = np.sin(angle)
    X["hour_cos1"] = np.cos(angle)
    X["hour_sin2"] = np.sin(2 * angle)
    X["hour_cos2"] = np.cos(2 * angle)

    # Weekend flag plus weekend x hour interactions, so weekends can have a
    # different daily shape, not just a different level.
    X["weekend"] = df["weekend"].astype(float)
    for term in HOUR_TERMS:
        X[f"weekend_x_{term}"] = X["weekend"] * X[term]

    X["holiday"] = (df["holiday"] == "Holiday").astype(float)

    for col in WEATHER:
        X[col] = df[col].astype(float)

    # One-hot seasons, dropping Autumn: with an intercept, all four dummies
    # would be perfectly collinear.
    for season in SEASONS[1:]:
        X[f"season_{season.lower()}"] = (df["season"] == season).astype(float)

    return X
