"""
Shared data code: used by train.py (to fit) and app.py (to rebuild the exact
same features for the saved models). Nothing in here trains anything.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hour.csv"
ARTIFACTS = ROOT / "artifacts"

START_DATE = pd.Timestamp("2011-01-01")
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15          # by calendar days, in time order
CONTINUOUS = ["temp", "temp_sq", "hum", "windspeed", "trend_years"]

# Feature sets, from the raw columns up to the final model. train.py fits every
# one of them with scikit-learn (the "ablation") and the last one three ways.
FEATURE_SETS = {
    "1. raw columns (hour as a number)": dict(hour="numeric"),
    "2. + one-hot hour": dict(hour="onehot"),
    "3. + hour x working-day": dict(hour="by_workingday"),
    "4. + temperature curve (temp^2)": dict(hour="by_workingday", temp_sq=True),
    "5. + growth trend (final)": dict(hour="by_workingday", temp_sq=True, trend=True),
}
FINAL_SET = "5. + growth trend (final)"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    # casual + registered == cnt: they ARE the target, so they can never be features.
    return df.drop(columns=["instant", "casual", "registered"])


def time_split(df: pd.DataFrame) -> pd.Series:
    """Label every row train / val / test by date: the past trains, the future tests."""
    days = np.sort(df["dteday"].unique())
    val_start = days[int(len(days) * TRAIN_FRAC)]
    test_start = days[int(len(days) * (TRAIN_FRAC + VAL_FRAC))]
    split = np.where(df["dteday"] < val_start, "train",
                     np.where(df["dteday"] < test_start, "val", "test"))
    return pd.Series(split, index=df.index, name="split")


def build_features(df: pd.DataFrame, hour="by_workingday", temp_sq=False, trend=False) -> pd.DataFrame:
    """Turn raw rows into the (unscaled) design matrix. Uses nothing but the row itself."""
    X = pd.DataFrame(index=df.index)
    X["temp"] = df["temp"]
    if temp_sq:
        X["temp_sq"] = df["temp"] ** 2     # lets demand flatten/drop when it gets very hot
    # 'atemp' (feels-like) is dropped: 0.99 correlated with temp, adds nothing but instability.
    X["hum"] = df["hum"]
    X["windspeed"] = df["windspeed"]

    # Weather: 1 = clear (reference), 2 = mist, 3 = light rain/snow, 4 = heavy storm.
    # Category 4 has only 3 rows in two years, so it is merged into 3.
    weather = df["weathersit"].clip(upper=3)
    X["weather_mist"] = (weather == 2).astype(float)
    X["weather_rain_snow"] = (weather == 3).astype(float)

    if hour == "numeric":
        X["hr"] = df["hr"].astype(float)
        X["workingday"] = df["workingday"].astype(float)
    elif hour == "onehot":
        for h in range(1, 24):             # hour 0 is the reference
            X[f"hr_{h:02d}"] = (df["hr"] == h).astype(float)
        X["workingday"] = df["workingday"].astype(float)
    elif hour == "by_workingday":
        # One dummy per (hour, day type): 48 cells, minus the reference (00h, non-working day).
        for wd, name in [(0, "off"), (1, "work")]:
            for h in range(24):
                if wd == 0 and h == 0:
                    continue
                X[f"{name}_{h:02d}h"] = ((df["hr"] == h) & (df["workingday"] == wd)).astype(float)
    if trend:
        X["trend_years"] = (df["dteday"] - START_DATE).dt.days / 365.0
    return X


def fit_scaler(X_train: pd.DataFrame) -> dict:
    """Mean/std of the continuous columns, computed on the TRAINING rows only."""
    cols = [c for c in X_train.columns if c in CONTINUOUS or c == "hr"]
    return {c: [float(X_train[c].mean()), float(X_train[c].std())] for c in cols}


def apply_scaler(X: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    X = X.copy()
    for c, (mean, std) in scaler.items():
        X[c] = (X[c] - mean) / std
    return X


def to_bikes(log_pred):
    """Models predict log(1 + rentals); convert back to a number of bikes."""
    return np.expm1(np.asarray(log_pred, dtype=float))
