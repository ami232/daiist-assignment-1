"""
Feature engineering shared by train.py and app.py.

Keeping it in one module guarantees the app builds features exactly the way
training did (same columns, same order). If the two ever drifted apart, the
saved models would silently receive the wrong inputs.
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/hour.csv"

# Columns that must never be used as inputs:
# - casual + registered == cnt exactly, so using them leaks the answer.
# - instant is just a row id.
LEAKY_OR_ID_COLUMNS = ["casual", "registered", "instant"]

WEEK_HOURS = 24 * 7

# Time-based split boundaries (see REPORT.md, "Business framing").
VAL_START = pd.Timestamp("2012-07-01")
TEST_START = pd.Timestamp("2012-10-01")


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the hourly file and add a proper timestamp column."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    return df.sort_values("timestamp").reset_index(drop=True)


def add_lag_week(df: pd.DataFrame) -> pd.DataFrame:
    """Add rentals in the same hour one week earlier.

    The file skips some hours (system outages), so the lag is looked up by
    timestamp, not by shifting 168 rows. A 168-row shift would point at the
    wrong hour after every gap.
    """
    counts = df.set_index("timestamp")["cnt"]
    df = df.copy()
    df["cnt_lag_week"] = counts.reindex(df["timestamp"] - pd.Timedelta(hours=WEEK_HOURS)).to_numpy()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw rows (with cnt_lag_week already present) into the model matrix.

    Every column below is something the operator knows when making a
    forecast up to a week ahead: the calendar, the weather forecast
    (approximated by observed weather, see Limitations), and last week's
    count for that hour.
    """
    X = pd.DataFrame(index=df.index)

    # Hour-of-day profile, separately for working and non-working days.
    # Working days have two commute peaks (8h, 17-18h). Weekends and
    # holidays have one broad afternoon hump. A linear model can only
    # express both shapes if each (hour, day type) pair gets its own
    # coefficient. Hour 0 of each day type is the baseline that gets dropped.
    work = df["workingday"].to_numpy() == 1
    for h in range(1, 24):
        X[f"hr{h:02d}_work"] = ((df["hr"] == h) & work).astype(float)
        X[f"hr{h:02d}_off"] = ((df["hr"] == h) & ~work).astype(float)
    # One shift between day types at hour 0. Without it the two profiles
    # would be forced to share the same midnight level.
    X["workingday"] = work.astype(float)
    X["holiday"] = df["holiday"].astype(float)

    # Month as categories: seasonality is not a straight line in month
    # number (December is not "more" than January). January is the baseline.
    # season is dropped because it's just a coarser version of month.
    for m in range(2, 13):
        X[f"month_{m:02d}"] = (df["mnth"] == m).astype(float)

    # Year: the system grew strongly from 2011 to 2012.
    X["year_2012"] = df["yr"].astype(float)

    # Weather situation (1 = clear is the baseline). Category 4 (heavy
    # rain/snow) appears only 3 times in the whole file, too rare to learn
    # its own coefficient, so it's merged into 3 ("bad weather").
    weather = df["weathersit"].clip(upper=3)
    X["weather_mist"] = (weather == 2).astype(float)
    X["weather_bad"] = (weather == 3).astype(float)

    # Temperature: demand rises with warmth but drops off when it gets very
    # hot, an inverted U. The squared term lets a linear model bend.
    # atemp ("feels like") is ~0.99 correlated with temp, so it's dropped
    # to avoid two near-identical features fighting over one coefficient.
    X["temp"] = df["temp"]
    X["temp_sq"] = df["temp"] ** 2
    X["humidity"] = df["hum"]
    X["windspeed"] = df["windspeed"]

    # Same hour last week, on the log scale to match the target.
    X["log_cnt_lag_week"] = np.log1p(df["cnt_lag_week"])

    return X


def make_target(df: pd.DataFrame) -> pd.Series:
    """log(1 + count): demand effects are multiplicative and the raw count is
    heavily right-skewed. Predictions are converted back with expm1."""
    return np.log1p(df["cnt"])


def prepare_dataset(path: str = DATA_PATH):
    """Load, add lag, drop rows without a lag, and label each row's split."""
    df = add_lag_week(load_raw(path))
    df = df.drop(columns=LEAKY_OR_ID_COLUMNS)
    # The first week of 2011 (and hours one week after an outage) have no
    # lag value. These are a few hundred rows, and dropping them is simpler
    # and more honest than inventing a value.
    df = df.dropna(subset=["cnt_lag_week"]).reset_index(drop=True)
    df["split"] = np.where(
        df["timestamp"] >= TEST_START, "test",
        np.where(df["timestamp"] >= VAL_START, "val", "train"),
    )
    return df
