"""
Shared code for train.py and app.py.

Everything that turns raw rows into model inputs lives here, so the app
transforms data exactly the way training did (no train/serve skew).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hour.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Temporal split (see REPORT.md, business framing).
# Train: 2011-01-01 .. 2012-06-30 | Val: Jul-Sep 2012 | Test: Oct-Dec 2012
# 2012 must appear in training so the model can learn the system's growth
# through `yr`; a random split would leak neighbouring hours into the test set.
# ---------------------------------------------------------------------------
VAL_START = pd.Timestamp("2012-07-01")
TEST_START = pd.Timestamp("2012-10-01")

TARGET = "cnt"
# casual + registered == cnt in every row: using them would leak the answer.
LEAKY_COLUMNS = ["casual", "registered"]

# The 48 hour x working-day dummies always sum to 1, which makes them perfectly
# collinear with the intercept (dummy-variable trap). Dropping one cell makes
# the solution unique, so the three implementations can be compared
# coefficient by coefficient. This cell becomes the baseline the others are
# measured against.
REFERENCE_CELL = "h00_off"


def load_data() -> pd.DataFrame:
    """Read the hourly CSV, add a timestamp, drop leaky/useless columns."""
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.drop(columns=LEAKY_COLUMNS + ["instant"])
    return df.sort_values("timestamp").reset_index(drop=True)


def split_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for the three time-ordered partitions."""
    ts = df["timestamp"]
    return {
        "train": ts < VAL_START,
        "val": (ts >= VAL_START) & (ts < TEST_START),
        "test": ts >= TEST_START,
    }


def fit_preprocessing(fit_df: pd.DataFrame) -> dict:
    """Learn preprocessing statistics from the fitting rows ONLY (no leakage)."""
    # hum == 0 only happens on 2011-03-10: physically impossible, a sensor fault.
    valid_hum = fit_df.loc[fit_df["hum"] > 0, "hum"]
    return {"hum_fill": float(valid_hum.median())}


def build_features(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """Turn raw rows into the model's feature matrix."""
    X = pd.DataFrame(index=df.index)

    X["yr"] = df["yr"]              # lets the model learn year-over-year growth
    X["holiday"] = df["holiday"]

    # Temperature: demand rises with heat but flattens above ~35 C.
    # temp^2 lets a linear model bend that curve. atemp is dropped: it is
    # 0.99 correlated with temp and only destabilises the coefficients.
    X["temp"] = df["temp"]
    X["temp_sq"] = df["temp"] ** 2

    X["hum"] = df["hum"].where(df["hum"] > 0, prep["hum_fill"])
    X["windspeed"] = df["windspeed"]

    # Weather severity (1 clear .. 3 rain/snow) kept as an ordered scale.
    # Level 4 occurs only 3 times: too rare to learn, so merged into 3.
    X["weathersit"] = df["weathersit"].clip(upper=3)

    # Cyclical month: December (12) and January (1) end up next to each other.
    angle = 2 * np.pi * df["mnth"] / 12
    X["month_sin"] = np.sin(angle)
    X["month_cos"] = np.cos(angle)

    # Hour of day x working day: one indicator per (hour, day type) cell.
    # Working days have two commute peaks (8h, 17-18h); other days one
    # midday bump. A single numeric `hr` column could only draw a line.
    for hour in range(24):
        for flag, label in [(1, "work"), (0, "off")]:
            name = f"h{hour:02d}_{label}"
            if name == REFERENCE_CELL:
                continue
            X[name] = ((df["hr"] == hour) & (df["workingday"] == flag)).astype(float)

    return X.astype(float)


# ---------------------------------------------------------------------------
# Target transform. Effects here are multiplicative (rain removes a % of
# riders, not a fixed number), and logs turn products into sums, which is
# what a linear model can represent. expm1 also guarantees non-negative output.
# ---------------------------------------------------------------------------
def to_model_space(y_bikes):
    return np.log1p(y_bikes)


def to_bikes(y_model):
    return np.expm1(y_model)


class LinearRegressionTorch(nn.Module):
    """Linear regression as a standard PyTorch module: y = Xw + b."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
