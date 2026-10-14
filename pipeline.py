"""
Data loading, cleaning, feature engineering, splitting and preprocessing for
the hourly bike-demand model.

Everything that touches the *raw* dataset lives here so that train.py and
app.py apply exactly the same transformations. The app never retrains; it
only re-applies the fitted preprocessor saved by train.py.

Business framing (see REPORT.md): a bike-share operator wants a day-ahead
hourly demand forecast to schedule rebalancing crews. That framing drives
three choices made in this file:

1. Target = log1p(cnt). Demand is a skewed, non-negative count whose drivers
   act multiplicatively (a rainy rush hour loses a *share* of rides, not a
   fixed number), so a linear model fits far better on the log scale.
2. Time-based split. A random split would leak the future (and the 2012
   growth trend) into training, giving optimistic scores that a deployed
   model would never achieve.
3. Lag features must only use data that is available the evening before the
   forecasted day, i.e. nothing newer than two days before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hour.csv"

# ---------------------------------------------------------------------------
# Split boundaries (inclusive). Chosen so that the training window covers
# every calendar month at least once and both years for Jan-Jun, validation
# is the quarter right before the test quarter, and test is the most recent
# quarter the operator would actually have had to forecast.
# ---------------------------------------------------------------------------
SPLITS = {
    "train": ("2011-01-01", "2012-06-30"),
    "val": ("2012-07-01", "2012-09-30"),
    "test": ("2012-10-01", "2012-12-31"),
}

# How far back the lag window looks. The forecast for day d is produced on the
# evening of day d-1, when day d-1 is still incomplete, so the newest complete
# day we may use is d-2. LAG_SHIFT_DAYS=2, LAG_WINDOW_DAYS=7 -> days d-2 .. d-8.
LAG_SHIFT_DAYS = 2
LAG_WINDOW_DAYS = 7
LAG_MIN_PERIODS = 3

TARGET = "cnt"
LOG_TARGET = "log_cnt"


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the committed UCI hour.csv and parse the date column."""
    df = pd.read_csv(path, parse_dates=["dteday"])
    return df.sort_values(["dteday", "hr"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality problems found during exploration.

    - 2011-03-10 has humidity == 0 for every hour (sensor defect; 0 % RH is
      physically implausible in Washington DC). Replaced with NaN here and
      imputed later from the *training* month median, so the fix never peeks
      at validation/test data.
    - `atemp` is dropped: it is 0.988-correlated with `temp`, and it has a
      known defect (constant value for the whole of 2012-08-17). `temp` is
      kept, rescaled to degrees Celsius so coefficients are interpretable.
    - `weathersit == 4` (heavy rain/ice) appears only 3 times in 17k rows;
      it is merged into category 3 (light rain/snow) so the one-hot column
      is not fit on three observations.
    - 12.5 % of rows have windspeed == 0, which is far more than a real
      anemometer would report. We keep the value but add a flag so the
      model can treat "0" as its own regime (calm *or* missing).
    """
    out = df.copy()
    out.loc[out["hum"] == 0, "hum"] = np.nan
    out["temp_c"] = out["temp"] * 41.0  # UCI normalised temp by dividing by 41
    out["weathersit"] = out["weathersit"].clip(upper=3)
    out["windspeed_zero"] = (out["windspeed"] == 0).astype(int)
    out = out.drop(columns=["atemp", "instant", "temp"])
    return out


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def add_lag_level(df: pd.DataFrame) -> pd.DataFrame:
    """Recent same-hour demand level, computed on a complete day x hour grid.

    For every (date d, hour h) this is the mean of log1p(cnt) at hour h over
    days d-2 .. d-8 (7 days), requiring at least 3 observed values. Missing
    hours in the raw file (system outages, dead-of-night hours with no
    rentals) are simply absent from the window rather than counted as zero.

    The 2-day shift is what makes the feature legal for a day-ahead
    forecast: it only uses days that were fully known when the forecast was
    made. Shifting by 1 instead would use "today", which is not complete yet.
    """
    grid = (
        df.pivot_table(index="dteday", columns="hr", values=LOG_TARGET, aggfunc="first")
        .reindex(pd.date_range(df["dteday"].min(), df["dteday"].max(), freq="D"))
        .reindex(columns=range(24))
    )
    lagged = grid.shift(LAG_SHIFT_DAYS).rolling(LAG_WINDOW_DAYS, min_periods=LAG_MIN_PERIODS).mean()
    long = lagged.stack(future_stack=True).rename("lag_level").reset_index()
    long.columns = ["dteday", "hr", "lag_level"]
    long["hr"] = long["hr"].astype(int)
    return df.merge(long, on=["dteday", "hr"], how="left")


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered columns (no history needed). Shared by the
    training table and by the dashboard's what-if form so a single row is
    transformed exactly like the training data."""
    out = df.copy()

    # Hour-of-day profile is completely different on working days (two
    # commute peaks) vs. non-working days (one afternoon hump). A single set
    # of hour dummies would average the two shapes, so we build one dummy per
    # (hour, workingday) combination instead: 48 categories.
    out["hr_x_workingday"] = out["hr"].astype(int).astype(str) + "_" + out["workingday"].astype(int).astype(str)

    # Temperature has a concave effect (rides stop increasing above ~30 C), so
    # we add a squared term. Humidity hurts more when it is hot, hence the
    # interaction; both are built on the Celsius scale before standardising.
    out["temp_c_sq"] = out["temp_c"] ** 2
    out["temp_x_hum"] = out["temp_c"] * out["hum"]

    # Bad weather hurts leisure riders (non-working days) more than commuters.
    out["badweather_x_nonworking"] = ((out["weathersit"] == 3) & (out["workingday"] == 0)).astype(int)
    out["windspeed_zero"] = (out["windspeed"] == 0).astype(int)

    # Cyclical encodings of hour; used only in the ablation study to show why
    # the interaction dummies above were chosen instead.
    for k in (1, 2, 3):
        out[f"hr_sin{k}"] = np.sin(2 * np.pi * k * out["hr"] / 24)
        out[f"hr_cos{k}"] = np.cos(2 * np.pi * k * out["hr"] / 24)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the modelling table from the cleaned frame.

    Returns one row per observed hour with: the target on both scales, the
    split label, and every raw (not yet encoded) feature column. Encoding and
    scaling happen in `make_preprocessor` so they can be fit on train only.
    """
    out = derive_columns(df)
    out[LOG_TARGET] = np.log1p(out[TARGET])

    # Linear trend in days; ablation-only (it extrapolates the 2011->2012
    # growth indefinitely, which the data does not support).
    out["days_since_start"] = (out["dteday"] - out["dteday"].min()).dt.days

    out = add_lag_level(out)
    out["split"] = assign_split(out["dteday"])
    return out


def make_whatif_row(hr: int, workingday: int, holiday: int, weekday: int, mnth: int, yr: int,
                    weathersit: int, temp_c: float, hum: float, windspeed: float,
                    lag_level_counts: float) -> pd.DataFrame:
    """One raw-feature row for the dashboard, from human-readable inputs.

    `lag_level_counts` is the recent same-hour demand in rides; it is turned
    into the log scale the model was trained on.
    """
    row = pd.DataFrame(
        [{
            "hr": int(hr), "workingday": int(workingday), "holiday": int(holiday),
            "weekday": int(weekday), "mnth": int(mnth), "yr": int(yr),
            "weathersit": int(min(weathersit, 3)), "temp_c": float(temp_c), "hum": float(hum),
            "windspeed": float(windspeed), "lag_level": float(np.log1p(max(lag_level_counts, 0))),
        }]
    )
    return derive_columns(row)


def assign_split(dates: pd.Series) -> pd.Series:
    split = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name, (start, end) in SPLITS.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        split[mask] = name
    return split


# ---------------------------------------------------------------------------
# 3. Feature sets (used by the ablation in train.py; FINAL is what ships)
# ---------------------------------------------------------------------------
# Each feature set is (numeric columns, categorical columns).
FEATURE_SETS: dict[str, tuple[list[str], list[str]]] = {
    # Everything as a raw number, the way a hasty first model would do it.
    "A_raw_numeric": (
        ["hr", "weekday", "mnth", "weathersit", "holiday", "workingday", "yr", "temp_c", "hum", "windspeed"],
        [],
    ),
    # Proper one-hot calendar encodings (hour, weekday, month, weather).
    "B_onehot_calendar": (
        ["holiday", "workingday", "yr", "temp_c", "hum", "windspeed"],
        ["hr", "weekday", "mnth", "weathersit"],
    ),
    # Hour x workingday interaction instead of plain hour.
    "C_hour_x_workingday": (
        ["holiday", "workingday", "yr", "temp_c", "hum", "windspeed"],
        ["hr_x_workingday", "weekday", "mnth", "weathersit"],
    ),
    # + non-linear weather terms and data-quality flags.
    "D_weather_terms": (
        ["holiday", "workingday", "yr", "temp_c", "temp_c_sq", "hum", "temp_x_hum", "windspeed",
         "windspeed_zero", "badweather_x_nonworking"],
        ["hr_x_workingday", "weekday", "mnth", "weathersit"],
    ),
    # + recent same-hour demand level (legal lag feature). This is FINAL.
    "E_plus_lag_level": (
        ["holiday", "workingday", "yr", "temp_c", "temp_c_sq", "hum", "temp_x_hum", "windspeed",
         "windspeed_zero", "badweather_x_nonworking", "lag_level"],
        ["hr_x_workingday", "weekday", "mnth", "weathersit"],
    ),
    # Alternative to E: cyclical hour encoding (3 harmonics) instead of the
    # 48 interaction dummies. Kept to justify the dummy choice.
    "E_alt_cyclical_hour": (
        ["holiday", "workingday", "yr", "temp_c", "temp_c_sq", "hum", "temp_x_hum", "windspeed",
         "windspeed_zero", "badweather_x_nonworking", "lag_level",
         "hr_sin1", "hr_cos1", "hr_sin2", "hr_cos2", "hr_sin3", "hr_cos3"],
        ["weekday", "mnth", "weathersit"],
    ),
    # Alternative to E: add a linear time trend. Kept to show it over-extrapolates.
    "E_alt_plus_trend": (
        ["holiday", "workingday", "yr", "temp_c", "temp_c_sq", "hum", "temp_x_hum", "windspeed",
         "windspeed_zero", "badweather_x_nonworking", "lag_level", "days_since_start"],
        ["hr_x_workingday", "weekday", "mnth", "weathersit"],
    ),
}
FINAL_FEATURE_SET = "E_plus_lag_level"


def feature_columns(name: str = FINAL_FEATURE_SET) -> list[str]:
    num, cat = FEATURE_SETS[name]
    return num + cat


# ---------------------------------------------------------------------------
# 4. Preprocessor: impute -> one-hot -> standardise. Fit on TRAIN ONLY.
# ---------------------------------------------------------------------------
class MonthMedianImputer:
    """Fill NaN humidity with the training median for the same calendar month.

    Implemented as a tiny sklearn-compatible transformer so it can live inside
    the saved Pipeline and be re-applied identically in the app.
    """

    def __init__(self, column: str = "hum", month_column: str = "mnth"):
        self.column = column
        self.month_column = month_column

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = X.groupby(self.month_column)[self.column].median().to_dict()
        self.global_median_ = float(X[self.column].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        fill = X[self.month_column].map(self.medians_).fillna(self.global_median_)
        X[self.column] = X[self.column].fillna(fill)
        # Interaction built from humidity has to be recomputed after imputing.
        if "temp_x_hum" in X.columns and "temp_c" in X.columns:
            X["temp_x_hum"] = X["temp_c"] * X[self.column]
        return X

    def get_params(self, deep=True):
        return {"column": self.column, "month_column": self.month_column}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def make_preprocessor(feature_set: str = FINAL_FEATURE_SET) -> Pipeline:
    """Build the (unfitted) preprocessing pipeline for a feature set.

    One-hot columns use drop='first' so the design matrix has no exact
    collinearity with the intercept, and *every* column (dummies included) is
    standardised afterwards. Standardising matters for two reasons: ridge
    penalises all coefficients equally, which only makes sense if the columns
    share a scale, and gradient descent in the two PyTorch versions converges
    orders of magnitude faster on a well-conditioned matrix.
    """
    num_cols, cat_cols = FEATURE_SETS[feature_set]
    encode = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )
    # The imputer needs the month column, which may not be a model feature,
    # so it runs on the full raw frame before the ColumnTransformer selects
    # columns.
    return Pipeline(
        [
            ("impute_hum", MonthMedianImputer()),
            ("encode", encode),
            ("scale", StandardScaler()),
        ]
    )


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    return list(preprocessor.named_steps["encode"].get_feature_names_out())


# ---------------------------------------------------------------------------
# 5. Convenience: full table + split frames
# ---------------------------------------------------------------------------
def load_model_table(path: Path = DATA_PATH) -> pd.DataFrame:
    """Raw CSV -> cleaned -> engineered table with a `split` column."""
    return build_features(clean(load_raw(path)))


def split_frames(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train/val/test frames. Rows without a lag level (the first days of
    January 2011, before the 7-day window has enough history) are dropped;
    they are all inside the training window so no evaluation row is lost."""
    usable = table.dropna(subset=["lag_level"])
    return {name: usable[usable["split"] == name].reset_index(drop=True) for name in SPLITS}


if __name__ == "__main__":
    t = load_model_table()
    print(t.shape)
    print(t["split"].value_counts(dropna=False))
    print("rows without lag_level:", t["lag_level"].isna().sum(), "->",
          sorted(t.loc[t["lag_level"].isna(), "dteday"].dt.date.unique()))
    frames = split_frames(t)
    print({k: len(v) for k, v in frames.items()})
    pre = make_preprocessor().fit(frames["train"])
    X = pre.transform(frames["train"])
    print("design matrix:", X.shape, "| columns:", len(get_feature_names(pre)))
    print(get_feature_names(pre)[:15], "...")
    print("hum NaN after impute (train):", np.isnan(X).sum())
