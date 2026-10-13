"""
Shared code used by BOTH train.py and app.py (so the dashboard applies exactly
the same feature engineering the models were trained with).

Business framing (details in REPORT.md): a dealership buys used cars at wholesale
auctions (some of them lease returns) and must decide how much to bid.
    target = sellingprice / mmr - 1        (+0.05 = closed 5% above mmr)
    max bid = mmr * (1 + prediction - X),  never above BID_CAP * mmr
`mmr` (a third-party market estimate) is the BASELINE (prediction 0), never a feature.
The dataset has no costs and no resale prices, so NO profit is modelled.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "car_prices_sample.csv"
ARTIFACTS = ROOT / "artifacts"

# ---- temporal split: train Dec-Mar, validate May, test June
TRAIN_END = pd.Timestamp("2015-03-31")
VAL_START, VAL_END = pd.Timestamp("2015-05-01"), pd.Timestamp("2015-05-31")
TEST_START = pd.Timestamp("2015-06-01")

# ---- bidding rule
# The cap is a BUSINESS ASSUMPTION (safety net for atypical cars/predictions), not estimated.
BID_CAP = 1.20
# X (the cushion) is fixed on VALIDATION errors, never on test: X is the CUSHION_QUANTILE
# quantile of (prediction - actual) in May, so that after subtracting X the model still
# overestimates in only (1 - CUSHION_QUANTILE) of validation sales. X >= 0 (it is a cushion).
CUSHION_QUANTILE = 0.75

NUM_FEATURES = ["age", "age_sq", "log_odometer", "odometer_per_year", "condition", "condition_sq",
                "condition_x_age", "condition_missing", "seller_is_lease"]
CAT_FEATURES = ["make", "body_group", "transmission", "state", "color", "seller"]

_BODY_KEYWORDS = [  # order matters: first match wins
    ("convertible", "convertible"), ("coupe", "coupe"), ("wagon", "wagon"),
    ("hatchback", "hatchback"), ("minivan", "minivan"), ("van", "van"),
    ("suv", "suv"), ("sedan", "sedan"),
    ("cab", "truck"), ("crew", "truck"), ("pickup", "truck"), ("koup", "coupe"),
]


def _body_group(b):
    if not isinstance(b, str):
        return "unknown"
    b = b.lower()
    for key, group in _BODY_KEYWORDS:
        if key in b:
            return group
    return "other"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["sale_date"])
    df["target"] = df["sellingprice"] / df["mmr"] - 1      # NOT clipped: RMSE must see the real tails
    return df


def split_name(dates: pd.Series) -> pd.Series:
    out = pd.Series("drop", index=dates.index)
    out[dates <= TRAIN_END] = "train"
    out[(dates >= VAL_START) & (dates <= VAL_END)] = "val"
    out[dates >= TEST_START] = "test"
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering (nothing learned here -> no leakage). Anything that
    needs training statistics (imputation, scaling, which categories are frequent) lives in
    the ColumnTransformer fitted on the training split only. `mmr` and `vin` are NOT used."""
    X = pd.DataFrame(index=df.index)
    sale_year = pd.to_datetime(df["sale_date"]).dt.year if "sale_date" in df else 2015
    X["age"] = (sale_year - df["year"]).clip(lower=0)
    X["age_sq"] = X["age"] ** 2
    odo = df["odometer"].clip(lower=0, upper=300_000)
    X["log_odometer"] = np.log1p(odo)
    X["odometer_per_year"] = (odo / (X["age"] + 1)).clip(upper=60_000)
    X["condition"] = df["condition"]                       # NaN kept -> imputed later
    X["condition_sq"] = df["condition"] ** 2
    X["condition_x_age"] = df["condition"] * X["age"]      # condition matters more on older cars
    X["condition_missing"] = df["condition"].isna().astype(float)
    seller = df["seller"].fillna("unknown").str.lower()
    X["seller_is_lease"] = seller.str.contains("lease", regex=False).astype(float)  # lease-return sellers
    X["make"] = df["make"].str.lower().fillna("unknown")
    X["body_group"] = df["body"].map(_body_group)
    X["transmission"] = df["transmission"].fillna("unknown").str.lower()
    X["state"] = df["state"].fillna("unknown").str.lower()
    X["color"] = df["color"].fillna("unknown").str.lower()
    X["seller"] = seller
    return X[NUM_FEATURES + CAT_FEATURES]


def build_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False)
    return ColumnTransformer([("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)])


# ---------------------------------------------------------------- metrics
def rmse(y, p):
    """Main metric: punishes big errors, i.e. very wrong bids."""
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def signed_error(y, p):
    """Mean(pred - actual): positive = the model OVERestimates (the costly side). It can
    sit near 0 when errors go both ways, so it complements the RMSE, never replaces it."""
    return float(np.mean(np.asarray(p) - np.asarray(y)))


# ---------------------------------------------------------------- bidding
def cushion_from_validation(pred_val, actual_val, q=CUSHION_QUANTILE):
    """X = q-quantile of the validation errors (pred - actual), floored at 0."""
    err = np.asarray(pred_val) - np.asarray(actual_val)
    return float(max(np.quantile(err, q), 0.0))


def simulate_bids(mmr, selling, pred, cushion, cap=BID_CAP):
    """Max bid = mmr*(1+pred-X), capped at cap*mmr. No profit is computed (no costs / resale
    prices in the data). We measure the two kinds of miss:
      * bid ABOVE the final price -> we were ready to pay more than the market did
        (overestimation, the costly error). excess = bid - final price.
      * bid BELOW the final price -> the car is lost (opportunity cost only)."""
    mmr, selling, pred = map(np.asarray, (mmr, selling, pred))
    uncapped = mmr * (1 + pred - cushion)
    bid = np.minimum(uncapped, cap * mmr)
    over = bid > selling
    lost = ~over
    n = len(mmr)
    return {
        "bid_above_price_share": float(over.mean()) if n else 0.0,
        "mean_excess_usd": float((bid - selling)[over].mean()) if over.any() else 0.0,
        "lost_share": float(lost.mean()) if n else 0.0,
        "mean_shortfall_usd": float((selling - bid)[lost].mean()) if lost.any() else 0.0,
        "cap_binding_share": float((uncapped > cap * mmr).mean()) if n else 0.0,
    }
