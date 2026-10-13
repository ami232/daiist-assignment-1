"""
Shared building blocks used by BOTH train.py and app.py.

Keeping them in one place guarantees the app transforms data exactly the way
training did (same features, same split, same cost definition).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

DATA_PATH = Path("data/loans_sample.csv")
ARTIFACTS = Path("artifacts")

# ---------------------------------------------------------------------------
# Business framing: a lender decides, at application time, whether to fund a
# loan. Both kinds of mistake cost money, but not the same amount (assumptions,
# expressed as a fraction of the amount lent):
# ---------------------------------------------------------------------------
LOSS_IF_DEFAULT = 0.60   # approve a loan that defaults -> lose ~60% of principal
PROFIT_IF_REPAID = 0.15  # reject a loan that would be repaid -> miss ~15% interest

# ---------------------------------------------------------------------------
# Time-based split: train on the past, test on the "future".
# ---------------------------------------------------------------------------
TEST_YEAR = 2015                  # test set: loans issued in 2015
VALIDATION_START = "2014-07-01"   # tuning slice: last 6 months of training data

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
EMP_YEARS = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}  # "NI" (not informed) is deliberately missing -> NaN + emp_missing flag

NUMERIC = [
    "fico", "dti", "log_income", "log_loan_amnt", "loan_to_income",
    "emp_years", "emp_missing",
]
CATEGORICAL = ["purpose", "home_ownership_n", "addr_state"]
# Dropped on purpose: id, experience_c (constant), zip_code (800+ levels),
# title (free text, duplicates purpose), desc (present in 61% of 2012 loans
# but 0% of 2015 loans, so it would encode *time*, not risk).


def load_data(path=DATA_PATH):
    return pd.read_csv(path, parse_dates=["issue_d"])


def engineer(df):
    """Stateless feature engineering: row by row, nothing learned from data."""
    X = pd.DataFrame(index=df.index)
    X["fico"] = df["fico_n"]
    X["dti"] = df["dti_n"].clip(0, 60)
    X["log_income"] = np.log(df["revenue"].clip(lower=1_000))  # heavy right tail
    X["log_loan_amnt"] = np.log(df["loan_amnt"])
    X["loan_to_income"] = (df["loan_amnt"] / df["revenue"]).clip(upper=1)
    X["emp_years"] = df["emp_length"].map(EMP_YEARS)
    X["emp_missing"] = X["emp_years"].isna().astype(float)
    for col in CATEGORICAL:
        X[col] = df[col]
    return X


def time_split(df):
    is_test = df["issue_d"].dt.year == TEST_YEAR
    return df[~is_test], df[is_test]


def make_preprocessor():
    """Learned preprocessing: medians, means/stds and categories are fitted on
    the training data only, then reused unchanged on validation/test."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(
        min_frequency=50,                     # rare levels share one column
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


class LogisticRegressionTorch(nn.Module):
    """Logistic regression = one linear layer; sigmoid is applied in the loss."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits


def business_cost(y, p_default, loan_amnt, threshold):
    """Total cost (USD) if we reject every loan with p_default >= threshold."""
    y = np.asarray(y)
    loan_amnt = np.asarray(loan_amnt, dtype=float)
    reject = np.asarray(p_default) >= threshold
    approved_defaults = ~reject & (y == 1)   # false negatives
    rejected_good = reject & (y == 0)        # false positives
    return float(
        loan_amnt[approved_defaults].sum() * LOSS_IF_DEFAULT
        + loan_amnt[rejected_good].sum() * PROFIT_IF_REPAID
    )
