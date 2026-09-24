"""
Shared code for train.ipynb and app.py.

Everything that BOTH stages need lives here, so the app transforms data and
rebuilds the models exactly the way training did:
  - load + clean the raw CSV
  - the feature engineering (a small fit/transform class, fitted on train only)
  - the torch nn.Module definition (needed to load the saved state_dict)
  - the business-cost function used for the decision threshold
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit_risk_dataset.csv"
ARTIFACTS = ROOT / "artifacts"

TARGET = "loan_status"  # 1 = the borrower defaulted, 0 = repaid
SEED = 42

# --- Business assumptions (see REPORT.md) -----------------------------------
# Approving someone who defaults: we lose LGD x loan amount.
# Rejecting someone who would have repaid: we lose the profit on that loan.
LOSS_GIVEN_DEFAULT = 0.60
PROFIT_MARGIN = 0.10

# Columns we are allowed to use at application time. loan_grade and
# loan_int_rate are deliberately NOT here: the lender sets them *after*
# assessing risk, so using them would leak the lender's own risk decision
# into a model that is supposed to support that decision.
RAW_FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_amnt", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
]
HOME_LEVELS = ["MORTGAGE", "OWN", "RENT", "OTHER"]          # MORTGAGE = reference
INTENT_LEVELS = ["EDUCATION", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT",
                 "MEDICAL", "PERSONAL", "VENTURE"]          # EDUCATION = reference
AFFORDABILITY_CUTOFF = 0.30  # loan > 30% of yearly income


def load_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and remove rows that cannot be real applications."""
    df = pd.read_csv(path)
    df = df.drop_duplicates()                 # exact duplicate rows would land in train AND test
    df = df[df["person_age"] <= 100]          # ages like 144 are data-entry errors
    return df.reset_index(drop=True)


class CreditFeatures:
    """Feature engineering with a scikit-learn-style fit/transform.

    fit() learns everything that depends on the data (imputation median,
    scaling mean/std) from the TRAINING split only, so no information from
    validation/test leaks into preprocessing.
    """

    def fit(self, df: pd.DataFrame) -> "CreditFeatures":
        emp = self._clean_emp_length(df)
        self.emp_median_ = float(emp.median())
        raw = self._build(df)
        self.feature_names_ = list(raw.columns)
        self.mean_ = raw.mean().to_numpy()
        self.std_ = raw.std(ddof=0).replace(0, 1).to_numpy()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._build(df)[self.feature_names_]
        return ((raw.to_numpy() - self.mean_) / self.std_).astype(np.float32)

    def unscaled(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineered features before standardisation (for plots)."""
        return self._build(df)[self.feature_names_]

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _clean_emp_length(df: pd.DataFrame) -> pd.Series:
        # Employment longer than (age - 14) years is impossible -> treat as missing.
        emp = df["person_emp_length"]
        return emp.where(emp <= df["person_age"] - 14)

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        emp = self._clean_emp_length(df)
        f["emp_length_missing"] = emp.isna().astype(float)   # missingness itself predicts default
        f["emp_length"] = emp.fillna(getattr(self, "emp_median_", emp.median()))
        f["age"] = df["person_age"].astype(float)
        f["log_income"] = np.log(df["person_income"])       # income is heavily right-skewed
        f["log_loan_amnt"] = np.log(df["loan_amnt"])
        # The provided loan_percent_income is rounded to 2 decimals; recompute it exactly.
        pct = df["loan_amnt"] / df["person_income"]
        f["loan_to_income"] = pct
        f["loan_to_income_sq"] = pct ** 2                    # risk rises faster than linearly
        f["over_affordability_cutoff"] = (pct > AFFORDABILITY_CUTOFF).astype(float)
        f["cred_hist_length"] = df["cb_person_cred_hist_length"].astype(float)
        f["prior_default"] = (df["cb_person_default_on_file"] == "Y").astype(float)
        for level in HOME_LEVELS[1:]:
            f[f"home_{level}"] = (df["person_home_ownership"] == level).astype(float)
        for level in INTENT_LEVELS[1:]:
            f[f"intent_{level}"] = (df["loan_intent"] == level).astype(float)
        # Renters have no housing asset as a buffer: a big loan hurts them more.
        f["rent_x_loan_to_income"] = f["home_RENT"] * pct
        return f


class LogisticRegressionModule(nn.Module):
    """Logistic regression as an nn.Module: one linear layer, sigmoid applied in the loss."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # logits


def business_cost(y_true, p_default, loan_amnt, threshold,
                  lgd=LOSS_GIVEN_DEFAULT, margin=PROFIT_MARGIN) -> dict:
    """Reject when p_default >= threshold. Returns the confusion matrix and money lost."""
    y_true = np.asarray(y_true)
    reject = np.asarray(p_default) >= threshold
    amnt = np.asarray(loan_amnt, dtype=float)
    fn = (~reject) & (y_true == 1)   # approved a defaulter
    fp = reject & (y_true == 0)      # rejected a good customer
    tp = reject & (y_true == 1)
    tn = (~reject) & (y_true == 0)
    return {
        "tn": int(tn.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "tp": int(tp.sum()),
        "default_losses": float(lgd * amnt[fn].sum()),
        "lost_profit": float(margin * amnt[fp].sum()),
        "total_cost": float(lgd * amnt[fn].sum() + margin * amnt[fp].sum()),
    }
