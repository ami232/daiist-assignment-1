"""
Shared code used by BOTH train.py and app.py.

Keeping these pieces in one place guarantees that the app transforms data and
computes the net gain exactly the way training did (no train/serve skew):

- load_raw()            : read the committed CSV and fix data-quality issues
- engineer_features()   : the feature engineering (pure function of one row,
                          so it is safe to apply before the split: nothing is
                          learned from the data here)
- net_gain()            : the money model behind the decision threshold
- LogisticRegressionNN  : the nn.Module used by the "standard PyTorch" model
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
ARTIFACTS = ROOT / "artifacts"

# The 6 internet add-on services. For customers without internet these say
# "No internet service", which carries no extra information beyond
# InternetService == "No", so we collapse it to "No" (avoids a redundant dummy
# column per service that would be perfectly collinear with InternetService_No).
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Columns fed to the ColumnTransformer (after engineer_features()).
NUMERIC_FEATURES = [
    "SeniorCitizen",     # already 0/1
    "log_tenure",        # log1p(tenure): churn risk falls fastest in the first months
    "MonthlyCharges",    # current price level
    "n_addons",          # number of add-on services (engagement / switching cost)
]
# Of the 6 add-ons only the two "protective" services (OnlineSecurity,
# TechSupport) are kept as individual dummies; the other four only enter via
# n_addons. Keeping all six dummies AND n_addons would be perfectly collinear
# (n_addons is exactly their sum), which makes the coefficients unidentifiable
# and plain gradient descent converge very slowly.
# gender is kept on purpose, for consistency with SeniorCitizen: both are
# sensitive attributes, so we treat them the same way and keep both visible in
# the model's weights (both end up near zero) rather than keeping only one.
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "TechSupport",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and fix the one known data-quality issue."""
    df = pd.read_csv(path)

    # TotalCharges is read as text because 11 rows contain a blank string " ".
    # All 11 have tenure == 0: brand-new customers who have not been billed yet.
    # They are NOT missing at random, so we do not drop or mean-impute them.
    # Choice: TotalCharges = MonthlyCharges (i.e. treat them as being in their
    # first billing month). The model no longer uses TotalCharges (see
    # engineer_features), but the dashboard's distribution plots still do.
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = total.fillna(df["MonthlyCharges"])

    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering. Learns nothing from the data, so applying
    it to the full dataset before splitting cannot leak information."""
    out = df.copy()

    # Collapse "No internet service" / "No phone service" into "No" (see above).
    for col in ADDON_COLS:
        out[col] = out[col].replace("No internet service", "No")
    out["MultipleLines"] = out["MultipleLines"].replace("No phone service", "No")

    # Tenure: churn hazard is highest in the first months and then flattens,
    # which a log transform makes closer to linear in the logit.
    out["log_tenure"] = np.log1p(out["tenure"])

    # Engagement: how many add-ons the customer pays for (0-6).
    out["n_addons"] = (out[ADDON_COLS] == "Yes").sum(axis=1)

    # Dropped on purpose:
    # - customerID: an identifier, no predictive meaning (would only memorise rows)
    # - TotalCharges: ~ MonthlyCharges * tenure, so it is nearly collinear with
    #   the two. A price-change feature built from it (charge_diff = MonthlyCharges
    #   - TotalCharges / tenure) was tested and dropped: its weight was ~0 and
    #   removing it did not change validation performance (see REPORT.md).
    # - tenure (raw): replaced by log_tenure
    # - fiber_x_monthly (fiber optic AND month-to-month) was tested and dropped:
    #   on the log-odds scale the two risks do not compound (interaction ~ -0.35),
    #   so its weight was ~0 and removing it did not change validation performance.
    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def net_gain(y_true, contact, monthly_charges, call_cost, horizon_months, save_rate):
    """Net gain (in $) of a call decision compared with doing nothing (calling nobody).

    y_true = 1 means the customer churned in the data, which was recorded with no
    retention calls: i.e. the customer would leave if we do nothing. Per customer:
      - called & would leave (TP): the offer keeps them with probability save_rate,
                                   saving MonthlyCharges * horizon_months of revenue;
                                   on average: save_rate * revenue - call_cost
      - called & would stay  (FP): their revenue was never at risk: - call_cost
      - not called           (FN, TN): same as doing nothing: 0
    """
    y_true = np.asarray(y_true).astype(bool)
    contact = np.asarray(contact).astype(bool)
    revenue = np.asarray(monthly_charges, dtype=float) * horizon_months
    gain = np.where(contact & y_true, save_rate * revenue, 0.0)
    gain -= np.where(contact, call_cost, 0.0)
    return float(gain.sum())


class LogisticRegressionNN(nn.Module):
    """Logistic regression as a PyTorch module: one linear layer.
    forward() returns logits; the sigmoid lives in the loss (BCEWithLogitsLoss)
    for numerical stability, and is applied explicitly at prediction time."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
