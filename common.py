"""
Shared code for train.py and app.py: paths, data cleaning, the feature-engineering
hook, the business-cost model, and the nn.Module used by the "standard PyTorch" model.

Everything marked  >>> YOUR DECISION <<<  is an assumption that was written as a
placeholder so the pipeline runs end to end. Replace it with your own reasoning
(and write that reasoning in REPORT.md) before you submit.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco.csv"
ART = ROOT / "artifacts"
SEED = 42

# ---------------------------------------------------------------------------
# >>> YOUR DECISION <<<  Business costs (euros per customer).
# Decision this model supports: "which customers get a retention offer this month?"
#   customer_value : what we lose if a churner leaves with no offer
#   offer_cost     : cost of making the offer (discount / call), paid for every flagged customer
#   save_rate      : probability that the offer actually keeps a customer who would have churned
# These numbers are made up. Justify or replace them.
# ---------------------------------------------------------------------------
COSTS = {"customer_value": 500.0, "offer_cost": 50.0, "save_rate": 0.30}


def cost_matrix(costs=COSTS):
    """Cost of each outcome when we flag a customer as 'will churn' (= make the offer).

    TP: flagged and would churn  -> pay the offer; with prob (1 - save_rate) they leave anyway
    FP: flagged but would stay   -> offer wasted
    FN: not flagged, churns      -> lose the customer
    TN: not flagged, stays       -> nothing happens
    """
    v, c, s = costs["customer_value"], costs["offer_cost"], costs["save_rate"]
    return {"TP": c + (1 - s) * v, "FP": c, "FN": v, "TN": 0.0}


def theoretical_threshold(costs=COSTS):
    """Flag a customer when p*cost_flag_if_churn + (1-p)*cost_flag_if_stay < p*cost_noflag_if_churn.

    With the matrix above this simplifies to p > offer_cost / (save_rate * customer_value).
    Only valid if p is a calibrated probability (logistic regression roughly is).
    """
    return costs["offer_cost"] / (costs["save_rate"] * costs["customer_value"])


def confusion_counts(y_true, p, threshold):
    y_true = np.asarray(y_true).astype(bool)
    flag = np.asarray(p) >= threshold
    return {
        "TP": int((flag & y_true).sum()),
        "FP": int((flag & ~y_true).sum()),
        "FN": int((~flag & y_true).sum()),
        "TN": int((~flag & ~y_true).sum()),
    }


def total_cost(y_true, p, threshold, costs=COSTS):
    cm, n = cost_matrix(costs), confusion_counts(y_true, p, threshold)
    return float(sum(n[k] * cm[k] for k in cm))


# ---------------------------------------------------------------------------
# Data loading / cleaning / feature engineering
# ---------------------------------------------------------------------------
def load_clean():
    """Return (features DataFrame, binary target Series). One row = one customer."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["customerID"])  # identifier, carries no signal

    # >>> YOUR DECISION <<<  TotalCharges is blank for 11 customers, all with tenure == 0
    # (just signed up, never billed). Here: fill with 0 (they have been charged nothing).
    # Alternative: drop the 11 rows. Say which you chose and why.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    y = (df.pop("Churn") == "Yes").astype(int)
    return df, y


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """>>> YOUR DECISION <<<  Feature engineering hook (worth the most marks).

    Currently a pass-through: the model sees the cleaned raw columns only.
    Add columns here; train.py and app.py pick them up automatically (numeric columns
    are standardised, text columns are one-hot encoded). Every feature must be
    something you could compute at the moment the offer decision is made, and you
    should be able to say in REPORT.md why it should help.
    """
    out = df.copy()
    return out


def split_columns(df: pd.DataFrame):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if c not in num]
    return num, cat


# ---------------------------------------------------------------------------
# The "standard PyTorch" model (the manual model is just two raw tensors)
# ---------------------------------------------------------------------------
class LogisticRegressionNN(nn.Module):
    """One linear layer producing a logit; sigmoid is applied inside the loss / at prediction."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        # Start at zero like the manual loop does, so all methods share the same start point.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_meta():
    with open(ART / "meta.json") as f:
        return json.load(f)
