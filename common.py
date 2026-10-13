"""
Shared pieces used by both train.py and app.py: paths, the business-cost
definition, threshold search, and the nn.Module (app.py needs the class to
load the saved weights).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit_default.csv"
ARTIFACTS = ROOT / "artifacts"

SEED = 42
DTYPE = torch.float64  # same precision as scikit-learn, so the 3 models are comparable

# Split: stratified 60 / 20 / 20 train / validation / test.
# Validation is used for tuning (C, learning rates) and for picking the
# decision threshold. Test is touched once, at the end.
VAL_SIZE = 0.20
TEST_SIZE = 0.20

# --- The business-cost definition (drives the threshold) ---------------------
# Decision supported: each month, flag a credit-card client as high risk
# (reject / cut the limit) or leave them as they are, given their repayment
# history. Two ways to be wrong:
#   * false negative: we keep a client who then defaults  -> COST_FN
#   * false positive: we flag a client who would have paid -> COST_FP
# Only the RATIO matters. 3:1 is an assumption: a defaulted balance (net of
# recoveries) costs about three times the margin lost by wrongly flagging a good
# client. Why 3 and not the 5 used in the German Credit benchmark: the
# break-even threshold is COST_FP / (COST_FP + COST_FN) = 0.25, just above the
# 22.1% base default rate, so "approve everyone" is the cheapest constant policy
# -- the lender's status quo, a meaningful baseline. At 5:1 the break-even
# (0.167) is below the base rate and the cheapest constant policy would be
# "reject everyone", which no lender actually does.
# Sensitivity to this assumption: `uv run python experiments/cost_sensitivity.py`.
# A flat cost per client is a simplification; in practice the loss scales with
# the client's balance or limit.
COST_FN = 3.0
COST_FP = 1.0


def confusion_counts(y_true, y_pred):
    """(tn, fp, fn, tp) for 0/1 arrays."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def total_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    _, fp, fn, _ = confusion_counts(y_true, y_pred)
    return fn * cost_fn + fp * cost_fp


def cost_curve(y_true, proba, thresholds=None, cost_fn=COST_FN, cost_fp=COST_FP):
    """Total cost at each threshold (predict default when proba >= threshold)."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    costs = np.array(
        [total_cost(y_true, (proba >= t).astype(int), cost_fn, cost_fp) for t in thresholds]
    )
    return thresholds, costs


def best_threshold(y_true, proba, cost_fn=COST_FN, cost_fp=COST_FP):
    thresholds, costs = cost_curve(y_true, proba, cost_fn=cost_fn, cost_fp=cost_fp)
    return float(thresholds[int(np.argmin(costs))])


def naive_cost(y_true, cost_fn=COST_FN, cost_fp=COST_FP):
    """Cheapest *constant* policy: approve everyone or reject everyone."""
    y_true = np.asarray(y_true).astype(int)
    approve_all = int(y_true.sum()) * cost_fn
    reject_all = int((1 - y_true).sum()) * cost_fp
    return min(approve_all, reject_all)


class LogisticRegressionNN(nn.Module):
    """One linear layer -> logit. Sigmoid is applied outside (in the loss / at inference)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-z))
