"""
The logistic regression as PyTorch sees it, plus predict helpers.

Shared by train.ipynb (fitting) and app.ipynb (loading saved weights), so
both stages compute probabilities with exactly the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DTYPE = torch.float64  # float64 so the three implementations can be compared to ~1e-6


class LogisticNet(nn.Module):
    """One linear layer; the sigmoid lives in the loss (BCEWithLogitsLoss)
    during training and in `predict_proba` at inference."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def manual_pytorch_proba(X: np.ndarray, w: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    X_t = torch.as_tensor(X, dtype=DTYPE)
    with torch.no_grad():
        return torch.sigmoid(X_t @ w + b).numpy().ravel()


def nn_module_proba(X: np.ndarray, model: LogisticNet) -> np.ndarray:
    X_t = torch.as_tensor(X, dtype=DTYPE)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_t)).numpy().ravel()


def monthly_errors(y_true: np.ndarray, estimate: np.ndarray, month: np.ndarray,
                   weight: np.ndarray | None = None) -> pd.Series:
    """Error of the estimated churn rate in each month, in percentage points
    (positive = the estimate says more voters leave PSOE than really do).

    `estimate` is a 0/1 assignment per voter (thresholded) or a probability
    per voter (expected value): either way, the month's estimate is its mean.
    `weight` is the corrected survey weight (data/raking.py): both the actual
    and the estimated rate are weighted, as a published figure would be.
    """
    w = np.ones(len(y_true)) if weight is None else np.asarray(weight, float)
    df = pd.DataFrame({"y": np.asarray(y_true, float) * w, "e": np.asarray(estimate, float) * w, "w": w,
                       "month": month})
    g = df.groupby("month")[["y", "e", "w"]].sum()
    return 100 * (g["e"] - g["y"]) / g["w"]


def monthly_abs_error(y_true: np.ndarray, estimate: np.ndarray, month: np.ndarray,
                      weight: np.ndarray | None = None) -> float:
    """The pollster's loss: mean absolute monthly error of the estimate, in points."""
    return float(monthly_errors(y_true, estimate, month, weight).abs().mean())
