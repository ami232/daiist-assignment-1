"""
Shared pieces used by both train.ipynb and app.py.

Keeping them in one file means the training notebook and the dashboard use
exactly the same model definition and the same business cost formula.
"""

import numpy as np
import torch
from torch import nn

# Business assumptions (explained in REPORT.md, "Business / real-life framing")
CALL_COST = 20.0        # euros of staff time for one outreach call
SAVE_RATE = 1 / 3       # share of leaving hosts that a call keeps on the platform
VALUE_LOST = 330.0      # euros of commission lost when one host leaves (one quarter)


class LogisticRegressionModule(nn.Module):
    """Logistic regression as a PyTorch module: one linear layer.

    forward() returns the raw score (logit). The sigmoid is applied by the
    loss during training and by predict_proba() at prediction time.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.sigmoid(self.forward(x)).numpy()


def manual_predict_proba(params: dict, x: np.ndarray) -> np.ndarray:
    """Probabilities from the manual PyTorch model (raw weight and bias tensors)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(x @ params["w"] + params["b"]).numpy()


def business_cost(y_true, proba, threshold,
                  call_cost=CALL_COST, save_rate=SAVE_RATE, value_lost=VALUE_LOST):
    """Total cost in euros of calling every host whose probability is >= threshold.

    - every call costs call_cost
    - a leaving host we do not call costs value_lost
    - a leaving host we call is kept with probability save_rate,
      so on average it still costs (1 - save_rate) * value_lost
    """
    y_true = np.asarray(y_true).astype(int)
    called = np.asarray(proba) >= threshold
    n_called = called.sum()
    leavers_missed = ((y_true == 1) & ~called).sum()
    leavers_called = ((y_true == 1) & called).sum()
    return float(n_called * call_cost
                 + leavers_missed * value_lost
                 + leavers_called * (1 - save_rate) * value_lost)
