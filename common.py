"""
Small pieces shared by train.ipynb and app.py, so the dashboard rebuilds the
exact same model class and the exact same profit calculation.
"""

import numpy as np
import torch
from torch import nn


class LogisticRegression(nn.Module):
    """One linear layer; the sigmoid lives in the loss (BCEWithLogitsLoss)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def voucher_profit(send, lapsed, spend, discount, margin, save_rate, retained_quarters, mail_cost):
    """
    Profit (GBP) of a voucher campaign, relative to sending nothing.

    send    : bool array, who gets a voucher
    lapsed  : 0/1 array, whether the customer actually made no purchase
    spend   : the customer's usual quarterly spend (GBP)

    - voucher to a would-be lapser: with probability `save_rate` they come
      back and keep ordering their usual amount for `retained_quarters`
      quarters at `margin`; the `discount` applies to the first quarter only
    - voucher to someone who would have ordered anyway: they use it on that
      order, so we just give away `discount` of it
    - no voucher: nothing changes (0)
    """
    send = np.asarray(send, dtype=bool)
    lapsed = np.asarray(lapsed, dtype=bool)
    spend = np.asarray(spend, dtype=float)
    win_back = save_rate * (margin * retained_quarters - discount) * spend
    leakage = -discount * spend
    per_customer = np.where(lapsed, win_back, leakage) - mail_cost
    return float(np.sum(np.where(send, per_customer, 0.0)))


def break_even_threshold(discount, margin, save_rate, retained_quarters):
    """
    Lapse probability above which a voucher pays off on average, ignoring the
    (small) mailing cost:  p * s * (g * L - d) > (1 - p) * d.
    """
    return discount / (save_rate * (margin * retained_quarters - discount) + discount)
