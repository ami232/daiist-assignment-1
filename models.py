"""
Model definitions shared by train.py (which fits and saves them) and app.py
(which only loads them). Keeping the nn.Module class in one place means the
app can rebuild it to load the saved state_dict without retraining anything.

All three implementations minimise the SAME objective, so they should land
on (almost) the same coefficients:

    mean binary cross-entropy  +  ||w||^2 / (2 * C * n)

That is scikit-learn's L2-penalised logistic regression,
    0.5 * ||w||^2 + C * sum(log-loss),
divided through by C * n. The intercept is not penalised in either.
"""

import numpy as np
import torch
from torch import nn


def l2_penalty(w: torch.Tensor, C: float, n: int) -> torch.Tensor:
    """sklearn-equivalent L2 term for a mean-reduced loss (bias excluded)."""
    return (w ** 2).sum() / (2.0 * C * n)


# ---------------------------------------------------------------------------
# 1) Manual PyTorch: raw tensors, autograd, hand-written gradient step
# ---------------------------------------------------------------------------
def fit_manual(X: np.ndarray, y: np.ndarray, C: float, lr: float, epochs: int):
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    n, d = Xt.shape

    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    history = []
    for _ in range(epochs):
        z = Xt @ w + b                                  # logits
        # BCE written out by hand: -[y log s(z) + (1-y) log(1-s(z))]
        # = softplus(z) - y*z, which is the numerically stable form.
        bce = (torch.nn.functional.softplus(z) - yt * z).mean()
        loss = bce + l2_penalty(w, C, n)
        loss.backward()                                 # autograd fills .grad
        with torch.no_grad():                           # manual update step
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())

    return {"w": w.detach(), "b": b.detach()}, history


def predict_manual(params: dict, X: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float64)
    return torch.sigmoid(Xt @ params["w"] + params["b"]).numpy()


# ---------------------------------------------------------------------------
# 2) Standard PyTorch: nn.Module + torch.optim
# ---------------------------------------------------------------------------
class LogisticRegressionModule(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)   # same start point as the manual loop
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)    # logits; sigmoid lives in the loss


def fit_standard(X: np.ndarray, y: np.ndarray, C: float, lr: float, epochs: int):
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    n, d = Xt.shape

    model = LogisticRegressionModule(d)
    criterion = nn.BCEWithLogitsLoss()
    # Not using SGD(weight_decay=...) because that would also shrink the bias,
    # which sklearn doesn't do. The penalty is added explicitly instead.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(Xt), yt) + l2_penalty(model.linear.weight, C, n)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return model, history


def predict_standard(model: LogisticRegressionModule, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float64))).numpy()
