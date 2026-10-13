"""The two PyTorch versions of the same L2-regularised logistic regression.

All three implementations minimise the same objective, so on the same data
they should land on the same weights:

    J(w, b) = mean binary cross-entropy  +  (lam / 2) * ||w||^2

scikit-learn writes it as  C * sum(loss) + 0.5 * ||w||^2. Dividing by C * n
gives the line above with lam = 1 / (C * n). The intercept b is not
penalised (sklearn does not penalise it either), which is why I add the L2
term by hand instead of using the optimiser's weight_decay (that would also
shrink the bias).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def l2_strength(C: float, n_rows: int) -> float:
    return 1.0 / (C * n_rows)


def objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam: float) -> float:
    """J(w, b) in float64 numpy, used to check that all three reached the same minimum."""
    z = X @ w + b
    bce = np.mean(np.logaddexp(0.0, z) - y * z)   # stable form of the BCE
    return float(bce + 0.5 * lam * np.dot(w, w))


# ---------------------------------------------------------------------------
# 1) Manual loop: raw tensors, autograd, hand-written update (Session 5 style)
# ---------------------------------------------------------------------------

def train_manual(X: np.ndarray, y: np.ndarray, lam: float, lr: float, epochs: int):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n_features = Xt.shape[1]

    w = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []

    for _ in range(epochs):
        # 1. forward pass: linear score -> probability
        p = torch.sigmoid(Xt @ w + b)
        p = torch.clamp(p, 1e-7, 1 - 1e-7)          # never take log(0)
        bce = -(yt * torch.log(p) + (1 - yt) * torch.log(1 - p)).mean()
        loss = bce + 0.5 * lam * (w ** 2).sum()

        # 2. backward pass: autograd fills w.grad and b.grad
        loss.backward()

        # 3. gradient step, outside the graph
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # 4. reset, or the next backward() adds on top of these gradients
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())

    return {"w": w.detach().clone(), "b": b.detach().clone()}, history


def predict_manual(params: dict, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = torch.tensor(X, dtype=torch.float32) @ params["w"] + params["b"]
        return torch.sigmoid(z).numpy().astype(float)


# ---------------------------------------------------------------------------
# 2) Standard workflow: nn.Module + loss module + torch.optim
# ---------------------------------------------------------------------------

class LogisticRegressionNet(nn.Module):
    """One linear layer. forward() returns logits; the sigmoid lives in the
    loss (BCEWithLogitsLoss), which is the numerically stable way to do it."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_module(X: np.ndarray, y: np.ndarray, lam: float, lr: float, epochs: int, seed: int = 0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    model = LogisticRegressionNet(Xt.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay of the step size to 0 over the run: with a constant rate Adam keeps
    # hovering around the minimum in float32; the decay lets it settle exactly on it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []

    # Full batch: 3k rows fit in memory, and a fixed batch makes the run
    # deterministic, so differences to sklearn are about the optimiser only.
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt) + 0.5 * lam * (model.linear.weight ** 2).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append(loss.item())

    return model, history


def predict_module(model: LogisticRegressionNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy().astype(float)


def module_params(model: LogisticRegressionNet) -> tuple[np.ndarray, float]:
    return (model.linear.weight.detach().numpy().ravel().astype(float),
            float(model.linear.bias.detach().item()))
