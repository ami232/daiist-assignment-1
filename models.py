"""
The two PyTorch implementations of ridge (L2-regularized linear) regression.

Kept out of train.py so app.py can import LinearRegressionModel to load the
saved weights without running any training code.

All three implementations (sklearn Ridge, manual loop, nn.Module + optim)
minimize the SAME objective, written in "mean" form:

    loss(w, b) = mean((X @ w + b - y)^2) + lam * sum(w^2),   lam = alpha / n

sklearn's Ridge minimizes  sum((X @ w + b - y)^2) + alpha * sum(w^2).
Dividing that by n (the number of training rows) gives the expression above,
and dividing by a positive constant does not move the minimum. So with
lam = alpha / n, both have exactly the same optimal w and b.
The intercept b is NOT penalized, again to match sklearn.

INITIALIZATION NOTE (found while comparing the three methods, see REPORT.md):
the one-hot columns of each categorical variable always add up to 1, exactly
like the intercept's column of ones. Starting from b = 0, gradient descent
has to climb to log(price) ~ 12 and spreads that height over the intercept
AND every group of dummies (e.g. Central Air_N and _Y both ~ +2.6). The
predictions are the same either way; only the tiny ridge penalty pulls the
coefficients back to sklearn's solution, which takes a very long time.
Starting the bias at mean(y) - the best constant prediction - removes that
detour. The weights still start at 0, so nothing is learned "for free".
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn


def ridge_loss(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor, lam: float) -> torch.Tensor:
    """Mean squared error + L2 penalty on the weights (not on the bias)."""
    return ((pred - y) ** 2).mean() + lam * (w ** 2).sum()


# ---------------------------------------------------------------------------
# 1) Manual PyTorch loop (Session 5 style): raw tensors + autograd +
#    a hand-written gradient-descent update. No nn.Module, no optim.
# ---------------------------------------------------------------------------
def train_manual(X, y, lam: float, lr: float, epochs: int) -> dict:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Parameters we want gradients for. Weights start at 0 and the bias at
    # mean(y) (see INITIALIZATION NOTE above), exactly like the nn.Module
    # version below, so the two PyTorch versions differ only in the API.
    w = torch.zeros(X_t.shape[1], requires_grad=True)
    b = torch.full((1,), float(y.mean()), requires_grad=True)

    history = []
    for _ in range(epochs):
        pred = X_t @ w + b                    # forward pass: linear prediction
        loss = ridge_loss(pred, y_t, w, lam)
        loss.backward()                       # autograd fills w.grad and b.grad

        with torch.no_grad():                 # the update itself must not be tracked
            w -= lr * w.grad                  # gradient-descent step
            b -= lr * b.grad
        w.grad.zero_()                        # gradients accumulate by default,
        b.grad.zero_()                        # so reset them every iteration

        history.append(loss.item())

    return {"w": w.detach().numpy(), "b": float(b.item()), "history": history}


# ---------------------------------------------------------------------------
# 2) Standard PyTorch workflow: nn.Module + torch.optim
# ---------------------------------------------------------------------------
class LinearRegressionModel(nn.Module):
    """y_hat = X @ w + b, i.e. a single nn.Linear layer with one output."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        # nn.Linear starts from small random values by default; we start the
        # weights from zero to match the manual loop exactly (see train_manual).
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)     # (n, 1) -> (n,)


def train_module(X, y, lam: float, lr: float, epochs: int) -> dict:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = LinearRegressionModel(X_t.shape[1])
    with torch.no_grad():
        model.linear.bias.fill_(float(y.mean()))   # same start as the manual loop
    # Plain SGD on the full training set each step = the same gradient
    # descent as the manual loop. We do NOT use SGD(weight_decay=...),
    # because that would also penalize the bias, unlike sklearn.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        optimizer.zero_grad()                 # reset gradients
        pred = model(X_t)                     # forward pass
        loss = ridge_loss(pred, y_t, model.linear.weight, lam)
        loss.backward()                       # gradients
        optimizer.step()                      # w -= lr * grad, done by optim
        history.append(loss.item())

    return {"model": model, "history": history}


# ---------------------------------------------------------------------------
# Saving / loading the three trained models
# ---------------------------------------------------------------------------
# Used by train.py (to save, and to re-load once as a check) and by app.py
# (to load only). Nothing in here trains anything.
ARTIFACTS_DIR = Path("artifacts")
MODEL_NAMES = ["sklearn", "manual_torch", "module_torch"]


def save_models(sk_model, manual: dict, module_model: nn.Module, out_dir: Path = ARTIFACTS_DIR) -> None:
    out_dir.mkdir(exist_ok=True)
    # sklearn objects are saved with joblib (pickle optimized for numpy arrays).
    joblib.dump(sk_model, out_dir / "sklearn_ridge.joblib")
    # The manual model IS just its two tensors, so we save exactly those.
    torch.save({"w": torch.tensor(manual["w"]), "b": torch.tensor([manual["b"]])},
               out_dir / "manual_torch.pt")
    # Standard PyTorch practice: save the state_dict (the parameter tensors),
    # not the whole Python object. To load it, rebuild the same class and
    # copy the tensors back in (load_models below).
    torch.save(module_model.state_dict(), out_dir / "module_torch.pt")


def load_models(n_features: int, in_dir: Path = ARTIFACTS_DIR) -> dict:
    sk_model = joblib.load(in_dir / "sklearn_ridge.joblib")

    manual = torch.load(in_dir / "manual_torch.pt")

    module_model = LinearRegressionModel(n_features)
    module_model.load_state_dict(torch.load(in_dir / "module_torch.pt"))
    module_model.eval()

    return {"sklearn": sk_model, "manual_torch": manual, "module_torch": module_model}


def predict_log_price(models: dict, X: np.ndarray) -> dict:
    """Predictions of the three models on the log-price scale."""
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return {
            "sklearn": models["sklearn"].predict(X),
            "manual_torch": (X_t @ models["manual_torch"]["w"] + models["manual_torch"]["b"]).numpy(),
            "module_torch": models["module_torch"](X_t).numpy(),
        }
