"""One model, L2-regularized logistic regression, implemented three ways on the same inputs.

All three minimize the same objective:

    mean binary cross-entropy  +  (lam / 2) * ||w||^2      (intercept not penalized)

scikit-learn writes it as  C * sum(cross-entropy) + 0.5 * ||w||^2.  Dividing by C * n gives the
form above with  lam = 1 / (C * n).  That conversion is what lets the three agree.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn


def l2_strength(C: float, n_train: int) -> float:
    """Convert scikit-learn's C into the per-sample L2 strength used in the PyTorch losses."""
    return 1.0 / (C * n_train)


def objective(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """The shared objective, in float64, so the three solutions can be compared on equal terms."""
    z = X.astype(np.float64) @ np.asarray(w, dtype=np.float64) + float(b)
    bce = np.mean(np.logaddexp(0.0, z) - y * z)
    return float(bce + lam / 2 * np.sum(np.asarray(w, dtype=np.float64) ** 2))


# 1. scikit-learn ---------------------------------------------------------------------------

def fit_sklearn(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    """L-BFGS solves the convex problem to (near) machine precision: the reference solution."""
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=10_000, tol=1e-10)
    return model.fit(X, y)


def predict_sklearn(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def sklearn_params(model: LogisticRegression) -> tuple[np.ndarray, float]:
    return model.coef_.ravel(), float(model.intercept_[0])


# 2. Manual PyTorch loop: raw tensors, autograd, hand-written update -------------------------

def fit_torch_manual(X: np.ndarray, y: np.ndarray, lam: float, lr: float, epochs: int):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    w = torch.zeros(X.shape[1], requires_grad=True)   # one weight per feature
    b = torch.zeros(1, requires_grad=True)            # intercept

    history = []
    for _ in range(epochs):
        z = X_t @ w + b                                # logits
        # Binary cross-entropy from its formula, log(1 + e^z) - y*z, written in the
        # numerically stable form so large |z| cannot overflow exp().
        bce = (torch.clamp(z, min=0) - y_t * z + torch.log1p(torch.exp(-z.abs()))).mean()
        loss = bce + lam / 2 * (w**2).sum()            # L2 on the weights only
        loss.backward()                                # autograd fills w.grad and b.grad

        with torch.no_grad():                          # the update itself must not be tracked
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                                 # gradients accumulate unless reset
        b.grad.zero_()
        history.append(loss.item())

    return {"w": w.detach().clone(), "b": b.detach().clone()}, history


def predict_torch_manual(params: dict, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = torch.tensor(X, dtype=torch.float32) @ params["w"] + params["b"]
        return torch.sigmoid(z).numpy()


def manual_params(params: dict) -> tuple[np.ndarray, float]:
    return params["w"].numpy(), float(params["b"][0])


# 3. Standard PyTorch workflow: nn.Module + loss module + torch.optim -----------------------

class LogisticRegressionNet(nn.Module):
    """A single linear layer. It returns logits: the sigmoid lives inside BCEWithLogitsLoss."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def fit_torch_module(X: np.ndarray, y: np.ndarray, lam: float, lr: float, epochs: int, seed: int):
    torch.manual_seed(seed)                            # nn.Linear starts from random weights
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = LogisticRegressionNet(X.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    # SGD's weight_decay adds lam * w to the gradient, which is exactly the gradient of
    # (lam / 2) * ||w||^2. Applied to the weights only, like scikit-learn, not to the bias.
    optimizer = torch.optim.SGD(
        [
            {"params": [model.linear.weight], "weight_decay": lam},
            {"params": [model.linear.bias], "weight_decay": 0.0},
        ],
        lr=lr,
    )

    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        with torch.no_grad():                          # log the full objective, penalty included
            history.append(loss.item() + lam / 2 * float((model.linear.weight**2).sum()))
    return model, history


def predict_torch_module(model: LogisticRegressionNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def module_params(model: LogisticRegressionNet) -> tuple[np.ndarray, float]:
    return model.linear.weight.detach().numpy().ravel(), float(model.linear.bias.detach()[0])


# Saving and loading -----------------------------------------------------------------------

def save_models(directory: Path, sk_model, manual: dict, module: LogisticRegressionNet) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(sk_model, directory / "model_sklearn.joblib")
    torch.save(manual, directory / "model_torch_manual.pt")
    torch.save(module.state_dict(), directory / "model_torch_module.pt")


def load_models(directory: Path, n_features: int):
    sk_model = joblib.load(directory / "model_sklearn.joblib")
    manual = torch.load(directory / "model_torch_manual.pt", weights_only=True)
    module = LogisticRegressionNet(n_features)
    module.load_state_dict(torch.load(directory / "model_torch_module.pt", weights_only=True))
    module.eval()
    return sk_model, manual, module


MODEL_NAMES = {
    "sklearn": "scikit-learn",
    "torch_manual": "PyTorch manual loop",
    "torch_module": "PyTorch nn.Module + optim",
}


def predict_all(sk_model, manual: dict, module: LogisticRegressionNet, X: np.ndarray) -> dict:
    return {
        "sklearn": predict_sklearn(sk_model, X),
        "torch_manual": predict_torch_manual(manual, X),
        "torch_module": predict_torch_module(module, X),
    }
