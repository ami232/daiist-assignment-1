"""The two PyTorch implementations of L2-regularised logistic regression.

Both minimise exactly the objective scikit-learn's LogisticRegression does,
rescaled by 1/(C*n) so it reads as a mean:

    sklearn:  C * sum_i logloss_i + 0.5 * ||w||^2
    here:     mean_i logloss_i   + ||w||^2 / (2 * C * n)

Same minimiser, so if both optimisers converge the coefficients should match
sklearn's. The bias is not penalised, matching sklearn.
"""

import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)  # float64 so tiny differences aren't just float32 rounding


def l2_strength(C: float, n: int) -> float:
    return 1.0 / (2.0 * C * n)


# ---------- 1. Manual loop: raw tensors + autograd + hand-written update ----------

def train_manual(X, y, C, lr, epochs, X_val=None, y_val=None, w_ref=None):
    X, y = torch.tensor(X), torch.tensor(y, dtype=torch.float64)
    n, d = X.shape
    lam = l2_strength(C, n)
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(epochs):
        z = X @ w + b
        p = torch.sigmoid(z)
        eps = 1e-12
        bce = -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()
        loss = bce + lam * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():  # the update itself must not be tracked by autograd
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch, "train_loss": loss.item(),
                            "val_bce": _val_bce(X_val, y_val, w.detach(), b.detach()),
                            "max_coef_gap": _gap(w.detach(), w_ref)})
    return {"w": w.detach(), "b": b.detach()}, history


def predict_manual(params, X):
    with torch.no_grad():
        return torch.sigmoid(torch.as_tensor(X) @ params["w"] + params["b"]).numpy()


# ---------- 2. Standard workflow: nn.Module + loss module + torch.optim ----------

class LogisticRegressionNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits; sigmoid lives in the loss for stability


def train_standard(X, y, C, lr, epochs, X_val=None, y_val=None, seed=0, w_ref=None):
    torch.manual_seed(seed)
    X, y = torch.tensor(X), torch.tensor(y, dtype=torch.float64)
    n, d = X.shape
    lam = l2_strength(C, n)
    model = LogisticRegressionNet(d)
    loss_fn = nn.BCEWithLogitsLoss()
    # Not using SGD(weight_decay=...) because it would also shrink the bias,
    # which sklearn does not do. The penalty is added to the loss instead.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X), y) + lam * (model.linear.weight ** 2).sum()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            w = model.linear.weight.detach().squeeze(0)
            b = model.linear.bias.detach()
            history.append({"epoch": epoch, "train_loss": loss.item(),
                            "val_bce": _val_bce(X_val, y_val, w, b),
                            "max_coef_gap": _gap(w, w_ref)})
    return model, history


def predict_standard(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X))).numpy()


def _gap(w, w_ref):
    """Largest |coefficient difference| to a reference (sklearn's) solution."""
    return None if w_ref is None else float((w - torch.as_tensor(w_ref)).abs().max())


def _val_bce(X_val, y_val, w, b):
    if X_val is None:
        return None
    p = torch.sigmoid(torch.as_tensor(X_val) @ w + b).clamp(1e-12, 1 - 1e-12)
    y = torch.as_tensor(y_val, dtype=torch.float64)
    return float(-(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean())


def as_numpy_coefs(w, b):
    return np.asarray(w).ravel(), float(np.asarray(b).ravel()[0])
