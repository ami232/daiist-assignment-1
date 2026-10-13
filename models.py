"""
The same model, logistic regression, implemented three ways.

All three minimise one objective on the same standardised matrix X:

    mean binary cross-entropy  +  (l2 / 2) * sum(weights ** 2)      (bias not penalised)

scikit-learn writes it as  C * sum(cross-entropy) + 0.5 * sum(weights ** 2).
Dividing that by C * n gives the form above with  l2 = 1 / (C * n),  so one
value of C fixes the regularisation of all three. The objective is convex,
which means it has a single minimum: if the three sets of weights differ,
it can only be because an optimiser stopped before reaching it.

The two PyTorch versions follow the Session 5 notebook: the manual loop is
its section 9.2 (raw tensors, a forward function, a hand-written loss and
update), the standard one is its appendix A.2 (nn.Linear + sigmoid,
nn.BCELoss, torch.optim).

train.py is the only caller of the train_* functions. app.py uses the
load_* and predict_* functions and never trains anything.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# scikit-learn computes in 64-bit floats. Using the same precision in PyTorch
# means any gap between the methods comes from the optimiser, not rounding.
torch.set_default_dtype(torch.float64)


def l2_from_C(C: float, n_rows: int) -> float:
    return 1.0 / (C * n_rows)


def as_tensors(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(X), torch.tensor(y, dtype=torch.get_default_dtype())


# --- The pieces of the manual model (Session 5, section 9.2) ---------------------
def forward(X: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Linear combination squashed into a probability."""
    return torch.sigmoid(X @ weights + bias)


def bce_loss(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy, averaged over the rows."""
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)  # never take log(0)
    return -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()


def objective(weights: torch.Tensor, bias: torch.Tensor, X: torch.Tensor, y: torch.Tensor, l2: float) -> torch.Tensor:
    """Cross-entropy plus the L2 penalty. Shared so every method is scored identically."""
    return bce_loss(forward(X, weights, bias), y) + 0.5 * l2 * (weights ** 2).sum()


# --- 1. scikit-learn ---------------------------------------------------------------
def train_sklearn(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    # Default solver (L-BFGS) with an L2 penalty. The tight tolerance makes it
    # run to the true minimum, so it can serve as the reference solution.
    return LogisticRegression(C=C, tol=1e-10, max_iter=10_000).fit(X, y)


# --- 2. Manual PyTorch loop --------------------------------------------------------
def train_manual(X: np.ndarray, y: np.ndarray, l2: float, lr: float, epochs: int):
    """Raw tensors, autograd for the gradient, and a hand-written update step.

    Full-batch gradient descent: every step uses all training rows, so the
    run is deterministic. The only addition to the Session 5 loop is the L2
    term inside `objective`.
    """
    X_t, y_t = as_tensors(X, y)
    weights = torch.zeros(X_t.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)

    history = []
    for _ in range(epochs):
        loss = objective(weights, bias, X_t, y_t, l2)

        loss.backward()                     # autograd fills weights.grad and bias.grad

        with torch.no_grad():               # the update itself must not be tracked
            weights -= lr * weights.grad
            bias -= lr * bias.grad

        weights.grad.zero_()                # gradients accumulate unless cleared
        bias.grad.zero_()

        history.append(loss.item())
    return weights.detach(), bias.detach(), history


# --- 3. Standard PyTorch workflow ----------------------------------------------------
class LogisticRegressionModule(nn.Module):
    """nn.Linear followed by a sigmoid, as in the Session 5 appendix."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)  # one probability per row


def train_module(
    X: np.ndarray, y: np.ndarray, l2: float, lr: float, epochs: int,
    batch_size: int = 256, lr_decay: float = 1.0, seed: int = 0,
):
    """nn.Module + loss function + torch.optim, fed by a DataLoader.

    Two additions to the Session 5 appendix. Mini-batches: each step
    estimates the gradient from `batch_size` shuffled rows instead of all of
    them, which is how PyTorch models are normally trained. The estimate is
    noisy, so the learning rate is multiplied by `lr_decay` after every epoch
    to let the weights settle.
    """
    torch.manual_seed(seed)  # fixes the weight initialisation
    X_t, y_t = as_tensors(X, y)
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),  # fixes the shuffling
    )

    model = LogisticRegressionModule(X_t.shape[1])
    loss_fn = nn.BCELoss()
    # weight_decay adds l2 * weight to each gradient, which is the gradient of
    # (l2 / 2) * weight ** 2. It is applied to the weights only, because
    # scikit-learn does not penalise the intercept.
    optimizer = torch.optim.SGD(
        [
            {"params": [model.linear.weight], "weight_decay": l2},
            {"params": [model.linear.bias], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    history = []
    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():  # full-data objective, comparable with the manual loop
            history.append(objective(model.linear.weight.squeeze(0), model.linear.bias, X_t, y_t, l2).item())
    return model, history


# --- Prediction ----------------------------------------------------------------------
def predict_sklearn(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def predict_manual(weights: torch.Tensor, bias: torch.Tensor, X: np.ndarray) -> np.ndarray:
    return forward(torch.tensor(X), weights, bias).numpy()


def predict_module(model: LogisticRegressionModule, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def predict(kind: str, model, X: np.ndarray) -> np.ndarray:
    """Probability of failing, whichever way the model was trained."""
    if kind == "sklearn":
        return predict_sklearn(model, X)
    if kind == "manual":
        return predict_manual(*model, X)
    return predict_module(model, X)


def coefficients(kind: str, model) -> tuple[np.ndarray, float]:
    """(weights, bias) as plain numbers, whichever way the model was trained."""
    if kind == "sklearn":
        return model.coef_.ravel(), float(model.intercept_[0])
    if kind == "manual":
        weights, bias = model
        return weights.numpy().ravel(), float(bias.item())
    return model.linear.weight.detach().numpy().ravel(), float(model.linear.bias.item())


# --- Saving and loading ----------------------------------------------------------------
SKLEARN_FILE = "model_sklearn.joblib"
MANUAL_FILE = "model_torch_manual.pt"
MODULE_FILE = "model_torch_module.pt"


def save_models(folder: Path, sklearn_model, manual_model, module_model) -> None:
    joblib.dump(sklearn_model, folder / SKLEARN_FILE)
    weights, bias = manual_model
    torch.save({"weights": weights, "bias": bias}, folder / MANUAL_FILE)
    torch.save(module_model.state_dict(), folder / MODULE_FILE)


def load_models(folder: Path, n_features: int) -> dict:
    manual = torch.load(folder / MANUAL_FILE)
    module = LogisticRegressionModule(n_features)
    module.load_state_dict(torch.load(folder / MODULE_FILE))
    return {
        "sklearn": joblib.load(folder / SKLEARN_FILE),
        "manual": (manual["weights"], manual["bias"]),
        "module": module,
    }
