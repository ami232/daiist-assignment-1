"""
One model, three implementations, plus the naive baselines and the metrics
used to compare them.

The model is ridge-regularised linear regression on the standardised design
matrix produced by pipeline.make_preprocessor, predicting log1p(cnt).

All three implementations minimise the *same* objective so they should land
on the same coefficients:

    scikit-learn Ridge:   sum_i (y_i - x_i.w - b)^2 + alpha * ||w||^2
    divided by n:         MSE(y, Xw + b)           + (alpha / n) * ||w||^2

* The manual PyTorch loop writes that second line literally as its loss.
* The torch.optim version uses nn.MSELoss and passes the penalty through the
  optimiser's `weight_decay`. PyTorch's weight_decay adds `wd * w` to the
  gradient, i.e. it corresponds to a penalty of (wd / 2) * ||w||^2, so to match
  sklearn we need wd = 2 * alpha / n, applied to the weights only (sklearn
  never penalises the intercept, so the bias gets weight_decay = 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Metrics. The model is trained on log1p(cnt) but the operator cares about
# rides, so every metric is reported on the count scale (after expm1), plus
# RMSLE, which is just RMSE on the scale the model actually optimises.
# ---------------------------------------------------------------------------
def to_counts(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y_log, dtype=float)).clip(min=0)


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray, peak_cutoff: float) -> dict:
    y_true = to_counts(y_true_log)
    y_pred = to_counts(y_pred_log)
    err = y_pred - y_true
    peak = y_true >= peak_cutoff
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": 1.0 - ss_res / ss_tot,
        "rmsle": float(np.sqrt(((np.asarray(y_pred_log) - np.asarray(y_true_log)) ** 2).mean())),
        "bias": float(err.mean()),  # >0 means over-forecasting on average
        "mae_peak_hours": float(np.abs(err[peak]).mean()) if peak.any() else float("nan"),
        "n_peak_hours": int(peak.sum()),
    }


# ---------------------------------------------------------------------------
# Naive baselines (no learning beyond group averages)
# ---------------------------------------------------------------------------
class MeanBaseline:
    """Predict the training mean of log1p(cnt) for every hour."""

    name = "baseline_mean"

    def fit(self, train_df, target_col: str):
        self.value_ = float(train_df[target_col].mean())
        return self

    def predict(self, df) -> np.ndarray:
        return np.full(len(df), self.value_)


class SeasonalNaiveBaseline:
    """Predict the training mean of log1p(cnt) for the same (hour, workingday).

    This is the "what a dispatcher with a spreadsheet would do" baseline: a
    fixed daily profile per day type, ignoring weather and growth.
    """

    name = "baseline_hour_profile"

    def fit(self, train_df, target_col: str):
        self.table_ = train_df.groupby(["hr", "workingday"])[target_col].mean()
        self.fallback_ = float(train_df[target_col].mean())
        return self

    def predict(self, df) -> np.ndarray:
        keys = list(zip(df["hr"], df["workingday"]))
        return np.array([self.table_.get(k, self.fallback_) for k in keys])


class LagOnlyBaseline:
    """Predict last week's same-hour level (the lag feature on its own)."""

    name = "baseline_last_week_level"

    def fit(self, train_df, target_col: str):
        return self

    def predict(self, df) -> np.ndarray:
        return df["lag_level"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 1. scikit-learn
# ---------------------------------------------------------------------------
class SklearnRidgeModel:
    name = "sklearn"

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self.model.coef_.ravel()

    @property
    def intercept_(self) -> float:
        return float(self.model.intercept_)

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "SklearnRidgeModel":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.alpha = float(obj.model.alpha)
        return obj


# ---------------------------------------------------------------------------
# 2. Manual PyTorch loop: raw tensors + autograd + hand-written update step
# ---------------------------------------------------------------------------
class ManualTorchModel:
    name = "torch_manual"

    def __init__(self, n_features: int, alpha: float, lr: float, n_steps: int):
        self.n_features = n_features
        self.alpha = alpha
        self.lr = lr
        self.n_steps = n_steps
        # Parameters are plain tensors that autograd tracks. Starting from zero
        # is fine for a convex problem (there is a single optimum).
        self.w = torch.zeros(n_features, 1, dtype=torch.float32, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.history: list[dict] = []

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None, log_every: int = 50):
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        n = X_t.shape[0]
        if X_val is not None:
            Xv = torch.as_tensor(X_val, dtype=torch.float32)
            yv = torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1)

        self.history = []
        for step in range(1, self.n_steps + 1):
            # Full-batch gradient descent: one forward pass over all rows.
            pred = self._forward(X_t)
            mse = ((pred - y_t) ** 2).mean()
            penalty = (self.alpha / n) * (self.w**2).sum()  # == sklearn Ridge / n
            loss = mse + penalty

            loss.backward()  # autograd fills w.grad and b.grad

            with torch.no_grad():  # the manual update step
                self.w -= self.lr * self.w.grad
                self.b -= self.lr * self.b.grad
            self.w.grad.zero_()
            self.b.grad.zero_()

            if step <= 100 or step % log_every == 0 or step == self.n_steps:
                rec = {"step": step, "train_mse": float(mse.detach())}
                if X_val is not None:
                    with torch.no_grad():
                        rec["val_mse"] = float(((self._forward(Xv) - yv) ** 2).mean())
                self.history.append(rec)
                if not np.isfinite(rec["train_mse"]):
                    break  # diverged (learning rate too high); stop wasting time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._forward(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()

    @property
    def coef_(self) -> np.ndarray:
        return self.w.detach().numpy().ravel()

    @property
    def intercept_(self) -> float:
        return float(self.b.detach())

    def save(self, path: Path) -> None:
        torch.save(
            {
                "w": self.w.detach().clone(),
                "b": self.b.detach().clone(),
                "config": {"n_features": self.n_features, "alpha": self.alpha, "lr": self.lr,
                           "n_steps": self.n_steps},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ManualTorchModel":
        state = torch.load(path, weights_only=True)
        obj = cls(**state["config"])
        with torch.no_grad():
            obj.w.copy_(state["w"])
            obj.b.copy_(state["b"])
        return obj


# ---------------------------------------------------------------------------
# 3. Standard PyTorch workflow: nn.Module + nn.MSELoss + torch.optim + DataLoader
# ---------------------------------------------------------------------------
class LinearRegressor(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class StandardTorchModel:
    name = "torch_standard"

    def __init__(self, n_features: int, alpha: float, lr: float, n_epochs: int, batch_size: int = 256):
        self.n_features = n_features
        self.alpha = alpha
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.net = LinearRegressor(n_features)
        self.history: list[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None):
        set_seed()
        with torch.no_grad():  # same starting point as the manual loop
            self.net.linear.weight.zero_()
            self.net.linear.bias.zero_()

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        n = X_t.shape[0]
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED))
        if X_val is not None:
            Xv = torch.as_tensor(X_val, dtype=torch.float32)
            yv = torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1)

        criterion = nn.MSELoss()
        # weight_decay = 2 * alpha / n reproduces sklearn's alpha (see module
        # docstring); the bias is left unpenalised exactly like sklearn does.
        optimizer = torch.optim.Adam(
            [
                {"params": [self.net.linear.weight], "weight_decay": 2.0 * self.alpha / n},
                {"params": [self.net.linear.bias], "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        # Decay the step size over training so the mini-batch noise of Adam
        # settles onto the optimum instead of bouncing around it.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        self.history = []
        for epoch in range(1, self.n_epochs + 1):
            self.net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            self.net.eval()
            with torch.no_grad():
                rec = {"epoch": epoch, "train_mse": float(criterion(self.net(X_t), y_t))}
                if X_val is not None:
                    rec["val_mse"] = float(criterion(self.net(Xv), yv))
            self.history.append(rec)
            if not np.isfinite(rec["train_mse"]):
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()

    @property
    def coef_(self) -> np.ndarray:
        return self.net.linear.weight.detach().numpy().ravel()

    @property
    def intercept_(self) -> float:
        return float(self.net.linear.bias.detach())

    def save(self, path: Path) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "config": {"n_features": self.n_features, "alpha": self.alpha, "lr": self.lr,
                           "n_epochs": self.n_epochs, "batch_size": self.batch_size},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "StandardTorchModel":
        state = torch.load(path, weights_only=True)
        obj = cls(**state["config"])
        obj.net.load_state_dict(state["state_dict"])
        obj.net.eval()
        return obj


# ---------------------------------------------------------------------------
# Loading helpers shared by app.py
# ---------------------------------------------------------------------------
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODEL_FILES = {
    "sklearn": "model_sklearn.joblib",
    "torch_manual": "model_torch_manual.pt",
    "torch_standard": "model_torch_standard.pt",
}
MODEL_LABELS = {
    "sklearn": "scikit-learn Ridge",
    "torch_manual": "PyTorch manual loop",
    "torch_standard": "PyTorch nn.Module + optim",
}


def load_models(artifacts: Path = ARTIFACTS) -> dict:
    return {
        "sklearn": SklearnRidgeModel.load(artifacts / MODEL_FILES["sklearn"]),
        "torch_manual": ManualTorchModel.load(artifacts / MODEL_FILES["torch_manual"]),
        "torch_standard": StandardTorchModel.load(artifacts / MODEL_FILES["torch_standard"]),
    }


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
