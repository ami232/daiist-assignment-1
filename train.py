"""
Assignment 1 — training pipeline.

Binary logistic regression on the Wine Quality dataset (red + white Vinho
Verde): predict whether a wine is "good" (quality >= 7) from its
physicochemical measurements.

The same model is trained three ways on the same train/val/test split:
  1. scikit-learn LogisticRegression
  2. a manual PyTorch loop (raw tensors, autograd, hand-written update)
  3. the standard torch.nn.Module + torch.optim workflow
and all three are compared against a naive baseline on the same test set.

All three minimise the same objective as sklearn's default L2 logistic
regression, rescaled by 1/n so it's a mean:

    mean(BCE) + 1 / (2 * C * n_train) * ||w||^2      (bias not penalised)

so with the same C they should converge to the same coefficients.

Run with:  uv run python main.py train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

# ---- Settings ------------------------------------------------------------
SEED = 42
GOOD_QUALITY = 7          # quality >= 7 counts as a "good" wine
C = 1.0                   # inverse L2 strength, shared by all three models
LR = 0.5                  # learning rate for both PyTorch versions
EPOCHS = 3000             # full-batch gradient-descent steps
DECISION_THRESHOLD = 0.5

SKEWED = ["residual sugar", "chlorides", "total sulfur dioxide", "sulphates"]
DROPPED = ["density", "free sulfur dioxide"]

MODEL_NAMES = ["Baseline", "scikit-learn", "Manual PyTorch", "nn.Module + optim"]


# ==========================================================================
# Data
# ==========================================================================
def load_data() -> pd.DataFrame:
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";").assign(color=0)
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";").assign(color=1)
    df = pd.concat([red, white], ignore_index=True)
    # 1,177 rows are exact duplicates; keeping them would let the same wine
    # land in both train and test.
    return df.drop_duplicates().reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Share of SO2 that is free (the part that actually protects the wine).
    out["free_so2_ratio"] = out["free sulfur dioxide"] / out["total sulfur dioxide"]
    # density is almost a linear function of alcohol + sugar; free SO2 is now
    # captured by the ratio.
    return out.drop(columns=DROPPED)


def split(df: pd.DataFrame):
    """Stratified 70/15/15 train/val/test split.

    There's no timestamp in this data, so a random split doesn't leak the
    future. Stratifying keeps the ~19% share of good wines in every split.
    """
    y = (df["quality"] >= GOOD_QUALITY).astype(int)
    X = df.drop(columns=["quality"])
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================================================
# Preprocessing (fitted on train only, then applied to every split)
# ==========================================================================
def fit_preprocessing(X_train: pd.DataFrame) -> dict:
    continuous = [c for c in X_train.columns if c != "color"]
    prep = {
        "lo": X_train[continuous].quantile(0.01),
        "hi": X_train[continuous].quantile(0.99),
        "skewed": SKEWED,
        "features": list(X_train.columns),
    }
    prep["scaler"] = StandardScaler().fit(_clip_and_log(X_train, prep))
    return prep


def _clip_and_log(X: pd.DataFrame, prep: dict) -> pd.DataFrame:
    X = X[prep["features"]].copy()
    continuous = list(prep["lo"].index)
    X[continuous] = X[continuous].clip(prep["lo"], prep["hi"], axis=1)  # winsorise 1%/99%
    X[prep["skewed"]] = np.log1p(X[prep["skewed"]])
    return X


def transform(X: pd.DataFrame, prep: dict) -> np.ndarray:
    return prep["scaler"].transform(_clip_and_log(X, prep))


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32)


# ==========================================================================
# 1. scikit-learn
# ==========================================================================
def train_sklearn(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=5000)  # L2 by default
    return model.fit(X, y)


# ==========================================================================
# 2. Manual PyTorch loop: raw tensors, autograd, hand-written update
# ==========================================================================
def manual_bce(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)  # avoid log(0)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def train_manual_torch(X, y, X_val, y_val):
    n, d = X.shape
    l2 = 1.0 / (2 * C * n)

    w = torch.zeros(d, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = {"train": [], "val": []}

    for _ in range(EPOCHS):
        bce = manual_bce(torch.sigmoid(X @ w + b), y)
        loss = bce + l2 * (w**2).sum()
        loss.backward()

        with torch.no_grad():
            w -= LR * w.grad
            b -= LR * b.grad
            w.grad.zero_()
            b.grad.zero_()
            val_bce = manual_bce(torch.sigmoid(X_val @ w + b), y_val)

        history["train"].append(bce.item())
        history["val"].append(val_bce.item())

    return w.detach(), b.detach(), history


def predict_manual(w, b, X) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(X @ w + b).squeeze(1).numpy()


# ==========================================================================
# 3. Standard torch.nn.Module + torch.optim workflow
# ==========================================================================
class LogisticRegressionNet(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)  # logits; the sigmoid lives inside the loss


def train_nn_module(X, y, X_val, y_val):
    n, d = X.shape
    model = LogisticRegressionNet(d)
    torch.nn.init.zeros_(model.linear.weight)
    torch.nn.init.zeros_(model.linear.bias)

    # SGD's weight_decay=λ adds λ*w to the gradient, i.e. a (λ/2)||w||^2
    # penalty. λ = 1/(C*n) reproduces sklearn's objective; the bias gets
    # no decay, matching sklearn (which never penalises the intercept).
    optimizer = torch.optim.SGD(
        [
            {"params": [model.linear.weight], "weight_decay": 1.0 / (C * n)},
            {"params": [model.linear.bias], "weight_decay": 0.0},
        ],
        lr=LR,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    history = {"train": [], "val": []}

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val)

        history["train"].append(loss.item())
        history["val"].append(val_loss.item())

    return model, history


def predict_nn(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).squeeze(1).numpy()


# ==========================================================================
# Evaluation & reporting
# ==========================================================================
def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, proba),
        "Log loss": log_loss(y_true, proba),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
    }


def coefficient_table(sk_model, w, b, nn_model, features) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scikit-learn": np.append(sk_model.coef_.ravel(), sk_model.intercept_),
            "Manual PyTorch": np.append(w.numpy().ravel(), b.numpy()),
            "nn.Module + optim": np.append(
                nn_model.linear.weight.detach().numpy().ravel(),
                nn_model.linear.bias.detach().numpy(),
            ),
        },
        index=features + ["intercept"],
    )


def print_report(results: pd.DataFrame, coefs: pd.DataFrame, test_proba: dict) -> None:
    print(f"\nTest-set results (threshold = {DECISION_THRESHOLD}):")
    print(results.round(4).to_string())

    print("\nCoefficients (standardised features):")
    print(coefs.round(4).to_string())

    sk = test_proba["scikit-learn"]
    print("\nMax |P(sklearn) - P(torch)| on test:")
    for name in ["Manual PyTorch", "nn.Module + optim"]:
        print(f"  {name:<18}: {np.abs(sk - test_proba[name]).max():.2e}")


def save_artifacts(prep, baseline, sk_model, w, b, nn_model, histories,
                   results, coefs, test_proba, y_test) -> None:
    """Save everything app.py needs, so the app never has to retrain."""
    ART_DIR.mkdir(exist_ok=True)
    joblib.dump(prep, ART_DIR / "preprocessing.joblib")
    joblib.dump(baseline, ART_DIR / "baseline.joblib")
    joblib.dump(sk_model, ART_DIR / "sklearn_model.joblib")
    torch.save({"w": w, "b": b}, ART_DIR / "manual_torch.pt")
    torch.save(nn_model.state_dict(), ART_DIR / "nn_module.pt")
    with open(ART_DIR / "loss_history.json", "w") as f:
        json.dump(histories, f)
    results.round(4).to_csv(ART_DIR / "results.csv")
    coefs.round(4).to_csv(ART_DIR / "coefficients.csv")
    pd.DataFrame({**test_proba, "y_true": y_test}).to_csv(
        ART_DIR / "test_predictions.csv", index=False
    )
    print(f"\nSaved artifacts to {ART_DIR}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    torch.manual_seed(SEED)

    # 1. Data, features, split
    df = engineer(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    print(f"Rows: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  "
          f"positive rate (train)={y_train.mean():.3f}")

    # 2. Preprocess (fit on train only)
    prep = fit_preprocessing(X_train)
    Xtr, Xva, Xte = (transform(X, prep) for X in (X_train, X_val, X_test))
    ytr, yva, yte = (s.to_numpy() for s in (y_train, y_val, y_test))

    Xtr_t, Xva_t, Xte_t = to_tensor(Xtr), to_tensor(Xva), to_tensor(Xte)
    ytr_t, yva_t = to_tensor(ytr).unsqueeze(1), to_tensor(yva).unsqueeze(1)

    # 3. Train: naive baseline + the same model three ways
    baseline = DummyClassifier(strategy="prior").fit(Xtr, ytr)  # always predicts base rate
    sk_model = train_sklearn(Xtr, ytr)
    w, b, manual_hist = train_manual_torch(Xtr_t, ytr_t, Xva_t, yva_t)
    nn_model, nn_hist = train_nn_module(Xtr_t, ytr_t, Xva_t, yva_t)

    # 4. Evaluate all four on the same test set
    test_proba = dict(zip(MODEL_NAMES, [
        baseline.predict_proba(Xte)[:, 1],
        sk_model.predict_proba(Xte)[:, 1],
        predict_manual(w, b, Xte_t),
        predict_nn(nn_model, Xte_t),
    ]))
    results = pd.DataFrame({name: evaluate(yte, p) for name, p in test_proba.items()}).T
    coefs = coefficient_table(sk_model, w, b, nn_model, prep["features"])
    print_report(results, coefs, test_proba)

    # 5. Save
    save_artifacts(prep, baseline, sk_model, w, b, nn_model,
                   {"manual": manual_hist, "nn_module": nn_hist},
                   results, coefs, test_proba, yte)


if __name__ == "__main__":
    main()
