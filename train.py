"""
Assignment 1 — training pipeline (Telco customer churn, logistic regression).

    uv run python main.py train

1. Load the committed dataset and engineer features (features.py).
2. Stratified train / validation / test split (60 / 20 / 20).
3. Fit the preprocessor (one-hot + standardisation) on train only.
4. Train the same L2-regularised logistic regression three ways:
   scikit-learn, a manual PyTorch loop, and nn.Module + torch.optim.
   All three minimise the *same* objective:
       mean BCE(w, b) + lambda/2 * ||w||^2,   lambda = 1 / (C * n_train)
   which is exactly scikit-learn's objective divided by C * n_train.
5. Tune only C (scikit-learn) and the learning rates (PyTorch) on validation.
6. Pick each model's decision threshold on validation by minimising the
   business cost, then report everything once on the untouched test set.
7. Save models, preprocessor, split assignment and metrics to artifacts/.
"""

import json
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn

from features import (ARTIFACTS, CHURN_COST, OFFER_COST, SEED, business_cost,
                      engineer, load_raw, make_preprocessor, target)

torch.manual_seed(SEED)
np.random.seed(SEED)

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
MANUAL_LR_GRID = [0.05, 0.2, 0.5, 1.0]
MANUAL_EPOCHS = 3000
STANDARD_LR_GRID = [0.001, 0.003, 0.01]
STANDARD_EPOCHS = 200
BATCH_SIZE = 256


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def split_data(raw: pd.DataFrame):
    """60/20/20 stratified random split.

    The dataset is a single snapshot with no date column, so there is no
    "future" to leak: a random split is valid. Stratifying keeps the ~26.5%
    churn rate identical across splits so validation-chosen thresholds
    transfer to test.
    """
    y = target(raw)
    idx = np.arange(len(raw))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, stratify=y[idx_tmp], random_state=SEED)
    return idx_train, idx_val, idx_test


# --------------------------------------------------------------------------
# The three implementations
# --------------------------------------------------------------------------
def fit_sklearn(X, y, C):
    model = LogisticRegression(C=C, max_iter=5000, tol=1e-8)
    return model.fit(X, y)


def fit_manual_torch(X, y, lam, lr, epochs=MANUAL_EPOCHS):
    """Session 5 style: raw tensors, autograd, manual gradient step."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(epochs):
        logits = X_t @ w + b
        p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        bce = -(y_t * torch.log(p) + (1 - y_t) * torch.log(1 - p)).mean()
        loss = bce + lam / 2 * (w ** 2).sum()   # bias is not regularised
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return {"w": w.detach(), "b": b.detach()}


def predict_manual(params, X):
    with torch.no_grad():
        logits = torch.tensor(X, dtype=torch.float32) @ params["w"] + params["b"]
        return torch.sigmoid(logits).numpy()


class LogisticModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)   # logits; sigmoid lives in the loss


def fit_standard_torch(X, y, lam, lr, epochs=STANDARD_EPOCHS):
    """Standard workflow: nn.Module, DataLoader mini-batches, Adam, BCEWithLogitsLoss."""
    gen = torch.Generator().manual_seed(SEED)
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                        torch.tensor(y, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
    model = LogisticModel(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Decay the step size so mini-batch noise dies out and Adam settles on the
    # optimum instead of bouncing around it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            # Explicit L2 on the weights only (not optimizer weight_decay, which
            # would also hit the bias and, with Adam, is not plain L2).
            loss = loss_fn(model(xb), yb) + lam / 2 * (model.linear.weight ** 2).sum()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def predict_standard(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def best_threshold(y, p):
    """Threshold that minimises the business cost on the given (validation) data."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [business_cost(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "offers_sent": int(pred.sum()),
        "business_cost": business_cost(y, pred),
        "cost_per_customer": business_cost(y, pred) / len(y),
    }


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    raw = load_raw()
    feats = engineer(raw)
    y_all = target(raw)
    idx_train, idx_val, idx_test = split_data(raw)

    pre = make_preprocessor()
    X_train = pre.fit_transform(feats.iloc[idx_train]).astype(np.float64)
    X_val = pre.transform(feats.iloc[idx_val]).astype(np.float64)
    X_test = pre.transform(feats.iloc[idx_test]).astype(np.float64)
    y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]
    feature_names = list(pre.get_feature_names_out())
    n = len(y_train)
    print(f"train={n}  val={len(y_val)}  test={len(y_test)}  features={len(feature_names)}")
    print(f"churn rate train/val/test: {y_train.mean():.3f} / {y_val.mean():.3f} / {y_test.mean():.3f}")

    # ---- 1. scikit-learn: tune C on validation log loss -------------------
    tuning = {"C": {}, "manual_lr": {}, "standard_lr": {}}
    for C in C_GRID:
        m = fit_sklearn(X_train, y_train, C)
        tuning["C"][C] = log_loss(y_val, m.predict_proba(X_val)[:, 1])
    best_C = min(tuning["C"], key=tuning["C"].get)
    lam = 1.0 / (best_C * n)
    sk_model = fit_sklearn(X_train, y_train, best_C)
    print(f"best C = {best_C}  (lambda = {lam:.2e})")

    # ---- 2. manual PyTorch: same C, tune only the learning rate -----------
    for lr in MANUAL_LR_GRID:
        params = fit_manual_torch(X_train, y_train, lam, lr)
        tuning["manual_lr"][lr] = log_loss(y_val, predict_manual(params, X_val))
    best_manual_lr = min(tuning["manual_lr"], key=tuning["manual_lr"].get)
    manual_params = fit_manual_torch(X_train, y_train, lam, best_manual_lr)
    print(f"manual loop best lr = {best_manual_lr}")

    # ---- 3. standard PyTorch: same C, tune only the learning rate ---------
    for lr in STANDARD_LR_GRID:
        torch.manual_seed(SEED)
        model = fit_standard_torch(X_train, y_train, lam, lr)
        tuning["standard_lr"][lr] = log_loss(y_val, predict_standard(model, X_val))
    best_standard_lr = min(tuning["standard_lr"], key=tuning["standard_lr"].get)
    torch.manual_seed(SEED)
    std_model = fit_standard_torch(X_train, y_train, lam, best_standard_lr)
    print(f"standard workflow best lr = {best_standard_lr}")

    # ---- Predictions ------------------------------------------------------
    prevalence = y_train.mean()
    val_probs = {
        "sklearn": sk_model.predict_proba(X_val)[:, 1],
        "manual_torch": predict_manual(manual_params, X_val),
        "standard_torch": predict_standard(std_model, X_val),
        "baseline": np.full(len(y_val), prevalence),
    }
    test_probs = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "manual_torch": predict_manual(manual_params, X_test),
        "standard_torch": predict_standard(std_model, X_test),
        "baseline": np.full(len(y_test), prevalence),
    }

    # Naive baseline: predict the training churn rate for everyone. Its
    # cost-optimal "threshold" is either contact nobody or contact everybody.
    # A second, slightly smarter rule-of-thumb baseline: contact every
    # month-to-month customer (what a retention team would do with no model).
    mtm_test = (raw.iloc[idx_test]["Contract"] == "Month-to-month").astype(float).to_numpy()

    metrics = {}
    for name in ["sklearn", "manual_torch", "standard_torch"]:
        t = best_threshold(y_val, val_probs[name])
        metrics[name] = evaluate(y_test, test_probs[name], t)
    # Baseline thresholds: 0.0 -> contact all, 1.0 -> contact none; pick cheaper on val.
    base_t = 0.0 if business_cost(y_val, np.ones_like(y_val)) < business_cost(y_val, np.zeros_like(y_val)) else 1.0
    metrics["baseline"] = evaluate(y_test, test_probs["baseline"], base_t)
    metrics["baseline_contact_none"] = evaluate(y_test, test_probs["baseline"], 1.0)
    metrics["rule_month_to_month"] = evaluate(y_test, mtm_test, 0.5)

    # ---- Agreement between the three implementations ----------------------
    w_sk = sk_model.coef_.ravel()
    w_man = manual_params["w"].numpy()
    w_std = std_model.linear.weight.detach().numpy().ravel()
    agreement = {
        "max_abs_prob_diff_sklearn_vs_manual": float(np.abs(test_probs["sklearn"] - test_probs["manual_torch"]).max()),
        "max_abs_prob_diff_sklearn_vs_standard": float(np.abs(test_probs["sklearn"] - test_probs["standard_torch"]).max()),
        "max_abs_coef_diff_sklearn_vs_manual": float(np.abs(w_sk - w_man).max()),
        "max_abs_coef_diff_sklearn_vs_standard": float(np.abs(w_sk - w_std).max()),
        "coef_corr_sklearn_vs_manual": float(np.corrcoef(w_sk, w_man)[0, 1]),
        "coef_corr_sklearn_vs_standard": float(np.corrcoef(w_sk, w_std)[0, 1]),
        "intercepts": {
            "sklearn": float(sk_model.intercept_[0]),
            "manual_torch": float(manual_params["b"][0]),
            "standard_torch": float(std_model.linear.bias.detach()[0]),
        },
    }

    # Train-set objective value each implementation reached (same objective).
    def objective(p, w):
        return float(log_loss(y_train, np.clip(p, 1e-7, 1 - 1e-7)) + lam / 2 * (w ** 2).sum())
    agreement["train_objective"] = {
        "sklearn": objective(sk_model.predict_proba(X_train)[:, 1], w_sk),
        "manual_torch": objective(predict_manual(manual_params, X_train), w_man),
        "standard_torch": objective(predict_standard(std_model, X_train), w_std),
    }

    # ---- Report to console ------------------------------------------------
    table = pd.DataFrame(metrics).T[["roc_auc", "pr_auc", "log_loss", "threshold", "precision",
                                     "recall", "f1", "offers_sent", "business_cost", "cost_per_customer"]]
    pd.set_option("display.width", 200)
    print("\nTest-set results (threshold chosen on validation by business cost):")
    print(table.round(4).to_string())
    print("\nAgreement:", json.dumps(agreement, indent=2))

    coefs = pd.DataFrame({"feature": feature_names, "sklearn": w_sk,
                          "manual_torch": w_man, "standard_torch": w_std})
    print("\nCoefficients (sorted by sklearn):")
    print(coefs.sort_values("sklearn").round(3).to_string(index=False))

    # ---- Save artifacts ---------------------------------------------------
    with open(ARTIFACTS / "preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)
    with open(ARTIFACTS / "sklearn_model.pkl", "wb") as f:
        pickle.dump(sk_model, f)
    torch.save(manual_params, ARTIFACTS / "manual_torch.pt")
    torch.save(std_model.state_dict(), ARTIFACTS / "standard_torch.pt")

    split = np.empty(len(raw), dtype=object)
    split[idx_train], split[idx_val], split[idx_test] = "train", "val", "test"
    pd.DataFrame({"customerID": raw["customerID"], "split": split}).to_csv(ARTIFACTS / "split.csv", index=False)
    coefs.to_csv(ARTIFACTS / "coefficients.csv", index=False)

    summary = {
        "best_C": best_C, "lambda": lam,
        "best_manual_lr": best_manual_lr, "best_standard_lr": best_standard_lr,
        "manual_epochs": MANUAL_EPOCHS, "standard_epochs": STANDARD_EPOCHS, "batch_size": BATCH_SIZE,
        "offer_cost": OFFER_COST, "churn_cost": CHURN_COST,
        "train_prevalence": float(prevalence),
        "n_features": len(feature_names), "feature_names": feature_names,
        "tuning_val_log_loss": {k: {str(kk): vv for kk, vv in v.items()} for k, v in tuning.items()},
        "test_metrics": metrics, "agreement": agreement,
    }
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved artifacts to {ARTIFACTS}")


if __name__ == "__main__":
    main()
