"""
Assignment 1 — training pipeline.

    uv run python main.py train

Loads the committed sample, builds features (common.py), splits
train/validation/test, trains the SAME regularized logistic regression three
ways, compares them with naive baselines, and saves everything app.py needs.
Runs on CPU in under a minute; no GPU needed.
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from common import (ARTIFACTS, DATA_PATH, SEED, TARGET, LogisticRegressionNet,
                    build_preprocessor, column_types, make_xy, sigmoid)

# ---------------------------------------------------------------------------
# Settings YOU own. Each one needs a sentence of justification in REPORT.md.
# ---------------------------------------------------------------------------
# Business costs: what does one missed case (false negative) cost relative to
# one unnecessary flag (false positive)? Together they decide the threshold and
# the dashboard's cost number.
#
# A missed case costs 8.1x an unnecessary flag ($2,436 vs $300). Only the RATIO
# moves the threshold; the absolute euros just scale cost_per_1000, which is why
# the numbers are kept in real dollars (the reported cost is then readable as
# money, not as an index).
COST_FN = 2436.0
COST_FP = 300.0

# Regularization: sklearn's C is the INVERSE strength (smaller C = stronger
# penalty). A small sweep on the validation set is the only tuning we do.
# Extended downward after the first run picked 0.01, the smallest value on the
# original grid — a winner at the edge means the grid never looked where the
# optimum actually is. Check this again if your feature set changes a lot.
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# Gradient-descent settings for both PyTorch versions (full batch).
LR = 0.5
EPOCHS = 3000

TEST_SIZE = 0.20   # fraction of all rows
VAL_SIZE = 0.20    # fraction of all rows

# float64 so any gap between the three methods reflects optimization, not
# float32 rounding.
DTYPE = torch.float64
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Data and split
# ---------------------------------------------------------------------------
def load_and_split():
    raw = pd.read_csv(DATA_PATH)
    raw = raw.dropna(subset=[TARGET])
    # No date column in this survey, so a random split can't leak the future.
    # Stratifying keeps the rare "Yes" share identical across the three parts.
    train_val, test = train_test_split(
        raw, test_size=TEST_SIZE, stratify=raw[TARGET], random_state=SEED)
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val[TARGET], random_state=SEED)
    return train, val, test


# ---------------------------------------------------------------------------
# 2. The three implementations of one model
# ---------------------------------------------------------------------------
# All three minimize the SAME objective:
#     mean binary cross-entropy  +  (lam / 2) * ||w||^2      (bias not penalized)
# sklearn writes it as  0.5*||w||^2 + C * SUM(cross-entropy). Dividing that by
# C*n gives the line above with  lam = 1 / (C * n).  Getting this conversion
# right is what makes the three agree.

def penalty_strength(C: float, n: int) -> float:
    return 1.0 / (C * n)


def train_sklearn(X, y, C):
    model = LogisticRegression(C=C, max_iter=10_000, tol=1e-10)
    model.fit(X, y)
    return model


def train_manual_torch(X, y, C):
    """Session-5 style: raw tensors, autograd for gradients, manual update."""
    Xt = torch.tensor(X, dtype=DTYPE)
    yt = torch.tensor(y, dtype=DTYPE)
    n, d = Xt.shape
    lam = penalty_strength(C, n)

    w = torch.zeros(d, dtype=DTYPE, requires_grad=True)
    b = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    history = []

    for _ in range(EPOCHS):
        z = Xt @ w + b                                            # logits
        p = (1 / (1 + torch.exp(-z))).clamp(1e-12, 1 - 1e-12)     # sigmoid
        bce = -(yt * torch.log(p) + (1 - yt) * torch.log(1 - p)).mean()
        loss = bce + 0.5 * lam * (w ** 2).sum()

        loss.backward()                  # autograd fills w.grad and b.grad
        with torch.no_grad():            # the update itself must not be tracked
            w -= LR * w.grad
            b -= LR * b.grad
        w.grad.zero_()                   # gradients accumulate unless reset
        b.grad.zero_()
        history.append(loss.item())

    return {"w": w.detach(), "b": b.detach()}, history


def train_module_torch(X, y, C):
    """Standard workflow: nn.Module + loss object + torch.optim optimizer."""
    Xt = torch.tensor(X, dtype=DTYPE)
    yt = torch.tensor(y, dtype=DTYPE)
    n, d = Xt.shape
    lam = penalty_strength(C, n)

    model = LogisticRegressionNet(d).to(DTYPE)
    # Same zero start as the manual loop, so any difference is the API, not luck.
    nn.init.zeros_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)

    # SGD's weight_decay=lam adds lam*w to the gradient, which is exactly the
    # gradient of (lam/2)*||w||^2. Applied to the weights only, like sklearn.
    optimizer = torch.optim.SGD(
        [{"params": [model.linear.weight], "weight_decay": lam},
         {"params": [model.linear.bias], "weight_decay": 0.0}],
        lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()     # sigmoid + cross-entropy, numerically stable
    history = []

    # Full batch (whole training set every step) to match the manual loop.
    # Mini-batches via DataLoader would also work but add noise to the comparison.
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()
        with torch.no_grad():  # log the full objective so curves are comparable
            history.append(loss.item() + 0.5 * lam * model.linear.weight.pow(2).sum().item())

    return model, history


# Prediction helpers: each returns P(y=1) as a numpy array.
def proba_sklearn(model, X):
    return model.predict_proba(X)[:, 1]


def proba_manual(params, X):
    return sigmoid(X @ params["w"].numpy() + params["b"].numpy())


def proba_module(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=DTYPE))).numpy()


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------
def business_cost(y, flagged):
    fn = int(((flagged == 0) & (y == 1)).sum())
    fp = int(((flagged == 1) & (y == 0)).sum())
    return COST_FN * fn + COST_FP * fp


def best_threshold(y, p):
    """Threshold minimizing business cost on the given (validation) data."""
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = [business_cost(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(costs))])


def objective(p, y, w, lam):
    """The shared objective, evaluated for any of the three fitted models."""
    return log_loss(y, np.clip(p, 1e-12, 1 - 1e-12)) + 0.5 * lam * float(np.sum(w ** 2))


def evaluate(name, p_test, y_test, threshold):
    flagged = (p_test >= threshold).astype(int)
    tp = int(((flagged == 1) & (y_test == 1)).sum())
    constant = np.ptp(p_test) == 0
    return {
        "model": name,
        "roc_auc": 0.5 if constant else roc_auc_score(y_test, p_test),
        "pr_auc": float(y_test.mean()) if constant else average_precision_score(y_test, p_test),
        "log_loss": log_loss(y_test, np.clip(p_test, 1e-12, 1 - 1e-12)),
        "brier": brier_score_loss(y_test, p_test),
        "threshold": threshold,
        "precision": tp / max(int(flagged.sum()), 1),
        "recall": tp / max(int(y_test.sum()), 1),
        "flag_rate": float(flagged.mean()),
        "cost_per_1000": 1000 * business_cost(y_test, flagged) / len(y_test),
    }


def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [f"{v:.4f}" for v in row]) + " |"
            for i, row in zip(df.index, df.to_numpy())]
    return "\n".join([header, sep, *body]) + "\n"


def main():
    ARTIFACTS.mkdir(exist_ok=True)

    # --- data -------------------------------------------------------------
    train, val, test = load_and_split()
    Xtr_df, y_train = make_xy(train)
    Xva_df, y_val = make_xy(val)
    Xte_df, y_test = make_xy(test)

    numeric, categorical = column_types(Xtr_df)
    pre = build_preprocessor(numeric, categorical).fit(Xtr_df)   # fit on TRAIN only
    X_train, X_val, X_test = (pre.transform(d) for d in (Xtr_df, Xva_df, Xte_df))
    feature_names = list(pre.named_steps["encode"].get_feature_names_out())
    n = len(y_train)

    print(f"Rows train/val/test: {len(y_train):,} / {len(y_val):,} / {len(y_test):,}")
    print(f"Positive rate train/val/test: "
          f"{y_train.mean():.2%} / {y_val.mean():.2%} / {y_test.mean():.2%}")
    print(f"Features after encoding: {X_train.shape[1]} "
          f"(from {len(numeric)} numeric + {len(categorical)} categorical columns)")

    # --- choose C on validation (sklearn is fast and exact, so sweep with it)
    print("\nC sweep (validation PR-AUC):")
    sweep = {}
    for C in C_GRID:
        p_val = proba_sklearn(train_sklearn(X_train, y_train, C), X_val)
        sweep[C] = float(average_precision_score(y_val, p_val))
        print(f"  C={C:<6} PR-AUC={sweep[C]:.4f}")
    C = max(sweep, key=sweep.get)
    lam = penalty_strength(C, n)
    print(f"Chosen C={C}  ->  lam = 1/(C*n) = {lam:.2e}")

    # --- train the three versions on the same data -------------------------
    sk = train_sklearn(X_train, y_train, C)
    manual, hist_manual = train_manual_torch(X_train, y_train, C)
    module, hist_module = train_module_torch(X_train, y_train, C)

    models = {
        "sklearn": lambda X: proba_sklearn(sk, X),
        "torch_manual": lambda X: proba_manual(manual, X),
        "torch_module": lambda X: proba_module(module, X),
    }
    coefs = {
        "sklearn": sk.coef_.ravel(),
        "torch_manual": manual["w"].numpy(),
        "torch_module": module.linear.weight.detach().numpy().ravel(),
    }
    intercepts = {
        "sklearn": float(sk.intercept_[0]),
        "torch_manual": float(manual["b"].item()),
        "torch_module": float(module.linear.bias.item()),
    }

    # --- evaluate: thresholds picked on VALIDATION, scores reported on TEST
    prevalence = float(y_train.mean())
    constant = np.full(len(y_test), prevalence)
    rows = [evaluate("baseline_flag_nobody", constant, y_test, 1.01),
            evaluate("baseline_flag_everyone", constant, y_test, 0.0)]
    thresholds = {}
    for name, predict in models.items():
        thresholds[name] = best_threshold(y_val, predict(X_val))
        rows.append(evaluate(name, predict(X_test), y_test, thresholds[name]))
    results = pd.DataFrame(rows).set_index("model")

    # --- how closely do the three agree? -----------------------------------
    p_test = {k: f(X_test) for k, f in models.items()}
    final_objective = {k: objective(models[k](X_train), y_train, coefs[k], lam) for k in models}
    pairs = [("sklearn", "torch_manual"), ("sklearn", "torch_module"),
             ("torch_manual", "torch_module")]
    agreement = {
        "max_abs_prob_diff": {f"{a} vs {b}": float(np.abs(p_test[a] - p_test[b]).max())
                              for a, b in pairs},
        "max_abs_coef_diff": {f"{a} vs {b}": float(np.abs(coefs[a] - coefs[b]).max())
                              for a, b in pairs},
        "final_train_objective": final_objective,
    }

    pd.set_option("display.width", 160)
    print("\nTest-set results (thresholds chosen on validation):")
    print(results.round(4).to_string())
    print(f"\nCost-optimal threshold IF probabilities were perfectly calibrated: "
          f"COST_FP/(COST_FP+COST_FN) = {COST_FP / (COST_FP + COST_FN):.3f}")
    print("\nAgreement between implementations:")
    print(json.dumps(agreement, indent=2))
    obj_gap = final_objective["torch_manual"] - final_objective["sklearn"]
    coef_gap = agreement["max_abs_coef_diff"]["sklearn vs torch_manual"]
    if coef_gap > 1e-3:
        # A tiny objective gap with a visible coefficient gap means gradient
        # descent is crawling along a flat valley of nearly equivalent solutions
        # (correlated features). Worth investigating and explaining in REPORT.md.
        print(f"\nNOTE: PyTorch is {obj_gap:.1e} above sklearn's objective but its "
              f"coefficients differ by up to {coef_gap:.3f}. Gradient descent hasn't "
              f"fully converged: look at which features differ, then try more EPOCHS "
              f"or a larger LR.")

    # --- save everything the app needs (it must never retrain) -------------
    joblib.dump(pre, ARTIFACTS / "preprocessor.joblib")
    joblib.dump(sk, ARTIFACTS / "sklearn_logreg.joblib")
    torch.save(manual, ARTIFACTS / "torch_manual.pt")
    torch.save(module.state_dict(), ARTIFACTS / "torch_module.pt")
    test.to_csv(ARTIFACTS / "test_raw.csv", index=False)  # raw rows: app reruns the pipeline

    meta = {
        "C": C, "lam": lam, "lr": LR, "epochs": EPOCHS,
        "c_sweep": {str(k): v for k, v in sweep.items()},
        "cost_fn": COST_FN, "cost_fp": COST_FP,
        "n_features": int(X_train.shape[1]), "feature_names": feature_names,
        "thresholds": thresholds,
        "coefs": {k: v.tolist() for k, v in coefs.items()},
        "intercepts": intercepts,
        "results": results.reset_index().to_dict(orient="records"),
        "agreement": agreement,
        "history": {"torch_manual": hist_manual, "torch_module": hist_module,
                    "sklearn_final": final_objective["sklearn"]},
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))

    # A table you can paste into REPORT.md (the interpretation is yours to write).
    cols = ["roc_auc", "pr_auc", "log_loss", "threshold", "precision", "recall", "cost_per_1000"]
    (ARTIFACTS / "results_table.md").write_text(to_markdown(results[cols]))
    print(f"\nSaved artifacts to {ARTIFACTS.name}/")


if __name__ == "__main__":
    main()
