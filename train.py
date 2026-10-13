"""
Assignment 1 -- training pipeline (run with: uv run python main.py train).

Trains the SAME model -- L2-regularised logistic regression -- three ways on
the SAME split, compares them against a naive baseline, and saves everything
the dashboard needs into artifacts/. This is the only stage that trains
anything; app.py only loads what is saved here.

    load -> engineer features -> stratified split -> fit preprocessor on train
         -> sklearn | manual PyTorch loop | nn.Module + optim
         -> choose threshold on validation -> evaluate once on test -> save

Why the three can be compared at all: scikit-learn minimises
    (1/2)||w||^2 + C * sum_i logloss_i
Dividing by C*n gives
    mean_i logloss_i + (lam/2)||w||^2     with   lam = 1 / (C * n_train)
and the intercept is NOT penalised. Both PyTorch versions minimise exactly that
second form, so all three are solving one optimisation problem.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import common as cm
from features import RAW_TARGET, TARGET, engineer_features, split_columns

# --- Tuning grids (the only hyperparameters the assignment asks you to touch) --
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # regularisation strength (sklearn's C)
C_TOLERANCE = 1e-4  # validation log-loss differences below this are treated as ties
MANUAL_LR_GRID = [0.1, 0.3, 1.0]
MANUAL_EPOCHS = 5000  # full-batch gradient descent steps
NN_LR_GRID = [0.003, 0.01, 0.03]
NN_EPOCHS = 60
NN_BATCH_SIZE = 512


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


# ------------------------------------------------------------------ data ----
def load_and_split():
    raw = pd.read_csv(cm.DATA_PATH)
    frame = engineer_features(raw)
    y = frame[TARGET].to_numpy().astype(int)

    # Stratified random split. Rows are independent clients (ID is unique) and
    # the dataset is a single snapshot with no time ordering between rows, so
    # a time-based split is not possible here -- see REPORT.md.
    idx = np.arange(len(frame))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=cm.TEST_SIZE, stratify=y, random_state=cm.SEED
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=cm.VAL_SIZE / (1 - cm.TEST_SIZE),
        stratify=y[trainval_idx],
        random_state=cm.SEED,
    )
    split = np.empty(len(frame), dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    frame = frame.assign(split=split)
    return frame, train_idx, val_idx, test_idx


def build_preprocessor(train_feats: pd.DataFrame) -> ColumnTransformer:
    """Scaler for numbers, one-hot for categories -- built from TRAIN rows only.

    One category per variable is dropped (the most frequent one in train).
    Without that, the dummies of a variable sum to 1 and are perfectly
    collinear with the intercept: with (almost) no regularisation the problem
    then has infinitely many solutions, and three correct optimisers can land
    on different coefficients that give the same predictions. (We hit exactly
    this on the first run.) The dropped category becomes the reference level.
    """
    nums, cats = split_columns(train_feats.assign(**{TARGET: 0}))
    transformers = [("num", StandardScaler(), nums)]
    if cats:
        reference = [train_feats[c].mode().iloc[0] for c in cats]
        transformers.append(
            ("cat", OneHotEncoder(drop=reference, handle_unknown="ignore", sparse_output=False), cats)
        )
    return ColumnTransformer(transformers)


# ----------------------------------------------------- the three trainers ----
def objective(w, b, X, y, lam) -> float:
    """Regularised training loss: mean logloss + (lam/2)||w||^2 (bias unpenalised)."""
    z = X @ w + b
    data_loss = (F.softplus(z) - y * z).mean()
    return float(data_loss + 0.5 * lam * (w**2).sum())


def train_sklearn(X, y, C):
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X, y)
    return model


def train_manual(X, y, lam, lr, epochs):
    """From-scratch loop: raw tensors, autograd for the gradient, hand-written update."""
    d = X.shape[1]
    w = torch.zeros(d, 1, dtype=cm.DTYPE, requires_grad=True)
    b = torch.zeros(1, dtype=cm.DTYPE, requires_grad=True)
    y_col = y.reshape(-1, 1)
    for _ in range(epochs):
        z = X @ w + b
        # logloss written out by hand: softplus(z) - y*z == -[y log s(z) + (1-y) log(1-s(z))]
        loss = (F.softplus(z) - y_col * z).mean() + 0.5 * lam * (w**2).sum()
        loss.backward()
        with torch.no_grad():  # the manual gradient-step update
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return w.detach(), b.detach()


def train_nn(X, y, lam, lr, epochs, batch_size):
    """Standard workflow: nn.Module + loss object + optimiser + DataLoader."""
    torch.manual_seed(cm.SEED)
    model = cm.LogisticRegressionNN(X.shape[1]).to(cm.DTYPE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # L2 penalty (lam/2)||w||^2 == weight_decay=lam on the weights only.
    # The bias must be excluded to match sklearn, hence two parameter groups.
    optimizer = torch.optim.Adam(
        [
            {"params": [model.linear.weight], "weight_decay": lam},
            {"params": [model.linear.bias], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    loader = DataLoader(
        TensorDataset(X, y.reshape(-1, 1)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cm.SEED),
    )
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def torch_proba(w, b, X):
    return torch.sigmoid(X @ w + b).numpy().ravel()


def nn_proba(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy().ravel()


# ------------------------------------------------------------ evaluation ----
def evaluate(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = cm.confusion_counts(y, pred)
    cost = fn * cm.COST_FN + fp * cm.COST_FP
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "log_loss": float(log_loss(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "cost_total": float(cost),
        "cost_per_client": float(cost / len(y)),
    }


def evaluate_baseline(y_train, y_test):
    """Naive baseline: every client gets the training default rate as probability,
    and the policy is the cheapest constant one (approve all / reject all)."""
    p = np.full(len(y_test), y_train.mean())
    cost = cm.naive_cost(y_test)
    approve_all = cost == int(y_test.sum()) * cm.COST_FN
    return {
        "threshold": None,
        "policy": "approve everyone" if approve_all else "reject everyone",
        "roc_auc": 0.5,
        "pr_auc": float(y_test.mean()),
        "log_loss": float(log_loss(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "accuracy": float(1 - y_test.mean()) if approve_all else float(y_test.mean()),
        "cost_total": float(cost),
        "cost_per_client": float(cost / len(y_test)),
    }


def val_logloss(proba, y):
    return float(log_loss(y, np.clip(proba, 1e-12, 1 - 1e-12)))


def results_markdown(results: dict) -> str:
    """Three-method comparison table (test set) as Markdown, ready to paste into REPORT.md."""
    head = ("| Model | Threshold | ROC AUC | PR AUC | Log loss | Recall | Precision | Cost per client |\n"
            "|---|---|---|---|---|---|---|---|")
    rows = []
    for name, r in results.items():
        t = r["test"]
        if t["threshold"] is None:  # naive baseline: constant policy, no threshold
            rows.append(f"| {name} ({t['policy']}) | - | {t['roc_auc']:.4f} | {t['pr_auc']:.4f} | "
                        f"{t['log_loss']:.4f} | - | - | {t['cost_per_client']:.4f} |")
        else:
            rows.append(f"| {name} | {t['threshold']:.2f} | {t['roc_auc']:.4f} | {t['pr_auc']:.4f} | "
                        f"{t['log_loss']:.4f} | {t['recall']:.3f} | {t['precision']:.3f} | "
                        f"{t['cost_per_client']:.4f} |")
    return "\n".join([head] + rows) + "\n"


# ------------------------------------------------------------------ main ----
def main():
    t0 = time.time()
    torch.manual_seed(cm.SEED)
    np.random.seed(cm.SEED)
    cm.ARTIFACTS.mkdir(exist_ok=True)

    banner("Data and split")
    frame, tr, va, te = load_and_split()
    y_all = frame[TARGET].to_numpy().astype(int)
    print(f"rows: train {len(tr)} | val {len(va)} | test {len(te)} | "
          f"default rate: train {y_all[tr].mean():.3f}, val {y_all[va].mean():.3f}, test {y_all[te].mean():.3f}")

    feats = frame.drop(columns=[TARGET, "split"])
    pre = build_preprocessor(feats.iloc[tr])
    X_tr_np = pre.fit_transform(feats.iloc[tr])  # fitted on TRAIN only
    X_va_np = pre.transform(feats.iloc[va])
    X_te_np = pre.transform(feats.iloc[te])
    feature_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    print(f"model inputs after preprocessing: {X_tr_np.shape[1]} columns")

    y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]
    X_tr, X_va, X_te = (torch.tensor(a, dtype=cm.DTYPE) for a in (X_tr_np, X_va_np, X_te_np))
    yt_tr = torch.tensor(y_tr, dtype=cm.DTYPE)
    n_train = len(tr)

    # ---- 1. scikit-learn ----------------------------------------------------
    banner("1/3 scikit-learn: tune C on validation")
    sk_scores = {}
    for C in C_GRID:
        m = train_sklearn(X_tr_np, y_tr, C)
        sk_scores[C] = val_logloss(m.predict_proba(X_va_np)[:, 1], y_va)
        print(f"  C={C:<8g} val log loss {sk_scores[C]:.5f}")
    # Several C values can be indistinguishable on validation (here C>=1 give the same
    # log loss to 5 decimals). Taking the plain argmin would pick one of them by
    # accident, so take the SMALLEST C (strongest regularisation) within C_TOLERANCE
    # of the best: same accuracy, better-conditioned problem.
    best_loss = min(sk_scores.values())
    best_C = min(c for c in C_GRID if sk_scores[c] <= best_loss + C_TOLERANCE)
    sk = train_sklearn(X_tr_np, y_tr, best_C)
    lam = 1.0 / (best_C * n_train)
    print(f"  -> C={best_C:g}  (equivalent lam = 1/(C*n) = {lam:.3e})")

    # ---- 2. manual PyTorch loop ---------------------------------------------
    banner("2/3 manual PyTorch loop: tune learning rate on validation")
    manual_scores, manual_fits = {}, {}
    for lr in MANUAL_LR_GRID:
        w, b = train_manual(X_tr, yt_tr, lam, lr, MANUAL_EPOCHS)
        manual_fits[lr] = (w, b)
        manual_scores[lr] = val_logloss(torch_proba(w, b, X_va), y_va)
        print(f"  lr={lr:<6g} val log loss {manual_scores[lr]:.5f}")
    best_lr_manual = min(manual_scores, key=manual_scores.get)
    w_man, b_man = manual_fits[best_lr_manual]
    print(f"  -> lr={best_lr_manual:g} ({MANUAL_EPOCHS} full-batch steps)")

    # ---- 3. standard PyTorch workflow ---------------------------------------
    banner("3/3 nn.Module + optim: tune learning rate on validation")
    nn_scores, nn_fits = {}, {}
    for lr in NN_LR_GRID:
        model = train_nn(X_tr, yt_tr, lam, lr, NN_EPOCHS, NN_BATCH_SIZE)
        nn_fits[lr] = model
        nn_scores[lr] = val_logloss(nn_proba(model, X_va), y_va)
        print(f"  lr={lr:<6g} val log loss {nn_scores[lr]:.5f}")
    best_lr_nn = min(nn_scores, key=nn_scores.get)
    nn_model = nn_fits[best_lr_nn]
    print(f"  -> lr={best_lr_nn:g} (Adam, batch {NN_BATCH_SIZE}, {NN_EPOCHS} epochs)")

    # ---- probabilities -------------------------------------------------------
    probs = {
        "scikit-learn": {
            "val": sk.predict_proba(X_va_np)[:, 1], "test": sk.predict_proba(X_te_np)[:, 1],
        },
        "manual PyTorch": {
            "val": torch_proba(w_man, b_man, X_va), "test": torch_proba(w_man, b_man, X_te),
        },
        "nn.Module + optim": {
            "val": nn_proba(nn_model, X_va), "test": nn_proba(nn_model, X_te),
        },
    }

    # ---- threshold chosen on VALIDATION, reported on TEST --------------------
    banner("Threshold (min business cost on validation) and test results")
    results = {}
    for name, p in probs.items():
        thr = cm.best_threshold(y_va, p["val"])
        results[name] = {"test": evaluate(y_te, p["test"], thr), "val": evaluate(y_va, p["val"], thr)}
    results["naive baseline"] = {"test": evaluate_baseline(y_tr, y_te)}

    cols = ["threshold", "roc_auc", "pr_auc", "log_loss", "recall", "precision", "cost_per_client"]
    table = pd.DataFrame({k: {c: v["test"].get(c) for c in cols} for k, v in results.items()}).T
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- agreement diagnostics (the 'do the three agree?' question) ----------
    banner("Do the three agree?")
    coef = {
        "scikit-learn": sk.coef_.ravel(),
        "manual PyTorch": w_man.numpy().ravel(),
        "nn.Module + optim": nn_model.linear.weight.detach().numpy().ravel(),
    }
    bias = {
        "scikit-learn": float(sk.intercept_[0]),
        "manual PyTorch": float(b_man.item()),
        "nn.Module + optim": float(nn_model.linear.bias.item()),
    }
    obj = {
        name: objective(torch.tensor(coef[name].reshape(-1, 1)), torch.tensor(bias[name], dtype=cm.DTYPE),
                        X_tr, yt_tr.reshape(-1, 1), lam)
        for name in coef
    }
    ref = "scikit-learn"
    agreement = {}
    for name in coef:
        agreement[name] = {
            "objective": obj[name],
            "objective_gap_vs_sklearn": obj[name] - obj[ref],
            "max_abs_coef_diff_vs_sklearn": float(np.abs(coef[name] - coef[ref]).max()),
            "max_abs_test_proba_diff_vs_sklearn": float(
                np.abs(probs[name]["test"] - probs[ref]["test"]).max()
            ),
            "test_proba_correlation_vs_sklearn": float(
                np.corrcoef(probs[name]["test"], probs[ref]["test"])[0, 1]
            ),
        }
        a = agreement[name]
        print(f"  {name:<18} objective {a['objective']:.6f} (gap {a['objective_gap_vs_sklearn']:+.2e}) | "
              f"max|dcoef| {a['max_abs_coef_diff_vs_sklearn']:.4f} | "
              f"max|dproba| {a['max_abs_test_proba_diff_vs_sklearn']:.4f}")

    # ---- save artifacts ------------------------------------------------------
    banner("Saving artifacts")
    joblib.dump(pre, cm.ARTIFACTS / "preprocessor.joblib")
    joblib.dump(sk, cm.ARTIFACTS / "sklearn_logreg.joblib")
    torch.save({"w": w_man, "b": b_man}, cm.ARTIFACTS / "manual_torch.pt")
    torch.save(
        {"state_dict": nn_model.state_dict(), "in_features": X_tr.shape[1]},
        cm.ARTIFACTS / "nn_torch.pt",
    )
    frame.to_csv(cm.ARTIFACTS / "dataset_engineered.csv.gz", index=False)
    pd.DataFrame({"feature": feature_names, **{k: v for k, v in coef.items()}}).to_csv(
        cm.ARTIFACTS / "coefficients.csv", index=False
    )
    config = {
        "seed": cm.SEED,
        "split_sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "raw_target": RAW_TARGET,
        "cost_fn": cm.COST_FN,
        "cost_fp": cm.COST_FP,
        "best_C": best_C,
        "lambda": lam,
        "manual_lr": best_lr_manual,
        "manual_epochs": MANUAL_EPOCHS,
        "nn_lr": best_lr_nn,
        "nn_epochs": NN_EPOCHS,
        "nn_batch_size": NN_BATCH_SIZE,
        "thresholds": {k: results[k]["test"]["threshold"] for k in probs},
        "feature_names": feature_names,
        "sweeps": {
            "sklearn_C_val_logloss": {str(k): v for k, v in sk_scores.items()},
            "manual_lr_val_logloss": {str(k): v for k, v in manual_scores.items()},
            "nn_lr_val_logloss": {str(k): v for k, v in nn_scores.items()},
        },
    }
    (cm.ARTIFACTS / "config.json").write_text(json.dumps(config, indent=2))
    (cm.ARTIFACTS / "metrics.json").write_text(
        json.dumps({"results": results, "agreement": agreement, "bias": bias}, indent=2)
    )
    (cm.ARTIFACTS / "results_table.md").write_text(results_markdown(results))
    print(f"saved to {cm.ARTIFACTS}/ in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
