"""
Training pipeline: diabetes screening prioritisation (NY State, BRFSS 2022).

Run with:  uv run python main.py train

1. Load data, define the target, stratified 80/20 train/test split.
2. Choose L2 strength C by 5-fold CV on the training set.
3. Ablation (train set only, 5-fold CV, same C): raw columns vs. engineered
   features vs. raw + "consequence" columns, to show what feature
   engineering buys.
4. Train the same L2-regularised logistic regression three ways:
   scikit-learn, a manual PyTorch loop, and nn.Module + torch.optim.
5. Pick the decision threshold from the 10% test-capacity constraint using
   out-of-fold training predictions (the test set is never used to tune).
6. Evaluate everything on the test set against naive baselines, with
   bootstrap confidence intervals, and audit by sex and race.
7. Save every artifact app.py needs, so the app never retrains.
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from torch import nn

from features import (
    ARTIFACTS, CAPACITY, COST_MISSED_CASE, COST_WASTED_TEST, SEED,
    build_preprocessor, build_raw_preprocessor, business_cost, engineer,
    load_raw, make_target, raw_features,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
ARTIFACTS.mkdir(exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Data and split
# ---------------------------------------------------------------------------
section("1. Data and split")
df = make_target(load_raw())
y = df["target"].to_numpy()
print(f"{len(df):,} respondents, diabetes prevalence {y.mean():.1%}")

# One survey year, no time dimension: a stratified random split is valid
# (no future to leak). Stratified so both sides keep the ~12.7% prevalence.
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.20, stratify=y, random_state=SEED
)
df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"train {len(train_idx):,} ({y_train.mean():.1%} positive) | test {len(test_idx):,} ({y_test.mean():.1%} positive)")

# ---------------------------------------------------------------------------
# 2. Regularisation strength (the only sklearn hyperparameter we tune)
# ---------------------------------------------------------------------------
section("2. Choose L2 strength C (5-fold CV on train, PR-AUC)")
X_train_df, X_test_df = engineer(df_train), engineer(df_test)
c_results = {}
for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
    pipe = make_pipeline(build_preprocessor(X_train_df), LogisticRegression(C=C, max_iter=5000))
    c_results[C] = float(cross_val_score(pipe, X_train_df, y_train, cv=CV, scoring="average_precision").mean())
    print(f"C={C:<6} PR-AUC {c_results[C]:.4f}")
# PR-AUC because the task is "find positives at the top of the ranking" on an
# imbalanced target. Within 0.001 of the best, prefer stronger regularisation
# (smaller C): same performance, simpler model.
best_score = max(c_results.values())
BEST_C = min(C for C, s in c_results.items() if s >= best_score - 0.001)
print(f"-> chosen C = {BEST_C}")

# ---------------------------------------------------------------------------
# 3. Ablation: what does feature engineering buy? (train set only)
# ---------------------------------------------------------------------------
section(f"3. Feature-set ablation (5-fold CV on train, sklearn, C={BEST_C})")
# Terms that exist only because the model is linear: squares, cut-offs and
# a max. Removing them shows what they add on top of the plain columns.
NONLINEAR_TERMS = ["age_sq", "bmi_sq", "obese", "short_sleep", "long_sleep",
                   "frequent_physical_distress", "frequent_mental_distress",
                   "cardiovascular_history"]
ablation = {}
for name, X_, prep in [
    ("raw columns", raw_features(df_train), build_raw_preprocessor),
    ("engineered features", engineer(df_train), build_preprocessor),
    ("engineered, minus non-linear terms", engineer(df_train).drop(columns=NONLINEAR_TERMS), build_preprocessor),
    ("raw + consequence columns (leaky)", raw_features(df_train, include_consequences=True), build_raw_preprocessor),
]:
    pipe = make_pipeline(prep(X_), LogisticRegression(C=BEST_C, max_iter=5000))
    auc = cross_val_score(pipe, X_, y_train, cv=CV, scoring="roc_auc")
    ap = cross_val_score(pipe, X_, y_train, cv=CV, scoring="average_precision")
    ablation[name] = {"roc_auc": float(auc.mean()), "roc_auc_sd": float(auc.std()),
                      "pr_auc": float(ap.mean()), "pr_auc_sd": float(ap.std())}
    print(f"{name:38s} ROC-AUC {auc.mean():.4f} ± {auc.std():.4f} | PR-AUC {ap.mean():.4f} ± {ap.std():.4f}")

# ---------------------------------------------------------------------------
# 4. Fit preprocessing on train only, then the three models
# ---------------------------------------------------------------------------
preprocessor = build_preprocessor(X_train_df).fit(X_train_df)
X_train = preprocessor.transform(X_train_df).astype(np.float32)
X_test = preprocessor.transform(X_test_df).astype(np.float32)
feature_names = list(preprocessor.get_feature_names_out())
n, d = X_train.shape
print(f"\n{d} model inputs after preprocessing")

# sklearn minimises  0.5*||w||^2 + C * sum(logloss).  Dividing by C*n gives the
# per-sample form PyTorch uses:  mean(logloss) + (lambda/2)*||w||^2  with
# lambda = 1/(C*n). Same objective -> the three models should agree.
LAMBDA = 1.0 / (BEST_C * n)

# 4a. scikit-learn ------------------------------------------------------------
section("4a. scikit-learn LogisticRegression")
sk_model = LogisticRegression(C=BEST_C, max_iter=10_000, tol=1e-10).fit(X_train, y_train)
print(f"fitted in {sk_model.n_iter_[0]} L-BFGS iterations")

Xt_train = torch.tensor(X_train)
yt_train = torch.tensor(y_train, dtype=torch.float32)


def manual_loop(lr: float, epochs: int, log_every: int = 0):
    """Session 5 style: raw tensors, autograd for the gradient, the update
    step written by hand. Full-batch gradient descent."""
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(epochs):
        logits = Xt_train @ w + b
        # Numerically stable binary cross-entropy written out by hand:
        # -[y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))] = softplus(z) - y*z
        bce = (torch.nn.functional.softplus(logits) - yt_train * logits).mean()
        loss = bce + 0.5 * LAMBDA * (w ** 2).sum()   # bias not penalised, like sklearn
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:5d}  loss {loss.item():.6f}")
    return w.detach(), b.detach(), history


class LogisticRegressionNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)   # logits; the sigmoid lives in the loss


def standard_loop(lr: float, epochs: int, log_every: int = 0):
    """nn.Module + torch.optim. Same objective: weight_decay on the weights
    only (adds lambda*w to the gradient = derivative of (lambda/2)*||w||^2)."""
    torch.manual_seed(SEED)
    model = LogisticRegressionNet(d)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD([
        {"params": [model.linear.weight], "weight_decay": LAMBDA},
        {"params": [model.linear.bias], "weight_decay": 0.0},
    ], lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        bce = loss_fn(model(Xt_train), yt_train)
        bce.backward()
        optimizer.step()
        # Log the full objective (incl. penalty) so it's comparable to the manual loop.
        with torch.no_grad():
            full = bce.item() + 0.5 * LAMBDA * (model.linear.weight ** 2).sum().item()
        history.append(full)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:5d}  loss {full:.6f}")
    return model, history


# Learning rate: the only other hyperparameter. Pick the rate that gets the
# training objective lowest in a short budget without diverging.
section("4b. Learning-rate check (300 full-batch epochs, manual loop)")
lr_results = {}
for lr in [0.01, 0.1, 0.5, 1.0, 2.0, 4.0]:
    *_, hist = manual_loop(lr, 300)
    lr_results[lr] = hist[-1] if np.isfinite(hist[-1]) else float("inf")
    print(f"lr={lr:<5} final loss {lr_results[lr]:.6f}")
LR = min(lr_results, key=lr_results.get)
EPOCHS = 3000
print(f"-> chosen lr = {LR}, {EPOCHS} epochs")

section("4c. Manual PyTorch loop")
w_manual, b_manual, hist_manual = manual_loop(LR, EPOCHS, log_every=500)

section("4d. Standard PyTorch (nn.Module + torch.optim.SGD)")
std_model, hist_std = standard_loop(LR, EPOCHS, log_every=500)

sk_objective = log_loss(y_train, sk_model.predict_proba(X_train)[:, 1]) + 0.5 * LAMBDA * float((sk_model.coef_ ** 2).sum())
print(f"\nsklearn objective at its optimum: {sk_objective:.6f}")


# ---------------------------------------------------------------------------
# 5. Predictions and the capacity threshold
# ---------------------------------------------------------------------------
def predict_all(X: np.ndarray) -> dict:
    Xt = torch.tensor(X)
    with torch.no_grad():
        p_manual = torch.sigmoid(Xt @ w_manual + b_manual).numpy()
        std_model.eval()
        p_std = torch.sigmoid(std_model(Xt)).numpy()
    return {
        "scikit-learn": sk_model.predict_proba(X)[:, 1],
        "PyTorch manual": p_manual,
        "PyTorch nn.Module": p_std,
    }


section("5. Decision threshold from the 10% capacity")
# Out-of-fold predictions: each training row is scored by a model that didn't
# see it, so the threshold isn't fitted to over-confident in-sample scores and
# the test set stays untouched.
oof = cross_val_predict(
    make_pipeline(build_preprocessor(X_train_df), LogisticRegression(C=BEST_C, max_iter=5000)),
    X_train_df, y_train, cv=CV, method="predict_proba",
)[:, 1]
THRESHOLD = float(np.quantile(oof, 1 - CAPACITY))
print(f"invite everyone with predicted risk >= {THRESHOLD:.4f} (top {CAPACITY:.0%} of out-of-fold scores)")

# Without the budget, the cost-optimal threshold is far lower: a missed case
# costs 33x a wasted test, so it pays to test almost everyone. The budget,
# not the cost ratio, is what binds.
grid = np.linspace(0.005, 0.8, 400)
oof_costs = [business_cost(y_train, oof >= t)["cost_eur"] for t in grid]
t_cost = float(grid[int(np.argmin(oof_costs))])
uncapped = business_cost(y_train, oof >= t_cost)
print(f"cost-optimal threshold ignoring budget: {t_cost:.3f} -> would invite {uncapped['invited_share']:.0%} of people")

# ---------------------------------------------------------------------------
# 6. Test-set evaluation
# ---------------------------------------------------------------------------
section("6. Test-set evaluation")
probs_test = predict_all(X_test)
prevalence = float(y_train.mean())
rng = np.random.default_rng(SEED)

# Naive baselines.
age_test = X_test_df["age"].fillna(X_train_df["age"].median()).to_numpy()
baselines = {
    # Everyone gets the training prevalence; who gets invited is pure chance.
    "Baseline: random invitations": rng.uniform(0, 1, len(y_test)),
    # A one-line rule a clinic could use with no model: invite the oldest.
    "Baseline: oldest first": age_test + rng.uniform(0, 1e-3, len(y_test)),
}


def top_share(scores: np.ndarray, share: float) -> np.ndarray:
    """Select exactly the top `share` of rows (baselines have no calibrated probabilities)."""
    k = int(round(share * len(scores)))
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(-scores)[:k]] = True
    return selected


def bootstrap_ci(y_true, scores, selected, n_boot=1000):
    aucs, precs = [], []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        aucs.append(roc_auc_score(y_true[s], scores[s]))
        sel = selected[s]
        precs.append(y_true[s][sel].mean() if sel.any() else 0.0)
    return np.percentile(aucs, [2.5, 97.5]).tolist(), np.percentile(precs, [2.5, 97.5]).tolist()


results = {}
for name, p in {**baselines, **probs_test}.items():
    is_model = name in probs_test
    selected = (p >= THRESHOLD) if is_model else top_share(p, CAPACITY)
    row = {
        "roc_auc": float(roc_auc_score(y_test, p)),
        "pr_auc": float(average_precision_score(y_test, p)),
        **business_cost(y_test, selected),
    }
    if is_model or name == "Baseline: random invitations":
        p_cal = p if is_model else np.full(len(y_test), prevalence)
        row["log_loss"] = float(log_loss(y_test, p_cal))
        row["brier"] = float(brier_score_loss(y_test, p_cal))
    if is_model:
        auc_ci, prec_ci = bootstrap_ci(y_test, p, selected)
        row["roc_auc_ci95"], row["precision_ci95"] = auc_ci, prec_ci
    results[name] = row

table = pd.DataFrame(results).T
cols = ["roc_auc", "pr_auc", "log_loss", "invited", "precision", "recall", "cost_eur"]
print(table[cols].to_string(float_format=lambda v: f"{v:.4f}"))
for name in probs_test:
    r = results[name]
    print(f"{name:18s} ROC-AUC 95% CI {r['roc_auc_ci95'][0]:.3f}-{r['roc_auc_ci95'][1]:.3f} | "
          f"precision@capacity 95% CI {r['precision_ci95'][0]:.3f}-{r['precision_ci95'][1]:.3f}")

# Do the three implementations agree?
section("6b. Agreement between the three implementations")
sk_w = sk_model.coef_.ravel()
weights = {
    "scikit-learn": (sk_w, float(sk_model.intercept_[0])),
    "PyTorch manual": (w_manual.numpy(), float(b_manual.item())),
    "PyTorch nn.Module": (std_model.linear.weight.detach().numpy().ravel(), float(std_model.linear.bias.item())),
}
agreement = {}
p_ref = probs_test["scikit-learn"]
for name in ["PyTorch manual", "PyTorch nn.Module"]:
    p = probs_test[name]
    w = weights[name][0]
    agreement[name] = {
        "max_abs_prob_diff": float(np.abs(p - p_ref).max()),
        "mean_abs_prob_diff": float(np.abs(p - p_ref).mean()),
        "max_abs_coef_diff": float(np.abs(w - sk_w).max()),
        "same_invitations": int(((p >= THRESHOLD) == (p_ref >= THRESHOLD)).sum()),
        "n_test": int(len(p)),
        "final_objective": (hist_manual if name == "PyTorch manual" else hist_std)[-1],
    }
    a = agreement[name]
    print(f"{name:18s} vs sklearn: max |Δp| {a['max_abs_prob_diff']:.2e}, mean |Δp| {a['mean_abs_prob_diff']:.2e}, "
          f"max |Δw| {a['max_abs_coef_diff']:.2e}, same decision for {a['same_invitations']}/{a['n_test']}")
agreement["sklearn_objective"] = sk_objective

# ---------------------------------------------------------------------------
# 6c. Fairness audit (sklearn model; the three agree)
# ---------------------------------------------------------------------------
section("6c. Audit by sex and race (test set)")
audit = []
selected_test = probs_test["scikit-learn"] >= THRESHOLD
for col in ["Sex", "RaceEthnicityCategory"]:
    groups = X_test_df[col].to_numpy()
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() < 50:
            continue
        r = business_cost(y_test[m], selected_test[m])
        audit.append({
            "attribute": col, "group": g, "n": int(m.sum()),
            "prevalence": float(y_test[m].mean()),
            "invited_share": r["invited_share"], "recall": r["recall"], "precision": r["precision"],
            "roc_auc": float(roc_auc_score(y_test[m], probs_test["scikit-learn"][m])),
        })
audit_df = pd.DataFrame(audit)
print(audit_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---------------------------------------------------------------------------
# 7. Save artifacts for app.py
# ---------------------------------------------------------------------------
section("7. Saving artifacts")
joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk_model, ARTIFACTS / "sklearn_model.joblib")
torch.save({"w": w_manual, "b": b_manual}, ARTIFACTS / "torch_manual.pt")
torch.save(std_model.state_dict(), ARTIFACTS / "torch_standard.pt")

coef_table = pd.DataFrame({
    "feature": feature_names,
    "scikit-learn": sk_w,
    "PyTorch manual": weights["PyTorch manual"][0],
    "PyTorch nn.Module": weights["PyTorch nn.Module"][0],
}).sort_values("scikit-learn", key=np.abs, ascending=False)
coef_table.to_csv(ARTIFACTS / "coefficients.csv", index=False)

with open(ARTIFACTS / "split.json", "w") as f:
    json.dump({"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}, f)

with open(ARTIFACTS / "metrics.json", "w") as f:
    json.dump({
        "n_rows": int(len(df)), "n_train": int(n), "n_test": int(len(test_idx)), "n_inputs": int(d),
        "prevalence_train": prevalence,
        "capacity": CAPACITY, "cost_wasted_test": COST_WASTED_TEST, "cost_missed_case": COST_MISSED_CASE,
        "threshold": THRESHOLD, "cost_optimal_threshold_uncapped": t_cost,
        "cost_optimal_uncapped_invited_share": uncapped["invited_share"],
        "best_C": BEST_C, "lambda": LAMBDA, "lr": LR, "epochs": EPOCHS,
        "c_search": {str(k): v for k, v in c_results.items()},
        "lr_search": {str(k): v for k, v in lr_results.items()},
        "ablation": ablation, "test_results": results, "agreement": agreement,
        "audit": audit, "loss_history": {"PyTorch manual": hist_manual, "PyTorch nn.Module": hist_std},
    }, f, indent=2)

print(f"saved to {ARTIFACTS}/")
