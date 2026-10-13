"""
Assignment 1 — training pipeline (run via `uv run python main.py train`).

1. Load the Bank Marketing data, engineer pre-call features, split by time.
2. Feature ablation on the validation set (which engineered groups help).
3. Train the same L2-regularized logistic regression three ways:
   scikit-learn, a manual PyTorch loop, and nn.Module + torch.optim.
4. Pick each model's call threshold on validation by expected profit,
   evaluate everything once on the test set against naive baselines.
5. Save models + metrics to artifacts/ for app.py (which never retrains).
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline import (ARTIFACTS, COST_PER_CALL, DEFAULT_GROUPS,
                      VALUE_PER_SUBSCRIPTION, TorchLogReg, call_mask,
                      chronological_split, engineer, expected_profit, l2_penalty,
                      load_raw, make_preprocessor, predict_manual, predict_module,
                      row_fraction_split)

SEED = 42
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
# Pick the strongest regularization whose validation log loss is within this
# tolerance of the best: smaller differences are noise on ~900 rows.
C_TOLERANCE = 0.001
MANUAL_LR_GRID = [0.1, 0.5, 1.0]
MANUAL_EPOCHS = 2000
MODULE_LR_GRID = [0.003, 0.01, 0.03]
MODULE_EPOCHS = 60
BATCH_SIZE = 256
THRESHOLDS = np.round(np.arange(0.01, 0.91, 0.01), 2)
CAPACITIES = [0.10, 0.25, 0.50]  # call-centre capacity as a share of the list

torch.manual_seed(SEED)
np.random.seed(SEED)


def ranking_metrics(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
    }


def decision_metrics(y, p, t, capacity=1.0):
    call = call_mask(p, t, capacity)
    tp = int((call & (y == 1)).sum())
    return {
        "threshold": float(t),
        "calls": int(call.sum()),
        "call_rate": float(call.mean()),
        "precision": tp / max(int(call.sum()), 1),
        "recall": tp / max(int(y.sum()), 1),
        "profit": expected_profit(y, p, t, capacity),
    }


def best_threshold(y, p):
    profits = [expected_profit(y, p, t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(profits))])


def fit_sklearn(X, y, C):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
raw = load_raw()
feats = engineer(raw)
train_df, val_df, test_df = chronological_split(raw)
splits = {"train": train_df, "val": val_df, "test": test_df}
y = {k: v["y"].to_numpy() for k, v in splits.items()}

split_summary = {
    k: {"rows": len(v), "from": v["period"].iloc[0], "to": v["period"].iloc[-1],
        "positive_rate": float(v["y"].mean())}
    for k, v in splits.items()
}
print(f"{len(raw)} rows after dropping duplicates; chronological split:")
for k, s in split_summary.items():
    print(f"  {k:5s} {s['rows']:6d} rows  {s['from']} -> {s['to']}  "
          f"subscription rate {s['positive_rate']:.1%}")


def X_for(groups, preprocessor=None):
    """Fit the preprocessor on train only, transform all three splits."""
    pre = preprocessor or make_preprocessor(groups)
    pre.fit(feats.loc[train_df.index])
    return pre, {k: pre.transform(feats.loc[v.index]).astype(np.float32)
                 for k, v in splits.items()}


# --------------------------------------------------------------------------
# Feature ablation (validation set only; test stays untouched)
# --------------------------------------------------------------------------
print("\nFeature ablation (sklearn, C=1, validation set):")

RAW_NUM = ["age", "campaign", "pdays", "previous", "emp.var.rate",
           "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
RAW_CAT = ["job", "marital", "education", "default", "housing", "loan",
           "contact", "month", "day_of_week", "poutcome"]
raw_pre = make_pipeline(ColumnTransformer([
    ("num", "passthrough", RAW_NUM),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                          sparse_output=False), RAW_CAT),
]), StandardScaler())
raw_pre.fit(raw.loc[train_df.index])
X_raw = {k: raw_pre.transform(raw.loc[v.index]) for k, v in splits.items()}

variants = {
    "raw columns (no engineering, duration dropped)": None,
    "engineered: final feature set": DEFAULT_GROUPS,
    **{f"final minus '{g}'": [x for x in DEFAULT_GROUPS if x != g]
       for g in DEFAULT_GROUPS},
    "final plus 'month'": DEFAULT_GROUPS + ["month"],
    "final plus collinear macro (emp.var.rate, nr.employed)":
        DEFAULT_GROUPS + ["macro_collinear"],
}
ablation = []
for name, groups in variants.items():
    Xs = X_raw if groups is None else X_for(groups)[1]
    m = fit_sklearn(Xs["train"], y["train"], C=1.0)
    p = m.predict_proba(Xs["val"])[:, 1]
    row = {"variant": name, "n_columns": Xs["train"].shape[1],
           **ranking_metrics(y["val"], p),
           "val_profit_at_best_t": expected_profit(y["val"], p, best_threshold(y["val"], p))}
    ablation.append(row)
    print(f"  {name:58s} AUC {row['roc_auc']:.4f}  PR-AUC {row['pr_auc']:.4f}  "
          f"logloss {row['log_loss']:.4f}")
ablation = pd.DataFrame(ablation)

# --------------------------------------------------------------------------
# Final features + regularization strength
# --------------------------------------------------------------------------
preprocessor, X = X_for(DEFAULT_GROUPS)
feature_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]
n_train, n_features = X["train"].shape
print(f"\nFinal design matrix: {n_features} columns")

c_search = []
for C in C_GRID:
    p = fit_sklearn(X["train"], y["train"], C).predict_proba(X["val"])[:, 1]
    c_search.append({"C": C, **ranking_metrics(y["val"], p)})
c_search = pd.DataFrame(c_search)
near_best = c_search["log_loss"] <= c_search["log_loss"].min() + C_TOLERANCE
BEST_C = float(c_search.loc[near_best, "C"].min())
print("C search (validation log loss):",
      ", ".join(f"C={r.C:g}: {r.log_loss:.4f}" for r in c_search.itertuples()),
      f"-> C={BEST_C:g}")

# --------------------------------------------------------------------------
# 1) scikit-learn
# --------------------------------------------------------------------------
sk_model = fit_sklearn(X["train"], y["train"], BEST_C)

# --------------------------------------------------------------------------
# 2) Manual PyTorch: raw tensors, autograd, hand-written update, full batch
# --------------------------------------------------------------------------
Xt = {k: torch.as_tensor(v) for k, v in X.items()}
yt = {k: torch.as_tensor(v.astype(np.float32)) for k, v in y.items()}


def train_manual(lr, epochs=MANUAL_EPOCHS):
    w = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(epochs):
        logits = Xt["train"] @ w + b
        # Numerically stable binary cross-entropy written out by hand:
        # -[y log s(z) + (1-y) log(1-s(z))] = softplus(z) - y z
        loss = (torch.nn.functional.softplus(logits) - yt["train"] * logits).mean()
        loss = loss + l2_penalty(w, BEST_C, n_train)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append((epoch, loss.item()))
    return {"w": w.detach(), "b": b.detach()}, history


manual_runs = {}
for lr in MANUAL_LR_GRID:
    params, hist = train_manual(lr)
    manual_runs[lr] = (params, hist, hist[-1][1])
# The learning rate changes how well we minimize a fixed objective, not the
# objective itself, so it's chosen by the final training objective.
MANUAL_LR = min(manual_runs, key=lambda lr: manual_runs[lr][2])
manual_params, manual_history, _ = manual_runs[MANUAL_LR]
print(f"Manual loop lr search (final train objective): "
      + ", ".join(f"lr={lr}: {r[2]:.6f}" for lr, r in manual_runs.items())
      + f" -> lr={MANUAL_LR}")

# --------------------------------------------------------------------------
# 3) Standard PyTorch: nn.Module + DataLoader + torch.optim (mini-batch Adam)
# --------------------------------------------------------------------------
def train_module(lr, epochs=MODULE_EPOCHS):
    torch.manual_seed(SEED)
    model = TorchLogReg(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Mini-batch gradients are noisy, so a constant step size leaves the
    # weights jittering around the optimum; annealing it to ~0 lets them settle.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt["train"], yt["train"]),
        batch_size=BATCH_SIZE, shuffle=True)
    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb) + l2_penalty(model.linear.weight, BEST_C, n_train)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            full = criterion(model(Xt["train"]), yt["train"]) + \
                l2_penalty(model.linear.weight, BEST_C, n_train)
        history.append((epoch, full.item()))
    return model, history


module_runs = {}
for lr in MODULE_LR_GRID:
    model, hist = train_module(lr)
    module_runs[lr] = (model, hist, hist[-1][1])
MODULE_LR = min(module_runs, key=lambda lr: module_runs[lr][2])
module_model, module_history, _ = module_runs[MODULE_LR]
print(f"nn.Module lr search (final train objective): "
      + ", ".join(f"lr={lr}: {r[2]:.6f}" for lr, r in module_runs.items())
      + f" -> lr={MODULE_LR}")

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
predictors = {
    "sklearn": lambda X_: sk_model.predict_proba(X_)[:, 1],
    "torch_manual": lambda X_: predict_manual(manual_params, X_),
    "torch_module": lambda X_: predict_module(module_model, X_),
}
proba = {name: {k: f(X[k]) for k in splits} for name, f in predictors.items()}

results = {}
for name in predictors:
    t = best_threshold(y["val"], proba[name]["val"])
    results[name] = {
        "val": {**ranking_metrics(y["val"], proba[name]["val"]),
                **decision_metrics(y["val"], proba[name]["val"], t)},
        "test": {**ranking_metrics(y["test"], proba[name]["test"]),
                 **decision_metrics(y["test"], proba[name]["test"], t)},
        # Break-even threshold if probabilities were perfectly calibrated.
        "test_at_breakeven": decision_metrics(
            y["test"], proba[name]["test"], COST_PER_CALL / VALUE_PER_SUBSCRIPTION),
        # Limited call capacity: ranking quality decides who gets the calls.
        "test_capacity": {str(c): decision_metrics(y["test"], proba[name]["test"], t, c)
                          for c in CAPACITIES},
    }

# Naive baselines: a constant probability (training base rate), and the two
# trivial policies a bank could run without any model.
prior = float(y["train"].mean())
const = np.full(len(y["test"]), prior)
baselines = {
    "constant_prior": {**ranking_metrics(y["test"], const)},
    "call_everyone": decision_metrics(y["test"], np.ones(len(y["test"])), 0.5),
    "call_no_one": decision_metrics(y["test"], np.zeros(len(y["test"])), 0.5),
    # With limited capacity and no model, the bank calls a random subset:
    # expected profit = calls * (base rate * value - cost).
    "random_at_capacity": {
        str(c): {"calls": int(c * len(y["test"])),
                 "precision": float(y["test"].mean()),
                 "profit": float(int(c * len(y["test"])) *
                                 (y["test"].mean() * VALUE_PER_SUBSCRIPTION - COST_PER_CALL))}
        for c in CAPACITIES},
}

# Do the three implementations agree?
coefs = pd.DataFrame({
    "feature": feature_names,
    "sklearn": sk_model.coef_[0],
    "torch_manual": manual_params["w"].numpy(),
    "torch_module": module_model.linear.weight.detach().numpy()[0],
})
intercepts = {"sklearn": float(sk_model.intercept_[0]),
              "torch_manual": float(manual_params["b"][0]),
              "torch_module": float(module_model.linear.bias.detach()[0])}
agreement = {}
for name in ["torch_manual", "torch_module"]:
    agreement[name] = {
        "max_abs_coef_diff": float((coefs[name] - coefs["sklearn"]).abs().max()),
        "max_abs_proba_diff_test": float(np.abs(proba[name]["test"] - proba["sklearn"]["test"]).max()),
        "mean_abs_proba_diff_test": float(np.abs(proba[name]["test"] - proba["sklearn"]["test"]).mean()),
        "intercept_diff": intercepts[name] - intercepts["sklearn"],
    }

# Diagnostic only: what a (leaky) stratified random split would have reported.
idx_tr, idx_te = train_test_split(raw.index, test_size=0.15, stratify=raw["y"],
                                  random_state=SEED)
pre_r = make_preprocessor(DEFAULT_GROUPS).fit(feats.loc[idx_tr])
m_r = fit_sklearn(pre_r.transform(feats.loc[idx_tr]), raw.loc[idx_tr, "y"], BEST_C)
p_r = m_r.predict_proba(pre_r.transform(feats.loc[idx_te]))[:, 1]
random_split_diag = ranking_metrics(raw.loc[idx_te, "y"], p_r)

# Diagnostic only: the first-attempt 70/15/15 row split (same features, C=1),
# which put all training data before the bank's mid-2009 policy change.
r_tr, _, r_te = row_fraction_split(raw)
pre_f = make_preprocessor(DEFAULT_GROUPS).fit(feats.loc[r_tr.index])
m_f = fit_sklearn(pre_f.transform(feats.loc[r_tr.index]), r_tr["y"], 1.0)
p_f = m_f.predict_proba(pre_f.transform(feats.loc[r_te.index]))[:, 1]
first_attempt_diag = {
    **ranking_metrics(r_te["y"], p_f),
    "test_from": r_te["period"].iloc[0], "test_to": r_te["period"].iloc[-1],
    "mean_predicted": float(p_f.mean()), "actual_rate": float(r_te["y"].mean()),
}

print(f"\nTest set ({split_summary['test']['from']} -> {split_summary['test']['to']}):")
print(f"  {'model':14s} {'AUC':>7s} {'PR-AUC':>7s} {'logloss':>8s} {'thr':>5s} "
      f"{'calls':>6s} {'prec':>6s} {'recall':>6s} {'profit':>8s}")
for name, r in results.items():
    t = r["test"]
    print(f"  {name:14s} {t['roc_auc']:7.4f} {t['pr_auc']:7.4f} {t['log_loss']:8.4f} "
          f"{t['threshold']:5.2f} {t['calls']:6d} {t['precision']:6.3f} "
          f"{t['recall']:6.3f} {t['profit']:8.0f}")
cp = baselines["constant_prior"]
print(f"  {'const. prior':14s} {cp['roc_auc']:7.4f} {cp['pr_auc']:7.4f} {cp['log_loss']:8.4f}")
for b in ["call_everyone", "call_no_one"]:
    print(f"  {b:14s} {'':32s} {baselines[b]['calls']:6d} "
          f"{baselines[b]['precision']:6.3f} {baselines[b]['recall']:6.3f} "
          f"{baselines[b]['profit']:8.0f}")
print("Agreement with sklearn:", json.dumps(agreement, indent=1))
for c in CAPACITIES:
    print(f"  capacity {c:.0%}: " + ", ".join(
        f"{n} {r['test_capacity'][str(c)]['profit']:.0f}" for n, r in results.items())
        + f", random {baselines['random_at_capacity'][str(c)]['profit']:.0f}")
print(f"Random-split diagnostic (sklearn): {random_split_diag}")
print(f"First-attempt 70/15/15 diagnostic (sklearn): {first_attempt_diag}")

# --------------------------------------------------------------------------
# Save artifacts (everything app.py needs; it never retrains)
# --------------------------------------------------------------------------
ARTIFACTS.mkdir(exist_ok=True)
joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk_model, ARTIFACTS / "sklearn_logreg.joblib")
torch.save(manual_params, ARTIFACTS / "torch_manual_params.pt")
torch.save(module_model.state_dict(), ARTIFACTS / "torch_module_state.pt")
coefs.to_csv(ARTIFACTS / "coefficients.csv", index=False)
ablation.to_csv(ARTIFACTS / "ablation.csv", index=False)
c_search.to_csv(ARTIFACTS / "c_search.csv", index=False)

metrics = {
    "splits": split_summary,
    "feature_groups": DEFAULT_GROUPS,
    "n_features": n_features,
    "hyperparameters": {"C": BEST_C, "manual_lr": MANUAL_LR,
                        "manual_epochs": MANUAL_EPOCHS, "module_lr": MODULE_LR,
                        "module_epochs": MODULE_EPOCHS, "batch_size": BATCH_SIZE},
    "economics": {"cost_per_call": COST_PER_CALL,
                  "value_per_subscription": VALUE_PER_SUBSCRIPTION},
    "results": results,
    "baselines": baselines,
    "agreement": agreement,
    "intercepts": intercepts,
    "random_split_diagnostic": random_split_diag,
    "first_attempt_row_split_diagnostic": first_attempt_diag,
    "loss_history": {"torch_manual": manual_history, "torch_module": module_history},
}
(ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nSaved artifacts to {ARTIFACTS}")
