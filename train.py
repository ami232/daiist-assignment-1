"""
Assignment 1 - training pipeline (run with `uv run python main.py train`).

Business problem: a telecom retention team decides which customers get a
retention offer (e.g. a discount). The model estimates P(churn) per customer;
the team contacts everyone above a threshold. The threshold is chosen to
maximise the net gain vs doing nothing (see the constants below), not fixed at 0.5.

Steps:
 1. load data + fix TotalCharges blanks             (pipeline.load_raw)
 2. feature engineering, row-wise                    (pipeline.engineer_features)
 3. stratified 60/20/20 train/val/test split
 4. fit ColumnTransformer on TRAIN only, transform val/test
 5. same L2 logistic regression three ways:
      a. scikit-learn  b. manual PyTorch loop  c. nn.Module + torch.optim
 6. pick each model's gain-maximising threshold on VALIDATION
 7. report metrics + net gain on TEST, compare against naive baselines and
    against each other, save everything the app needs to artifacts/
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline import (
    ARTIFACTS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,
    LogisticRegressionNN, engineer_features, load_raw, net_gain,
)

# ---------------------------------------------------------------------------
# Money assumptions (ASSUMED, not from the data - change them here).
# ---------------------------------------------------------------------------
CALL_COST = 50.0      # $ we spend on every customer we call with a retention offer
HORIZON_MONTHS = 12   # months of revenue lost when a customer churns
SAVE_RATE = 0.30      # share of contacted would-be churners the offer retains
                      # (default in Verbraken, Verbeke & Baesens 2013: Beta(6,14), mean 0.30)
# => vs doing nothing, calling a would-be churner gains on average
#    0.30 * MonthlyCharges * 12 - $50 (~ +$220 for a $75/month customer);
#    calling a customer who would stay anyway gains -$50.
COSTS = dict(call_cost=CALL_COST, horizon_months=HORIZON_MONTHS, save_rate=SAVE_RATE)

SEED = 42
C_GRID = [0.01, 0.1, 1.0, 10.0]          # light tuning of L2 strength only
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
MANUAL_LR, MANUAL_EPOCHS = 2.0, 30_000   # plain full-batch GD; lr < 2/L (L = max curvature ~0.97, so 2/L ~ 2.06) is stable

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)   # float64 so the 3 fits can match closely
ARTIFACTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1-2. Load and engineer features
# ---------------------------------------------------------------------------
raw = load_raw()
X_all = engineer_features(raw)
y_all = raw[TARGET].to_numpy()
print(f"Data: {len(raw)} rows, churn rate {y_all.mean():.3f}, "
      f"{X_all.shape[1]} columns after feature engineering")

# ---------------------------------------------------------------------------
# 3. Split
# The dataset is a single snapshot with no date column, so a time-based split
# is impossible: we cannot know which customers came "later". `tenure` is how
# long each customer has been with the company at snapshot time - a customer
# attribute, not a time index - so it is used as a feature, not to split.
# A stratified random split keeps the ~26.5% churn rate equal in all three sets.
# ---------------------------------------------------------------------------
idx = np.arange(len(raw))
idx_train, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y_all, random_state=SEED)
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, stratify=y_all[idx_tmp], random_state=SEED)
y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]
mc_val = raw["MonthlyCharges"].to_numpy()[idx_val]     # needed for the $ net gain
mc_test = raw["MonthlyCharges"].to_numpy()[idx_test]
print(f"Split: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

# ---------------------------------------------------------------------------
# 4. Preprocessing, fitted on TRAIN only (no leakage of val/test statistics).
# - StandardScaler on numerics: puts features on one scale, so a single L2
#   penalty treats them equally and gradient descent converges well.
# - OneHotEncoder on categoricals with drop="first": one level per feature is
#   the reference category (absorbed by the intercept). Keeping all levels
#   would be perfectly collinear with the intercept -> non-unique coefficients
#   and slow gradient descent. Reference levels (alphabetical first): e.g.
#   Contract=Month-to-month, InternetService=DSL, PaymentMethod=Bank transfer.
#   handle_unknown="ignore": an unseen category at inference is encoded as the
#   reference level (all zeros) instead of crashing.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
])
X_train = preprocessor.fit_transform(X_all.iloc[idx_train])
X_val = preprocessor.transform(X_all.iloc[idx_val])
X_test = preprocessor.transform(X_all.iloc[idx_test])
X_train, X_val, X_test = (np.asarray(m, dtype=np.float64) for m in (X_train, X_val, X_test))
feature_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]
n_train, n_features = X_train.shape
print(f"Model matrix: {n_features} features")


# ---------------------------------------------------------------------------
# 5a. scikit-learn. sklearn minimises  C * sum(logloss) + 0.5 * ||w||^2
# (bias not penalised). C chosen on validation log loss.
# ---------------------------------------------------------------------------
def fit_sklearn(C):
    return LogisticRegression(C=C, solver="lbfgs", max_iter=10_000, tol=1e-10).fit(X_train, y_train)


val_logloss = {C: log_loss(y_val, fit_sklearn(C).predict_proba(X_val)[:, 1]) for C in C_GRID}
BEST_C = min(val_logloss, key=val_logloss.get)
print("Validation log loss by C:", {c: round(v, 5) for c, v in val_logloss.items()}, "-> C =", BEST_C)
sk_model = fit_sklearn(BEST_C)

# Dividing sklearn's objective by C*n gives the SAME minimiser written as a mean:
#   mean(logloss) + ||w||^2 / (2*C*n)   -> L2 coefficient used by both torch fits
L2 = 1.0 / (2.0 * BEST_C * n_train)

Xt_train = torch.tensor(X_train)
yt_train = torch.tensor(y_train, dtype=torch.float64)


def penalised_loss(logits, w):
    """Mean BCE (numerically stable form) + L2 on the weights, not the bias.
    BCE(z, y) = max(z,0) - z*y + log(1 + exp(-|z|)) never overflows exp()."""
    bce = torch.clamp(logits, min=0) - logits * yt_train + torch.log1p(torch.exp(-logits.abs()))
    return bce.mean() + L2 * (w ** 2).sum()


# ---------------------------------------------------------------------------
# 5b. Manual PyTorch: raw tensors + autograd + hand-written gradient step.
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
w = torch.zeros(n_features, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(MANUAL_EPOCHS):
    logits = Xt_train @ w + b                 # linear part; sigmoid is inside the loss
    loss = penalised_loss(logits, w)
    loss.backward()                           # autograd fills w.grad and b.grad
    with torch.no_grad():                     # the update itself must not be tracked
        w -= MANUAL_LR * w.grad
        b -= MANUAL_LR * b.grad
    w.grad.zero_()                            # gradients accumulate unless reset
    b.grad.zero_()
    if epoch % 5000 == 0:
        print(f"  manual epoch {epoch:>6}: loss {loss.item():.6f}")
manual_loss = penalised_loss(Xt_train @ w + b, w).item()
print(f"Manual PyTorch final penalised loss: {manual_loss:.8f}")

# ---------------------------------------------------------------------------
# 5c. Standard PyTorch: nn.Module + BCEWithLogitsLoss + torch.optim.LBFGS.
# LBFGS is a quasi-Newton method (the same family as sklearn's lbfgs solver),
# so on this small convex problem it reaches the optimum in few iterations.
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
nn_model = LogisticRegressionNN(n_features)
bce_loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.LBFGS(nn_model.parameters(), lr=1.0, max_iter=500,
                              tolerance_grad=1e-12, tolerance_change=1e-14,
                              line_search_fn="strong_wolfe")


def closure():
    optimizer.zero_grad()
    loss = bce_loss(nn_model(Xt_train), yt_train) + L2 * (nn_model.linear.weight ** 2).sum()
    loss.backward()
    return loss


for _ in range(5):          # a few outer calls; each runs up to max_iter steps
    optimizer.step(closure)
nn_loss = closure().item()
print(f"Standard PyTorch final penalised loss: {nn_loss:.8f}")

sk_coef = np.r_[sk_model.intercept_, sk_model.coef_.ravel()]
sk_loss = penalised_loss(torch.tensor(X_train @ sk_coef[1:] + sk_coef[0]), torch.tensor(sk_coef[1:])).item()
print(f"scikit-learn   final penalised loss: {sk_loss:.8f}")


# ---------------------------------------------------------------------------
# 6-7. Predictions, thresholds, metrics
# ---------------------------------------------------------------------------
def torch_proba(X, weight, bias):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X) @ weight.reshape(-1) + bias.reshape(-1)).numpy()


predictors = {
    "sklearn": lambda X: sk_model.predict_proba(X)[:, 1],
    "torch_manual": lambda X: torch_proba(X, w.detach(), b.detach()),
    "torch_nn": lambda X: torch_proba(X, nn_model.linear.weight.detach(), nn_model.linear.bias.detach()),
}
coefs = {
    "sklearn": sk_coef,
    "torch_manual": np.r_[b.detach().numpy(), w.detach().numpy()],
    "torch_nn": np.r_[nn_model.linear.bias.detach().numpy(), nn_model.linear.weight.detach().numpy().ravel()],
}


def best_threshold(y, p, mc):
    """Threshold on the grid that maximises the total net gain on (y, p)."""
    gains = [net_gain(y, p >= t, mc, **COSTS) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(gains))])


def evaluate(y, p, threshold, mc):
    pred = p >= threshold
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "n_contacted": int(pred.sum()),
        "net_gain": net_gain(y, pred, mc, **COSTS),
        "net_gain_at_0.5": net_gain(y, p >= 0.5, mc, **COSTS),
    }


val_proba = {m: f(X_val) for m, f in predictors.items()}
test_proba = {m: f(X_test) for m, f in predictors.items()}
thresholds = {m: best_threshold(y_val, val_proba[m], mc_val) for m in predictors}
results = {m: evaluate(y_test, test_proba[m], thresholds[m], mc_test) for m in predictors}

# Naive baselines (no model): contact nobody (= predict "no churn" for all)
# and contact everyone.
for name, contact in [("baseline_contact_nobody", np.zeros_like(y_test)),
                      ("baseline_contact_everyone", np.ones_like(y_test))]:
    results[name] = {
        "roc_auc": 0.5, "pr_auc": float(y_test.mean()), "threshold": None,
        "precision": precision_score(y_test, contact, zero_division=0),
        "recall": recall_score(y_test, contact), "f1": f1_score(y_test, contact),
        "n_contacted": int(contact.sum()),
        "net_gain": net_gain(y_test, contact, mc_test, **COSTS),
        "net_gain_at_0.5": None,
    }

# Agreement between the three implementations.
pairs = [("sklearn", "torch_manual"), ("sklearn", "torch_nn"), ("torch_manual", "torch_nn")]
agreement = {
    f"{a}_vs_{b}": {
        "max_abs_coef_diff": float(np.abs(coefs[a] - coefs[b]).max()),
        "max_abs_test_proba_diff": float(np.abs(test_proba[a] - test_proba[b]).max()),
        "roc_auc_diff": abs(results[a]["roc_auc"] - results[b]["roc_auc"]),
    }
    for a, b in pairs
}
final_losses = {"sklearn": sk_loss, "torch_manual": manual_loss, "torch_nn": nn_loss}

table = pd.DataFrame(results).T
print("\nTEST SET RESULTS (threshold chosen on validation to maximise net gain):")
print(table.to_string(float_format=lambda v: f"{v:,.4f}"))
print("\nAgreement between implementations:")
print(pd.DataFrame(agreement).T.to_string(float_format=lambda v: f"{v:.2e}"))
for key, d in agreement.items():
    if d["max_abs_test_proba_diff"] > 1e-3:
        print(f"WARNING {key}: probabilities differ by up to {d['max_abs_test_proba_diff']:.2e}. "
              f"Final penalised losses: {final_losses}. The one with the higher loss has "
              f"not fully converged (same convex objective => same unique optimum).")

# ---------------------------------------------------------------------------
# Save artifacts (the app only reads these; it never trains)
# ---------------------------------------------------------------------------
joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk_model, ARTIFACTS / "sklearn_model.joblib")
torch.save({"w": w.detach(), "b": b.detach()}, ARTIFACTS / "torch_manual.pt")
torch.save(nn_model.state_dict(), ARTIFACTS / "torch_nn.pt")

meta = {
    "n_features": n_features, "feature_names": feature_names,
    "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
    "thresholds": thresholds, "costs": COSTS, "best_C": BEST_C,
    "l2_coefficient_torch": L2, "val_logloss_by_C": {str(k): v for k, v in val_logloss.items()},
    "manual_lr": MANUAL_LR, "manual_epochs": MANUAL_EPOCHS, "seed": SEED,
    "split": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)},
}
(ARTIFACTS / "model_meta.json").write_text(json.dumps(meta, indent=2))
(ARTIFACTS / "metrics.json").write_text(json.dumps(
    {"test_metrics": results, "agreement": agreement,
     "final_penalised_train_loss": final_losses,
     "coefficients": {m: dict(zip(["intercept"] + feature_names, map(float, c)))
                      for m, c in coefs.items()}},
    indent=2))
table.to_csv(ARTIFACTS / "metrics.csv")

# Test-set predictions (for the record) and the raw test rows (the app
# re-scores them with the saved models), plus all rows for distribution plots.
pd.DataFrame({"y_true": y_test, "MonthlyCharges": mc_test,
              **{f"p_{m}": p for m, p in test_proba.items()}}).to_csv(
    ARTIFACTS / "test_predictions.csv", index=False)
raw.iloc[idx_test].to_csv(ARTIFACTS / "test_set.csv", index=False)
split = np.empty(len(raw), dtype=object)
split[idx_train], split[idx_val], split[idx_test] = "train", "val", "test"
# Also keep raw columns the model does not use directly (the 4 add-ons that
# only enter via n_addons) so the dashboard can show why they were dropped.
extra = ["tenure", "TotalCharges", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
dist = pd.concat([raw[extra], X_all, raw[[TARGET]]], axis=1)
dist["split"] = split
dist.to_csv(ARTIFACTS / "distributions.csv", index=False)
print(f"\nSaved artifacts to {ARTIFACTS}")
