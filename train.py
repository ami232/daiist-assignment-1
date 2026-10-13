"""
Assignment 1 - training pipeline.  Run with:  uv run python main.py train

Stages
  1. Load data and split it by time (train / val / test).
  2. Model selection on the VALIDATION set (models fitted on train only):
       - regularisation strength alpha (scikit-learn)
       - learning rate (PyTorch)
  3. Refit the same linear regression three ways on train+val with the chosen
     settings, then evaluate ONCE on test against two naive baselines.
  4. Save everything app.py needs into artifacts/ (app.py never trains).
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from pipeline import (
    ARTIFACTS_DIR, TARGET, TEST_START, VAL_START, LinearRegressionTorch,
    build_features, fit_preprocessing, load_data, split_masks, to_bikes,
    to_model_space,
)

SEED = 42
PRIMARY_METRIC = "MAE"            # must match the metric argued in REPORT.md
ALPHAS = [0.0, 1.0, 10.0, 100.0]  # 0 = plain least squares (no regularisation)
LEARNING_RATES = [0.01, 0.1, 0.3, 0.5]  # stability limit is ~0.35, so 0.5 should diverge
EPOCHS = 3000                     # full-batch gradient steps

torch.manual_seed(SEED)
np.random.seed(SEED)
MODELS = ["sklearn", "manual_torch", "standard_torch"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def evaluate(y_true_bikes, y_pred_bikes) -> dict:
    """Business metrics in bikes/hour, plus the training objective (log-space MSE)."""
    if not np.all(np.isfinite(y_pred_bikes)):
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "log_MSE": np.nan}
    return {
        "MAE": mean_absolute_error(y_true_bikes, y_pred_bikes),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_bikes, y_pred_bikes))),
        "R2": r2_score(y_true_bikes, y_pred_bikes),
        "log_MSE": mean_squared_error(to_model_space(y_true_bikes),
                                      to_model_space(np.clip(y_pred_bikes, 0, None))),
    }


def make_design(fit_df, *other_dfs):
    """Learn preprocessing + scaling on fit_df only, then apply it to the others."""
    prep = fit_preprocessing(fit_df)
    X_fit = build_features(fit_df, prep)
    # Standardising puts every feature on the same scale: gradient descent
    # converges much faster, and all three models see identical inputs.
    scaler = StandardScaler().fit(X_fit)
    matrices = [scaler.transform(X_fit)]
    matrices += [scaler.transform(build_features(d, prep)) for d in other_dfs]
    return prep, scaler, list(X_fit.columns), matrices


def as_tensor(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


# ---------------------------------------------------------------------------
# The same model, three ways. All minimise:  mean((Xw + b - y)^2) + alpha/n * ||w||^2
# (sklearn's Ridge minimises sum(...) + alpha*||w||^2; dividing by n gives the
#  same minimiser, which is why the penalty below is alpha/n. Bias not penalised.)
# ---------------------------------------------------------------------------
def fit_sklearn(X, y, alpha):
    model = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
    return model.fit(X, y)


def fit_manual_torch(X, y, lr, epochs, alpha):
    """Session 5 style: raw tensors, autograd for gradients, manual update."""
    X_t, y_t = as_tensor(X), as_tensor(y)
    n, p = X_t.shape
    w = torch.zeros(p, requires_grad=True)   # convex problem: any start works
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for _ in range(epochs):
        pred = X_t @ w + b                                  # forward pass
        mse = ((pred - y_t) ** 2).mean()
        loss = mse + (alpha / n) * (w ** 2).sum()
        loss.backward()                                     # autograd fills .grad
        with torch.no_grad():                               # update is not part of the graph
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()                                  # otherwise gradients accumulate
            b.grad.zero_()
        losses.append(mse.item())
        if not np.isfinite(losses[-1]):                    # learning rate too high
            break
    return w.detach(), b.detach(), losses


def predict_manual(w, b, X):
    return (as_tensor(X) @ w + b).numpy()


def fit_standard_torch(X, y, lr, epochs, alpha):
    """Standard workflow: nn.Module + loss function + torch.optim optimiser."""
    X_t, y_t = as_tensor(X), as_tensor(y)
    n = len(X_t)
    model = LinearRegressionTorch(X_t.shape[1])   # PyTorch's default random init
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        mse = loss_fn(model(X_t), y_t)
        loss = mse + (alpha / n) * model.linear.weight.pow(2).sum()
        loss.backward()
        optimizer.step()
        losses.append(mse.item())
        if not np.isfinite(losses[-1]):
            break
    return model, losses


def predict_standard(model, X):
    with torch.no_grad():
        return model(as_tensor(X)).numpy()


# ---------------------------------------------------------------------------
# 1. Load + temporal split
# ---------------------------------------------------------------------------
start = time.time()
df = load_data()
masks = split_masks(df)
train_df, val_df, test_df = df[masks["train"]], df[masks["val"]], df[masks["test"]]
print(f"Rows  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

# ---------------------------------------------------------------------------
# 2. Model selection on validation (fit on train only)
# ---------------------------------------------------------------------------
_, _, _, (X_tr, X_va) = make_design(train_df, val_df)
y_tr = to_model_space(train_df[TARGET].to_numpy())
y_va = val_df[TARGET].to_numpy()

tuning = []
for alpha in ALPHAS:
    model = fit_sklearn(X_tr, y_tr, alpha)
    tuning.append({"model": "sklearn", "param": "alpha", "value": alpha,
                   **evaluate(y_va, to_bikes(model.predict(X_va)))})
best_alpha = min((r for r in tuning if r["param"] == "alpha"),
                 key=lambda r: r[PRIMARY_METRIC])["value"]

for lr in LEARNING_RATES:
    w, b, losses = fit_manual_torch(X_tr, y_tr, lr, EPOCHS, best_alpha)
    with np.errstate(over="ignore", invalid="ignore"):
        pred = to_bikes(predict_manual(w, b, X_va))
    row = {"model": "manual_torch", "param": "lr", "value": lr, **evaluate(y_va, pred)}
    row["diverged"] = not np.isfinite(losses[-1])
    tuning.append(row)
best_lr = min((r for r in tuning if r["param"] == "lr" and not r["diverged"]),
              key=lambda r: r[PRIMARY_METRIC])["value"]

tuning_df = pd.DataFrame(tuning)
print("\nValidation tuning (Jul-Sep 2012):")
print(tuning_df.round(3).to_string(index=False))
print(f"-> chosen alpha={best_alpha}, learning rate={best_lr} (by validation {PRIMARY_METRIC})")

# ---------------------------------------------------------------------------
# 3. Final fit on train+val (most recent data before the test period), test once
# ---------------------------------------------------------------------------
fit_df = pd.concat([train_df, val_df])
prep, scaler, feature_names, (X_fit, X_te) = make_design(fit_df, test_df)
y_fit = to_model_space(fit_df[TARGET].to_numpy())
y_te = test_df[TARGET].to_numpy()

sk_model = fit_sklearn(X_fit, y_fit, best_alpha)
w, b, manual_losses = fit_manual_torch(X_fit, y_fit, best_lr, EPOCHS, best_alpha)
std_model, std_losses = fit_standard_torch(X_fit, y_fit, best_lr, EPOCHS, best_alpha)

preds_log = {
    "sklearn": sk_model.predict(X_te),
    "manual_torch": predict_manual(w, b, X_te),
    "standard_torch": predict_standard(std_model, X_te),
}
preds = {name: to_bikes(p) for name, p in preds_log.items()}

# Naive baselines: what you could do with no model at all.
preds["baseline_mean"] = np.full(len(y_te), fit_df[TARGET].mean())
hourly_profile = fit_df.groupby(["hr", "workingday"])[TARGET].mean()
preds["baseline_hour_profile"] = hourly_profile.reindex(
    pd.MultiIndex.from_frame(test_df[["hr", "workingday"]])).to_numpy()

metrics_df = pd.DataFrame([{"model": name, **evaluate(y_te, p)} for name, p in preds.items()])
print("\nTest set (Oct-Dec 2012), evaluated once:")
print(metrics_df.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# Do the three implementations agree?
# ---------------------------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": feature_names,
    "sklearn": sk_model.coef_,
    "manual_torch": w.numpy(),
    "standard_torch": std_model.linear.weight.detach().numpy().ravel(),
})
intercepts = {"sklearn": float(sk_model.intercept_), "manual_torch": float(b.item()),
              "standard_torch": float(std_model.linear.bias.item())}
optimum_mse = mean_squared_error(y_fit, sk_model.predict(X_fit))  # sklearn = exact optimum

agreement = {}
for name in ["manual_torch", "standard_torch"]:
    agreement[name] = {
        "max_abs_coef_diff_vs_sklearn": float((coef_df[name] - coef_df["sklearn"]).abs().max()),
        "intercept_diff_vs_sklearn": intercepts[name] - intercepts["sklearn"],
        "max_abs_pred_diff_bikes": float(np.abs(preds[name] - preds["sklearn"]).max()),
        "final_train_mse_gap_vs_optimum": (manual_losses if name == "manual_torch"
                                           else std_losses)[-1] - optimum_mse,
    }
print("\nAgreement with scikit-learn (same split, same features):")
print(pd.DataFrame(agreement).T.to_string(float_format=lambda v: f"{v:.2e}"))

# ---------------------------------------------------------------------------
# 4. Save artifacts for app.py
# ---------------------------------------------------------------------------
ARTIFACTS_DIR.mkdir(exist_ok=True)
joblib.dump({"prep": prep, "scaler": scaler, "feature_names": feature_names},
            ARTIFACTS_DIR / "preprocessing.joblib")
joblib.dump(sk_model, ARTIFACTS_DIR / "sklearn_model.joblib")
torch.save({"w": w, "b": b}, ARTIFACTS_DIR / "manual_torch.pt")
torch.save(std_model.state_dict(), ARTIFACTS_DIR / "standard_torch.pt")

pred_out = test_df[["timestamp", "hr", "workingday", "temp", "weathersit"]].copy()
pred_out["actual"] = y_te
for name, p in preds.items():
    pred_out[name] = p
pred_out.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
metrics_df.to_csv(ARTIFACTS_DIR / "test_metrics.csv", index=False)
tuning_df.to_csv(ARTIFACTS_DIR / "val_tuning.csv", index=False)
coef_df.to_csv(ARTIFACTS_DIR / "coefficients.csv", index=False)
pd.DataFrame({"epoch": range(1, EPOCHS + 1), "manual_torch": manual_losses,
              "standard_torch": std_losses, "sklearn_optimum": optimum_mse}
             ).to_csv(ARTIFACTS_DIR / "loss_curves.csv", index=False)
with open(ARTIFACTS_DIR / "summary.json", "w") as f:
    json.dump({
        "split": {"val_start": str(VAL_START.date()), "test_start": str(TEST_START.date()),
                  "rows_train": len(train_df), "rows_val": len(val_df), "rows_test": len(test_df)},
        "chosen": {"alpha": best_alpha, "learning_rate": best_lr, "epochs": EPOCHS,
                   "selected_by": f"validation {PRIMARY_METRIC}"},
        "intercepts": intercepts,
        "agreement": agreement,
    }, f, indent=2)

print(f"\nArtifacts saved to {ARTIFACTS_DIR.name}/  ({time.time() - start:.1f}s)")
