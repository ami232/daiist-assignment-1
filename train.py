"""
Assignment 1: training pipeline (Bike Sharing hourly demand).

Run with:  uv run python main.py train

Steps:
1. Load data/hour.csv, build features (features.py), time-based split.
2. Naive baselines.
3. Tune on train -> validation: ridge strength (alpha) with scikit-learn,
   learning rate for the PyTorch versions.
4. Refit all three implementations of the same ridge regression on
   train+validation and evaluate once on the untouched test period.
5. Save models, scaler, metrics and test predictions to artifacts/ for app.py.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from features import build_features, make_target, prepare_dataset

SEED = 42
ARTIFACTS = Path("artifacts")
ALPHAS = [0.0, 1.0, 10.0, 100.0, 1000.0]      # ridge strength candidates
LEARNING_RATES = [0.01, 0.03, 0.1, 0.3, 1.0]  # for both PyTorch versions
EPOCHS = 3000                                  # full-batch gradient steps


# --------------------------------------------------------------------------
# Metrics: always reported in bikes/hour (the business unit), not log space.
# --------------------------------------------------------------------------
def to_counts(log_pred: np.ndarray) -> np.ndarray:
    # Undo log1p, and a forecast can't be a negative number of bikes.
    return np.clip(np.expm1(log_pred), 0, None)


def evaluate(y_true_counts, y_pred_counts) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true_counts, y_pred_counts)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_counts, y_pred_counts))),
        "R2": float(r2_score(y_true_counts, y_pred_counts)),
    }


# --------------------------------------------------------------------------
# The three implementations of the same model: ridge regression on log(1+cnt)
#
# All three minimise the same objective. scikit-learn's Ridge minimises
#     sum((y - Xw - b)^2) + alpha * ||w||^2
# Dividing by n (which doesn't move the minimum) gives the form used in both
# PyTorch versions:
#     mean((y - Xw - b)^2) + (alpha / n) * ||w||^2
# The intercept b is not penalised, matching scikit-learn.
# --------------------------------------------------------------------------
def fit_sklearn(X, y, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def fit_torch_manual(X, y, alpha, lr, epochs=EPOCHS):
    """Session 5 style: raw tensors, autograd, hand-written update."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    n, d = X_t.shape

    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for _ in range(epochs):
        pred = X_t @ w + b
        loss = ((pred - y_t) ** 2).mean() + (alpha / n) * (w ** 2).sum()
        loss.backward()                      # autograd fills w.grad, b.grad
        with torch.no_grad():                # the update itself isn't tracked
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()                   # grads accumulate unless cleared
            b.grad.zero_()
        losses.append(loss.item())
    return {"w": w.detach(), "b": b.detach()}, losses


def predict_torch_manual(params, X):
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        return (X_t @ params["w"] + params["b"]).numpy()


class LinearRegressionModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        # Same starting point as the manual loop (all zeros) so the two
        # PyTorch versions take exactly the same path.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def fit_torch_nn(X, y, alpha, lr, epochs=EPOCHS):
    """Standard workflow: nn.Module + loss function + torch.optim."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    n = X_t.shape[0]

    model = LinearRegressionModel(X_t.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        # Explicit L2 penalty on the weights only (not the bias). SGD's
        # weight_decay option would also shrink the bias and uses a
        # different scaling, so it would no longer match scikit-learn.
        loss = loss_fn(pred, y_t) + (alpha / n) * model.linear.weight.pow(2).sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


def predict_torch_nn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


# --------------------------------------------------------------------------
# Feature ablation: does each engineering step earn its place?
# --------------------------------------------------------------------------
def feature_ablation(df, X_full, is_train, is_val):
    """Add the engineered features one step at a time and score each version
    on validation (plain least squares, alpha=0, train -> val)."""
    raw = df[["hr", "workingday", "holiday", "mnth", "yr", "weathersit", "temp", "hum", "windspeed"]].astype(float)
    hours_shared = pd.DataFrame(
        {f"hr{h:02d}": X_full[f"hr{h:02d}_work"] + X_full[f"hr{h:02d}_off"] for h in range(1, 24)})
    calendar = X_full[[c for c in X_full.columns if c.startswith(("month_", "weather_"))]
                      + ["workingday", "holiday", "year_2012", "temp", "humidity", "windspeed"]]
    hour_by_daytype = X_full[[c for c in X_full.columns if c.startswith("hr")]]

    steps = [
        ("1. Raw columns as numbers, raw target", raw, False),
        ("2. + hour/month/weather as categories", pd.concat([hours_shared, calendar], axis=1), False),
        ("3. + separate hour profile per day type", pd.concat([hour_by_daytype, calendar], axis=1), False),
        ("4. + log(1+cnt) target", pd.concat([hour_by_daytype, calendar], axis=1), True),
        ("5. + temperature squared", pd.concat([hour_by_daytype, calendar, X_full[["temp_sq"]]], axis=1), True),
        ("6. + same hour last week (final)", X_full, True),
    ]
    counts = df["cnt"].to_numpy()
    rows = []
    for label, X, log_target in steps:
        sc = StandardScaler().fit(X[is_train])
        y = np.log1p(counts) if log_target else counts
        m = Ridge(alpha=0.0).fit(sc.transform(X[is_train]), y[is_train])
        pred = m.predict(sc.transform(X[is_val]))
        pred = to_counts(pred) if log_target else np.clip(pred, 0, None)
        rows.append({"step": label, "n_features": X.shape[1], **evaluate(counts[is_val], pred)})
    return rows


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)

    df = prepare_dataset()
    X_all = build_features(df)
    y_all = make_target(df)
    feature_names = list(X_all.columns)

    is_train = (df["split"] == "train").to_numpy()
    is_val = (df["split"] == "val").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    is_fit = is_train | is_val       # final models are refit on train+val
    for name, mask in [("train", is_train), ("val", is_val), ("test", is_test)]:
        part = df[mask]
        print(f"{name:>5}: {mask.sum():>5} rows  "
              f"{part['timestamp'].min():%Y-%m-%d} -> {part['timestamp'].max():%Y-%m-%d}")
    print(f"{len(feature_names)} features")

    counts = df["cnt"].to_numpy()

    # ---- Naive baselines -------------------------------------------------
    # 1) Same hour last week: what an operator might do with no model at all.
    # 2) Average count for that (hour, working-day) pair over the fit period.
    #    This is the hour profile with no weather, trend or season.
    profile = df[is_fit].groupby(["hr", "workingday"])["cnt"].mean()
    profile_pred = df.set_index(["hr", "workingday"]).index.map(profile).to_numpy()
    baselines = {
        "Baseline: same hour last week": df["cnt_lag_week"].to_numpy(),
        "Baseline: hour x day-type average": profile_pred,
    }

    ablation = feature_ablation(df, X_all, is_train, is_val)
    print("\nFeature ablation (validation, bikes/hour):")
    print(pd.DataFrame(ablation).round(2).to_string(index=False))

    # ---- Stage 1: tune on train, judge on validation ----------------------
    scaler = StandardScaler().fit(X_all[is_train])
    X_tr, X_va = scaler.transform(X_all[is_train]), scaler.transform(X_all[is_val])
    y_tr, y_va = y_all[is_train].to_numpy(), y_all[is_val].to_numpy()

    alpha_results = []
    for alpha in ALPHAS:
        m = fit_sklearn(X_tr, y_tr, alpha)
        alpha_results.append({"alpha": alpha, **evaluate(counts[is_val], to_counts(m.predict(X_va)))})
    alpha_table = pd.DataFrame(alpha_results)
    best_alpha = float(alpha_table.loc[alpha_table["MAE"].idxmin(), "alpha"])
    print("\nRidge alpha search (validation, bikes/hour):")
    print(alpha_table.round(2).to_string(index=False))
    print(f"-> alpha = {best_alpha}")

    # Learning rate: pick the one that gets closest to the optimum in EPOCHS
    # steps, judged by validation MAE. Too high diverges (loss -> inf/nan),
    # too low hasn't finished converging.
    lr_results = []
    for lr in LEARNING_RATES:
        params, losses = fit_torch_manual(X_tr, y_tr, best_alpha, lr)
        pred = predict_torch_manual(params, X_va)
        if not np.all(np.isfinite(pred)):
            lr_results.append({"lr": lr, "final_train_loss": float("nan"), "MAE": float("nan")})
            continue
        lr_results.append({"lr": lr, "final_train_loss": losses[-1],
                           "MAE": evaluate(counts[is_val], to_counts(pred))["MAE"]})
    lr_table = pd.DataFrame(lr_results)
    best_lr = float(lr_table.loc[lr_table["MAE"].idxmin(), "lr"])
    print("\nLearning-rate search (manual PyTorch, validation):")
    print(lr_table.round(4).to_string(index=False))
    print(f"-> lr = {best_lr}")

    # ---- Stage 2: refit on train+val, evaluate once on test --------------
    # Validation is the most recent data before the test period. Leaving it
    # out of the final fit would throw away the months closest to what we
    # forecast, which matters with a growing system.
    scaler = StandardScaler().fit(X_all[is_fit])
    X_fit, X_te = scaler.transform(X_all[is_fit]), scaler.transform(X_all[is_test])
    y_fit = y_all[is_fit].to_numpy()

    sk_model = fit_sklearn(X_fit, y_fit, best_alpha)
    manual_params, manual_losses = fit_torch_manual(X_fit, y_fit, best_alpha, best_lr)
    nn_model, nn_losses = fit_torch_nn(X_fit, y_fit, best_alpha, best_lr)

    test_preds = {
        "scikit-learn Ridge": to_counts(sk_model.predict(X_te)),
        "PyTorch manual loop": to_counts(predict_torch_manual(manual_params, X_te)),
        "PyTorch nn.Module + optim": to_counts(predict_torch_nn(nn_model, X_te)),
    }
    results = {name: evaluate(counts[is_test], pred[is_test]) for name, pred in baselines.items()}
    results.update({name: evaluate(counts[is_test], pred) for name, pred in test_preds.items()})
    results_table = pd.DataFrame(results).T
    print("\nTest set (Oct-Dec 2012), bikes/hour:")
    print(results_table.round(2).to_string())

    # ---- Do the three implementations agree? ------------------------------
    coefs = pd.DataFrame({
        "scikit-learn Ridge": sk_model.coef_,
        "PyTorch manual loop": manual_params["w"].numpy(),
        "PyTorch nn.Module + optim": nn_model.linear.weight.detach().numpy().ravel(),
    }, index=feature_names)
    intercepts = {
        "scikit-learn Ridge": float(sk_model.intercept_),
        "PyTorch manual loop": float(manual_params["b"].item()),
        "PyTorch nn.Module + optim": float(nn_model.linear.bias.item()),
    }
    sk = test_preds["scikit-learn Ridge"]
    agreement = {
        "max_abs_coef_diff_manual_vs_sklearn": float((coefs["PyTorch manual loop"] - coefs["scikit-learn Ridge"]).abs().max()),
        "max_abs_coef_diff_nn_vs_sklearn": float((coefs["PyTorch nn.Module + optim"] - coefs["scikit-learn Ridge"]).abs().max()),
        "max_abs_coef_diff_manual_vs_nn": float((coefs["PyTorch manual loop"] - coefs["PyTorch nn.Module + optim"]).abs().max()),
        "max_abs_pred_diff_manual_vs_sklearn_bikes": float(np.abs(test_preds["PyTorch manual loop"] - sk).max()),
        "max_abs_pred_diff_nn_vs_sklearn_bikes": float(np.abs(test_preds["PyTorch nn.Module + optim"] - sk).max()),
    }
    print("\nAgreement between implementations:")
    for k, v in agreement.items():
        print(f"  {k}: {v:.6f}")

    # ---- Save everything app.py needs (it never retrains) ----------------
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
    joblib.dump(sk_model, ARTIFACTS / "sklearn_ridge.joblib")
    torch.save(manual_params, ARTIFACTS / "torch_manual.pt")
    torch.save(nn_model.state_dict(), ARTIFACTS / "torch_nn.pt")

    test_df = df.loc[is_test, ["timestamp", "hr", "workingday", "cnt"]].reset_index(drop=True)
    for name, pred in baselines.items():
        test_df[name] = pred[is_test]
    for name, pred in test_preds.items():
        test_df[name] = pred
    test_df.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    coefs.to_csv(ARTIFACTS / "coefficients.csv", index_label="feature")

    meta = {
        "feature_names": feature_names,
        "alpha": best_alpha,
        "learning_rate": best_lr,
        "epochs": EPOCHS,
        "split_rows": {"train": int(is_train.sum()), "val": int(is_val.sum()), "test": int(is_test.sum())},
        "feature_ablation": ablation,
        "alpha_search": alpha_results,
        "lr_search": lr_results,
        "test_metrics": results,
        "intercepts": intercepts,
        "agreement": agreement,
        "loss_curves": {"PyTorch manual loop": manual_losses, "PyTorch nn.Module + optim": nn_losses},
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved artifacts to {ARTIFACTS.resolve()}")


if __name__ == "__main__":
    main()
