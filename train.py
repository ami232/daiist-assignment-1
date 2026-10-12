# Assignment 1 - training pipeline

# Trains the same linear regression model three ways on an identical train/val/test split, compares all three against a naive baseline

import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from common import (
    RANDOM_STATE, load_raw, split_data, fit_preprocessing, transform, all_metrics, mae,
)

ARTIFACT_DIR = Path("models")
ARTIFACT_DIR.mkdir(exist_ok=True)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def run():
    t0 = time.time()

    # 1. data
    df = load_raw()
    train, val, test = split_data(df)
    bedroom_imputer, ohe = fit_preprocessing(train)

    X_train, y_train, scaler = transform(train, bedroom_imputer, ohe, fit_scaler=True)
    X_val, y_val, _ = transform(val, bedroom_imputer, ohe, scaler=scaler)
    X_test, y_test, _ = transform(test, bedroom_imputer, ohe, scaler=scaler)

    feature_names = list(X_train.columns)
    print(f"[data] {len(X_train)} train / {len(X_val)} val / {len(X_test)} test rows, "
          f"{len(feature_names)} features")

    # 2. baseline: predict the training mean for everyone
    baseline_pred_val = np.full_like(y_val, y_train.mean())
    baseline_pred_test = np.full_like(y_test, y_train.mean())
    results = {"baseline": {"val": all_metrics(y_val, baseline_pred_val),
                             "test": all_metrics(y_test, baseline_pred_test)}}
    print(f"[baseline] val MAE=${results['baseline']['val']['MAE']:,.0f}  "
          f"test MAE=${results['baseline']['test']['MAE']:,.0f}")

    # 3. scikit-learn Ridge: tune alpha on validation MAE 
    from sklearn.linear_model import Ridge
    alpha_grid = [0.01, 0.1, 1, 10, 30, 100, 300]
    best_alpha, best_val_mae = None, float("inf")
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        v_mae = mae(y_val, model.predict(X_val))
        if v_mae < best_val_mae:
            best_val_mae, best_alpha = v_mae, alpha
    sk_model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    sk_model.fit(X_train, y_train)
    results["sklearn"] = {
        "alpha": best_alpha,
        "val": all_metrics(y_val, sk_model.predict(X_val)),
        "test": all_metrics(y_test, sk_model.predict(X_test)),
    }
    print(f"[sklearn Ridge] best alpha={best_alpha}  "
          f"val MAE=${results['sklearn']['val']['MAE']:,.0f}  "
          f"test MAE=${results['sklearn']['test']['MAE']:,.0f}")

    # 4. tensors + target scaling (gradient-descent methods only)
    y_mean, y_std = y_train.mean(), y_train.std()

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32)

    X_train_t, X_val_t, X_test_t = to_t(X_train.values), to_t(X_val.values), to_t(X_test.values)
    y_train_s = to_t((y_train - y_mean) / y_std).view(-1, 1)
    y_val_s = to_t((y_val - y_mean) / y_std).view(-1, 1)
    n_features = X_train_t.shape[1]

    def unscale(pred_s):
        return (pred_s * y_std + y_mean).detach().numpy().ravel()

    # 5. manual PyTorch loop: raw tensors + autograd + manual update
    def train_manual(lr, lam, epochs=3000):
        torch.manual_seed(RANDOM_STATE)
        w = torch.zeros(n_features, 1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        for _ in range(epochs):
            y_pred = X_train_t @ w + b
            mse = torch.mean((y_pred - y_train_s) ** 2)
            loss = mse + lam * torch.sum(w ** 2)
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
        return w.detach(), b.detach()

    lr_grid = [0.01, 0.05, 0.1, 0.3]
    lam_grid = [0.0, 0.001, 0.01]
    best_lr, best_lam, best_val_mae = None, None, float("inf")
    for lr in lr_grid:
        for lam in lam_grid:
            w, b = train_manual(lr, lam, epochs=1500)
            val_pred = unscale(X_val_t @ w + b)
            v_mae = mae(y_val, val_pred)
            if np.isfinite(v_mae) and v_mae < best_val_mae:
                best_val_mae, best_lr, best_lam = v_mae, lr, lam

    w_manual, b_manual = train_manual(best_lr, best_lam, epochs=4000)
    manual_val_pred = unscale(X_val_t @ w_manual + b_manual)
    manual_test_pred = unscale(X_test_t @ w_manual + b_manual)
    results["manual_pytorch"] = {
        "lr": best_lr, "lam": best_lam,
        "val": all_metrics(y_val, manual_val_pred),
        "test": all_metrics(y_test, manual_test_pred),
    }
    print(f"[manual pytorch] best lr={best_lr} lam={best_lam}  "
          f"val MAE=${results['manual_pytorch']['val']['MAE']:,.0f}  "
          f"test MAE=${results['manual_pytorch']['test']['MAE']:,.0f}")

    # 6. standard PyTorch: nn.Module + torch.optim
    torch.manual_seed(RANDOM_STATE)
    linear = torch.nn.Linear(n_features, 1)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.SGD([
        {"params": [linear.weight], "weight_decay": 2 * best_lam},
        {"params": [linear.bias], "weight_decay": 0.0},
    ], lr=best_lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(4000):
        optimizer.zero_grad()
        y_pred = linear(X_train_t)
        loss = loss_fn(y_pred, y_train_s)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        std_val_pred = unscale(linear(X_val_t))
        std_test_pred = unscale(linear(X_test_t))
    results["standard_pytorch"] = {
        "lr": best_lr, "lam": best_lam,
        "val": all_metrics(y_val, std_val_pred),
        "test": all_metrics(y_test, std_test_pred),
    }
    print(f"[standard pytorch] (reused lr/lam)  "
          f"val MAE=${results['standard_pytorch']['val']['MAE']:,.0f}  "
          f"test MAE=${results['standard_pytorch']['test']['MAE']:,.0f}")

    manual_vs_standard_gap = abs(
        results["manual_pytorch"]["val"]["MAE"] - results["standard_pytorch"]["val"]["MAE"]
    )
    print(f"[sanity check] manual vs standard val MAE gap: ${manual_vs_standard_gap:,.2f} "
          f"(should be small - both implement the same update rule)")

    # 7. save everything needed for app.py
    joblib.dump(sk_model, ARTIFACT_DIR / "sklearn_ridge.joblib")
    joblib.dump({"bedroom_imputer": bedroom_imputer, "ohe": ohe, "scaler": scaler,
                 "feature_names": feature_names, "y_mean": float(y_mean), "y_std": float(y_std)},
                ARTIFACT_DIR / "preprocessing.joblib")
    torch.save({"w": w_manual, "b": b_manual}, ARTIFACT_DIR / "manual_pytorch.pt")
    torch.save(linear.state_dict(), ARTIFACT_DIR / "standard_pytorch.pt")

    # Test-set predictions from all methods (for the dashboard)
    test_predictions = {
        "y_true": y_test.tolist(),
        "baseline": baseline_pred_test.tolist(),
        "sklearn": sk_model.predict(X_test).tolist(),
        "manual_pytorch": manual_test_pred.tolist(),
        "standard_pytorch": std_test_pred.tolist(),
    }
    with open(ARTIFACT_DIR / "test_predictions.json", "w") as f:
        json.dump(test_predictions, f)

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Feature/target distributions (for the dashboard)
    dist_data = {
        "median_house_value": train["median_house_value"].tolist(),
        "median_income": train["median_income"].tolist(),
        "housing_median_age": train["housing_median_age"].tolist(),
        "ocean_proximity": train["ocean_proximity"].tolist(),
        "latitude": train["latitude"].tolist(),
        "longitude": train["longitude"].tolist(),
    }
    with open(ARTIFACT_DIR / "distributions.json", "w") as f:
        json.dump(dist_data, f)

    print(f"\n[done] saved artifacts to {ARTIFACT_DIR}/  ({time.time()-t0:.1f}s total)")
    print("\nComparison (test set)")
    for name, r in results.items():
        t = r["test"]
        print(f"{name:20s}  MAE=${t['MAE']:>10,.0f}   RMSE=${t['RMSE']:>10,.0f}   R2={t['R2']:.3f}")


if __name__ == "__main__":
    run()
