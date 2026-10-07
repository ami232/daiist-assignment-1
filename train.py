"""Train the same ridge-regularised linear regression three ways and compare them.

    1. scikit-learn Ridge   (exact solver)
    2. Manual PyTorch       (tensors + autograd, parameters updated by hand)
    3. Standard PyTorch     (nn.Module + nn.MSELoss + torch.optim.SGD)

All three minimise the same objective on the same features and the same split:

    mean((X @ w + b - y)^2) + (alpha / n) * sum(w^2)

This is sklearn's Ridge objective, sum((X @ w + b - y)^2) + alpha * sum(w^2), divided by n,
so it has the same minimum. As in sklearn, the bias b is not penalised.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch import nn

from features import FLAGS, NUMERIC, TARGET, add_features, build_preprocessor, load_data, split_by_time
from models import LinearRegressionModel

ARTIFACTS = Path(__file__).parent / "artifacts"
SEED = 0
ALPHAS = [0.001, 0.01, 0.1, 1, 10]     # regularisation strengths to try (tuned on validation)
LEARNING_RATES = [0.1, 0.5, 0.8, 1.0]  # for the two PyTorch versions (tuned on validation)
EPOCHS = 20_000                        # full-batch gradient descent steps
SPECIAL_WEEK_WEIGHT = 5                # holiday weeks count 5x in WMAE: they matter most to the business


# ---------------------------------------------------------------- metrics
def evaluate(df, log_pred):
    """Business metrics in dollars, after undoing the log transform."""
    y = df["Weekly_Sales"].to_numpy()
    pred = np.exp(log_pred)
    weights = np.where(df[FLAGS].sum(axis=1) > 0, SPECIAL_WEEK_WEIGHT, 1)
    abs_err = np.abs(y - pred)
    return {
        "WMAE": float(np.sum(weights * abs_err) / np.sum(weights)),
        "MAE": float(abs_err.mean()),
        "MAPE_%": float(100 * np.mean(abs_err / y)),
        "R2": float(r2_score(y, pred)),
    }


def to_tensor(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


# ---------------------------------------------------------------- method 2: manual PyTorch
def train_manual(X, y, alpha, lr, epochs=EPOCHS):
    """We create the parameters, compute the loss, and apply the gradient-descent update ourselves."""
    X_t, y_t = to_tensor(X), to_tensor(y).reshape(-1, 1)
    n, d = X_t.shape
    w = torch.zeros(d, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(epochs):
        y_hat = X_t @ w + b
        loss = torch.mean((y_hat - y_t) ** 2) + (alpha / n) * torch.sum(w ** 2)
        if not torch.isfinite(loss):
            return None, history          # learning rate too high: the loss blew up
        loss.backward()                   # autograd computes d(loss)/dw and d(loss)/db
        with torch.no_grad():             # the update itself must not be tracked by autograd
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()                # gradients accumulate unless reset each step
            b.grad.zero_()
        if epoch % 100 == 0:
            history.append(loss.item())
    return {"w": w.detach(), "b": b.detach()}, history


def predict_manual(params, X):
    return (to_tensor(X) @ params["w"] + params["b"]).numpy().ravel()


# ---------------------------------------------------------------- method 3: standard PyTorch
def train_standard(X, y, alpha, lr, epochs=EPOCHS):
    """Same maths, packaged: nn.Module holds the parameters and the optimiser applies the update."""
    torch.manual_seed(SEED)               # nn.Linear starts from random weights; make it repeatable
    X_t, y_t = to_tensor(X), to_tensor(y).reshape(-1, 1)
    n, d = X_t.shape
    model = LinearRegressionModel(d)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = mse(model(X_t), y_t) + (alpha / n) * model.linear.weight.pow(2).sum()
        if not torch.isfinite(loss):
            return None, history
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            history.append(loss.item())
    return model, history


def predict_standard(model, X):
    with torch.no_grad():
        return model(to_tensor(X)).numpy().ravel()


# ---------------------------------------------------------------- pipeline
def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = add_features(load_data())
    train, val, test = split_by_time(df)
    print(f"Split: train {len(train)} rows, validation {len(val)}, test {len(test)}")

    # 1. Tuning: fit on train, score on validation. The test set is not touched here.
    pre = build_preprocessor().fit(train)   # scaler statistics come from training rows only
    X_tr, y_tr = pre.transform(train), train[TARGET].to_numpy()
    X_val = pre.transform(val)

    alpha_scores = {a: evaluate(val, Ridge(alpha=a).fit(X_tr, y_tr).predict(X_val))["WMAE"] for a in ALPHAS}
    best_alpha = min(alpha_scores, key=alpha_scores.get)
    print("Validation WMAE by alpha:", {a: round(s) for a, s in alpha_scores.items()}, "-> best", best_alpha)

    lr_scores = {"manual": {}, "standard": {}}
    for lr in LEARNING_RATES:
        params, _ = train_manual(X_tr, y_tr, best_alpha, lr)
        lr_scores["manual"][lr] = evaluate(val, predict_manual(params, X_val))["WMAE"] if params else None
        model, _ = train_standard(X_tr, y_tr, best_alpha, lr)
        lr_scores["standard"][lr] = evaluate(val, predict_standard(model, X_val))["WMAE"] if model else None
        print(f"  lr={lr}: manual {lr_scores['manual'][lr]}, standard {lr_scores['standard'][lr]}  (None = diverged)")
    best_lr = {m: min((lr for lr, s in sc.items() if s is not None), key=sc.get) for m, sc in lr_scores.items()}
    print("Best learning rates:", best_lr)

    # 2. Final fit on train + validation with the chosen settings.
    full = pd.concat([train, val])
    pre = build_preprocessor().fit(full)
    X_full, y_full = pre.transform(full), full[TARGET].to_numpy()
    X_test = pre.transform(test)

    sk_model = Ridge(alpha=best_alpha).fit(X_full, y_full)
    manual_params, manual_hist = train_manual(X_full, y_full, best_alpha, best_lr["manual"])
    torch_model, standard_hist = train_standard(X_full, y_full, best_alpha, best_lr["standard"])
    if manual_params is None or torch_model is None:
        raise RuntimeError("Final PyTorch training diverged: lower the learning rate.")
    store_avg = full.groupby("Store")["Weekly_Sales"].mean()   # the naive baseline's "model"

    # 3. Evaluate everything once on the untouched test year.
    preds = {
        "Baseline (store average)": np.log(test["Store"].map(store_avg).to_numpy()),
        "scikit-learn": sk_model.predict(X_test),
        "Manual PyTorch": predict_manual(manual_params, X_test),
        "Standard PyTorch": predict_standard(torch_model, X_test),
    }
    results = pd.DataFrame({name: evaluate(test, p) for name, p in preds.items()}).T
    print("\nTest-year results (dollars):")
    print(results.to_string(formatters={"WMAE": "{:,.0f}".format, "MAE": "{:,.0f}".format,
                                        "MAPE_%": "{:.2f}".format, "R2": "{:.4f}".format}))

    # 4. Do the three methods agree?
    names = list(pre.get_feature_names_out())
    coefs = pd.DataFrame({
        "scikit-learn": np.append(sk_model.coef_, sk_model.intercept_),
        "Manual PyTorch": np.append(manual_params["w"].numpy().ravel(), manual_params["b"].numpy()),
        "Standard PyTorch": np.append(torch_model.linear.weight.detach().numpy().ravel(),
                                      torch_model.linear.bias.detach().numpy()),
    }, index=names + ["intercept"])
    sk_pred = np.exp(preds["scikit-learn"])
    agreement = {
        f"{m} vs scikit-learn": {
            "max_abs_coef_diff": float((coefs[m] - coefs["scikit-learn"]).abs().max()),
            "max_test_pred_diff_%": float(100 * np.max(np.abs(np.exp(preds[m]) / sk_pred - 1))),
        }
        for m in ["Manual PyTorch", "Standard PyTorch"]
    }
    print("\nAgreement with scikit-learn:", json.dumps(agreement, indent=2))

    drivers = coefs.loc[NUMERIC + FLAGS, "scikit-learn"]
    print("\nDrivers (scikit-learn): % change in weekly sales")
    print((100 * (np.exp(drivers) - 1)).round(2).to_string())

    # 5. Save everything the app needs. The app loads these and never retrains.
    joblib.dump(pre, ARTIFACTS / "preprocessor.joblib")
    joblib.dump(sk_model, ARTIFACTS / "sklearn_ridge.joblib")
    torch.save(manual_params, ARTIFACTS / "manual_torch.pt")
    torch.save(torch_model.state_dict(), ARTIFACTS / "standard_torch.pt")
    store_avg.to_csv(ARTIFACTS / "baseline_store_avg.csv")
    coefs.to_csv(ARTIFACTS / "coefficients.csv")
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({
            "best_alpha": best_alpha,
            "best_lr": best_lr,
            "validation_wmae_by_alpha": {str(a): s for a, s in alpha_scores.items()},
            "validation_wmae_by_lr": {m: {str(lr): s for lr, s in sc.items()} for m, sc in lr_scores.items()},
            "test_results": results.to_dict(orient="index"),
            "agreement": agreement,
            "loss_history_every_100_epochs": {"manual": manual_hist, "standard": standard_hist},
        }, f, indent=2)
    print(f"\nSaved artifacts to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
