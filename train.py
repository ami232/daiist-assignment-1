"""
Assignment 1 - training pipeline (run with: uv run python main.py train).

Trains ONE model - ridge-regularised linear regression on log(1 + hourly rentals) -
three ways on the same chronological split, compares them with two naive
baselines, and saves everything app.py needs into artifacts/.
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from features import (ARTIFACTS, FEATURE_SETS, FINAL_SET, apply_scaler, build_features,
                      fit_scaler, load_data, time_split, to_bikes)

SEED = 0
ALPHAS = [0.1, 1.0, 10.0]            # ridge strengths tried (chosen on validation)
LEARNING_RATES = [0.01, 0.1, 0.5]    # tried for the PyTorch loops
EPOCHS = 3000                        # full-batch gradient steps


def metrics(y_bikes, pred_bikes) -> dict:
    err = np.asarray(pred_bikes) - np.asarray(y_bikes)
    return {"MAE (bikes)": float(np.abs(err).mean()),
            "RMSE (bikes)": float(np.sqrt((err ** 2).mean())),
            "Bias (bikes)": float(err.mean())}       # negative = under-forecast


def prepare(df, split, feature_set):
    """Build + scale one feature set. Scaler statistics come from train rows only."""
    X_raw = build_features(df, **FEATURE_SETS[feature_set])
    scaler = fit_scaler(X_raw[split == "train"])
    return apply_scaler(X_raw, scaler), scaler


# ---------------------------------------------------------------- 1) scikit-learn
def train_sklearn(X, y, alpha):
    return Ridge(alpha=alpha).fit(X, y)


# ------------------------------------------- shared loss for both PyTorch versions
def ridge_loss(pred, y, w, alpha):
    """Mean squared error + L2 penalty on the weights (not the bias).
    Dividing alpha by n makes it the same objective scikit-learn's Ridge minimises
    (sum of squared errors + alpha * ||w||^2), just divided by n."""
    return ((pred - y) ** 2).mean() + alpha / len(y) * (w ** 2).sum()


# ------------------------------------------------------- 2) manual PyTorch loop
def train_manual(X, y, alpha, lr, epochs=EPOCHS):
    """Raw tensors, autograd for the gradients, and a hand-written update step."""
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for _ in range(epochs):
        loss = ridge_loss(X @ w + b, y, w, alpha)
        loss.backward()                      # fills w.grad and b.grad
        with torch.no_grad():                # the update itself must not be tracked
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        history.append(loss.item())
    return w.detach(), b.detach(), history


# ----------------------------------------------- 3) standard PyTorch workflow
def train_standard(X, y, alpha, lr, epochs=EPOCHS):
    """nn.Module + torch.optim: same maths, PyTorch does the bookkeeping."""
    model = torch.nn.Linear(X.shape[1], 1)
    torch.nn.init.zeros_(model.weight)       # same starting point as the manual loop
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = ridge_loss(model(X).squeeze(1), y, model.weight, alpha)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return model, history


def main():
    torch.manual_seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)

    df = load_data()
    split = time_split(df)
    y_bikes = df["cnt"].to_numpy(dtype=float)
    y_log = np.log1p(y_bikes)
    tr, va, te = (split == "train").to_numpy(), (split == "val").to_numpy(), (split == "test").to_numpy()
    for name, mask in [("train", tr), ("val", va), ("test", te)]:
        d = df.loc[mask, "dteday"]
        print(f"{name:5s} {mask.sum():6d} rows  {d.min().date()} -> {d.max().date()}  mean rentals {y_bikes[mask].mean():.0f}")

    # ---- naive baselines (computed from training rows only) ----
    results = {}
    base_mean = np.full(len(df), to_bikes(y_log[tr].mean()))
    profile = (df[tr].assign(y=y_log[tr]).groupby(["hr", "workingday"])["y"].mean().rename("p").reset_index())
    base_profile = to_bikes(df.merge(profile, on=["hr", "workingday"], how="left")["p"].to_numpy())
    preds = {"Baseline: training average": base_mean,
             "Baseline: average for that hour & day type": base_profile}

    # ---- feature ablation with scikit-learn (what did each feature step buy?) ----
    ablation = []
    for fs in FEATURE_SETS:
        X, _ = prepare(df, split, fs)
        m = train_sklearn(X[tr], y_log[tr], alpha=1.0)
        row = {"Feature set": fs, "Features": X.shape[1]}
        for name, mask in [("val", va), ("test", te)]:
            mm = metrics(y_bikes[mask], to_bikes(m.predict(X[mask])))
            row[f"{name} MAE"], row[f"{name} bias"] = mm["MAE (bikes)"], mm["Bias (bikes)"]
        ablation.append(row)
    ablation = pd.DataFrame(ablation)
    print("\nFeature ablation (scikit-learn, alpha=1):\n", ablation.round(1).to_string(index=False))

    # ---- final feature set ----
    X, scaler = prepare(df, split, FINAL_SET)
    feature_names = list(X.columns)
    Xn = X.to_numpy(dtype=float)

    # regularisation strength: pick on the validation set, never on test
    alpha_search = []
    for a in ALPHAS:
        m = train_sklearn(Xn[tr], y_log[tr], a)
        alpha_search.append({"alpha": a, "val MAE": metrics(y_bikes[va], to_bikes(m.predict(Xn[va])))["MAE (bikes)"]})
    alpha = min(alpha_search, key=lambda r: r["val MAE"])["alpha"]
    print("\nalpha search:", alpha_search, "-> chosen", alpha)

    sk = train_sklearn(Xn[tr], y_log[tr], alpha)

    Xt = torch.tensor(Xn[tr], dtype=torch.float32)
    yt = torch.tensor(y_log[tr], dtype=torch.float32)

    # learning rate: run the manual loop for each candidate, keep the lowest final training loss
    lr_curves = {}
    for lr in LEARNING_RATES:
        _, _, hist = train_manual(Xt, yt, alpha, lr)
        lr_curves[str(lr)] = hist
        print(f"lr={lr}: final training loss {hist[-1]:.5f}")
    lr = float(min(lr_curves, key=lambda k: lr_curves[k][-1] if np.isfinite(lr_curves[k][-1]) else np.inf))
    print("chosen learning rate:", lr)

    w, b, manual_hist = train_manual(Xt, yt, alpha, lr)
    std_model, std_hist = train_standard(Xt, yt, alpha, lr)

    # ---- predictions of the three implementations on every row ----
    X_all = torch.tensor(Xn, dtype=torch.float32)
    with torch.no_grad():
        preds["scikit-learn"] = to_bikes(sk.predict(Xn))
        preds["PyTorch manual loop"] = to_bikes((X_all @ w + b).numpy())
        preds["PyTorch nn.Module + optim"] = to_bikes(std_model(X_all).squeeze(1).numpy())

    for name, p in preds.items():
        results[name] = {f"{s} {k}": v for s, mask in [("val", va), ("test", te)]
                         for k, v in metrics(y_bikes[mask], p[mask]).items()}
    results = pd.DataFrame(results).T.rename_axis("Model").reset_index()
    print("\n", results.round(2).to_string(index=False))

    coefs = pd.DataFrame({
        "feature": feature_names + ["(intercept)"],
        "scikit-learn": list(sk.coef_) + [sk.intercept_],
        "PyTorch manual loop": list(w.numpy()) + [b.item()],
        "PyTorch nn.Module + optim": list(std_model.weight.detach().numpy().ravel()) + [std_model.bias.item()],
    })
    agree = {
        "max |coef diff| manual vs sklearn": float((coefs["PyTorch manual loop"] - coefs["scikit-learn"]).abs().max()),
        "max |coef diff| nn.Module vs sklearn": float((coefs["PyTorch nn.Module + optim"] - coefs["scikit-learn"]).abs().max()),
        "max |test prediction diff| manual vs sklearn (bikes)": float(np.abs(preds["PyTorch manual loop"] - preds["scikit-learn"])[te].max()),
        "max |test prediction diff| nn.Module vs sklearn (bikes)": float(np.abs(preds["PyTorch nn.Module + optim"] - preds["scikit-learn"])[te].max()),
    }
    print("\nagreement:", json.dumps(agree, indent=1))

    # ---- save everything the dashboard needs ----
    joblib.dump(sk, ARTIFACTS / "model_sklearn.joblib")
    torch.save({"w": w, "b": b}, ARTIFACTS / "model_manual.pt")
    torch.save(std_model.state_dict(), ARTIFACTS / "model_standard.pt")
    results.to_csv(ARTIFACTS / "results.csv", index=False)
    ablation.to_csv(ARTIFACTS / "ablation.csv", index=False)
    coefs.to_csv(ARTIFACTS / "coefficients.csv", index=False)
    step = 10                                   # keep the saved loss curves small
    curves = {"epoch": list(range(0, EPOCHS, step)),
              "lr_search_manual": {k: v[::step] for k, v in lr_curves.items()},
              "final_manual": manual_hist[::step], "final_standard": std_hist[::step]}
    (ARTIFACTS / "curves.json").write_text(json.dumps(curves))
    (ARTIFACTS / "meta.json").write_text(json.dumps({
        "feature_set": FINAL_SET, "feature_names": feature_names, "scaler": scaler,
        "alpha": alpha, "alpha_search": alpha_search, "learning_rate": lr, "epochs": EPOCHS,
        "agreement": agree}, indent=1))
    print(f"\nSaved models and reports to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
