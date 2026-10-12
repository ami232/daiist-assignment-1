"""Assignment 1 - training pipeline (step 2: feature engineering)."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import json
from pathlib import Path
import joblib

DATA_PATH = "data/hour.csv"
TRAIN_END = "2012-06-30"   # train: 2011-01-01 to 2012-06-30
VAL_END = "2012-09-30"     # validation: 2012-07-01 to 2012-09-30
                           # test: 2012-10-01 to 2012-12-31
NUMERIC = ["temp", "hum"]  # columns that get scaled


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    # casual + registered = cnt, so keeping them would give away the answer
    df = df.drop(columns=["casual", "registered"])
    return df


def split_by_time(df):
    train = df[df["dteday"] <= TRAIN_END]
    val = df[(df["dteday"] > TRAIN_END) & (df["dteday"] <= VAL_END)]
    test = df[df["dteday"] > VAL_END]
    return train, val, test


def one_hot(values, categories, prefix, index):
    """Yes/no columns with a fixed set of categories, so train, val and
    test always end up with exactly the same columns."""
    cat = pd.Series(pd.Categorical(values, categories=categories), index=index)
    return pd.get_dummies(cat, prefix=prefix, drop_first=True, dtype=float)


def build_features(df):
    out = pd.DataFrame(index=df.index)
    out["yr"] = df["yr"]
    out["temp"] = df["temp"]
    out["hum"] = df["hum"]
    # hour x workday: 48 groups (24 hours for non-working days + 24 for working days)
    hr_wd = df["workingday"] * 24 + df["hr"]
    parts = [
        out,
        one_hot(hr_wd, range(48), "hr_wd", df.index),
        one_hot(df["season"], [1, 2, 3, 4], "season", df.index),
        # weather 4 (heavy rain/snow) is very rare, so it is merged into 3
        one_hot(df["weathersit"].clip(upper=3), [1, 2, 3], "weather", df.index),
    ]
    return pd.concat(parts, axis=1).astype(float)


def fit_scaler(X_train):
    """Mean and standard deviation from the TRAINING rows only."""
    return X_train[NUMERIC].mean(), X_train[NUMERIC].std()


def apply_scaler(X, mean, std):
    X = X.copy()
    X[NUMERIC] = (X[NUMERIC] - mean) / std
    return X

def evaluate(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)  # positive = predicted too low
    return {
        "MAE": round(float(np.abs(err).mean()), 1),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 1),
        "signed_error": round(float(err.mean()), 1),
        "pct_too_low": round(float((err > 0).mean() * 100), 1),
    }


def to_tensor(data):
    return torch.tensor(np.asarray(data, dtype=np.float32))


def mse_loss(y_pred, y):
    return ((y_pred - y) ** 2).mean()


def train_manual(X_train, y_train, lr, epochs=5000):
    """Method 2: linear regression by hand (raw tensors, autograd, manual update)."""
    X = to_tensor(X_train)
    y = to_tensor(y_train)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.tensor(float(y.mean()), requires_grad=True)  # start at the average demand
    loss_history = []
    for epoch in range(epochs):
        y_pred = X @ w + b                 # 1. forward pass
        loss = mse_loss(y_pred, y)         # 2. loss
        loss.backward()                    # 3. autograd computes the gradients
        with torch.no_grad():              # 4. update, not tracked by autograd
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                     # reset, or gradients pile up
        b.grad.zero_()
        loss_history.append(loss.item())
    return w.detach(), b.detach(), loss_history


def predict_manual(X, w, b):
    return (to_tensor(X) @ w + b).numpy()


def train_standard(X_train, y_train, lr, epochs=2000):
    """Method 3: the standard workflow, nn.Linear + torch.optim."""
    torch.manual_seed(0)
    X = to_tensor(X_train)
    y = to_tensor(y_train)
    model = nn.Linear(X.shape[1], 1)
    with torch.no_grad():                  # same starting point as the manual loop
        model.weight.zero_()
        model.bias.fill_(float(y.mean()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X).squeeze(-1)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return model, loss_history


def predict_standard(X, model):
    with torch.no_grad():
        return model(to_tensor(X)).squeeze(-1).numpy()


def pick_best_lr(results):
    """results: {lr: validation MAE}. Ignores runs that blew up (nan)."""
    valid = {lr: mae for lr, mae in results.items() if np.isfinite(mae)}
    return min(valid, key=valid.get)

ARTIFACT_DIR = Path("artifacts")


def save_artifacts(sk_model, w, b, std_model, mean, std, columns, results, pred_table):
    """Everything app.py needs, so the dashboard never has to train anything."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(sk_model, ARTIFACT_DIR / "sklearn_model.joblib")
    torch.save({"w": w, "b": b}, ARTIFACT_DIR / "manual_model.pt")
    torch.save(std_model.state_dict(), ARTIFACT_DIR / "standard_model.pt")
    joblib.dump({"mean": mean, "std": std, "columns": columns}, ARTIFACT_DIR / "preprocessing.joblib")
    with open(ARTIFACT_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    pred_table.to_csv(ARTIFACT_DIR / "predictions.csv", index=False)

if __name__ == "__main__":
    df = load_data()
    train, val, test = split_by_time(df)

    X_train, X_val, X_test = (build_features(p) for p in (train, val, test))
    y_train, y_val, y_test = train["cnt"], val["cnt"], test["cnt"]

    mean, std = fit_scaler(X_train)
    X_train, X_val, X_test = (apply_scaler(X, mean, std) for X in (X_train, X_val, X_test))

    # naive baseline: always predict the average of the training set
    baseline = np.full(len(y_val), y_train.mean())
    print("baseline (val)", evaluate(y_val, baseline))

    # method 1: scikit-learn
    sk_model = LinearRegression().fit(X_train, y_train)
    print("sklearn (val) ", evaluate(y_val, sk_model.predict(X_val)))

    # method 2: manual loop, learning rate chosen on the validation set
    print("\nmanual loop, learning rate search (validation MAE):")
    manual_runs, manual_mae = {}, {}
    for lr in [0.01, 0.1, 0.5, 1.0]:
        w, b, hist = train_manual(X_train, y_train, lr)
        manual_runs[lr] = (w, b)
        manual_mae[lr] = evaluate(y_val, predict_manual(X_val, w, b))["MAE"]
        print(f"  lr={lr}: MAE {manual_mae[lr]}")
    best_lr_manual = pick_best_lr(manual_mae)
    w, b = manual_runs[best_lr_manual]
    print(f"manual (val), lr={best_lr_manual}:", evaluate(y_val, predict_manual(X_val, w, b)))

    # method 3: nn.Linear + torch.optim.Adam
    print("\nstandard workflow, learning rate search (validation MAE):")
    std_runs, std_mae = {}, {}
    for lr in [0.1, 0.5, 1.0]:
        model, hist = train_standard(X_train, y_train, lr)
        std_runs[lr] = model
        std_mae[lr] = evaluate(y_val, predict_standard(X_val, model))["MAE"]
        print(f"  lr={lr}: MAE {std_mae[lr]}")
    best_lr_std = pick_best_lr(std_mae)
    print(f"standard (val), lr={best_lr_std}:", evaluate(y_val, predict_standard(X_val, std_runs[best_lr_std])))
        # final evaluation: the test set is used only here, once
    std_model = std_runs[best_lr_std]
    test_preds = {
        "baseline": np.full(len(y_test), y_train.mean()),
        "sklearn": sk_model.predict(X_test),
        "manual": predict_manual(X_test, w, b),
        "standard": predict_standard(X_test, std_model),
    }
    results = {name: evaluate(y_test, p) for name, p in test_preds.items()}
    print("\nTEST SET RESULTS")
    for name, r in results.items():
        print(f"  {name:9s}", r)

    # predictions on validation and test, for the dashboard plots
    val_preds = {
        "baseline": np.full(len(y_val), y_train.mean()),
        "sklearn": sk_model.predict(X_val),
        "manual": predict_manual(X_val, w, b),
        "standard": predict_standard(X_val, std_model),
    }
    tables = []
    for split, part, preds in [("val", val, val_preds), ("test", test, test_preds)]:
        t = part[["dteday", "hr", "workingday", "cnt"]].copy()
        t["split"] = split
        for name, p in preds.items():
            t[f"pred_{name}"] = p
        tables.append(t)
    pred_table = pd.concat(tables, ignore_index=True)

    save_artifacts(sk_model, w, b, std_model, mean, std, list(X_train.columns), results, pred_table)
    print("\nsaved models and predictions to artifacts/")