import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_prep import load_data, split
from features import build_features


def metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def train_manual(X, y, alpha, lr=0.05, epochs=3000):
    X = torch.tensor(X)
    y = torch.tensor(y)
    n, d = X.shape
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for _ in range(epochs):
        pred = X @ w + b
        loss = ((pred - y) ** 2).mean() + (alpha / n) * (w ** 2).sum()  # b not penalized
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        losses.append(loss.item())
    return w.detach(), b.detach(), losses


class LinearModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)
        nn.init.zeros_(self.linear.weight)  # same start as the manual loop
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def train_module(X, y, alpha, lr=0.05, epochs=3000):
    X = torch.tensor(X)
    y = torch.tensor(y)
    n, d = X.shape
    model = LinearModel(d)
    optimizer = torch.optim.SGD(
        [
            {"params": [model.linear.weight], "weight_decay": 2 * alpha / n},
            {"params": [model.linear.bias], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


def pick_lr(d, alpha):
    best_lr, best_rmse = None, np.inf
    for lr in [0.005, 0.02, 0.05, 0.2]:
        w, b, _ = train_manual(d["X_train"], d["y_train"], alpha, lr=lr)
        pred = (torch.tensor(d["X_val"]) @ w + b).numpy()
        rmse = metrics(d["y_val"], pred)["rmse"]
        print(f"lr={lr:<6}: val rmse {rmse:.4f}")
        if rmse < best_rmse:
            best_lr, best_rmse = lr, rmse
    return best_lr


def main():
    df = load_data()
    tr, va, te = split(df)
    d = build_features(tr, va, te)

    # naive baseline: predict the train mean for every song
    base = float(d["y_train"].mean())
    print("baseline  val:", metrics(d["y_val"], np.full(len(d["y_val"]), base)))
    print("baseline test:", metrics(d["y_test"], np.full(len(d["y_test"]), base)))

    # sklearn Ridge: choose alpha on validation only
    best_alpha, best_rmse = None, np.inf
    for alpha in [0.1, 1, 10, 100, 1000, 3000, 10000, 30000, 100000]:
        m = Ridge(alpha=alpha).fit(d["X_train"], d["y_train"])
        rmse = metrics(d["y_val"], m.predict(d["X_val"]))["rmse"]
        print(f"alpha={alpha:>7}: val rmse {rmse:.4f}")
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse
    print("best alpha:", best_alpha)

    m = Ridge(alpha=best_alpha).fit(d["X_train"], d["y_train"])
    for s in ["train", "val", "test"]:
        print(f"ridge {s}:", metrics(d["y_" + s], m.predict(d["X_" + s])))

    # learning rate for the two PyTorch versions, chosen on validation
    best_lr = pick_lr(d, best_alpha)
    print("best lr:", best_lr)

    # manual PyTorch loop, same loss as Ridge
    w, b, losses = train_manual(d["X_train"], d["y_train"], best_alpha, lr=best_lr)
    print("manual loss first/last:", round(losses[0], 4), round(losses[-1], 4))
    for s in ["train", "val", "test"]:
        pred = (torch.tensor(d["X_" + s]) @ w + b).numpy()
        print(f"manual {s}:", metrics(d["y_" + s], pred))
    print("manual max |coef diff| vs sklearn:", float(np.abs(w.numpy() - m.coef_).max()))
    print("manual intercept diff:", float(abs(b.item() - m.intercept_)))

    # standard PyTorch workflow: nn.Module + torch.optim
    model, mlosses = train_module(d["X_train"], d["y_train"], best_alpha, lr=best_lr)
    print("module loss first/last (MSE only):", round(mlosses[0], 4), round(mlosses[-1], 4))
    with torch.no_grad():
        for s in ["train", "val", "test"]:
            pred = model(torch.tensor(d["X_" + s])).numpy()
            print(f"module {s}:", metrics(d["y_" + s], pred))
        wm = model.linear.weight.squeeze(0).numpy()
        print("module max |coef diff| vs sklearn:", float(np.abs(wm - m.coef_).max()))
        print("module intercept diff:", float(abs(model.linear.bias.item() - m.intercept_)))

    # save artifacts: the app only loads these, it never retrains
    os.makedirs("models", exist_ok=True)
    joblib.dump(m, "models/sklearn_ridge.joblib")
    torch.save({"w": w, "b": b}, "models/torch_manual.pt")
    torch.save(model.state_dict(), "models/torch_module.pt")
    joblib.dump(
        {"feature_names": d["feature_names"], "mean": d["mean"], "std": d["std"],
         "artist_stats": d["artist_stats"], "global_mean": d["global_mean"],
         "alpha": best_alpha, "lr": best_lr, "baseline": base}, "models/prep.joblib")
    with torch.no_grad():
        np.savez("models/predictions.npz",
                 y_val=d["y_val"], y_test=d["y_test"],
                 sklearn=m.predict(d["X_test"]),
                 manual=(torch.tensor(d["X_test"]) @ w + b).numpy(),
                 module=model(torch.tensor(d["X_test"])).numpy())
    print("saved models/ and predictions")

    # metrics.json: every model on every split, plus the coefficient comparison
    with torch.no_grad():
        preds = {
            "baseline": lambda X: np.full(len(X), base),
            "sklearn": lambda X: m.predict(X),
            "manual": lambda X: (torch.tensor(X) @ w + b).numpy(),
            "module": lambda X: model(torch.tensor(X)).numpy(),
        }
        report = {
            name: {s: metrics(d["y_" + s], f(d["X_" + s])) for s in ["train", "val", "test"]}
            for name, f in preds.items()
        }
    report["max_coef_diff"] = {
        "manual": float(np.abs(w.numpy() - m.coef_).max()),
        "module": float(np.abs(wm - m.coef_).max()),
    }
    report["coefficients"] = {
        "feature": d["feature_names"],
        "sklearn": m.coef_.tolist(),
        "manual": w.numpy().tolist(),
        "module": wm.tolist(),
    }
    with open("models/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("saved models/metrics.json")


if __name__ == "__main__":
    main()
    