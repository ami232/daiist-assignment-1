"""
Training stage: `uv run python main.py train`

Kepler KOI vetting: predict whether a transit signal is a real planet (CONFIRMED)
or a FALSE POSITIVE, so follow-up telescope time goes to the most promising signals.

Steps
  1. Load data, define the target
  2. Feature engineering
  3. Split by star (train / val / test)
  4. Fill missing values + scale (fit on train only)
  5. Train logistic regression three ways + a naive baseline
  6. Pick each model's decision threshold on val by minimising business cost
  7. Evaluate on test, save models and data for the app
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

DATA = Path("data/koi_cumulative.csv")
OUT = Path("models")
SEED = 42

# Business costs: a wasted follow-up slot costs 1, a missed planet costs 3.
COST_FP = 1
COST_FN = 3

FEATURES = [
    "log_period", "log_depth", "log_snr", "log_prad", "log_duration", "log_teq", "log_srad",
    "koi_impact", "koi_steff", "koi_kepmag",
    "log_snr_per_transit", "log_centroid_offset", "is_giant", "is_grazing", "is_multi",
]


# ---------------------------------------------------------------- 1. load + target
def load_data():
    df = pd.read_csv(DATA)
    # CANDIDATE = not decided yet -> it is not a negative, so we leave it out.
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df["y"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    # Columns like koi_score, koi_pdisposition, koi_fpflag_* and kepler_name are the
    # vetting result itself (leakage). We never use them: only FEATURES go into the model.
    return df


# ---------------------------------------------------------------- 2. features
def add_features(df):
    # Skewed quantities (span several orders of magnitude) -> log10
    df["log_period"] = np.log10(df["koi_period"])
    df["log_depth"] = np.log10(df["koi_depth"].clip(lower=1))
    df["log_snr"] = np.log10(df["koi_model_snr"].clip(lower=1))
    df["log_prad"] = np.log10(df["koi_prad"].clip(lower=0.1))
    df["log_duration"] = np.log10(df["koi_duration"].clip(lower=0.1))
    df["log_teq"] = np.log10(df["koi_teq"].clip(lower=10))
    df["log_srad"] = np.log10(df["koi_srad"].clip(lower=0.01))

    # Signal strength of ONE transit: a few strong dips vs many weak ones
    snr_per_transit = df["koi_model_snr"] / np.sqrt(df["koi_num_transits"].clip(lower=1))
    df["log_snr_per_transit"] = np.log10(snr_per_transit.clip(lower=0.1))

    # How far the light centre moves during the dip, in units of its error.
    # Large -> the dip probably comes from a nearby star, not our target.
    # log10(1 + x) because x is 0 or more and very skewed.
    offset_sig = df["koi_dikco_msky"] / df["koi_dikco_msky_err"].clip(lower=0.01)
    df["log_centroid_offset"] = np.log10(1 + offset_sig)

    # Impact parameter: 0 = crosses the star's centre, >1 = only grazes it. A handful of
    # extreme values (up to ~100) are fit failures, so cap at 2.
    df["koi_impact"] = df["koi_impact"].clip(upper=2)

    # Simple yes/no flags
    df["is_giant"] = (df["koi_prad"] > 15).astype(int)    # bigger than any planet -> likely a star
    df["is_grazing"] = (df["koi_impact"] > 1).astype(int)  # only clips the star's edge -> typical of binaries
    df["is_multi"] = (df["koi_count"] > 1).astype(int)     # several signals on one star -> rarely false
    return df


# ---------------------------------------------------------------- 3. split by star
def split_by_star(df):
    # One star can have several KOIs that share all the star's values.
    # Splitting by star (kepid) keeps them together, so test stars are never seen in training.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, rest_idx = next(gss.split(df, groups=df["kepid"]))
    train, rest = df.iloc[train_idx], df.iloc[rest_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(gss2.split(rest, groups=rest["kepid"]))
    return train.copy(), rest.iloc[val_idx].copy(), rest.iloc[test_idx].copy()


# ---------------------------------------------------------------- 4. preprocessing
def preprocess(train, val, test):
    medians = train[FEATURES].median()          # learned from train only
    for d in (train, val, test):
        d[FEATURES] = d[FEATURES].fillna(medians)
    scaler = StandardScaler().fit(train[FEATURES])
    return scaler, medians


# ---------------------------------------------------------------- 5a. scikit-learn
def train_sklearn(X_train, y_train, X_val, y_val):
    best = None
    for C in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=C, max_iter=2000).fit(X_train, y_train)
        loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
        print(f"  sklearn C={C:<5} val log loss {loss:.4f}")
        if best is None or loss < best[0]:
            best = (loss, C, model)
    return best[2], best[1]


# ---------------------------------------------------------------- 5b. manual PyTorch loop
def train_manual(X, y, lam, lr, epochs=5000):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(epochs):
        p = torch.sigmoid(X @ w + b)
        bce = -(y * torch.log(p + 1e-7) + (1 - y) * torch.log(1 - p + 1e-7)).mean()
        loss = bce + lam / 2 * (w ** 2).sum()        # same L2 penalty as sklearn
        loss.backward()                              # autograd computes the gradients
        with torch.no_grad():                        # manual gradient step
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    return w.detach(), b.detach()


def predict_manual(w, b, X):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X, dtype=torch.float32) @ w + b).numpy()


# ---------------------------------------------------------------- 5c. standard PyTorch
class LogReg(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)   # logits; sigmoid is inside the loss


def train_nn(X, y, lam, lr, epochs=5000):
    torch.manual_seed(SEED)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    model = LogReg(X.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y) + lam / 2 * (model.linear.weight ** 2).sum()
        loss.backward()
        optimizer.step()
    return model


def predict_nn(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def pick_lr(train_fn, predict_fn, X_train, y_train, X_val, y_val, lam, name):
    best = None
    for lr in [0.01, 0.1, 1.0]:
        model = train_fn(X_train, y_train, lam, lr)
        p = predict_fn(*model, X_val) if isinstance(model, tuple) else predict_fn(model, X_val)
        loss = log_loss(y_val, np.clip(p, 1e-7, 1 - 1e-7))
        print(f"  {name} lr={lr:<5} val log loss {loss:.4f}")
        if best is None or loss < best[0]:
            best = (loss, lr, model)
    return best[2], best[1]


# ---------------------------------------------------------------- 6-7. threshold + evaluation
def cost(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return COST_FP * fp + COST_FN * fn


def best_threshold(y, p):
    thresholds = np.arange(0.05, 0.96, 0.01)
    costs = [cost(y, p, t) for t in thresholds]
    return round(float(thresholds[np.argmin(costs)]), 2)


def evaluate(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return {
        "ROC-AUC": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        "log loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "cost per 100 KOIs": 100 * (COST_FP * fp + COST_FN * fn) / len(y),
    }


def main():
    OUT.mkdir(exist_ok=True)
    df = add_features(load_data())
    train, val, test = split_by_star(df)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  (positive rate {train['y'].mean():.2f})")

    scaler, medians = preprocess(train, val, test)
    X_train, X_val, X_test = (scaler.transform(d[FEATURES]) for d in (train, val, test))
    y_train, y_val, y_test = train["y"].values, val["y"].values, test["y"].values

    # Same regularisation for all three: sklearn's C  <->  lam = 1 / (C * n_train)
    sk_model, C = train_sklearn(X_train, y_train, X_val, y_val)
    lam = 1 / (C * len(y_train))
    (w, b), manual_lr = pick_lr(train_manual, predict_manual, X_train, y_train, X_val, y_val, lam, "manual")
    nn_model, nn_lr = pick_lr(train_nn, predict_nn, X_train, y_train, X_val, y_val, lam, "nn    ")

    # Save models
    joblib.dump(scaler, OUT / "scaler.joblib")
    joblib.dump(sk_model, OUT / "sklearn.joblib")
    torch.save({"w": w, "b": b}, OUT / "manual.pt")
    torch.save(nn_model.state_dict(), OUT / "nn.pt")

    # Predictions (naive baseline = always predict the training positive rate)
    prior = y_train.mean()
    preds = {
        "baseline": (np.full(len(y_val), prior), np.full(len(y_test), prior)),
        "sklearn": (sk_model.predict_proba(X_val)[:, 1], sk_model.predict_proba(X_test)[:, 1]),
        "manual": (predict_manual(w, b, X_val), predict_manual(w, b, X_test)),
        "nn": (predict_nn(nn_model, X_val), predict_nn(nn_model, X_test)),
    }

    results, thresholds = {}, {}
    for name, (p_val, p_test) in preds.items():
        # baseline gives everyone the same p, so "follow up all" (t=0) is its only useful choice
        thresholds[name] = 0.0 if name == "baseline" else best_threshold(y_val, p_val)
        results[name] = evaluate(y_test, p_test, thresholds[name])

    table = pd.DataFrame(results).T.round(3)
    print("\nTEST SET\n", table.to_string())

    coefs = pd.DataFrame({
        "sklearn": sk_model.coef_[0],
        "manual": w.numpy(),
        "nn": nn_model.linear.weight.detach().numpy()[0],
    }, index=FEATURES).round(3)
    print("\nCOEFFICIENTS (should match)\n", coefs.to_string())

    # Everything the app needs
    test.to_csv(OUT / "test.csv", index=False)
    train[FEATURES + ["y"]].to_csv(OUT / "train.csv", index=False)
    table.to_csv(OUT / "results.csv")
    coefs.to_csv(OUT / "coefficients.csv")
    json.dump({"prior": prior, "thresholds": thresholds, "C": C,
               "manual_lr": manual_lr, "nn_lr": nn_lr}, open(OUT / "meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
