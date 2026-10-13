"""Assignment 1 - training pipeline (Telco churn, logistic regression)."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT_DIR = "artifacts"
EPOCHS = 1000
DATA_PATH = "data/telco_churn.csv"
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def load_and_clean(path):
    df = pd.read_csv(path)
    # TotalCharges has 11 blanks, all clients with tenure 0 (just joined): 0 charged so far
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def engineer_features(df):
    df = df.copy()
    # how many services the client has contracted
    df["n_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS) \
        + (df["InternetService"] != "No").astype(int)
    # tenure in buckets (churn risk is not linear in tenure)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    # flags for the two strongest known risk patterns
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["autopay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    # monthly price relative to what the client has paid on average so far
    avg_paid = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
    df["price_vs_avg_paid"] = df["MonthlyCharges"] - avg_paid
    return df


def prepare_data():
    df = engineer_features(load_and_clean(DATA_PATH))
    y = df["Churn"].values.astype(np.float32)
    X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True).astype(float)
    feature_names = list(X.columns)

    # 60/20/20 stratified split, done ONCE and shared by all three models
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X.values, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    # scaler fitted on train only (otherwise val/test info leaks into training)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = [scaler.transform(a).astype(np.float32) for a in (X_tr, X_val, X_te)]
    return X_tr, X_val, X_te, y_tr, y_val, y_te, feature_names, scaler


def evaluate(y_true, p):
    """Metrics from predicted probabilities p (threshold 0.5 for accuracy/F1)."""
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, pred), "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, p), "log_loss": log_loss(y_true, p)}


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def train_manual(X, y, lr, lam, epochs=EPOCHS):
    """Session 5 style: raw tensors, autograd, manual gradient step."""
    X, y = torch.tensor(X), torch.tensor(y)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(epochs):
        p = sigmoid(X @ w + b).clamp(1e-7, 1 - 1e-7)
        loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean() + 0.5 * lam * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return w.detach(), b.detach()


def predict_manual(w, b, X):
    return sigmoid(torch.tensor(X) @ w + b).numpy()


class LogReg(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)  # logits


def train_nn(X, y, lr, lam, epochs=EPOCHS):
    """Standard workflow: nn.Module + loss + torch.optim."""
    torch.manual_seed(SEED)
    X, y = torch.tensor(X), torch.tensor(y)
    model = LogReg(X.shape[1])
    torch.nn.init.zeros_(model.linear.weight)
    torch.nn.init.zeros_(model.linear.bias)
    # L2 on weights only (not on bias), same penalty as the manual version
    opt = torch.optim.SGD([{"params": [model.linear.weight], "weight_decay": lam},
                           {"params": [model.linear.bias], "weight_decay": 0.0}], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    return model


def predict_nn(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def feature_ablation(X_tr, X_val, y_tr, y_val, names, C):
    """Refit sklearn without each engineered feature (group) and compare on the VALIDATION set."""
    groups = {"n_services": ["n_services"], "tenure_group": ["tenure_group_"],
              "is_month_to_month": ["is_month_to_month"], "autopay": ["autopay"],
              "price_vs_avg_paid": ["price_vs_avg_paid"],
              "ALL 5 engineered": ["n_services", "tenure_group_", "is_month_to_month", "autopay", "price_vs_avg_paid"]}

    def score(drop_prefixes):
        keep = [i for i, nme in enumerate(names) if not any(nme.startswith(p) for p in drop_prefixes)]
        m = LogisticRegression(C=C, max_iter=1000).fit(X_tr[:, keep], y_tr)
        p = m.predict_proba(X_val[:, keep])[:, 1]
        return {"roc_auc": roc_auc_score(y_val, p), "log_loss": log_loss(y_val, p)}

    rows = {"all features (full model)": score([])}
    for label, prefixes in groups.items():
        rows[f"without {label}"] = score(prefixes)
    return rows


def main():
    X_tr, X_val, X_te, y_tr, y_val, y_te, names, scaler = prepare_data()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(X_tr)

    # --- naive baseline: always predict the majority class (no churn)
    base = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    p_base = base.predict_proba(X_te)[:, 1]

    # --- scikit-learn: tune C (regularization strength) on validation
    best_C = min([0.01, 0.1, 1, 10],
                 key=lambda C: log_loss(y_val, LogisticRegression(C=C, max_iter=1000).fit(X_tr, y_tr).predict_proba(X_val)[:, 1]))
    sk = LogisticRegression(C=best_C, max_iter=1000).fit(X_tr, y_tr)
    # same L2 strength for the two PyTorch versions (sklearn C <-> lam = 1 / (C * n))
    lam = 1 / (best_C * n)

    # --- PyTorch manual and nn.Module: tune learning rate on validation
    lrs = [0.05, 0.2, 0.5]
    lr_manual = min(lrs, key=lambda lr: log_loss(y_val, predict_manual(*train_manual(X_tr, y_tr, lr, lam), X_val)))
    lr_nn = min(lrs, key=lambda lr: log_loss(y_val, predict_nn(train_nn(X_tr, y_tr, lr, lam), X_val)))
    w, b = train_manual(X_tr, y_tr, lr_manual, lam)
    net = train_nn(X_tr, y_tr, lr_nn, lam)

    # --- evaluate everything on the same test set
    p_test = {"baseline": p_base, "sklearn": sk.predict_proba(X_te)[:, 1],
              "manual_torch": predict_manual(w, b, X_te), "nn_torch": predict_nn(net, X_te)}
    p_val = {"baseline": base.predict_proba(X_val)[:, 1], "sklearn": sk.predict_proba(X_val)[:, 1],
             "manual_torch": predict_manual(w, b, X_val), "nn_torch": predict_nn(net, X_val)}
    metrics = {k: evaluate(y_te, p) for k, p in p_test.items()}
    print(pd.DataFrame(metrics).T.round(4))
    print("best C:", best_C, "| lr manual:", lr_manual, "| lr nn:", lr_nn)

    # --- feature-engineering ablation (validation set)
    ablation = feature_ablation(X_tr, X_val, y_tr, y_val, names, best_C)
    print(pd.DataFrame(ablation).T.round(4))

    # --- save everything app.py needs (it never retrains)
    joblib.dump(sk, f"{OUT_DIR}/sklearn_model.joblib")
    joblib.dump(scaler, f"{OUT_DIR}/scaler.joblib")
    torch.save({"w": w, "b": b}, f"{OUT_DIR}/manual_torch.pt")
    torch.save(net.state_dict(), f"{OUT_DIR}/nn_torch.pt")
    np.savez(f"{OUT_DIR}/data.npz", X_val=X_val, y_val=y_val, X_te=X_te, y_te=y_te,
             **{f"p_test_{k}": v for k, v in p_test.items()}, **{f"p_val_{k}": v for k, v in p_val.items()})
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump({"feature_names": names, "metrics": metrics, "best_C": best_C,
                   "lr_manual": lr_manual, "lr_nn": lr_nn, "lam": lam, "ablation": ablation}, f, indent=2)
    print("saved to", OUT_DIR)


main()