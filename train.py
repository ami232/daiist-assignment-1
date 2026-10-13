"""Assignment 1: training pipeline (Telco customer churn)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

SEED = 42
DATA_PATH = "data/telco.csv"
ARTIFACTS = Path("artifacts")

# Business costs (assumptions, see REPORT.md): a missed churner vs. a wasted offer
COST_FN = 400.0
COST_FP = 50.0

# PyTorch optimisation settings (same for both PyTorch versions)
LR = 0.5
EPOCHS = 300000

# Regularisation: pick the strongest (smallest C) within this log-loss tolerance of the best
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
C_TOLERANCE = 0.001

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
NUM_COLS = ["tenure_log", "MonthlyCharges", "SeniorCitizen", "is_echeck", "contract_ord"]
CAT_COLS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService"] + ADDONS + ["PaperlessBilling"]


# ---------- data ----------
def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])  # identifier, no signal
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank = total.isna()
    assert (df.loc[blank, "tenure"] == 0).all(), "blank TotalCharges with tenure > 0"
    df["TotalCharges"] = total.fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def split_data(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.4, stratify=df["Churn"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["Churn"], random_state=SEED)
    return train_df, val_df, test_df


def engineer_features(df):
    """Row-wise transformations only: nothing here learns from the data."""
    X = pd.DataFrame(index=df.index)
    X["tenure_log"] = np.log1p(df["tenure"])
    X["MonthlyCharges"] = df["MonthlyCharges"]
    X["SeniorCitizen"] = df["SeniorCitizen"]
    X["is_echeck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    X["contract_ord"] = df["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2})
    for c in ["gender", "Partner", "Dependents", "PhoneService", "InternetService",
              "PaperlessBilling"]:
        X[c] = df[c]
    for c in ADDONS + ["MultipleLines"]:
        X[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    return X[NUM_COLS + CAT_COLS]


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            # drop="first" removes one reference category per feature, which
            # removes the exact collinearity with the intercept
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
        ],
        sparse_threshold=0,
    )


# ---------- business metric ----------
def expected_cost(y, p, thr):
    pred = p >= thr
    fn = np.sum((y == 1) & ~pred)
    fp = np.sum((y == 0) & pred)
    return COST_FN * fn + COST_FP * fp


def best_threshold(y, p):
    grid = np.linspace(0.01, 0.99, 99)
    return float(grid[int(np.argmin([expected_cost(y, p, t) for t in grid]))])


# ---------- model 1: scikit-learn ----------
def fit_sklearn(X_tr, y_tr, X_val, y_val):
    fits = {}
    for C in C_GRID:  # the only tuned knob
        m = LogisticRegression(C=C, max_iter=10000, tol=1e-10).fit(X_tr, y_tr)
        fits[C] = (log_loss(y_val, m.predict_proba(X_val)[:, 1]), m)
    best_ll = min(ll for ll, _ in fits.values())
    print("C grid (validation log-loss):")
    for C, (ll, _) in fits.items():
        print(f"  C={C:<6} val log-loss {ll:.4f}")
    # strongest regularisation (smallest C) within tolerance of the best
    chosen = min(C for C, (ll, _) in fits.items() if ll <= best_ll + C_TOLERANCE)
    return chosen, fits[chosen][1]


# ---------- model 2: manual PyTorch loop ----------
def fit_manual(X, y, lam, lr=LR, epochs=EPOCHS):
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    for _ in range(epochs):
        logits = Xt @ w + b
        # binary cross-entropy written out: softplus(z) - y*z == -[y log s(z) + (1-y) log(1-s(z))]
        bce = (F.softplus(logits) - yt * logits).mean()
        loss = bce + 0.5 * lam * (w ** 2).sum()  # L2 on weights only, not the bias
        loss.backward()
        with torch.no_grad():  # manual gradient step
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return w.detach().numpy(), b.detach().numpy()


# ---------- model 3: nn.Module + torch.optim ----------
class LogReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def fit_module(X, y, lam, lr=LR, epochs=EPOCHS):
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    model = LogReg(X.shape[1]).double()
    # weight_decay adds lam*w to the gradient, matching the manual L2 term; bias gets none
    opt = torch.optim.SGD(
        [{"params": [model.linear.weight], "weight_decay": lam},
         {"params": [model.linear.bias], "weight_decay": 0.0}],
        lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(Xt), yt).backward()
        opt.step()
    return model


# ---------- evaluation ----------
def full_grad_norm(w, b, X, y, lam):
    """Gradient norm of the regularised mean-loss objective; ~0 means converged."""
    p = 1 / (1 + np.exp(-(X @ w + b)))
    g_w = X.T @ (p - y) / len(y) + lam * w
    g_b = np.mean(p - y)
    return float(np.sqrt(np.sum(g_w ** 2) + np.sum(g_b ** 2)))


def evaluate(name, y_val, p_val, y_test, p_test):
    thr = best_threshold(y_val, p_val)  # threshold chosen on VALIDATION only
    cost = expected_cost(y_test, p_test, thr)
    return {"model": name, "threshold": thr,
            "test_auc": float(roc_auc_score(y_test, p_test)),
            "test_logloss": float(log_loss(y_test, p_test)),
            "test_cost": float(cost), "cost_per_customer": float(cost / len(y_test))}


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_and_clean()
    train_df, val_df, test_df = split_data(df)

    prep = build_preprocessor()
    X_tr = prep.fit_transform(engineer_features(train_df)).astype(np.float64)  # fit on TRAIN only
    X_va = prep.transform(engineer_features(val_df)).astype(np.float64)
    X_te = prep.transform(engineer_features(test_df)).astype(np.float64)
    y_tr, y_va, y_te = (d["Churn"].to_numpy() for d in (train_df, val_df, test_df))
    joblib.dump(prep, ARTIFACTS / "preprocessor.joblib")
    feature_names = [str(n) for n in prep.get_feature_names_out()]
    print("feature matrix shapes:", X_tr.shape, X_va.shape, X_te.shape)

    # Fit all three on the same train split
    C, sk = fit_sklearn(X_tr, y_tr, X_va, y_va)
    lam = 1.0 / (C * len(y_tr))  # maps sklearn's C onto a mean-loss L2 strength
    print(f"sklearn: chosen C={C}  ->  PyTorch L2 strength lam=1/(C*n)={lam:.3e}")
    w_man, b_man = fit_manual(X_tr, y_tr, lam)
    mod = fit_module(X_tr, y_tr, lam)

    # Do the three agree?
    w_mod = mod.linear.weight.detach().numpy().ravel()
    b_mod = mod.linear.bias.detach().numpy()
    w_sk = sk.coef_.ravel()
    for name, w, b in [("sklearn", w_sk, sk.intercept_), ("manual", w_man, b_man),
                       ("module", w_mod, b_mod)]:
        print(f"gradient norm {name:<8} = {full_grad_norm(w, b, X_tr, y_tr, lam):.2e}")
    print(f"max |w_sklearn - w_manual|  = {np.abs(w_sk - w_man).max():.2e}")
    print(f"max |w_sklearn - w_module|  = {np.abs(w_sk - w_mod).max():.2e}")
    print(f"max |w_manual  - w_module|  = {np.abs(w_man - w_mod).max():.2e}")

    def probs(X):
        out = {"sklearn": sk.predict_proba(X)[:, 1]}
        out["manual"] = 1 / (1 + np.exp(-(X @ w_man + b_man)))
        with torch.no_grad():
            out["module"] = torch.sigmoid(mod(torch.tensor(X))).numpy()
        return out

    p_va, p_te = probs(X_va), probs(X_te)
    print(f"max |p_sklearn - p_manual| (val) = {np.abs(p_va['sklearn'] - p_va['manual']).max():.2e}")
    print(f"max |p_sklearn - p_module| (val) = {np.abs(p_va['sklearn'] - p_va['module']).max():.2e}")

    # Results on the TEST set, threshold picked on validation
    rows = [evaluate(k, y_va, p_va[k], y_te, p_te[k]) for k in ["sklearn", "manual", "module"]]
    base_p = np.full(len(y_te), y_tr.mean())  # naive: everyone gets the train churn rate
    rows.append({"model": "naive (train rate)", "threshold": float("nan"),
                 "test_auc": 0.5, "test_logloss": float(log_loss(y_te, base_p)),
                 "test_cost": float(COST_FN * y_te.sum()),  # = contact nobody
                 "cost_per_customer": float(COST_FN * y_te.sum() / len(y_te))})
    table = pd.DataFrame(rows)
    print("\nTEST results:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"(contact everyone would cost {COST_FP * (y_te == 0).sum():.0f}; "
          f"contact nobody costs {COST_FN * y_te.sum():.0f})")

    # Save everything the app needs
    joblib.dump(sk, ARTIFACTS / "sklearn_model.joblib")
    torch.save({"w": torch.tensor(w_man), "b": torch.tensor(b_man)}, ARTIFACTS / "manual_params.pt")
    torch.save(mod.state_dict(), ARTIFACTS / "module_state.pt")
    np.savez(ARTIFACTS / "predictions.npz", y_val=y_va, y_test=y_te,
             **{f"p_val_{k}": v for k, v in p_va.items()},
             **{f"p_test_{k}": v for k, v in p_te.items()})
    df.to_csv(ARTIFACTS / "clean_data.csv", index=False)
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({"C": C, "lam": lam, "lr": LR, "epochs": EPOCHS,
                   "cost_fn": COST_FN, "cost_fp": COST_FP,
                   "feature_names": feature_names, "results": rows}, f, indent=2)
    print("\nSaved artifacts to", ARTIFACTS)


if __name__ == "__main__":
    main()
