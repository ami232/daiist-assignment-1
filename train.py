"""
Assignment 1 — training pipeline.

Run with `uv run python main.py train`. Start to finish, this:

1. Loads data/UCI_Credit_Card.csv and builds the engineered features
   (engineer_features below).
2. Splits clients 70/15/15 into train/validation/test, stratified on the
   target (a random split is fine: every client is observed over the same
   six months, so there is no future to leak).
3. Fits the scaler on train only.
4. Trains the same L2-regularised logistic regression three ways:
   scikit-learn, a manual PyTorch loop (raw tensors + autograd + manual
   gradient step), and the standard nn.Module + torch.optim workflow.
   Regularisation strength is tuned on validation with scikit-learn and then
   reused by both PyTorch versions so all three optimise the same objective;
   only the learning rate is tuned for PyTorch.
5. Picks, for each model, the decision threshold that minimises the
   business cost (FN = 5, FP = 1) on validation.
6. Evaluates everything once on the test set, against naive baselines.
7. Saves the models, fitted preprocessing, loss curves and metrics to
   artifacts/ for app.py.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
ART = Path("artifacts")
COST_FN, COST_FP = 5, 1
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0]
MANUAL_LR_GRID = [0.01, 0.05, 0.1, 0.5, 1.0]
MANUAL_EPOCHS = 2000          # full-batch gradient descent steps
STD_LR_GRID = [0.003, 0.01, 0.03, 0.1]
STD_EPOCHS = 60               # passes over the training set, mini-batches of 256
BATCH_SIZE = 256
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)

np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
# Every feature is computed from that client's own row only, so nothing here
# learns from other rows and it is safe to apply before the split. Scaling,
# which does learn from data, is fitted on the training rows only (main()).
# app.py can reuse this with `from train import engineer_features`; nothing is
# retrained on import because training only runs under __main__.

TARGET = "DEFAULT"

# PAY_0 is renamed to PAY_1 so the three monthly blocks share the same index:
# 1 = September 2005 (most recent) ... 6 = April 2005 (oldest).
MONTHS = range(1, 7)
PAY = [f"PAY_{i}" for i in MONTHS]
BILL = [f"BILL_AMT{i}" for i in MONTHS]
PAY_AMT = [f"PAY_AMT{i}" for i in MONTHS]

# Excluded on purpose (see REPORT.md): SEX and MARRIAGE are protected
# attributes in lending, and ID carries no information.
EXCLUDED = ["ID", "SEX", "MARRIAGE"]

# Raw columns kept for the "no feature engineering" comparison model.
RAW_FEATURES = ["LIMIT_BAL", "EDUCATION", "AGE"] + PAY + BILL + PAY_AMT


def load_raw(path="data/UCI_Credit_Card.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"default.payment.next.month": TARGET, "PAY_0": "PAY_1"})


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the engineered feature matrix (one row per client)."""
    out = pd.DataFrame(index=df.index)
    limit = df["LIMIT_BAL"]
    pay = df[PAY].to_numpy()
    bill = df[BILL].to_numpy()
    pay_amt = df[PAY_AMT].to_numpy()

    # --- Profile -------------------------------------------------------------
    out["LOG_LIMIT"] = np.log(limit)
    out["AGE"] = df["AGE"]
    # EDUCATION 0, 5, 6 are undocumented and 4 is "others"; together they are
    # <2% of clients, so they are merged into one "other" level.
    edu = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
    out["EDU_GRAD"] = (edu == 1).astype(int)
    out["EDU_UNI"] = (edu == 2).astype(int)
    out["EDU_HS"] = (edu == 3).astype(int)
    # "other" is the reference level (dropped to avoid perfect collinearity).

    # --- Repayment status ----------------------------------------------------
    # -2 (no consumption), -1 (paid in full) and 0 (revolving / minimum paid)
    # all mean "not late" and have similar default rates in the EDA; the jump
    # happens at >= 1 month late. Code 1 is used almost only in September, so
    # the number of months late is treated as "late or not" plus a capped
    # count rather than trusting the exact code in every month.
    delay = np.clip(pay, 0, None)
    out["DELAY_RECENT"] = np.minimum(delay[:, 0], 3)            # Sep, capped at 3+
    out["LATE_RECENT"] = (pay[:, 0] >= 1).astype(int)
    out["LATE_PREV"] = (pay[:, 1] >= 1).astype(int)              # Aug
    out["N_MONTHS_LATE"] = (pay >= 1).sum(axis=1)
    out["MAX_DELAY"] = np.minimum(delay.max(axis=1), 3)
    out["N_MONTHS_PAID_FULL"] = (pay == -1).sum(axis=1)
    out["N_MONTHS_INACTIVE"] = (pay == -2).sum(axis=1)

    # --- Balance relative to limit (utilisation) -----------------------------
    # Bills are divided by the client's own limit so a 50k bill means
    # something different for a 50k-limit and a 500k-limit client. Negative
    # bills (credit balances, overpayment) are kept as negative utilisation;
    # values are clipped to limit the influence of extreme rows.
    util = bill / limit.to_numpy()[:, None]
    out["UTIL_RECENT"] = np.clip(util[:, 0], -1, 2)
    out["UTIL_MEAN"] = np.clip(util.mean(axis=1), -1, 2)
    out["UTIL_TREND"] = np.clip(util[:, 0] - util[:, 5], -2, 2)  # Sep minus Apr
    out["ANY_OVER_LIMIT"] = (util > 1).any(axis=1).astype(int)
    out["ANY_NEG_BILL"] = (bill < 0).any(axis=1).astype(int)

    # --- Payment behaviour ---------------------------------------------------
    # PAY_AMT_i is paid during month i against the statement of month i+1,
    # so the share of the previous bill that was repaid is
    # PAY_AMT_i / BILL_AMT_{i+1}. When nothing was owed (bill <= 0) the
    # client is treated as having repaid everything (ratio = 1).
    prev_bill = bill[:, 1:]
    paid = pay_amt[:, :5]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev_bill > 0, paid / prev_bill, 1.0)
    ratio = np.clip(ratio, 0, 1)
    out["PAY_RATIO_RECENT"] = ratio[:, 0]
    out["PAY_RATIO_MEAN"] = ratio.mean(axis=1)
    out["N_ZERO_PAYMENTS"] = (pay_amt == 0).sum(axis=1)
    # Payment amounts are heavily right-skewed, so they enter on a log scale.
    out["LOG_PAY_AMT_MEAN"] = np.log1p(pay_amt.mean(axis=1))
    out["LOG_PAY_AMT_RECENT"] = np.log1p(pay_amt[:, 0])

    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def business_cost(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return COST_FN * fn + COST_FP * fp, (tn, fp, fn, tp)


def best_threshold(y, p):
    costs = np.array([business_cost(y, p, t)[0] for t in THRESHOLDS])
    return float(THRESHOLDS[costs.argmin()]), costs


def evaluate(y, p, threshold):
    cost, (tn, fp, fn, tp) = business_cost(y, p, threshold)
    n = len(y)
    return {
        "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "total_cost": int(cost),
        "cost_per_client": cost / n,
    }


def l2_lambda(C, n):
    # sklearn minimises  0.5*||w||^2 + C * sum(log-loss).
    # Dividing by C*n gives the equivalent per-sample objective used in
    # PyTorch:  mean(log-loss) + (1 / (2*C*n)) * ||w||^2   (bias not penalised).
    return 1.0 / (2.0 * C * n)


def objective(X, y, w, b, lam):
    """Regularised training objective, identical for all three models."""
    z = X @ w + b
    bce = np.mean(np.logaddexp(0, z) - y * z)
    return float(bce + lam * np.sum(w ** 2))


# --------------------------------------------------------------------------- #
# Model 2: manual PyTorch (raw tensors, autograd, manual update)
# --------------------------------------------------------------------------- #
def train_manual(X, y, lam, lr, epochs=MANUAL_EPOCHS):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(epochs):
        z = X_t @ w + b
        p = torch.sigmoid(z)
        eps = 1e-7
        bce = -(y_t * torch.log(p + eps) + (1 - y_t) * torch.log(1 - p + eps)).mean()
        loss = bce + lam * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())
    return w.detach().numpy(), b.detach().numpy()[0], history


# --------------------------------------------------------------------------- #
# Model 3: standard PyTorch (nn.Module + torch.optim + DataLoader)
# --------------------------------------------------------------------------- #
class LogisticRegressionNet(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits; sigmoid applied outside


def train_standard(X, y, lam, lr, epochs=STD_EPOCHS):
    torch.manual_seed(SEED)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True
    )
    model = LogisticRegressionNet(X.shape[1])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb) + lam * (model.linear.weight ** 2).sum()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full = loss_fn(model(X_t), y_t) + lam * (model.linear.weight ** 2).sum()
        history.append(full.item())
    return model, history


def predict_standard(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main():
    ART.mkdir(exist_ok=True)

    # 1. Data and features ---------------------------------------------------
    raw = load_raw()
    X_all = engineer_features(raw)
    y_all = raw[TARGET].to_numpy()
    feature_names = list(X_all.columns)
    print(f"Loaded {len(raw):,} clients, {len(feature_names)} engineered features")

    # 2. Stratified 70 / 15 / 15 split ---------------------------------------
    idx = np.arange(len(raw))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y_all, random_state=SEED)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, stratify=y_all[idx_tmp], random_state=SEED
    )
    split = pd.Series("train", index=raw.index)
    split.iloc[idx_val] = "val"
    split.iloc[idx_test] = "test"
    for name, ix in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        print(f"  {name:<5} n={len(ix):>6,}  default rate={y_all[ix].mean():.3f}")

    # 3. Scale (fit on train only) -------------------------------------------
    scaler = StandardScaler().fit(X_all.iloc[idx_train])
    X = scaler.transform(X_all)
    X_tr, X_va, X_te = X[idx_train], X[idx_val], X[idx_test]
    y_tr, y_va, y_te = y_all[idx_train], y_all[idx_val], y_all[idx_test]
    n_tr = len(y_tr)

    # 4a. scikit-learn: tune C on validation log-loss ------------------------
    tuning_rows = []
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=5000)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        tuning_rows.append({"model": "sklearn", "C": C, "lr": None,
                            "val_log_loss": log_loss(y_va, p),
                            "val_roc_auc": roc_auc_score(y_va, p)})
    tune_sk = pd.DataFrame(tuning_rows)
    best_C = float(tune_sk.loc[tune_sk["val_log_loss"].idxmin(), "C"])
    lam = l2_lambda(best_C, n_tr)
    sk = LogisticRegression(C=best_C, max_iter=5000).fit(X_tr, y_tr)
    print(f"sklearn: best C = {best_C}  (lambda for PyTorch = {lam:.2e})")

    # Feature-engineering check: same model on the raw columns only.
    raw_scaler = StandardScaler().fit(raw.iloc[idx_train][RAW_FEATURES])
    R = raw_scaler.transform(raw[RAW_FEATURES])
    raw_rows = []
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=5000).fit(R[idx_train], y_tr)
        raw_rows.append((log_loss(y_va, m.predict_proba(R[idx_val])[:, 1]), C))
    raw_C = min(raw_rows)[1]
    sk_raw = LogisticRegression(C=raw_C, max_iter=5000).fit(R[idx_train], y_tr)

    # 4b. Manual PyTorch: tune learning rate on validation -------------------
    manual_runs = {}
    for lr in MANUAL_LR_GRID:
        w, b, hist = train_manual(X_tr, y_tr, lam, lr)
        p = sigmoid(X_va @ w + b)
        manual_runs[lr] = (w, b, hist)
        tuning_rows.append({"model": "manual_torch", "C": best_C, "lr": lr,
                            "val_log_loss": log_loss(y_va, p),
                            "val_roc_auc": roc_auc_score(y_va, p)})
    tune = pd.DataFrame(tuning_rows)
    t = tune[tune.model == "manual_torch"]
    best_lr_manual = float(t.loc[t["val_log_loss"].idxmin(), "lr"])
    w_man, b_man, hist_man = manual_runs[best_lr_manual]
    print(f"manual torch: best lr = {best_lr_manual}")

    # 4c. Standard PyTorch: tune learning rate on validation -----------------
    std_runs = {}
    for lr in STD_LR_GRID:
        model, hist = train_standard(X_tr, y_tr, lam, lr)
        p = predict_standard(model, X_va)
        std_runs[lr] = (model, hist)
        tuning_rows.append({"model": "standard_torch", "C": best_C, "lr": lr,
                            "val_log_loss": log_loss(y_va, p),
                            "val_roc_auc": roc_auc_score(y_va, p)})
    tune = pd.DataFrame(tuning_rows)
    t = tune[tune.model == "standard_torch"]
    best_lr_std = float(t.loc[t["val_log_loss"].idxmin(), "lr"])
    std_model, hist_std = std_runs[best_lr_std]
    print(f"standard torch: best lr = {best_lr_std}")

    # 5. Probabilities for every split + threshold on validation -------------
    w_std = std_model.linear.weight.detach().numpy().ravel()
    b_std = float(std_model.linear.bias.detach().numpy()[0])
    prevalence = y_tr.mean()

    def probs(Xs, Rs):
        return {
            "baseline": np.full(len(Xs), prevalence),
            "raw_features_sklearn": sk_raw.predict_proba(Rs)[:, 1],
            "sklearn": sk.predict_proba(Xs)[:, 1],
            "manual_torch": sigmoid(Xs @ w_man + b_man),
            "standard_torch": predict_standard(std_model, Xs),
        }

    p_val = probs(X_va, R[idx_val])
    p_test = probs(X_te, R[idx_test])

    thresholds = {name: best_threshold(y_va, p)[0] for name, p in p_val.items()}

    # 6. Evaluate on validation and test ------------------------------------
    rows = []
    for split_name, y_s, p_s in [("val", y_va, p_val), ("test", y_te, p_test)]:
        for name, p in p_s.items():
            rows.append({"split": split_name, "model": name, **evaluate(y_s, p, thresholds[name])})
        # Trivial policies for reference: flag nobody / flag everybody.
        rows.append({"split": split_name, "model": "flag_nobody", **evaluate(y_s, np.zeros(len(y_s)), 0.5)})
        rows.append({"split": split_name, "model": "flag_everyone", **evaluate(y_s, np.ones(len(y_s)), 0.5)})
    metrics = pd.DataFrame(rows)

    # Agreement between the three implementations.
    coefs = pd.DataFrame(
        {"sklearn": np.r_[sk.intercept_[0], sk.coef_.ravel()],
         "manual_torch": np.r_[b_man, w_man],
         "standard_torch": np.r_[b_std, w_std]},
        index=["(intercept)"] + feature_names,
    )
    objectives = {
        "sklearn": objective(X_tr, y_tr, sk.coef_.ravel(), sk.intercept_[0], lam),
        "manual_torch": objective(X_tr, y_tr, w_man, b_man, lam),
        "standard_torch": objective(X_tr, y_tr, w_std, b_std, lam),
    }
    pt = p_test
    agreement = {
        "max_abs_coef_diff_vs_sklearn": {
            "manual_torch": float((coefs.manual_torch - coefs.sklearn).abs().max()),
            "standard_torch": float((coefs.standard_torch - coefs.sklearn).abs().max()),
        },
        "max_abs_test_prob_diff_vs_sklearn": {
            "manual_torch": float(np.abs(pt["manual_torch"] - pt["sklearn"]).max()),
            "standard_torch": float(np.abs(pt["standard_torch"] - pt["sklearn"]).max()),
        },
        "test_decision_agreement_vs_sklearn": {
            m: float(((pt[m] >= thresholds[m]) == (pt["sklearn"] >= thresholds["sklearn"])).mean())
            for m in ["manual_torch", "standard_torch"]
        },
        "train_objective": objectives,
    }

    # 7. Save artifacts -----------------------------------------------------
    # Everything that comes out of training: models, fitted preprocessing and
    # tuning history. Predictions, confusion matrices and cost curves are
    # cheap inference, so the app computes those live from the saved models.
    joblib.dump(sk, ART / "sklearn_logreg.joblib")
    joblib.dump(sk_raw, ART / "sklearn_raw_features.joblib")  # "did features help?" comparison
    torch.save({"w": torch.tensor(w_man), "b": torch.tensor(b_man)}, ART / "manual_torch.pt")
    torch.save(std_model.state_dict(), ART / "standard_torch.pt")
    joblib.dump({
        "scaler": scaler,
        "features": feature_names,
        "raw_scaler": raw_scaler,
        "raw_features": RAW_FEATURES,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "thresholds": thresholds,          # chosen on validation, per model
        "train_prevalence": float(prevalence),
        "cost_fn": COST_FN, "cost_fp": COST_FP,
    }, ART / "preprocessing.joblib")

    # Training-loss curves for every learning rate tried, so the dashboard can
    # show convergence (and why the chosen LR won) without retraining.
    curves = {f"manual_torch|lr={lr}": h for lr, (_, _, h) in manual_runs.items()}
    curves.update({f"standard_torch|lr={lr}": h for lr, (_, h) in std_runs.items()})
    pd.DataFrame({k: pd.Series(v) for k, v in curves.items()}).to_csv(
        ART / "loss_history.csv", index_label="epoch")

    # Small record of this run's results, so REPORT.md numbers are traceable.
    with open(ART / "metrics.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)},
            "best_C": best_C, "raw_best_C": raw_C, "lambda": lam,
            "best_lr": {"manual_torch": best_lr_manual, "standard_torch": best_lr_std},
            "thresholds": thresholds,
            "metrics": metrics.to_dict(orient="records"),
            "tuning": tune.to_dict(orient="records"),
            "coefficients": coefs.to_dict(),
            "agreement": agreement,
        }, f, indent=2, default=float)

    # Console summary ----------------------------------------------------------
    cols = ["model", "roc_auc", "pr_auc", "log_loss", "threshold", "precision",
            "recall", "fp", "fn", "total_cost", "cost_per_client"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nTEST SET (thresholds chosen on validation)")
        print(metrics[metrics.split == "test"][cols].to_string(index=False))
        print("\nTraining objective at each solution:", {k: round(v, 6) for k, v in objectives.items()})
        print("Agreement:", json.dumps(agreement, indent=2))
    print(f"\nArtifacts written to {ART.resolve()}")


if __name__ == "__main__":
    main()
