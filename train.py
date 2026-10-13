
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    ART, COSTS, SEED, LogisticRegressionNN, confusion_counts, engineer_features,
    load_clean, sigmoid_np, split_columns, theoretical_threshold, total_cost,
)


C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LR_GRID = [0.03, 0.1, 0.3, 0.6]
EPOCHS = 10_000          # full-batch gradient descent
LOG_EVERY = 100


def main():
    ART.mkdir(exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    
    X_raw, y = load_clean()
    X_fe = engineer_features(X_raw)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_fe, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)  # 0.25*0.8 = 0.2
    print(f"split  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}  "
          f"churn rate train={y_train.mean():.3f} val={y_val.mean():.3f} test={y_test.mean():.3f}")

 
    num_cols, cat_cols = split_columns(X_train)
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    A_train = pre.fit_transform(X_train)
    A_val, A_test = pre.transform(X_val), pre.transform(X_test)
    feature_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    n, d = A_train.shape
    print(f"features: {len(num_cols)} numeric + {len(cat_cols)} categorical -> {d} model inputs")

    # torch tensors (float32), same arrays for both PyTorch versions
    Xt = torch.tensor(A_train, dtype=torch.float32)
    Xv = torch.tensor(A_val, dtype=torch.float32)
    yt = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    yv = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

   
    val_ll = {}
    for C in C_GRID:
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=5000, tol=1e-10).fit(A_train, y_train)
        val_ll[C] = log_loss(y_val, m.predict_proba(A_val)[:, 1])
    best_C = min(val_ll, key=val_ll.get)
    print("sklearn val log loss by C:", {k: round(v, 4) for k, v in val_ll.items()}, "-> C =", best_C)
    sk = LogisticRegression(C=best_C, solver="lbfgs", max_iter=5000, tol=1e-10).fit(A_train, y_train)

    lam = 1.0 / (best_C * n)   # the equivalent L2 strength for the mean-loss form

    def objective_np(w, b, A, yy):
        z = A @ w + b
        return float(np.mean(np.logaddexp(0, z) - yy * z) + 0.5 * lam * np.sum(w ** 2))

    sk_w, sk_b = sk.coef_.ravel(), float(sk.intercept_[0])
    sk_obj = objective_np(sk_w, sk_b, A_train, y_train.to_numpy())

    def train_manual(lr):
        """Raw tensors + autograd + a hand-written gradient step (Session 5 style)."""
        w = torch.zeros(d, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        history = []
        for epoch in range(EPOCHS):
            z = Xt @ w + b
            # binary cross-entropy written out: softplus(z) - y*z  ==  -[y log s(z) + (1-y) log(1-s(z))]
            loss = (torch.nn.functional.softplus(z) - yt * z).mean() + 0.5 * lam * (w ** 2).sum()
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
            if epoch % LOG_EVERY == 0:
                history.append(float(loss.detach()))
        return w.detach(), b.detach(), history

    def train_nn(lr):
        """nn.Module + loss object + optimizer. weight_decay=lam on the weights only equals the
        (lam/2)||w||^2 penalty, and the bias is left out, matching the other two methods."""
        model = LogisticRegressionNN(d)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            [{"params": [model.linear.weight], "weight_decay": lam},
             {"params": [model.linear.bias], "weight_decay": 0.0}], lr=lr)
        history = []
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model(Xt), yt)                      # data term only
            loss.backward()
            optimizer.step()
            if epoch % LOG_EVERY == 0:
                with torch.no_grad():   # log the full objective so curves are comparable
                    history.append(float(loss) + 0.5 * lam * float((model.linear.weight ** 2).sum()))
        return model, history

    def val_loss_from_logits(z):
        return float(torch.nn.functional.softplus(z).sub(yv * z).mean())

    manual_runs, nn_runs = {}, {}
    for lr in LR_GRID:
        w, b, h = train_manual(lr)
        manual_runs[lr] = (w, b, h, val_loss_from_logits(Xv @ w + b) if np.isfinite(h[-1]) else np.inf)
        model, h2 = train_nn(lr)
        with torch.no_grad():
            nn_runs[lr] = (model, h2, val_loss_from_logits(model(Xv)) if np.isfinite(h2[-1]) else np.inf)
    lr_manual = min(manual_runs, key=lambda k: manual_runs[k][3])
    lr_nn = min(nn_runs, key=lambda k: nn_runs[k][2])
    print("manual  val loss by lr:", {k: round(v[3], 5) for k, v in manual_runs.items()}, "-> lr =", lr_manual)
    print("nn      val loss by lr:", {k: round(v[2], 5) for k, v in nn_runs.items()}, "-> lr =", lr_nn)
    w_man, b_man, hist_man, _ = manual_runs[lr_manual]
    model_nn, hist_nn, _ = nn_runs[lr_nn]

    At = torch.tensor(A_test, dtype=torch.float32)
    with torch.no_grad():
        p_man = torch.sigmoid(At @ w_man + b_man).numpy()
        p_nn = torch.sigmoid(model_nn(At)).numpy()
    p_sk = sk.predict_proba(A_test)[:, 1]

    base = DummyClassifier(strategy="prior").fit(A_train, y_train)   # predicts the train churn rate for everyone
    p_base = base.predict_proba(A_test)[:, 1]

    t_star = theoretical_threshold()
    preds = {"naive baseline": p_base, "scikit-learn": p_sk, "manual PyTorch": p_man, "nn.Module PyTorch": p_nn}
    yt_np = y_test.to_numpy()
    rows = []
    for name, p in preds.items():
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        rows.append({
            "model": name,
            "ROC-AUC": roc_auc_score(yt_np, p),
            "log loss": log_loss(yt_np, pc),
            "Brier": brier_score_loss(yt_np, p),
            "accuracy@0.5": accuracy_score(yt_np, p >= 0.5),
            "customers flagged @t*": int((p >= t_star).sum()),
            "total cost @t* (EUR)": total_cost(yt_np, p, t_star),
        })
    metrics = pd.DataFrame(rows).set_index("model")
    n_test = len(yt_np)
    metrics["cost per customer (EUR)"] = metrics["total cost @t* (EUR)"] / n_test
    do_nothing = total_cost(yt_np, np.zeros(n_test), 0.5)
    flag_all = total_cost(yt_np, np.ones(n_test), 0.5)

   
    agree = {
        "max |p_sklearn - p_manual|": float(np.abs(p_sk - p_man).max()),
        "max |p_sklearn - p_nn|": float(np.abs(p_sk - p_nn).max()),
        "max |p_manual - p_nn|": float(np.abs(p_man - p_nn).max()),
        "||w_sklearn - w_manual||": float(np.linalg.norm(sk_w - w_man.numpy())),
        "||w_sklearn - w_nn||": float(np.linalg.norm(sk_w - model_nn.linear.weight.detach().numpy().ravel())),
        "customers where flag differs (sk vs manual)": int(((p_sk >= t_star) != (p_man >= t_star)).sum()),
        "customers where flag differs (sk vs nn)": int(((p_sk >= t_star) != (p_nn >= t_star)).sum()),
        "final objective sklearn": sk_obj,
        "final objective manual": hist_man[-1],
        "final objective nn": hist_nn[-1],
    }

    pd.set_option("display.width", 200)
    print("\n=== TEST SET RESULTS ===")
    print(metrics.round(4).to_string())
    print(f"\nbusiness reference (test set, {n_test} customers): do nothing = {do_nothing:,.0f} EUR, "
          f"offer to everyone = {flag_all:,.0f} EUR, theoretical threshold t* = {t_star:.3f}")
    print("\n=== DO THE THREE METHODS AGREE? ===")
    for k, v in agree.items():
        print(f"  {k:48s} {v:.6g}")

    
    joblib.dump(pre, ART / "preprocessor.joblib")
    joblib.dump(sk, ART / "sklearn_logreg.joblib")
    torch.save({"w": w_man, "b": b_man}, ART / "manual_torch.pt")
    torch.save({"state_dict": model_nn.state_dict(), "n_features": d}, ART / "nn_torch.pt")
    test_out = X_test.copy()
    test_out["Churn"] = y_test.to_numpy()
    test_out.to_csv(ART / "test_set.csv", index=False)
    pd.DataFrame({"epoch": np.arange(len(hist_man)) * LOG_EVERY,
                  "manual PyTorch": hist_man, "nn.Module PyTorch": hist_nn}).to_csv(ART / "loss_history.csv", index=False)
    meta = {
        "feature_names": feature_names, "numeric_cols": num_cols, "categorical_cols": cat_cols,
        "seed": SEED, "n_train": int(n), "n_val": int(len(y_val)), "n_test": int(n_test),
        "best_C": best_C, "lambda": lam, "lr_manual": lr_manual, "lr_nn": lr_nn, "epochs": EPOCHS,
        "train_prior": float(y_train.mean()), "sklearn_objective": sk_obj,
        "costs": COSTS, "theoretical_threshold": t_star,
        "baseline_do_nothing_cost": do_nothing, "baseline_flag_all_cost": flag_all,
        "metrics": json.loads(metrics.reset_index().to_json(orient="records")),
        "agreement": agree,
    }
    with open(ART / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved artifacts to {ART}")


if __name__ == "__main__":
    main()
