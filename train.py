"""Training pipeline: baselines + logistic regression three ways
(scikit-learn, manual PyTorch loop, torch.nn.Module + torch.optim)."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from evaluation import LGD, MARGIN, evaluate
from features import Preprocessor, build_raw_features, load_data, split_by_date

MODELS_DIR = Path("models")
C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
LR_GRID = [0.1, 0.5, 1.0]
EPOCHS = 3000


# ---------- 2) manual PyTorch loop ----------
def train_manual(X, y, C, lr, epochs=EPOCHS):
    """Full-batch gradient descent for L2-regularized logistic regression,
    written by hand: raw tensors, autograd for gradients, manual update."""
    n, d = X.shape
    X = torch.tensor(X)
    y = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lam = 1.0 / (C * n)  # L2 strength equivalent to sklearn's C

    for _ in range(epochs):
        logits = X @ w + b                                    # forward pass
        loss = F.binary_cross_entropy_with_logits(logits, y)  # mean log-loss
        loss = loss + 0.5 * lam * (w ** 2).sum()              # L2 on weights only
        loss.backward()                                       # autograd: dloss/dw, dloss/db
        with torch.no_grad():                                 # the update is not part of the graph
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                                        # otherwise gradients accumulate
        b.grad.zero_()
    return w.detach(), b.detach()


def predict_manual(w, b, X):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X) @ w + b).numpy()


# ---------- 3) standard PyTorch workflow ----------
class LogReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)  # logits


def train_module(X, y, C, lr, epochs=EPOCHS):
    torch.manual_seed(0)
    n, d = X.shape
    X = torch.tensor(X)
    y = torch.tensor(y, dtype=torch.float32)
    model = LogReg(d)
    lam = 1.0 / (C * n)
    # weight_decay adds lam * w to the gradient, i.e. the gradient of
    # 0.5 * lam * w^2. Bias gets no decay, like sklearn and the manual loop.
    optimizer = torch.optim.SGD(
        [{"params": model.linear.weight, "weight_decay": lam},
         {"params": model.linear.bias, "weight_decay": 0.0}],
        lr=lr,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()            # reset gradients
        loss = loss_fn(model(X), y)      # forward + loss
        loss.backward()                  # gradients
        optimizer.step()                 # update (including weight decay)
    return model


def predict_module(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data()
    train, val, test = split_by_date(df)
    prep = Preprocessor().fit(train)  # fitted on TRAIN only
    X_tr, X_val, X_te = (prep.transform(d) for d in (train, val, test))
    y_tr, y_val, y_te = (d["default"].to_numpy() for d in (train, val, test))
    amt_val, amt_te = (d["loan_amnt"].to_numpy() for d in (val, test))

    results = {}

    def run(name, s_val, s_te):
        results[name] = evaluate(y_val, s_val, amt_val, y_te, s_te, amt_te)

    # Baseline 1: approve everyone (constant score)
    run("baseline_approve_all", y_val * 0.0, y_te * 0.0)

    # Baseline 2: FICO rule (lower FICO = riskier, so we negate it)
    fico_val = -build_raw_features(val)["fico_mid"].to_numpy()
    fico_te = -build_raw_features(test)["fico_mid"].to_numpy()
    run("baseline_fico", fico_val, fico_te)

    # 1) scikit-learn; C chosen by validation log-loss
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, tol=1e-8, max_iter=10000).fit(X_tr, y_tr)
        ll = log_loss(y_val, m.predict_proba(X_val)[:, 1])
        print(f"sklearn C={C:<7} val log-loss={ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, C, m)
    _, best_C, sk = best
    run("sklearn", sk.predict_proba(X_val)[:, 1], sk.predict_proba(X_te)[:, 1])

    # 2) manual PyTorch: same C as sklearn, learning rate chosen on validation
    best_m = None
    for lr in LR_GRID:
        w, b = train_manual(X_tr, y_tr, best_C, lr)
        ll = log_loss(y_val, predict_manual(w, b, X_val))
        print(f"manual  lr={lr:<7} val log-loss={ll:.4f}")
        if best_m is None or ll < best_m[0]:
            best_m = (ll, lr, w, b)
    _, best_lr, w_m, b_m = best_m
    run("pytorch_manual", predict_manual(w_m, b_m, X_val), predict_manual(w_m, b_m, X_te))
    w_gap = float(np.abs(w_m.numpy() - sk.coef_[0]).max())
    b_gap = float(abs(b_m.item() - sk.intercept_[0]))
    print(f"max |w_manual - w_sklearn| = {w_gap:.5f}   |b gap| = {b_gap:.5f}")

    # 3) nn.Module + torch.optim: same C, learning rate chosen on validation
    best_n = None
    for lr in LR_GRID:
        model = train_module(X_tr, y_tr, best_C, lr)
        ll = log_loss(y_val, predict_module(model, X_val))
        print(f"module  lr={lr:<7} val log-loss={ll:.4f}")
        if best_n is None or ll < best_n[0]:
            best_n = (ll, lr, model)
    _, best_lr_n, net = best_n
    run("pytorch_module", predict_module(net, X_val), predict_module(net, X_te))
    w_gap_n = float(np.abs(net.linear.weight.detach().numpy()[0] - sk.coef_[0]).max())
    b_gap_n = float(abs(net.linear.bias.item() - sk.intercept_[0]))
    print(f"max |w_module - w_sklearn| = {w_gap_n:.5f}   |b gap| = {b_gap_n:.5f}")

    # Save artifacts: the app only loads these, it never retrains
    joblib.dump(prep, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(sk, MODELS_DIR / "sklearn_logreg.joblib")
    torch.save({"w": w_m, "b": b_m}, MODELS_DIR / "pytorch_manual.pt")
    torch.save(net.state_dict(), MODELS_DIR / "pytorch_module.pt")
    config = {"best_C": best_C, "best_lr_manual": best_lr, "best_lr_module": best_lr_n,
              "epochs": EPOCHS,
              "weight_gap_manual_vs_sklearn": w_gap,
              "weight_gap_module_vs_sklearn": w_gap_n,
              "lgd": LGD, "margin": MARGIN,
              "n_features": len(prep.feature_names),
              "feature_names": prep.feature_names, "results": results}
    (MODELS_DIR / "results.json").write_text(json.dumps(config, indent=2))

    print(f"\nbest C = {best_C}, best lr: manual = {best_lr}, module = {best_lr_n}")
    print(f"{'model':<24}{'test AUC':>9}{'PR-AUC':>9}{'test cost':>12}{'vs approve-all':>16}")
    for name, r in results.items():
        saving = 1 - r["test_cost"] / r["test_cost_approve_all"]
        print(f"{name:<24}{r['test_auc']:>9.3f}{r['test_pr_auc']:>9.3f}"
              f"{r['test_cost']:>12,.0f}{saving:>15.1%}")


if __name__ == "__main__":
    main()