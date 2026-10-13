"""
Assignment 1 -- training pipeline (run with `uv run python main.py train`).

Steps:
  1. Load the committed Bank Marketing data and engineer features (pipeline.py).
  2. Time-ordered split: oldest 70% train / next 15% validation / newest 15% test.
  3. Small, honest tuning on VALIDATION with scikit-learn only:
       - which feature groups to keep (ablation)
       - regularisation strength C
  4. Train the SAME L2-regularised logistic regression three ways on TRAIN:
       a) scikit-learn LogisticRegression
       b) manual PyTorch: raw tensors + autograd + hand-written update step
       c) standard PyTorch: nn.Module + nn.BCEWithLogitsLoss + torch.optim.SGD
  5. Pick each model's profit-maximising threshold on VALIDATION.
  6. Evaluate everything once on TEST against a naive baseline.
  7. Save every artifact the Gradio app needs (it never retrains).
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn

from pipeline import (
    ARTIFACTS, CALL_COST, FEATURE_SETS, SEED, SUB_VALUE,
    LogisticRegressionNet, best_threshold, build_preprocessor, business_outcome,
    columns_for, engineer, load_raw, predict_manual, predict_standard,
    regularised_objective, temporal_split,
)

C_GRID = [0.01, 0.1, 1.0, 10.0]
LR_GRID = [0.1, 0.5, 1.0]
EPOCHS = 4000

torch.manual_seed(SEED)
np.random.seed(SEED)
ARTIFACTS.mkdir(exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ---------------------------------------------------------------------------
# 1-2. Data + split
# ---------------------------------------------------------------------------
section("1. Data and time-ordered split")
df = engineer(load_raw())
train_df, val_df, test_df = temporal_split(df)
split_info = {}
for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    split_info[name] = {
        "rows": len(part),
        "positive_rate": round(float(part["y"].mean()), 4),
        "from": part["period"].iloc[0],
        "to": part["period"].iloc[-1],
    }
    print(f"{name:>10}: {len(part):>6} rows | {part['period'].iloc[0]} -> "
          f"{part['period'].iloc[-1]} | subscription rate {part['y'].mean():.3f}")

y_train = train_df["y"].to_numpy()
y_val = val_df["y"].to_numpy()
y_test = test_df["y"].to_numpy()


# ---------------------------------------------------------------------------
# 3. Feature-set ablation + C, chosen on validation average precision
#    (the business ranks clients and calls the top ones, and positives are
#    rare, so precision-recall quality is what matters)
# ---------------------------------------------------------------------------
section("2. Tuning on validation (scikit-learn only)")
tuning_rows = []
for fs_name in FEATURE_SETS:
    numeric, categorical = columns_for(fs_name)
    pre = build_preprocessor(numeric, categorical)
    Xtr = pre.fit_transform(train_df)
    Xva = pre.transform(val_df)
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=10_000, tol=1e-6)
        clf.fit(Xtr, y_train)
        p = clf.predict_proba(Xva)[:, 1]
        row = {
            "feature_set": fs_name, "C": C, "n_features": Xtr.shape[1],
            "val_avg_precision": average_precision_score(y_val, p),
            "val_roc_auc": roc_auc_score(y_val, p),
            "val_log_loss": log_loss(y_val, p),
        }
        tuning_rows.append(row)
        print(f"{fs_name:<24} C={C:<6} AP={row['val_avg_precision']:.4f} "
              f"AUC={row['val_roc_auc']:.4f} logloss={row['val_log_loss']:.4f}")

tuning = pd.DataFrame(tuning_rows)
tuning.to_csv(ARTIFACTS / "tuning.csv", index=False)
best = tuning.loc[tuning["val_avg_precision"].idxmax()]
FEATURE_SET, C = str(best["feature_set"]), float(best["C"])
print(f"\nChosen: feature set '{FEATURE_SET}', C = {C}")


# ---------------------------------------------------------------------------
# Final preprocessing (fit on TRAIN only)
# ---------------------------------------------------------------------------
numeric, categorical = columns_for(FEATURE_SET)
preprocessor = build_preprocessor(numeric, categorical)
X_train = preprocessor.fit_transform(train_df).astype(np.float32)
X_val = preprocessor.transform(val_df).astype(np.float32)
X_test = preprocessor.transform(test_df).astype(np.float32)
feature_names = list(preprocessor.get_feature_names_out())
n_train, n_features = X_train.shape

# sklearn minimises  C * sum(logloss) + 0.5 * ||w||^2.
# Divide by C * n:    mean(logloss) + 0.5 * (1 / (C * n)) * ||w||^2.
# So the PyTorch versions use lam = 1 / (C * n) to optimise the SAME objective.
LAM = 1.0 / (C * n_train)
print(f"{n_features} features after one-hot encoding; lambda for PyTorch = {LAM:.3e}")


# ---------------------------------------------------------------------------
# 4a. scikit-learn
# ---------------------------------------------------------------------------
section("3a. scikit-learn LogisticRegression")
sk_model = LogisticRegression(C=C, max_iter=10_000, tol=1e-8)
sk_model.fit(X_train, y_train)
sk_w, sk_b = sk_model.coef_.ravel(), float(sk_model.intercept_[0])
sk_obj = regularised_objective(X_train, y_train, sk_w, sk_b, LAM)
print(f"train objective = {sk_obj:.6f}")


# ---------------------------------------------------------------------------
# 4b. Manual PyTorch: raw tensors, autograd, hand-written gradient step
# ---------------------------------------------------------------------------
Xt = torch.tensor(X_train)
yt = torch.tensor(y_train, dtype=torch.float32)


def train_manual(lr: float, epochs: int = EPOCHS):
    w = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for _ in range(epochs):
        z = Xt @ w + b                                     # logits
        # numerically stable log-loss: max(z,0) - z*y + log(1 + exp(-|z|))
        data_loss = (torch.clamp(z, min=0) - z * yt + torch.log1p(torch.exp(-z.abs()))).mean()
        loss = data_loss + 0.5 * LAM * (w * w).sum()       # L2 on weights, not bias
        loss.backward()                                    # autograd fills w.grad, b.grad
        with torch.no_grad():                              # manual update step
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())
    return {"w": w.detach(), "b": b.detach()}, history


# ---------------------------------------------------------------------------
# 4c. Standard PyTorch: nn.Module + loss module + optimizer
# ---------------------------------------------------------------------------
def train_standard(lr: float, epochs: int = EPOCHS):
    torch.manual_seed(SEED)
    model = LogisticRegressionNet(n_features)
    nn.init.zeros_(model.linear.weight)   # same start point as the manual loop
    nn.init.zeros_(model.linear.bias)
    loss_fn = nn.BCEWithLogitsLoss()
    # SGD's weight_decay adds lam * w to the gradient, which is exactly the
    # gradient of 0.5 * lam * ||w||^2. The bias gets no decay (like sklearn).
    optimizer = torch.optim.SGD(
        [{"params": [model.linear.weight], "weight_decay": LAM},
         {"params": [model.linear.bias], "weight_decay": 0.0}],
        lr=lr,
    )
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        with torch.no_grad():  # log the full objective, comparable to the manual loop
            # (penalty on the SAME pre-step weights the loss was computed with)
            penalty = 0.5 * LAM * (model.linear.weight ** 2).sum().item()
        optimizer.step()
        history.append(loss.item() + penalty)
    return model, history


def final_objective(history):
    last = history[-1]
    return last if np.isfinite(last) else np.inf


# Learning rate is an OPTIMISATION setting, not a generalisation one: pick the
# rate that reaches the lowest training objective in the fixed epoch budget.
section("3b/3c. PyTorch learning-rate check (lowest final training objective)")
lr_rows = []
manual_runs, standard_runs = {}, {}
for lr in LR_GRID:
    manual_runs[lr] = train_manual(lr)
    standard_runs[lr] = train_standard(lr)
    m_obj, s_obj = final_objective(manual_runs[lr][1]), final_objective(standard_runs[lr][1])
    lr_rows.append({"lr": lr, "manual_final_objective": m_obj, "standard_final_objective": s_obj})
    print(f"lr={lr:<4} manual={m_obj:.6f} standard={s_obj:.6f} (sklearn optimum {sk_obj:.6f})")

LR_MANUAL = min(LR_GRID, key=lambda lr: final_objective(manual_runs[lr][1]))
LR_STANDARD = min(LR_GRID, key=lambda lr: final_objective(standard_runs[lr][1]))
manual_params, manual_hist = manual_runs[LR_MANUAL]
std_model, std_hist = standard_runs[LR_STANDARD]
print(f"Chosen lr: manual={LR_MANUAL}, standard={LR_STANDARD}")
pd.DataFrame(lr_rows).to_csv(ARTIFACTS / "lr_check.csv", index=False)
pd.DataFrame({"epoch": np.arange(1, EPOCHS + 1), "manual": manual_hist,
              "standard": std_hist, "sklearn_optimum": sk_obj}).to_csv(
    ARTIFACTS / "loss_history.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Thresholds on validation, 6. evaluation on test
# ---------------------------------------------------------------------------
section("4. Thresholds (validation) and test results")
models = {
    "scikit-learn": lambda X: sk_model.predict_proba(X)[:, 1],
    "Manual PyTorch": lambda X: predict_manual(manual_params, X),
    "Standard PyTorch": lambda X: predict_standard(std_model, X),
}
val_probs = {name: f(X_val) for name, f in models.items()}
test_probs = {name: f(X_test) for name, f in models.items()}
thresholds = {name: best_threshold(y_val, p) for name, p in val_probs.items()}

# Naive baseline: predict the training subscription rate for everyone.
# As a decision rule that means "call everyone" (it is above break-even 0.125
# only if the base rate is; we report both call-everyone and call-nobody).
base_rate = float(y_train.mean())
test_probs_baseline = np.full(len(y_test), base_rate)


def evaluate(name, p, threshold):
    out = business_outcome(y_test, p, threshold)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, p) if np.unique(p).size > 1 else 0.5,
        "avg_precision": average_precision_score(y_test, p),
        "log_loss": log_loss(y_test, np.clip(p, 1e-7, 1 - 1e-7)),
        "threshold (from val)": threshold,
        "calls": out["calls"],
        "precision": out["precision"],
        "recall": out["recall"],
        "profit_eur": out["profit"],
    }


rows = [evaluate(name, p, thresholds[name]) for name, p in test_probs.items()]
rows.append(evaluate("Baseline: call everyone", test_probs_baseline, 0.0))
rows.append(evaluate("Baseline: call nobody", test_probs_baseline, 1.01))
metrics = pd.DataFrame(rows)
print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# Agreement between the three implementations
weights = {
    "scikit-learn": (sk_w, sk_b),
    "Manual PyTorch": (manual_params["w"].numpy(), float(manual_params["b"][0])),
    "Standard PyTorch": (std_model.linear.weight.detach().numpy().ravel(),
                         float(std_model.linear.bias.detach()[0])),
}
agreement = {}
for name in ["Manual PyTorch", "Standard PyTorch"]:
    w, b = weights[name]
    agreement[name] = {
        "train_objective": regularised_objective(X_train, y_train, w, b, LAM),
        "objective_gap_vs_sklearn": regularised_objective(X_train, y_train, w, b, LAM) - sk_obj,
        "max_abs_coef_diff_vs_sklearn": float(np.max(np.abs(w - sk_w))),
        "intercept_diff_vs_sklearn": b - sk_b,
        "max_abs_test_prob_diff_vs_sklearn": float(np.max(np.abs(test_probs[name] - test_probs["scikit-learn"]))),
        "mean_abs_test_prob_diff_vs_sklearn": float(np.mean(np.abs(test_probs[name] - test_probs["scikit-learn"]))),
    }
agreement["scikit-learn"] = {"train_objective": sk_obj}
print("\nAgreement with scikit-learn:")
print(json.dumps(agreement, indent=2))

coefs = pd.DataFrame({"feature": feature_names,
                      **{name: w for name, (w, _) in weights.items()}})
coefs.to_csv(ARTIFACTS / "coefficients.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Save artifacts
# ---------------------------------------------------------------------------
joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk_model, ARTIFACTS / "sklearn_model.joblib")
torch.save(manual_params, ARTIFACTS / "manual_torch.pt")
torch.save(std_model.state_dict(), ARTIFACTS / "standard_torch.pt")
metrics.to_csv(ARTIFACTS / "metrics.csv", index=False)

meta = {
    "feature_set": FEATURE_SET, "numeric": numeric, "categorical": categorical,
    "feature_names": feature_names, "n_features": n_features,
    "C": C, "lambda": LAM, "epochs": EPOCHS,
    "lr_manual": LR_MANUAL, "lr_standard": LR_STANDARD,
    "thresholds": thresholds, "train_base_rate": base_rate,
    "call_cost": CALL_COST, "sub_value": SUB_VALUE,
    "splits": split_info, "agreement": agreement,
}
(ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))

# A ready-to-paste markdown table for REPORT.md
with open(ARTIFACTS / "results.md", "w", encoding="utf-8") as f:
    f.write("| Model | ROC-AUC | Avg precision | Log-loss | Threshold | Calls | Precision | Recall | Profit (EUR) |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['model']} | {r['roc_auc']:.4f} | {r['avg_precision']:.4f} | {r['log_loss']:.4f} | "
                f"{r['threshold (from val)']:.2f} | {r['calls']} | {r['precision']:.3f} | "
                f"{r['recall']:.3f} | {r['profit_eur']:,.0f} |\n")

print(f"\nSaved artifacts to {ARTIFACTS}")
