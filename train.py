"""
Assignment 1 — training pipeline: student dropout early-warning model.

Trains the same L2-regularised logistic regression three ways on the same
cohort-based split (sklearn, manual PyTorch loop, nn.Module + torch.optim),
compares them with a naive baseline, and saves everything app.py needs.
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from pipeline import (
    ARTIFACTS_DIR, COST_FN, COST_FP, MODELS_DIR, SEED, TEST_COHORTS,
    TRAIN_COHORTS, VAL_COHORTS, LogisticRegressionNet, build_preprocessor,
    build_target, business_cost, engineer_features, load_raw,
)

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LR_GRID = [0.05, 0.1, 0.5, 1.0]
EPOCHS = 4000
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


# ---------------------------------------------------------------- data
def split_by_cohort(df):
    parts = {}
    for name, cohorts in [("train", TRAIN_COHORTS), ("val", VAL_COHORTS), ("test", TEST_COHORTS)]:
        part = df[df["cohort"].isin(cohorts)]
        parts[name] = (engineer_features(part), part["dropout"].to_numpy(), part)
    return parts


# ---------------------------------------------------------------- models
def train_manual_torch(X, y, lam, lr, epochs=EPOCHS):
    """Session-5 style: raw tensors, autograd, manual gradient step."""
    torch.manual_seed(SEED)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for _ in range(epochs):
        p = torch.sigmoid(X @ w + b).clamp(1e-7, 1 - 1e-7)
        bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        loss = bce + lam / 2 * (w ** 2).sum()  # same objective as sklearn (bias not penalised)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        losses.append(loss.item())
    return {"w": w.detach(), "b": b.detach()}, losses


def train_module_torch(X, y, lam, lr, epochs=EPOCHS):
    """Standard workflow: nn.Module + loss module + optimizer."""
    torch.manual_seed(SEED)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    model = LogisticRegressionNet(X.shape[1])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # weight_decay=lam adds lam*w to the gradient == gradient of lam/2*||w||^2;
    # the bias gets no decay, like sklearn's intercept.
    optimizer = torch.optim.SGD([
        {"params": [model.linear.weight], "weight_decay": lam},
        {"params": [model.linear.bias], "weight_decay": 0.0},
    ], lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        bce = loss_fn(model(X), y)
        bce.backward()
        optimizer.step()
        losses.append(bce.item() + lam / 2 * (model.linear.weight.detach() ** 2).sum().item())
    return model, losses


def proba_manual(params, X):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X, dtype=torch.float32) @ params["w"] + params["b"]).numpy()


def proba_module(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# ---------------------------------------------------------------- evaluation
def best_threshold(y, proba):
    costs = [business_cost(y, proba, t)["cost"] for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmin(costs))])


def evaluate(y, proba, threshold):
    c = business_cost(y, proba, threshold)
    flagged = c["tp"] + c["fp"]
    return {
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(proba)) > 1 else 0.5,
        "pr_auc": average_precision_score(y, proba),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "recall": c["tp"] / max(c["tp"] + c["fn"], 1),
        "precision": c["tp"] / max(flagged, 1),
        "flagged_share": flagged / len(y),
        "cost_eur": c["cost"],
        "cost_per_student_eur": c["cost"] / len(y),
        **{k: c[k] for k in ("tp", "fp", "fn", "tn")},
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    df = build_target(load_raw())
    parts = split_by_cohort(df)
    (Xtr_df, ytr, _), (Xva_df, yva, _), (Xte_df, yte, te_raw) = parts["train"], parts["val"], parts["test"]
    print(f"rows  train={len(ytr)}  val={len(yva)}  test={len(yte)}")
    print(f"dropout rate  train={ytr.mean():.3f}  val={yva.mean():.3f}  test={yte.mean():.3f}")

    pre = build_preprocessor().fit(Xtr_df)  # fitted on train cohorts only
    Xtr, Xva, Xte = (pre.transform(d) for d in (Xtr_df, Xva_df, Xte_df))
    feature_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    n = len(ytr)

    # --- 1) scikit-learn: pick C on validation log loss
    val_scores = {}
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=10_000).fit(Xtr, ytr)
        val_scores[C] = log_loss(yva, m.predict_proba(Xva)[:, 1])
    best_C = min(val_scores, key=val_scores.get)
    sk = LogisticRegression(C=best_C, max_iter=10_000).fit(Xtr, ytr)
    # sklearn minimises 0.5*||w||^2 + C*sum(logloss) -> dividing by C*n gives
    # mean(logloss) + lam/2*||w||^2 with lam = 1/(C*n). Torch versions use this lam.
    lam = 1.0 / (best_C * n)
    print(f"best C={best_C}  (val log loss {val_scores[best_C]:.4f})  -> lam={lam:.2e}")

    # --- 2) + 3) torch: pick lr by final training objective (an optimisation
    # setting, so it is judged on how well it minimises the loss, not on val)
    lr_scores = {}
    for lr in LR_GRID:
        _, l_manual = train_manual_torch(Xtr, ytr, lam, lr)
        lr_scores[lr] = l_manual[-1] if np.isfinite(l_manual[-1]) else np.inf
    best_lr = min(lr_scores, key=lr_scores.get)
    print("lr -> final train objective:", {k: round(v, 5) for k, v in lr_scores.items()}, f"best={best_lr}")
    manual_params, manual_losses = train_manual_torch(Xtr, ytr, lam, best_lr)
    module_model, module_losses = train_module_torch(Xtr, ytr, lam, best_lr)

    # sklearn objective on the same scale, for the loss-curve reference line
    p_sk_tr = sk.predict_proba(Xtr)[:, 1]
    sk_objective = log_loss(ytr, p_sk_tr) + lam / 2 * float((sk.coef_ ** 2).sum())

    # --- naive baseline: every student gets the train dropout rate
    prevalence = float(ytr.mean())

    probas = {
        "baseline": {"val": np.full(len(yva), prevalence), "test": np.full(len(yte), prevalence)},
        "sklearn": {"val": sk.predict_proba(Xva)[:, 1], "test": sk.predict_proba(Xte)[:, 1]},
        "torch_manual": {"val": proba_manual(manual_params, Xva), "test": proba_manual(manual_params, Xte)},
        "torch_module": {"val": proba_module(module_model, Xva), "test": proba_module(module_model, Xte)},
    }

    # threshold chosen on validation cohorts by business cost, applied to test
    results = {}
    for name, p in probas.items():
        t = best_threshold(yva, p["val"])
        results[name] = evaluate(yte, p["test"], t)
    # alternative naive policies, for context
    results["mentor_nobody"] = evaluate(yte, np.zeros(len(yte)), 0.5)
    results["mentor_everybody"] = evaluate(yte, np.ones(len(yte)), 0.5)

    coefs = pd.DataFrame({
        "feature": feature_names,
        "sklearn": sk.coef_.ravel(),
        "torch_manual": manual_params["w"].numpy(),
        "torch_module": module_model.linear.weight.detach().numpy().ravel(),
    })
    intercepts = {
        "sklearn": float(sk.intercept_[0]),
        "torch_manual": float(manual_params["b"][0]),
        "torch_module": float(module_model.linear.bias.detach()[0]),
    }
    agreement = {
        name: {
            "max_abs_proba_diff_vs_sklearn": float(np.abs(probas[name]["test"] - probas["sklearn"]["test"]).max()),
            "max_abs_coef_diff_vs_sklearn": float(np.abs(coefs[name] - coefs["sklearn"]).max()),
            "same_decision_share": float(((probas[name]["test"] >= results[name]["threshold"])
                                          == (probas["sklearn"]["test"] >= results["sklearn"]["threshold"])).mean()),
            # separates "the models disagree" from "threshold selection picked a different value"
            "same_decision_share_at_sklearn_threshold": float(
                ((probas[name]["test"] >= results["sklearn"]["threshold"])
                 == (probas["sklearn"]["test"] >= results["sklearn"]["threshold"])).mean()),
        }
        for name in ("torch_manual", "torch_module")
    }
    # validation cost around the chosen thresholds, per model: one student near the
    # cut-off is enough to move the cost-minimising threshold
    agreement["val_cost_by_threshold_near_optimum"] = {
        name: {str(t): business_cost(yva, probas[name]["val"], t)["cost"] for t in (0.10, 0.11, 0.12, 0.13, 0.14)}
        for name in ("sklearn", "torch_manual", "torch_module")
    }

    # --- save models
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(sk, MODELS_DIR / "sklearn_logreg.joblib")
    torch.save(manual_params, MODELS_DIR / "torch_manual.pt")
    torch.save({"state_dict": module_model.state_dict(), "n_features": Xtr.shape[1]},
               MODELS_DIR / "torch_module.pt")

    # --- save artifacts for the app (data + metadata, no model fitting there)
    Xte_df.assign(dropout=yte, cohort=te_raw["cohort"].to_numpy()).to_csv(ARTIFACTS_DIR / "test_set.csv", index=False)
    Xtr_df.assign(dropout=ytr, cohort=parts["train"][2]["cohort"].to_numpy()).to_csv(ARTIFACTS_DIR / "train_set.csv", index=False)
    coefs.to_csv(ARTIFACTS_DIR / "coefficients.csv", index=False)
    meta = {
        "cost_fn": COST_FN, "cost_fp": COST_FP, "best_C": best_C, "lambda": lam, "best_lr": best_lr,
        "epochs": len(manual_losses), "baseline_prevalence": prevalence,
        "val_log_loss_by_C": {str(k): v for k, v in val_scores.items()},
        "final_train_objective_by_lr": {str(k): v for k, v in lr_scores.items()},
        "sklearn_train_objective": sk_objective, "intercepts": intercepts,
        "results_test": results, "agreement": agreement,
        "rows": {"train": len(ytr), "val": len(yva), "test": len(yte)},
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACTS_DIR / "loss_curves.json").write_text(json.dumps(
        {"torch_manual": manual_losses, "torch_module": module_losses, "sklearn_objective": sk_objective}))

    table = pd.DataFrame(results).T[["roc_auc", "pr_auc", "log_loss", "threshold", "recall",
                                      "precision", "flagged_share", "cost_eur", "cost_per_student_eur"]]
    print("\nTest cohorts 2016-2017 (threshold chosen on 2014-2015):")
    print(table.round(3).to_string())
    print("\nAgreement with sklearn:", json.dumps(agreement, indent=2))
    print("Intercepts:", intercepts)


if __name__ == "__main__":
    main()
