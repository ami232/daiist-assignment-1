"""
Assignment 1 — training pipeline (run with: uv run python main.py train).

Business question: which wine batches should a winery send to its (expensive)
expert tasting panel as premium candidates (quality >= 7)?
See REPORT.md for the framing and why it drives each step below.

Steps:
  1. load + profile, drop exact duplicates (leakage), build target
  2. stratified 60/20/20 train/val/test split (no time column -> no temporal leak)
  3. feature engineering ablation on validation (sklearn, fast)
  4. tune C (sklearn) and learning rate (PyTorch) on validation, sparingly
  5. train the same logistic regression three ways on the same split
  6. choose each model's cost-minimising threshold on validation
  7. evaluate once on test vs naive baselines, save everything to artifacts/
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from data_utils import ARTIFACTS_DIR, load_wine
from features import FINAL_GROUPS, build_preprocessor, engineer, make_target
from models import (as_numpy_coefs, predict_manual, predict_standard,
                    train_manual, train_standard)

SEED = 42

# Business costs (euros per batch), see REPORT.md.
COST_FP = 50    # ordinary batch sent to the tasting panel: wasted panel time
COST_FN = 200   # premium batch sold as ordinary: lost premium margin
COSTS = {"fp": COST_FP, "fn": COST_FN}

C_GRID = [0.01, 0.1, 1.0, 10.0]
LR_GRID = [0.1, 0.5, 1.0, 2.0]
EPOCHS = 20_000  # 4,000 was not enough: see the convergence section in REPORT.md


def profile(df):
    return {
        "n_rows": int(len(df)),
        "n_red": int(df.is_red.sum()),
        "n_white": int((1 - df.is_red).sum()),
        "n_missing": int(df.isna().sum().sum()),
        "n_exact_duplicates": int(df.duplicated().sum()),
        "quality_counts": {int(k): int(v) for k, v in df.quality.value_counts().sort_index().items()},
    }


def split(df):
    # Stratify on target x colour so all three splits keep the same premium
    # rate and the same red/white mix.
    strata = make_target(df).astype(str) + "_" + df["is_red"].astype(str)
    train, rest = train_test_split(df, test_size=0.4, stratify=strata, random_state=SEED)
    strata_rest = make_target(rest).astype(str) + "_" + rest["is_red"].astype(str)
    val, test = train_test_split(rest, test_size=0.5, stratify=strata_rest, random_state=SEED)
    return train, val, test


def expected_cost(y, p, t):
    pred = p >= t
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return fp * COST_FP + fn * COST_FN


def best_threshold(y, p):
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [expected_cost(y, p, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    has_two = len(np.unique(p)) > 1
    return {
        "roc_auc": float(roc_auc_score(y, p)) if has_two else 0.5,
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))),
        "brier": float(brier_score_loss(y, p)),
        "threshold": t,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost_eur": fp * COST_FP + fn * COST_FN,
        "cost_per_batch_eur": (fp * COST_FP + fn * COST_FN) / len(y),
    }


def objective(X, y, w, b, C):
    """The shared training objective (mean log loss + L2), see models.py."""
    p = np.clip(1 / (1 + np.exp(-(X @ w + b))), 1e-12, 1 - 1e-12)
    return float(log_loss(y, p) + (w ** 2).sum() / (2 * C * len(y)))


def fit_sklearn(X, y, C):
    # lbfgs + tight tol so sklearn reaches the true optimum we compare against.
    return LogisticRegression(C=C, max_iter=10_000, tol=1e-10).fit(X, y)


def ablation(dev, y_dev):
    """Which feature groups actually help?

    5-fold stratified CV on train+val (test untouched). A single 1,064-row
    validation split turned out too noisy to rank variants that differ by
    ~0.003 AUC, so CV mean +/- std is used instead.
    """
    configs = {
        "raw (12 features)": (),
        "+ log transforms": ("log",),
        "+ log + chemistry ratios": ("log", "chem"),
        "+ log + colour interactions": ("log", "color"),
        "+ log + colour + chemistry": ("log", "color", "chem"),
        "+ log + colour + splines (final)": FINAL_GROUPS,
        "final, without winsorizing": FINAL_GROUPS,
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, groups in configs.items():
        clip = {} if "without winsorizing" not in name else {"lower_q": 0.0, "upper_q": 1.0}
        aucs, aps, lls = [], [], []
        for i, j in folds.split(dev, y_dev):
            a_df, b_df = engineer(dev.iloc[i], groups), engineer(dev.iloc[j], groups)
            pre = build_preprocessor(groups, **clip).fit(a_df)
            Xa = pre.transform(a_df)
            p = fit_sklearn(Xa, y_dev[i], C=1.0).predict_proba(pre.transform(b_df))[:, 1]
            aucs.append(roc_auc_score(y_dev[j], p))
            aps.append(average_precision_score(y_dev[j], p))
            lls.append(log_loss(y_dev[j], p))
        Xa_full = np.c_[Xa, np.ones(len(Xa))]
        rows.append({"features": name, "n_features": Xa.shape[1],
                     "cv_roc_auc": np.mean(aucs), "cv_roc_auc_std": np.std(aucs),
                     "cv_pr_auc": np.mean(aps), "cv_log_loss": np.mean(lls),
                     # conditioning of X^T X: how hard the problem is for plain gradient descent
                     "condition_number": float(np.linalg.cond(Xa_full.T @ Xa_full))})
    return pd.DataFrame(rows)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ---- 1. load, profile, dedupe ----
    raw = load_wine()
    info = profile(raw)
    df = raw.drop_duplicates().reset_index(drop=True)
    info["n_rows_after_dedup"] = int(len(df))
    info["premium_rate_after_dedup"] = float(make_target(df).mean())
    (ARTIFACTS_DIR / "data_profile.json").write_text(json.dumps(info, indent=2))
    print(f"{info['n_rows']} rows, {info['n_exact_duplicates']} exact duplicates dropped "
          f"-> {len(df)} rows, premium rate {info['premium_rate_after_dedup']:.3f}")

    # ---- 2. split ----
    train, val, test = split(df)
    y_tr, y_va, y_te = (make_target(d).to_numpy() for d in (train, val, test))
    print(f"split: train {len(train)}, val {len(val)}, test {len(test)}")

    # ---- 3. feature ablation (5-fold CV on train+val) ----
    dev = pd.concat([train, val])
    abl = ablation(dev, make_target(dev).to_numpy())
    print("\nFeature ablation (5-fold CV on train+val, sklearn C=1):")
    print(abl.round(4).to_string(index=False))
    abl.to_csv(ARTIFACTS_DIR / "ablation.csv", index=False)

    # ---- final features + preprocessing, fit on train only ----
    Xtr_df, Xva_df, Xte_df = engineer(train), engineer(val), engineer(test)
    pre = build_preprocessor().fit(Xtr_df)
    Xtr, Xva, Xte = pre.transform(Xtr_df), pre.transform(Xva_df), pre.transform(Xte_df)
    feature_names = pre.columns

    # ---- 4a. tune C with sklearn on validation log loss ----
    c_scores = {C: log_loss(y_va, fit_sklearn(Xtr, y_tr, C).predict_proba(Xva)[:, 1]) for C in C_GRID}
    best_C = min(c_scores, key=c_scores.get)
    print(f"\nC search (val log loss): { {k: round(v, 4) for k, v in c_scores.items()} } -> C={best_C}")

    # ---- 4b. tune learning rate for the manual loop (same lr reused by nn.Module) ----
    lr_scores = {}
    for lr in LR_GRID:
        params, hist = train_manual(Xtr, y_tr, best_C, lr, EPOCHS, Xva, y_va)
        final = hist[-1]["train_loss"]
        lr_scores[lr] = final if np.isfinite(final) else float("inf")  # diverged
    best_lr = min(lr_scores, key=lr_scores.get)
    print(f"lr search (final train objective after {EPOCHS} epochs): "
          f"{ {k: round(v, 6) for k, v in lr_scores.items()} } -> lr={best_lr}")

    # ---- 5. train three ways on the same split ----
    t0 = time.perf_counter()
    sk = fit_sklearn(Xtr, y_tr, best_C)
    t_sk = time.perf_counter() - t0

    t0 = time.perf_counter()
    manual_params, manual_hist = train_manual(Xtr, y_tr, best_C, best_lr, EPOCHS, Xva, y_va,
                                              w_ref=sk.coef_.ravel())
    t_man = time.perf_counter() - t0

    t0 = time.perf_counter()
    net, std_hist = train_standard(Xtr, y_tr, best_C, best_lr, EPOCHS, Xva, y_va, seed=SEED,
                                   w_ref=sk.coef_.ravel())
    t_std = time.perf_counter() - t0

    predict = {
        "sklearn": lambda X: sk.predict_proba(X)[:, 1],
        "manual_pytorch": lambda X: predict_manual(manual_params, X),
        "nn_module_pytorch": lambda X: predict_standard(net, X),
    }

    # ---- baselines ----
    prior = y_tr.mean()
    alcohol_idx = feature_names.index("alcohol")
    alc_only = fit_sklearn(Xtr[:, [alcohol_idx]], y_tr, best_C)
    predict["baseline_prior"] = lambda X: np.full(len(X), prior)
    predict["baseline_alcohol_only"] = lambda X: alc_only.predict_proba(X[:, [alcohol_idx]])[:, 1]

    # ---- 6. thresholds on validation, 7. evaluate on test ----
    results = {}
    for name, f in predict.items():
        p_va, p_te = f(Xva), f(Xte)
        t = best_threshold(y_va, p_va) if not name == "baseline_prior" else 1.0  # prior: never flag
        results[name] = {"val": metrics(y_va, p_va, t), "test": metrics(y_te, p_te, t)}
    # "flag everything" is the other trivial policy: send every batch to the panel
    results["baseline_flag_all"] = {s: metrics(y, np.ones(len(y)), 0.5)
                                    for s, y in (("val", y_va), ("test", y_te))}
    for s in ("val", "test"):
        results["baseline_flag_all"][s]["roc_auc"] = 0.5

    # ---- agreement between the three implementations ----
    coef = {
        "sklearn": as_numpy_coefs(sk.coef_, sk.intercept_),
        "manual_pytorch": as_numpy_coefs(manual_params["w"].numpy(), manual_params["b"].numpy()),
        "nn_module_pytorch": as_numpy_coefs(net.linear.weight.detach().numpy(),
                                            net.linear.bias.detach().numpy()),
    }
    p_te = {k: predict[k](Xte) for k in coef}
    agreement = {"sklearn": {"train_objective": objective(Xtr, y_tr, *coef["sklearn"], best_C)}}
    for k in ("manual_pytorch", "nn_module_pytorch"):
        agreement[k] = {
            "train_objective": objective(Xtr, y_tr, *coef[k], best_C),
            "objective_gap_vs_sklearn": objective(Xtr, y_tr, *coef[k], best_C)
                                        - agreement["sklearn"]["train_objective"],
            "max_abs_coef_diff_vs_sklearn": float(np.max(np.abs(coef[k][0] - coef["sklearn"][0]))),
            "intercept_diff_vs_sklearn": float(coef[k][1] - coef["sklearn"][1]),
            "max_abs_prob_diff_vs_sklearn": float(np.max(np.abs(p_te[k] - p_te["sklearn"]))),
            "test_label_disagreements_at_sklearn_threshold": int(
                ((p_te[k] >= results["sklearn"]["test"]["threshold"]) !=
                 (p_te["sklearn"] >= results["sklearn"]["test"]["threshold"])).sum()),
            "test_label_disagreements_at_own_threshold": int(
                ((p_te[k] >= results[k]["test"]["threshold"]) !=
                 (p_te["sklearn"] >= results["sklearn"]["test"]["threshold"])).sum()),
        }

    print("\nTest-set results (threshold chosen on validation):")
    table = pd.DataFrame({k: v["test"] for k, v in results.items()}).T
    print(table[["roc_auc", "pr_auc", "log_loss", "threshold", "precision", "recall",
                 "fp", "fn", "cost_eur"]].round(4).to_string())
    print("\nAgreement with sklearn:")
    print(json.dumps(agreement, indent=2))
    print(f"\nTrain time: sklearn {t_sk:.2f}s, manual {t_man:.2f}s, nn.Module {t_std:.2f}s")

    # ---- save artifacts (the app only ever loads these) ----
    joblib.dump(pre, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(sk, ARTIFACTS_DIR / "sklearn_logreg.joblib")
    joblib.dump(alc_only, ARTIFACTS_DIR / "baseline_alcohol_only.joblib")
    torch.save(manual_params, ARTIFACTS_DIR / "manual_pytorch_params.pt")
    torch.save(net.state_dict(), ARTIFACTS_DIR / "nn_module_pytorch.pt")

    test.assign(split="test").to_csv(ARTIFACTS_DIR / "test_set.csv", index=False)

    coef_df = pd.DataFrame({k: v[0] for k, v in coef.items()}, index=feature_names)
    coef_df.loc["(intercept)"] = [v[1] for v in coef.values()]
    coef_df.to_csv(ARTIFACTS_DIR / "coefficients.csv")

    pd.concat([pd.DataFrame(manual_hist).assign(model="manual_pytorch"),
               pd.DataFrame(std_hist).assign(model="nn_module_pytorch")]).to_csv(
        ARTIFACTS_DIR / "training_history.csv", index=False)

    summary = {
        "costs_eur": COSTS,
        "bayes_threshold_if_calibrated": COST_FP / (COST_FP + COST_FN),
        "best_C": best_C, "C_search_val_log_loss": {str(k): v for k, v in c_scores.items()},
        "best_lr": best_lr, "lr_search_final_train_objective": {str(k): v for k, v in lr_scores.items()},
        "epochs": EPOCHS, "feature_names": feature_names, "n_features": len(feature_names),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "train_time_s": {"sklearn": t_sk, "manual_pytorch": t_man, "nn_module_pytorch": t_std},
        "results": results, "agreement_vs_sklearn": agreement,
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved artifacts to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
