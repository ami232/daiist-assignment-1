"""
Assignment 1 - training pipeline: hotel booking cancellation, logistic
regression trained three ways (scikit-learn, manual PyTorch, nn.Module +
torch.optim). The reasoning behind each step is in REPORT.md.

Run with: uv run python main.py train
"""

import json
import random
import time
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)

from pipeline import (ARTIFACTS, EXCLUDED, LEAKY_FEATURES, TARGET, FeatureConfig, LogisticRegressionNet,
                      booking_revenue, business_cost, clean, cost_curve, load_data, make_preprocessor,
                      manual_predict_proba)

SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
FN_MULT, FP_MULT = 1.0, 2.0
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LR_GRID = [0.1, 0.3, 1.0, 3.0, 5.0]
EPOCHS = 10_000
CURVE_EVERY = 10
SNAPSHOT_EPOCHS = (1000, 3000, 10_000)
AGREEMENT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.6667, 0.7)
N_BOOTSTRAP = 2000


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- split

def temporal_split(df: pd.DataFrame):
    n = len(df)
    cut1 = df.arrival_date.iloc[int(SPLIT_FRACTIONS[0] * n)]
    cut2 = df.arrival_date.iloc[int((SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]) * n)]
    train = df[df.arrival_date < cut1]
    val = df[(df.arrival_date >= cut1) & (df.arrival_date < cut2)]
    test = df[df.arrival_date >= cut2]
    return train, val, test, cut1, cut2


def split_report(splits: dict) -> dict:
    rep = {}
    for name, d in splits.items():
        nonref = d.deposit_type == "Non Refund"
        rep[name] = {
            "rows": len(d),
            "from": str(d.arrival_date.min().date()), "to": str(d.arrival_date.max().date()),
            "cancel_rate": round(float(d[TARGET].mean()), 4),
            "city_share": round(float((d.hotel == "City Hotel").mean()), 4),
            "non_refund_share": round(float(nonref.mean()), 4),
            "non_refund_cancel_rate": round(float(d.loc[nonref, TARGET].mean()), 4) if nonref.any() else None,
            "months": sorted(d.arrival_date.dt.month.unique().tolist()),
            "adr_mean_canceled": round(float(d.loc[d[TARGET] == 1, "adr"].mean()), 2),
            "adr_mean_not_canceled": round(float(d.loc[d[TARGET] == 0, "adr"].mean()), 2),
        }
    return rep


# ---------------------------------------------------------------- metrics

def prob_metrics(y, p) -> dict:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return {"log_loss": log_loss(y, p), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


def threshold_metrics(y, p, revenue, t) -> dict:
    pred = (p >= t).astype(int)
    out = {"threshold": float(t), "accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}
    out.update(business_cost(y, p, revenue, t, FN_MULT, FP_MULT))
    return out


def best_threshold(y, p, revenue) -> float:
    grid, costs = cost_curve(y, p, revenue, FN_MULT, FP_MULT)
    return float(grid[int(np.argmin(costs))])


def rounded(d: dict, nd: int = 4) -> dict:
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items()}


# ---------------------------------------------------------------- the three models

def fit_sklearn(X, y, C) -> LogisticRegression:
    model = LogisticRegression(C=C, max_iter=10_000, tol=1e-10)
    return model.fit(X, y)


def fit_manual_torch(X, y, lam, lr, epochs=EPOCHS, snapshots=()):
    """Full-batch gradient descent with raw tensors and autograd, no optimizer.
    Loss = mean BCE + lam/2 * ||w||^2 (bias not penalised), same objective as sklearn.
    `snapshots`: epoch counts at which to keep a copy of (w, b) for the convergence table."""
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(X_t.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    curve, snaps = [], {}
    for epoch in range(epochs):
        z = X_t @ w + b
        bce = -(y_t * torch.nn.functional.logsigmoid(z) + (1 - y_t) * torch.nn.functional.logsigmoid(-z)).mean()
        loss = bce + 0.5 * lam * (w @ w)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if epoch % CURVE_EVERY == 0:
            curve.append(loss.item())
        if epoch + 1 in snapshots:
            snaps[epoch + 1] = (w.detach().numpy().copy(), float(b.detach()[0]))
        if not np.isfinite(loss.item()):
            break
    return {"w": w.detach(), "b": b.detach()}, curve, snaps


def fit_standard_torch(X, y, lam, lr, epochs=EPOCHS, snapshots=()):
    """nn.Module + BCEWithLogitsLoss + optim.SGD (full batch). L2 via weight_decay
    on the weight only, which adds lam*w to the gradient = d/dw of lam/2*||w||^2."""
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    model = LogisticRegressionNet(X_t.shape[1])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD([
        {"params": [model.linear.weight], "weight_decay": lam},
        {"params": [model.linear.bias], "weight_decay": 0.0},
    ], lr=lr)
    curve, snaps = [], {}
    for epoch in range(epochs):
        optimizer.zero_grad()
        bce = loss_fn(model(X_t), y_t)
        if epoch % CURVE_EVERY == 0:
            # Penalty from the same (pre-step) weights as the BCE, matching the manual curve.
            w = model.linear.weight.detach()
            curve.append(bce.item() + 0.5 * lam * (w * w).sum().item())
        bce.backward()
        optimizer.step()
        if epoch + 1 in snapshots:
            snaps[epoch + 1] = (model.linear.weight.detach().numpy().ravel().copy(),
                                float(model.linear.bias.detach()[0]))
        if not np.isfinite(bce.item()):
            break
    return model, curve, snaps


def objective(y, p, w, lam) -> float:
    """Shared objective: mean BCE + lam/2 ||w||^2, to check all three reach the same optimum."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(log_loss(y, p) + 0.5 * lam * np.dot(w, w))


# ---------------------------------------------------------------- pipeline

def main() -> None:
    t0 = time.time()
    set_seeds()
    ARTIFACTS.mkdir(exist_ok=True)

    raw = load_data()
    df = clean(raw)
    print(f"Loaded {len(raw):,} rows, {len(raw) - len(df)} dropped (0 nights or 0 guests) -> {len(df):,}")

    train, val, test, cut1, cut2 = temporal_split(df)
    splits_info = split_report({"train": train, "val": val, "test": test})
    for k, v in splits_info.items():
        print(f"  {k:5s} {v['rows']:6,} rows  {v['from']} -> {v['to']}  cancel={v['cancel_rate']:.3f}  "
              f"non_refund={v['non_refund_share']:.3f} (cancel {v['non_refund_cancel_rate']})")
    missing_months = sorted(set(splits_info["test"]["months"] + splits_info["val"]["months"])
                            - set(splits_info["train"]["months"]))
    print(f"  months in val/test missing from train: {missing_months or 'none'}")

    y_tr, y_va, y_te = (d[TARGET].to_numpy(copy=True) for d in (train, val, test))

    # 1) Feature-variant selection on validation (sklearn, C=1): one axis at a time.
    print("\n[1] Feature variants (val log-loss, sklearn C=1)")
    variant_rows = []

    def score_config(cfg: FeatureConfig, C: float = 1.0, label: str = "") -> dict:
        pre = make_preprocessor(cfg).fit(train, y_tr)
        Xtr, Xva = pre.transform(train), pre.transform(val)
        p = fit_sklearn(Xtr, y_tr, C).predict_proba(Xva)[:, 1]
        rev = booking_revenue(val, pre.named_steps["features"].adr_cap_)
        t = best_threshold(y_va, p, rev)
        row = {"label": label, "config": cfg.as_dict(), "C": C, "n_features": Xtr.shape[1],
               **rounded(prob_metrics(y_va, p)), "best_threshold": t,
               "min_cost": round(business_cost(y_va, p, rev, t, FN_MULT, FP_MULT)["total_cost"], 2)}
        print(f"  {label:32s} feats={row['n_features']:3d} logloss={row['log_loss']:.4f} "
              f"pr_auc={row['pr_auc']:.4f} cost={row['min_cost']:,.0f}")
        return row

    cfg = FeatureConfig()
    axes = [("lead", ["log", "bucket", "both"]),
            ("month", ["cyclic", "onehot", "onehot_week"]),
            ("interactions", [True, False]),
            ("meal_room", [True, False])]
    for field, options in axes:
        results = []
        for opt in options:
            row = score_config(replace(cfg, **{field: opt}), label=f"{field}={opt}")
            row["axis"] = field
            variant_rows.append(row)
            results.append((row["log_loss"], opt))
        best = min(results, key=lambda r: r[0])[1]
        cfg = replace(cfg, **{field: best})
        print(f"  -> keep {field}={best}")
    print(f"  final feature config: {cfg.as_dict()}")

    # 2) Regularisation strength C on validation.
    print("\n[2] Regularisation (val log-loss)")
    c_rows = [score_config(cfg, C=C, label=f"C={C}") for C in C_GRID]
    best_C = min(c_rows, key=lambda r: r["log_loss"])["C"]
    print(f"  -> C={best_C}")

    # 3) Final preprocessing, fit on train only.
    pre = make_preprocessor(cfg).fit(train, y_tr)
    X_tr, X_va, X_te = (pre.transform(d).astype(np.float64) for d in (train, val, test))
    feature_names = pre.named_steps["encode"].get_feature_names_out().tolist()
    adr_cap = pre.named_steps["features"].adr_cap_
    rev_va, rev_te = booking_revenue(val, adr_cap), booking_revenue(test, adr_cap)
    n_train = X_tr.shape[0]
    lam = 1.0 / (best_C * n_train)
    print(f"\n[3] X_train {X_tr.shape}, lambda = 1/(C*n) = {lam:.3e}, ADR cap = {adr_cap:.2f}")

    # 4) Learning rate for the two PyTorch versions (same C/lambda, val log-loss).
    print("\n[4] Learning rate (final train objective after %d full-batch epochs)" % EPOCHS)
    lr_rows = []
    for method in ("manual", "standard"):
        for lr in LR_GRID:
            set_seeds()
            if method == "manual":
                params, curve, _ = fit_manual_torch(X_tr, y_tr, lam, lr)
                p = manual_predict_proba(params, X_va)
            else:
                model, curve, _ = fit_standard_torch(X_tr, y_tr, lam, lr)
                p = model.predict_proba(X_va)
            ok = np.all(np.isfinite(p)) and np.isfinite(curve[-1])
            ll = float(log_loss(y_va, np.clip(p, 1e-15, 1 - 1e-15))) if ok else float("nan")
            lr_rows.append({"method": method, "lr": lr, "val_log_loss": round(ll, 5),
                            "final_train_objective": round(curve[-1], 6) if ok else None})
            print(f"  {method:8s} lr={lr:<5} val_logloss={ll:.5f}  train_obj={curve[-1]:.6f}")
    # The learning rate is an optimisation setting: pick the one that gets closest to the optimum
    # of the shared objective (lowest final train objective). Generalisation is C's job; picking
    # by val log-loss would reward stopping short of the optimum (implicit early stopping).
    best_lr = {}
    for method in ("manual", "standard"):
        rows = [r for r in lr_rows if r["method"] == method and r["final_train_objective"] is not None]
        best_lr[method] = min(rows, key=lambda r: r["final_train_objective"])["lr"]
    print(f"  -> {best_lr}")

    # 5) Final models: same split, same features, same C.
    print("\n[5] Training final models")
    set_seeds()
    sk = fit_sklearn(X_tr, y_tr, best_C)
    set_seeds()
    manual_params, manual_curve, manual_snaps = fit_manual_torch(X_tr, y_tr, lam, best_lr["manual"],
                                                                  snapshots=SNAPSHOT_EPOCHS)
    set_seeds()
    std_model, std_curve, std_snaps = fit_standard_torch(X_tr, y_tr, lam, best_lr["standard"],
                                                         snapshots=SNAPSHOT_EPOCHS)

    predict = {
        "sklearn": lambda X: sk.predict_proba(X)[:, 1],
        "manual_torch": lambda X: manual_predict_proba(manual_params, X),
        "standard_torch": lambda X: std_model.predict_proba(X),
    }
    weights = {
        "sklearn": (sk.coef_.ravel(), float(sk.intercept_[0])),
        "manual_torch": (manual_params["w"].numpy(), float(manual_params["b"][0])),
        "standard_torch": (std_model.linear.weight.detach().numpy().ravel(),
                           float(std_model.linear.bias.detach()[0])),
    }

    # 6) Decision threshold. Both costs scale with the booking's revenue, so the expected-cost
    #    rule "flag if p*FN > (1-p)*FP" gives p > FP/(FP+FN) for every booking: the official
    #    threshold. The validation-optimal one is kept for comparison (it did not transfer to
    #    the test period - see REPORT.md).
    theory_t = round(FP_MULT / (FP_MULT + FN_MULT), 4)
    p_va = {m: f(X_va) for m, f in predict.items()}
    p_te = {m: f(X_te) for m, f in predict.items()}
    val_thresholds = {m: best_threshold(y_va, p, rev_va) for m, p in p_va.items()}
    thresholds = {m: theory_t for m in predict}
    print(f"\n[6] official threshold {theory_t} (FP/(FP+FN)); val-optimal for comparison: {val_thresholds}")

    # 7) Baselines.
    base_rate = float(y_tr.mean())
    rule = lambda d: ((d.deposit_type == "Non Refund") | (d.previous_cancellations > 0)).astype(float).to_numpy()
    baselines = {
        "accept_all (majority)": (np.full(len(test), base_rate), np.full(len(val), base_rate), 0.5),
        "rule: non_refund or prev_cancel": (rule(test), rule(val), 0.5),
    }

    # 8) Evaluation on the untouched test set.
    results = {}
    for name, (pt, _, t) in baselines.items():
        r = threshold_metrics(y_te, pt, rev_te, t)
        if name.startswith("accept_all"):
            r.update(prob_metrics(y_te, pt))
        results[name] = rounded(r)
    for m in predict:
        r = prob_metrics(y_te, p_te[m])
        r.update(threshold_metrics(y_te, p_te[m], rev_te, thresholds[m]))
        r["cost_at_0.5"] = business_cost(y_te, p_te[m], rev_te, 0.5, FN_MULT, FP_MULT)["total_cost"]
        r["val_optimal_threshold"] = val_thresholds[m]
        r["cost_at_val_optimal"] = business_cost(y_te, p_te[m], rev_te, val_thresholds[m],
                                                 FN_MULT, FP_MULT)["total_cost"]
        r["val_log_loss"] = float(log_loss(y_va, np.clip(p_va[m], 1e-15, 1 - 1e-15)))
        results[m] = rounded(r)
    print("\n[8] Test results")
    print(pd.DataFrame(results).T[["log_loss", "roc_auc", "pr_auc", "threshold", "precision", "recall",
                                   "total_cost", "cost_at_val_optimal"]].to_string())

    # 9) Do the three implementations agree?
    agreement = {}
    names = list(predict)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = names[i], names[j]
            dp = np.abs(p_te[a] - p_te[b])
            dw = np.abs(weights[a][0] - weights[b][0])
            agreement[f"{a} vs {b}"] = {
                "max_abs_prob_diff": float(dp.max()), "mean_abs_prob_diff": float(dp.mean()),
                "max_abs_coef_diff": float(dw.max()),
                "intercept_diff": abs(weights[a][1] - weights[b][1]),
                "same_decision_at_sklearn_threshold": float(
                    ((p_te[a] >= thresholds["sklearn"]) == (p_te[b] >= thresholds["sklearn"])).mean()),
                "same_decision_by_threshold": {
                    str(t): float(((p_te[a] >= t) == (p_te[b] >= t)).mean()) for t in AGREEMENT_THRESHOLDS},
            }
    objectives = {m: objective(y_tr, predict[m](X_tr), weights[m][0], lam) for m in predict}
    print("\n[9] Agreement")
    for k, v in agreement.items():
        print(f"  {k:32s} max|dp|={v['max_abs_prob_diff']:.2e} max|dw|={v['max_abs_coef_diff']:.2e} "
              f"min same-decision over thresholds={min(v['same_decision_by_threshold'].values()):.4f}")
    print(f"  train objective: {objectives}")

    # Gap to sklearn after 1k / 3k / 10k epochs: if the gap shrinks with training, the
    # differences are convergence, not a different model.
    sig = lambda X, w, b: 1.0 / (1.0 + np.exp(-(X @ w + b)))
    convergence = []
    for method, snaps in (("manual_torch", manual_snaps), ("standard_torch", std_snaps)):
        for ep, (w, b) in sorted(snaps.items()):
            dp = np.abs(sig(X_te, w, b) - p_te["sklearn"])
            convergence.append({
                "method": method, "epochs": ep,
                "max_abs_coef_diff": float(np.abs(w - weights["sklearn"][0]).max()),
                "intercept_diff": abs(b - weights["sklearn"][1]),
                "max_abs_prob_diff": float(dp.max()), "mean_abs_prob_diff": float(dp.mean()),
                "objective_gap": objective(y_tr, sig(X_tr, w, b), w, lam) - objectives["sklearn"],
            })
            print(f"  {method:15s} {ep:>6,} epochs: max|dw|={convergence[-1]['max_abs_coef_diff']:.2e} "
                  f"max|dp|={dp.max():.2e} objective gap={convergence[-1]['objective_gap']:.2e}")

    # 10) Leakage ablation: what the post-booking variables would add (sklearn, same C).
    print("\n[10] Leakage ablation (sklearn, same C)")
    ablation = []
    for label, leaky in [("at-booking features (final)", ())] + [(f"+ {f}", (f,)) for f in LEAKY_FEATURES] \
            + [("+ all post-booking", tuple(LEAKY_FEATURES))]:
        c = replace(cfg, leaky=leaky)
        pp = make_preprocessor(c).fit(train, y_tr)
        m = fit_sklearn(pp.transform(train), y_tr, best_C)
        pv, pt = m.predict_proba(pp.transform(val))[:, 1], m.predict_proba(pp.transform(test))[:, 1]
        row = {"variant": label, "val_roc_auc": roc_auc_score(y_va, pv), "test_roc_auc": roc_auc_score(y_te, pt),
               "test_pr_auc": average_precision_score(y_te, pt), "test_log_loss": log_loss(y_te, pt),
               "test_cost": business_cost(y_te, pt, rev_te, theory_t, FN_MULT, FP_MULT)["total_cost"]}
        ablation.append(rounded(row))
        print(f"  {label:30s} test AUC={row['test_roc_auc']:.4f} PR-AUC={row['test_pr_auc']:.4f} "
              f"cost={row['test_cost']:,.0f}")

    # 11) Diagnostics behind the REPORT's claims about drift, Non Refund and model vs rule.
    print("\n[11] Diagnostics")
    p_sk_va, p_sk_te = p_va["sklearn"], p_te["sklearn"]

    # a) Drift or seasonality? Compare each evaluation window with the same calendar dates
    #    one year earlier (those 2016 rows are part of train).
    def window_stats(d: pd.DataFrame, label: str) -> dict:
        nonref = d.deposit_type == "Non Refund"
        return {"window": label, "rows": len(d), "cancel_rate": float(d[TARGET].mean()),
                "city_cancel_rate": float(d.loc[d.hotel == "City Hotel", TARGET].mean()),
                "resort_cancel_rate": float(d.loc[d.hotel == "Resort Hotel", TARGET].mean()),
                "cancel_rate_excl_non_refund": float(d.loc[~nonref, TARGET].mean()),
                "mean_lead_time": float(d.lead_time.mean()), "non_refund_share": float(nonref.mean()),
                "mean_adr": float(d.adr.mean()), "mean_nights": float(d.total_nights.mean())}

    seasonality = []
    for name, d in (("val", val), ("test", test)):
        lo, hi = d.arrival_date.min(), d.arrival_date.max()
        prior = df[(df.arrival_date >= lo - pd.DateOffset(years=1)) & (df.arrival_date <= hi - pd.DateOffset(years=1))]
        seasonality.append(rounded(window_stats(prior, f"{name} dates, one year earlier "
                                                       f"({(lo - pd.DateOffset(years=1)).date()} -> "
                                                       f"{(hi - pd.DateOffset(years=1)).date()})")))
        seasonality.append(rounded(window_stats(d, f"{name} ({lo.date()} -> {hi.date()})")))
    seasonality.append(rounded(window_stats(train, "all of train")))
    print(pd.DataFrame(seasonality).set_index("window").to_string())

    # b) Is the AUC drop degradation or composition? Non Refund bookings are the easy ones.
    composition = []
    for name, d, y, p in (("val", val, y_va, p_sk_va), ("test", test, y_te, p_sk_te)):
        keep = (d.deposit_type != "Non Refund").to_numpy()
        composition.append(rounded({
            "split": name, "non_refund_share": float(1 - keep.mean()),
            "roc_auc_all": roc_auc_score(y, p), "roc_auc_excl_non_refund": roc_auc_score(y[keep], p[keep]),
            "pr_auc_excl_non_refund": average_precision_score(y[keep], p[keep]),
            "log_loss_excl_non_refund": log_loss(y[keep], p[keep]),
            "mean_pred_excl_non_refund": float(p[keep].mean()),
            "actual_rate_excl_non_refund": float(y[keep].mean()),
        }))
    print(pd.DataFrame(composition).set_index("split").to_string())

    # c) Is the model vs rule cost difference real? Block bootstrap over arrival weeks
    #    (bookings in the same week share season/events, so weeks are resampled, not rows).
    rng = np.random.default_rng(SEED)
    week = test.arrival_date.dt.to_period("W").to_numpy()
    rule_te = baselines["rule: non_refund or prev_cancel"][0]
    per_booking = {"accept_all": FN_MULT * rev_te * (y_te == 1)}
    for key, p, t in (("model", p_sk_te, theory_t), ("rule", rule_te, 0.5)):
        flag = p >= t
        per_booking[key] = FN_MULT * rev_te * (~flag & (y_te == 1)) + FP_MULT * rev_te * (flag & (y_te == 0))
    weekly = pd.DataFrame({k: v for k, v in per_booking.items()}).groupby(week).sum()
    idx = rng.integers(0, len(weekly), size=(N_BOOTSTRAP, len(weekly)))
    boot = {k: weekly[k].to_numpy()[idx].sum(axis=1) for k in weekly}
    bootstrap = {}
    for label, a, b in (("model - rule", "model", "rule"), ("model - accept_all", "model", "accept_all"),
                        ("rule - accept_all", "rule", "accept_all")):
        diff = boot[a] - boot[b]
        bootstrap[label] = rounded({
            "observed": float(weekly[a].sum() - weekly[b].sum()), "ci_low": float(np.percentile(diff, 2.5)),
            "ci_high": float(np.percentile(diff, 97.5)), "share_negative": float((diff < 0).mean()),
        }, 2)
        print(f"  {label:20s} observed={bootstrap[label]['observed']:>10,.0f}  95% CI "
              f"[{bootstrap[label]['ci_low']:,.0f}, {bootstrap[label]['ci_high']:,.0f}]  "
              f"P(<0)={bootstrap[label]['share_negative']:.3f}")
    bootstrap["weeks"] = len(weekly)
    bootstrap["resamples"] = N_BOOTSTRAP

    # d) How much does the model depend on deposit_type? Same C, without it and its interaction.
    no_dep = []
    for label, c in (("final (with deposit_type)", cfg), ("without deposit_type", replace(cfg, deposit=False))):
        pp = make_preprocessor(c).fit(train, y_tr)
        m = fit_sklearn(pp.transform(train), y_tr, best_C)
        pv, pt = m.predict_proba(pp.transform(val))[:, 1], m.predict_proba(pp.transform(test))[:, 1]
        keep = (test.deposit_type != "Non Refund").to_numpy()
        no_dep.append(rounded({
            "variant": label, "val_roc_auc": roc_auc_score(y_va, pv), "test_roc_auc": roc_auc_score(y_te, pt),
            "test_roc_auc_excl_non_refund": roc_auc_score(y_te[keep], pt[keep]),
            "test_pr_auc": average_precision_score(y_te, pt), "test_log_loss": log_loss(y_te, pt),
            "test_cost": business_cost(y_te, pt, rev_te, theory_t, FN_MULT, FP_MULT)["total_cost"],
        }))
    print(pd.DataFrame(no_dep).set_index("variant").to_string())

    # 12) Save artifacts. The app loads these and never retrains.
    joblib.dump(pre, ARTIFACTS / "preprocessor.joblib")
    joblib.dump(sk, ARTIFACTS / "sklearn_logreg.joblib")
    torch.save(manual_params, ARTIFACTS / "manual_torch.pt")
    torch.save(std_model.state_dict(), ARTIFACTS / "standard_torch.pt")

    pred_frames = []
    for split_name, d, p, rev in (("val", val, p_va, rev_va), ("test", test, p_te, rev_te)):
        f = pd.DataFrame({"split": split_name, "arrival_date": d.arrival_date.dt.date.astype(str).to_numpy(),
                          "hotel": d.hotel.to_numpy(), TARGET: d[TARGET].to_numpy(), "revenue": rev})
        for m in predict:
            f[f"p_{m}"] = p[m]
        pred_frames.append(f)
    pd.concat(pred_frames).to_csv(ARTIFACTS / "predictions.csv", index=False)

    coef_table = pd.DataFrame({m: weights[m][0] for m in predict}, index=feature_names)
    coef_table.loc["(intercept)"] = [weights[m][1] for m in predict]
    coef_table.to_csv(ARTIFACTS / "coefficients.csv", index_label="feature")

    metadata = {
        "seed": SEED,
        "prediction_moment": "at booking time",
        "excluded_columns": EXCLUDED,
        "rows": {"raw": len(raw), "after_cleaning": len(df)},
        "split": {"by": "arrival_date", "fractions": SPLIT_FRACTIONS,
                  "val_start": str(cut1.date()), "test_start": str(cut2.date()), "info": splits_info,
                  "val_test_months_missing_from_train": missing_months},
        "feature_config": cfg.as_dict(),
        "feature_variants": variant_rows,
        "c_search": c_rows,
        "lr_search": lr_rows,
        "chosen": {"C": best_C, "lambda": lam, "lr": best_lr, "epochs": EPOCHS},
        "feature_names": feature_names,
        "adr_cap": adr_cap,
        "costs": {"fn_mult": FN_MULT, "fp_mult": FP_MULT,
                  "definition": "FN = fn_mult x ADR x nights; FP = fp_mult x ADR x nights"},
        "thresholds": thresholds,
        "val_optimal_thresholds": val_thresholds,
        "theoretical_threshold": theory_t,
        "test_results": results,
        "agreement": agreement,
        "convergence": convergence,
        "train_objective": objectives,
        "seasonality": seasonality,
        "composition": composition,
        "bootstrap": bootstrap,
        "without_deposit": no_dep,
        "loss_curves": {"manual_torch": manual_curve, "standard_torch": std_curve, "every": CURVE_EVERY},
        "leakage_ablation": ablation,
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))
    print(f"\nSaved artifacts to {ARTIFACTS} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
