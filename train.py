"""Training pipeline: run with `uv run python main.py train`.

Raw data -> features and target -> chronological split -> preprocessing fit on train ->
tune C (and the PyTorch learning rate) -> train logistic regression three ways ->
pick the cost-minimizing threshold on validation -> score everything once on test ->
save models and results to artifacts/ for the dashboard.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from volmodel import config, evaluation as ev, models as M
from volmodel.dataset import build_dataset
from volmodel.features import CANDIDATE_FEATURES, DROPPED_FEATURES, FEATURES
from volmodel.preprocess import Preprocessor

FEATURE_GROUPS = {
    "HAR realized vol (1d, 5d, 21d)": ["log_rv_1d", "log_rv_5d", "log_rv_21d"],
    "Mean reversion (21d vs 1y vol)": ["log_rv_21d_vs_1y"],
    "VIX (gap to realized, 5d change)": ["vix_gap", "vix_change_5d"],
    "Intraday range (Parkinson vs close)": ["range_vs_close_5d"],
    "Window shape (spike share, down share)": ["spike_share_21d", "down_share_21d"],
    "Price path (21d return, 1y drawdown)": ["return_21d", "drawdown_1y"],
}


def tune_C(X_tr, y_tr, X_va, y_va) -> tuple[float, list]:
    """C is a property of the model, not of the solver: tune it once with the exact solver
    on validation log loss (a threshold-free score of probability quality)."""
    rows = []
    for C in config.C_GRID:
        p = M.predict_sklearn(M.fit_sklearn(X_tr, y_tr, C), X_va)
        rows.append({"C": C, "val_log_loss": log_loss(y_va, p), "val_auc": roc_auc_score(y_va, p)})
    best = min(rows, key=lambda r: r["val_log_loss"])
    return best["C"], rows


def tune_lr(fit_fn, X_tr, y_tr, X_va, y_va, lam, reference_objective, **kw):
    """The learning rate only changes how fast gradient descent gets to the optimum, not the
    optimum itself. So it is judged on how close the training objective gets to the exact
    scikit-learn optimum after EPOCHS steps, measured in float64. Gaps below 1e-6 are within
    float32 rounding, so they count as a tie, and ties go to the smaller (safer) rate."""
    rows, curves = [], {}
    for lr in config.LR_GRID:
        model, history = fit_fn(X_tr, y_tr, lam, lr, config.EPOCHS, **kw)
        if isinstance(model, dict):
            w, b = M.manual_params(model)
            p = M.predict_torch_manual(model, X_va)
        else:
            w, b = M.module_params(model)
            p = M.predict_torch_module(model, X_va)
        diverged = not np.isfinite(history[-1])
        gap = np.inf if diverged else M.objective(w, b, X_tr, y_tr, lam) - reference_objective
        rows.append({"lr": lr, "objective_gap": float(gap), "diverged": bool(diverged),
                     "val_log_loss": None if diverged else float(log_loss(y_va, p))})
        curves[lr] = history
    finite = [r for r in rows if not r["diverged"]]
    best_gap = min(r["objective_gap"] for r in finite)
    best = min((r for r in finite if r["objective_gap"] <= best_gap + 1e-6), key=lambda r: r["lr"])
    return best["lr"], rows, curves


def feature_ablation(f, y_tr, y_va, C) -> dict:
    """Which feature groups earn their place, measured on validation with the exact solver.

    Runs on every candidate feature, including the ones later dropped: this is the evidence
    behind the final feature list, so it has to include what was rejected."""
    tr, va = f[f["split"] == "train"], f[f["split"] == "val"]

    def score(cols):
        pp = Preprocessor(cols).fit(tr)
        p = M.predict_sklearn(M.fit_sklearn(pp.transform(tr), y_tr, C), pp.transform(va))
        return {"val_auc": roc_auc_score(y_va, p), "val_log_loss": log_loss(y_va, p)}

    cumulative, cols = [], []
    for group, members in FEATURE_GROUPS.items():
        cols = cols + members
        cumulative.append({"added": group, **score(cols)})
    leave_one_out = [{"removed": g, **score([c for c in CANDIDATE_FEATURES if c not in m])} for g, m in FEATURE_GROUPS.items()]
    return {"cumulative": cumulative, "leave_one_group_out": leave_one_out,
            "final_feature_set": {"features": FEATURES, **score(FEATURES)}}


def main() -> None:
    np.random.seed(config.SEED)
    out = config.ARTIFACTS_DIR
    out.mkdir(exist_ok=True)

    # 1. Data, features, target, split ---------------------------------------------------------
    f, quality = build_dataset()
    blocks = {s: f[f["split"] == s] for s in ("train", "val", "test", "live")}
    y = {s: blocks[s]["vol_rises"].to_numpy() for s in ("train", "val", "test")}
    print(f"Data: {quality['joined_rows']} trading days, {quality['first_date']} to {quality['last_date']}")
    print("Rows by block:", quality["rows_by_split"])

    # 2. Preprocessing, fit on train only -------------------------------------------------------
    pp = Preprocessor(FEATURES).fit(blocks["train"])
    pp.save(out / "preprocessor.json")
    X = {s: pp.transform(blocks[s]) for s in blocks}
    n_train = len(y["train"])

    # 3. Tuning --------------------------------------------------------------------------------
    C, c_rows = tune_C(X["train"], y["train"], X["val"], y["val"])
    lam = M.l2_strength(C, n_train)
    sk_model = M.fit_sklearn(X["train"], y["train"], C)
    reference = M.objective(*M.sklearn_params(sk_model), X["train"], y["train"], lam)
    lr_manual, lr_manual_rows, curves_manual = tune_lr(M.fit_torch_manual, X["train"], y["train"], X["val"], y["val"], lam, reference)
    lr_module, lr_module_rows, curves_module = tune_lr(M.fit_torch_module, X["train"], y["train"], X["val"], y["val"], lam, reference, seed=config.SEED)
    print(f"Tuning: C = {C} (lam = {lam:.2e}), lr manual = {lr_manual}, lr nn.Module = {lr_module}")

    # 4. Final models, trained once on train ---------------------------------------------------
    manual, hist_manual = M.fit_torch_manual(X["train"], y["train"], lam, lr_manual, config.EPOCHS)
    module, hist_module = M.fit_torch_module(X["train"], y["train"], lam, lr_module, config.EPOCHS, config.SEED)
    M.save_models(out, sk_model, manual, module)

    params = {"sklearn": M.sklearn_params(sk_model), "torch_manual": M.manual_params(manual), "torch_module": M.module_params(module)}
    objectives = {k: M.objective(w, b, X["train"], y["train"], lam) for k, (w, b) in params.items()}
    coefficients = {k: dict(zip(FEATURES, map(float, w)), intercept=b) for k, (w, b) in params.items()}

    # 5. Probabilities, thresholds and metrics -------------------------------------------------
    probs = {s: M.predict_all(sk_model, manual, module, X[s]) for s in ("train", "val", "test", "live")}
    c_fp = ev.false_alarm_cost_bps(blocks["train"])
    k = config.DEFAULT_MISS_MULTIPLE
    thresholds = {m: ev.best_threshold(y["val"], probs["val"][m], c_fp, k) for m in M.MODEL_NAMES}

    results = {}
    for s in ("val", "test"):
        res = {}
        for m in M.MODEL_NAMES:
            p = probs[s][m]
            res[m] = ev.classification_metrics(y[s], p, p >= thresholds[m], c_fp, k)
        base_rate = y["train"].mean()
        res["baseline_majority"] = ev.classification_metrics(
            y[s], np.full(len(y[s]), base_rate), np.zeros(len(y[s]), bool), c_fp, k)
        res["baseline_mean_reversion"] = ev.classification_metrics(
            y[s], None, ev.mean_reversion_rule(blocks[s]), c_fp, k, score=ev.mean_reversion_score(blocks[s]))
        res["policy_always_cut"] = ev.business_cost(y[s], np.ones(len(y[s]), bool), c_fp, k)
        results[s] = res

    decisions_test = {m: probs["test"][m] >= thresholds[m] for m in M.MODEL_NAMES}
    agree = ev.agreement(probs["test"], decisions_test)

    # 6. Overlay backtest on test, monthly rebalancing -----------------------------------------
    sk_decision = pd.Series(probs["test"]["sklearn"] >= thresholds["sklearn"], index=blocks["test"].index)
    mr_decision = pd.Series(ev.mean_reversion_rule(blocks["test"]), index=blocks["test"].index)
    bt_model = ev.overlay_backtest(f, "test", sk_decision)
    bt_rule = ev.overlay_backtest(f, "test", mr_decision)
    bt_always = ev.overlay_backtest(f, "test", pd.Series(True, index=blocks["test"].index))

    # 7. How much the conclusion depends on the miss multiple (threshold always re-chosen on val)
    sensitivity = []
    for mult in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        t = ev.best_threshold(y["val"], probs["val"]["sklearn"], c_fp, mult)
        cost = lambda d: ev.business_cost(y["test"], d, c_fp, mult)["cost_per_decision_bps"]
        p = probs["test"]["sklearn"]
        sensitivity.append({
            "miss_multiple": mult, "threshold_from_val": t, "share_flagged_test": float((p >= t).mean()),
            "model": cost(p >= t), "always_cut": cost(np.ones(len(p), bool)),
            "never_cut": cost(np.zeros(len(p), bool)), "mean_reversion_rule": cost(ev.mean_reversion_rule(blocks["test"])),
        })

    ablation = feature_ablation(f, y["train"], y["val"], C)

    # 8. Save everything the dashboard needs ---------------------------------------------------
    keep = ["split", *FEATURES, "vol_rises", "rv_21d", "fwd_vol_21d", "rv_1y", "vix", "close",
            "daily_return", "leverage", "fund_fwd_return_21d"]
    f[keep].to_csv(out / "dataset.csv", index_label="date", float_format="%.8g")

    curves = pd.DataFrame({f"manual_lr_{lr}": h for lr, h in curves_manual.items()}
                          | {f"module_lr_{lr}": h for lr, h in curves_module.items()})
    curves.index.name = "epoch"
    curves.iloc[::10].to_csv(out / "training_curves.csv", float_format="%.8g")

    metrics = {
        "data_quality": quality,
        "features": {"used": FEATURES, "dropped": DROPPED_FEATURES},
        "config": {"horizon": config.HORIZON, "haircut": config.HAIRCUT, "target_vol": config.TARGET_VOL,
                   "max_leverage": config.MAX_LEVERAGE, "epochs": config.EPOCHS, "miss_multiple": k},
        "tuning": {"C": C, "lam": lam, "C_grid": c_rows, "lr_manual": lr_manual, "lr_module": lr_module,
                   "lr_grid_manual": lr_manual_rows, "lr_grid_module": lr_module_rows},
        "objective_on_train": objectives,
        "coefficients": coefficients,
        "costs": {"false_alarm_bps": c_fp, "miss_bps": c_fp * k, "evidence": ev.cost_evidence(blocks["train"])},
        "thresholds": {"chosen_on_val": thresholds, "theoretical_if_calibrated": 1 / (1 + k)},
        "results": results,
        "agreement_test": agree,
        "backtest_test": {name: {k2: v for k2, v in bt.items() if k2 != "equity"}
                          for name, bt in (("model", bt_model), ("mean_reversion_rule", bt_rule), ("always_cut", bt_always))},
        "cost_sensitivity_test": sensitivity,
        "feature_ablation_val": ablation,
        "live": {"dates": [d.strftime("%Y-%m-%d") for d in blocks["live"].index],
                 "prob_sklearn": probs["live"]["sklearn"].tolist()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # 9. Summary -------------------------------------------------------------------------------
    print(f"\nCosts: false alarm {c_fp:.1f} bps, miss {c_fp * k:.1f} bps ({k:g}x)")
    print("Thresholds chosen on validation:", thresholds, f"(theory if calibrated: {1 / (1 + k):.2f})")
    print("Training objective reached:", {m: round(v, 8) for m, v in objectives.items()})
    table = pd.DataFrame({m: {key: results["test"][m].get(key) for key in
                              ("auc", "log_loss", "accuracy", "precision", "recall", "cost_per_decision_bps")}
                          for m in [*M.MODEL_NAMES, "baseline_majority", "baseline_mean_reversion"]}).T
    print("\nTest block (2020 to 2026):")
    print(table.round(4).to_string())
    print("\nAgreement on test:", json.dumps(agree, indent=1))
    print(f"\nBacktest on test, vol target only: {bt_model['vol_target_only']}")
    print(f"Backtest on test, with model overlay: {bt_model['with_overlay']}")
    print(f"Backtest on test, with mean-reversion rule overlay: {bt_rule['with_overlay']}")
    print(f"Backtest on test, always cut: {bt_always['with_overlay']}")
    print("\nCost per decision on test (bps) by miss multiple:")
    print(pd.DataFrame(sensitivity).round(2).to_string(index=False))
    print(f"\nArtifacts saved to {out}")


if __name__ == "__main__":
    main()
