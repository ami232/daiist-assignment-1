"""
Assignment 1 - training pipeline (run with `uv run python main.py train`).

Start to finish, no manual steps:

1. Load the committed UCI Bike Sharing hourly data, clean it, engineer
   features and split it by time (pipeline.py).
2. Score three naive baselines.
3. Ablation: fit the same ridge regression on progressively richer feature
   sets and record validation error, to show which engineering decisions
   actually mattered (and which alternatives did not).
4. Tune the only two hyperparameters worth touching on the validation
   quarter: ridge strength (alpha) and, for the two PyTorch versions, the
   learning rate.
5. Train the final model three ways on the identical design matrix:
   scikit-learn Ridge, a manual PyTorch gradient-descent loop, and the
   standard nn.Module + torch.optim workflow.
6. Compare them (metrics, coefficients, objective value) on the held-out
   test quarter and translate the forecast into the dispatch decision cost.
7. Save every artifact the dashboard needs under artifacts/ so app.py never
   trains anything.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from business import (  # noqa: E402
    COST_MISSED_SURGE, COST_UNNEEDED_CREW, SURGE_THRESHOLD_RIDES, best_threshold, cost_curve, surge_decision,
)
from models import (  # noqa: E402
    ARTIFACTS, MODEL_FILES, MODEL_LABELS, LagOnlyBaseline, ManualTorchModel, MeanBaseline, SeasonalNaiveBaseline,
    SklearnRidgeModel, StandardTorchModel, regression_metrics, set_seed, to_counts,
)
from pipeline import (  # noqa: E402
    FEATURE_SETS, FINAL_FEATURE_SET, LOG_TARGET, SPLITS, TARGET, get_feature_names, load_model_table,
    make_preprocessor, split_frames,
)

# Hyperparameter grids. Deliberately tiny: the brief says to spend effort on
# features, not on a grid search.
ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
ALPHA_TOLERANCE = 0.001  # pick the largest alpha within 0.1 % of the best validation RMSE
MANUAL_LRS = [0.03, 0.1, 0.2, 0.3]
MANUAL_STEPS = 4_000  # a first pass with 10k steps showed lr=0.2 within 1e-5 of its final loss by step ~1000
STANDARD_LRS = [0.001, 0.003, 0.01, 0.03]
STANDARD_EPOCHS = 60
BATCH_SIZE = 256
ABLATION_ALPHA = 1.0

FIGURES = ARTIFACTS / "figures"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def safe_rmse(metrics: dict) -> float:
    v = metrics.get("rmse", float("inf"))
    return v if np.isfinite(v) else float("inf")


def metrics_table(results: dict, split: str) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = res[split]
        rows.append({"model": name, "label": MODEL_LABELS.get(name, name), "mae": m["mae"], "rmse": m["rmse"],
                     "r2": m["r2"], "rmsle": m["rmsle"], "bias": m["bias"], "mae_peak_hours": m["mae_peak_hours"]})
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, ndigits: int = 3) -> str:
    """Minimal markdown table writer (avoids a dependency on `tabulate`)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{ndigits}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def ridge_objective(X: np.ndarray, y: np.ndarray, coef: np.ndarray, intercept: float, alpha: float) -> float:
    """MSE + (alpha / n) * ||w||^2: the quantity all three implementations minimise."""
    pred = X @ coef + intercept
    return float(((pred - y) ** 2).mean() + alpha / len(y) * (coef**2).sum())


def main() -> None:
    t_start = time.time()
    set_seed()
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ data
    log("=" * 78)
    log("1. Data: load -> clean -> engineer -> time split")
    table = load_model_table()
    frames = split_frames(table)
    train, val, test = frames["train"], frames["val"], frames["test"]
    y = {s: f[LOG_TARGET].to_numpy(dtype=np.float32) for s, f in frames.items()}
    counts = {s: f[TARGET].to_numpy(dtype=float) for s, f in frames.items()}
    peak_cutoff = float(np.percentile(train[TARGET], 90))
    for s, (a, b) in SPLITS.items():
        log(f"   {s:5s} {a} .. {b}: {len(frames[s]):6d} rows, mean demand {frames[s][TARGET].mean():6.1f} rides/h")
    log(f"   peak-hour cutoff (90th pct of training demand): {peak_cutoff:.0f} rides/h")

    results: dict[str, dict] = {}
    test_preds: dict[str, np.ndarray] = {}
    val_preds: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- baselines
    log("\n2. Naive baselines")
    for b in (MeanBaseline(), SeasonalNaiveBaseline(), LagOnlyBaseline()):
        b.fit(train, LOG_TARGET)
        results[b.name] = {s: regression_metrics(y[s], b.predict(frames[s]), peak_cutoff) for s in ("val", "test")}
        val_preds[b.name] = b.predict(val)
        test_preds[b.name] = b.predict(test)
        log(f"   {b.name:28s} val RMSE {results[b.name]['val']['rmse']:7.1f} | test RMSE {results[b.name]['test']['rmse']:7.1f}")

    # -------------------------------------------------------------- ablation
    log("\n3. Feature-set ablation (sklearn Ridge, alpha=1, scored on validation)")
    ablation_rows = []

    # Row 0: the hasty model - raw numeric columns AND the raw count target.
    pre0 = make_preprocessor("A_raw_numeric").fit(train)
    m0 = SklearnRidgeModel(ABLATION_ALPHA).fit(pre0.transform(train), counts["train"])
    pred0_counts = np.clip(m0.predict(pre0.transform(val)), 0, None)
    met0 = regression_metrics(y["val"], np.log1p(pred0_counts), peak_cutoff)
    ablation_rows.append({"feature_set": "A_raw_numeric", "target": "cnt (raw)",
                          "n_columns": pre0.transform(train).shape[1], **met0})
    log(f"   {'A_raw_numeric + raw target':34s} cols {ablation_rows[-1]['n_columns']:3d} | val RMSE {met0['rmse']:7.1f} | R2 {met0['r2']:6.3f}")

    for name in FEATURE_SETS:
        pre = make_preprocessor(name).fit(train)
        X_tr, X_va = pre.transform(train), pre.transform(val)
        m = SklearnRidgeModel(ABLATION_ALPHA).fit(X_tr, y["train"])
        met = regression_metrics(y["val"], m.predict(X_va), peak_cutoff)
        ablation_rows.append({"feature_set": name, "target": "log1p(cnt)", "n_columns": X_tr.shape[1], **met})
        log(f"   {name:34s} cols {X_tr.shape[1]:3d} | val RMSE {met['rmse']:7.1f} | R2 {met['r2']:6.3f} | RMSLE {met['rmsle']:.3f}")
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(ARTIFACTS / "ablation.csv", index=False)

    # ---------------------------------------------- final preprocessing + X
    log(f"\n4. Final feature set: {FINAL_FEATURE_SET}")
    preprocessor = make_preprocessor(FINAL_FEATURE_SET).fit(train)
    feature_names = get_feature_names(preprocessor)
    X = {s: preprocessor.transform(frames[s]).astype(np.float32) for s in frames}
    n_features = X["train"].shape[1]
    log(f"   design matrix: {X['train'].shape[0]} x {n_features} (train), fit on train only")

    # ---------------------------------------------------------- tune alpha
    tuning_rows = []
    sk_candidates = {}
    for a in ALPHAS:
        m = SklearnRidgeModel(a).fit(X["train"], y["train"])
        met = regression_metrics(y["val"], m.predict(X["val"]), peak_cutoff)
        tuning_rows.append({"model": "sklearn", "param": "alpha", "value": a, **met})
        sk_candidates[a] = m
        log(f"   alpha={a:<6g} val RMSE {met['rmse']:7.2f} | RMSLE {met['rmsle']:.4f}")
    # The validation curve is flat for small alpha, and at the smallest values
    # sklearn warns that the design matrix is ill-conditioned (temp_c, its
    # square and temp x hum are nearly collinear). Rule: take the LARGEST
    # alpha whose validation RMSE is within 0.1 % of the best. Same accuracy,
    # a better-posed problem, and gradient descent converges to the same
    # point faster.
    alpha_rmse = {r["value"]: safe_rmse(r) for r in tuning_rows if r["param"] == "alpha"}
    best_rmse = min(alpha_rmse.values())
    best_alpha = max(a for a, v in alpha_rmse.items() if v <= best_rmse * (1 + ALPHA_TOLERANCE))
    sk_model = sk_candidates[best_alpha]
    log(f"   -> alpha = {best_alpha} (largest alpha within {ALPHA_TOLERANCE:.1%} of best val RMSE {best_rmse:.2f})")

    # ------------------------------------------------ manual PyTorch loop
    log(f"\n5a. Manual PyTorch loop (full-batch GD, {MANUAL_STEPS} steps, alpha={best_alpha})")
    manual_candidates = {}
    for lr in MANUAL_LRS:
        t0 = time.time()
        m = ManualTorchModel(n_features, best_alpha, lr, MANUAL_STEPS).fit(X["train"], y["train"], X["val"], y["val"])
        with np.errstate(over="ignore", invalid="ignore"):
            met = regression_metrics(y["val"], m.predict(X["val"]), peak_cutoff)
        tuning_rows.append({"model": "torch_manual", "param": "lr", "value": lr, **met})
        manual_candidates[lr] = m
        status = "diverged" if not np.isfinite(met["rmse"]) else f"val RMSE {met['rmse']:7.2f}"
        log(f"   lr={lr:<5g} {status} | final train MSE {m.history[-1]['train_mse']:.5f} ({time.time() - t0:.1f}s)")
    best_lr_manual = min(MANUAL_LRS, key=lambda lr: safe_rmse(next(r for r in tuning_rows if r["model"] == "torch_manual" and r["value"] == lr)))
    manual_model = manual_candidates[best_lr_manual]
    log(f"   -> lr = {best_lr_manual}")

    # ------------------------------------- standard nn.Module + optim loop
    log(f"\n5b. Standard PyTorch workflow (Adam, {STANDARD_EPOCHS} epochs, batch {BATCH_SIZE}, weight_decay=2*alpha/n)")
    standard_candidates = {}
    for lr in STANDARD_LRS:
        t0 = time.time()
        m = StandardTorchModel(n_features, best_alpha, lr, STANDARD_EPOCHS, BATCH_SIZE).fit(X["train"], y["train"], X["val"], y["val"])
        with np.errstate(over="ignore", invalid="ignore"):
            met = regression_metrics(y["val"], m.predict(X["val"]), peak_cutoff)
        tuning_rows.append({"model": "torch_standard", "param": "lr", "value": lr, **met})
        standard_candidates[lr] = m
        status = "diverged" if not np.isfinite(met["rmse"]) else f"val RMSE {met['rmse']:7.2f}"
        log(f"   lr={lr:<6g} {status} | final train MSE {m.history[-1]['train_mse']:.5f} ({time.time() - t0:.1f}s)")
    best_lr_standard = min(STANDARD_LRS, key=lambda lr: safe_rmse(next(r for r in tuning_rows if r["model"] == "torch_standard" and r["value"] == lr)))
    standard_model = standard_candidates[best_lr_standard]
    log(f"   -> lr = {best_lr_standard}")
    pd.DataFrame(tuning_rows).to_csv(ARTIFACTS / "tuning.csv", index=False)

    # ------------------------------------------------------- comparison
    log("\n6. Three implementations, same split")
    models = {"sklearn": sk_model, "torch_manual": manual_model, "torch_standard": standard_model}
    for name, m in models.items():
        results[name] = {s: regression_metrics(y[s], m.predict(X[s]), peak_cutoff) for s in ("val", "test")}
        val_preds[name] = m.predict(X["val"])
        test_preds[name] = m.predict(X["test"])

    objective = {name: ridge_objective(X["train"], y["train"], m.coef_, m.intercept_, best_alpha) for name, m in models.items()}
    agreement = {}
    for a, b in (("sklearn", "torch_manual"), ("sklearn", "torch_standard"), ("torch_manual", "torch_standard")):
        ca, cb = models[a].coef_, models[b].coef_
        agreement[f"{a}_vs_{b}"] = {
            "max_abs_coef_diff": float(np.abs(ca - cb).max()),
            "mean_abs_coef_diff": float(np.abs(ca - cb).mean()),
            "coef_correlation": float(np.corrcoef(ca, cb)[0, 1]),
            "intercept_diff": float(models[a].intercept_ - models[b].intercept_),
            "rmse_between_predictions_log": float(np.sqrt(((test_preds[a] - test_preds[b]) ** 2).mean())),
            "rmse_between_predictions_counts": float(np.sqrt(((to_counts(test_preds[a]) - to_counts(test_preds[b])) ** 2).mean())),
            "max_abs_pred_diff_counts": float(np.abs(to_counts(test_preds[a]) - to_counts(test_preds[b])).max()),
        }
    log("   test-set metrics (rides/hour):")
    for name, res in results.items():
        m = res["test"]
        log(f"   {MODEL_LABELS.get(name, name):28s} MAE {m['mae']:6.1f} | RMSE {m['rmse']:6.1f} | R2 {m['r2']:6.3f} | RMSLE {m['rmsle']:.3f} | bias {m['bias']:+6.1f} | peak MAE {m['mae_peak_hours']:6.1f}")
    log("   training objective MSE + alpha/n*||w||^2 reached by each implementation:")
    for name, v in objective.items():
        log(f"   {name:16s} {v:.6f}")
    for pair, a in agreement.items():
        log(f"   {pair:32s} max|dw| {a['max_abs_coef_diff']:.4f} | pred RMSE between models {a['rmse_between_predictions_counts']:.2f} rides")

    # ----------------------------------------------------- decision cost
    log(f"\n7. Dispatch decision: surge >= {SURGE_THRESHOLD_RIDES} rides/h, missed surge {COST_MISSED_SURGE:.0f} USD, unneeded crew {COST_UNNEEDED_CREW:.0f} USD")
    decision = {}
    for name in list(models) + ["baseline_hour_profile", "baseline_last_week_level"]:
        pv, pt = to_counts(val_preds[name]), to_counts(test_preds[name])
        bt = best_threshold(counts["val"], pv)
        decision[name] = {
            "val_best_forecast_threshold": bt["forecast_threshold"],
            "val_cost_at_best": bt["total_cost"],
            "test_at_best_threshold": surge_decision(counts["test"], pt, bt["forecast_threshold"]),
            "test_at_naive_threshold": surge_decision(counts["test"], pt, SURGE_THRESHOLD_RIDES),
        }
        d = decision[name]
        log(f"   {MODEL_LABELS.get(name, name):28s} best threshold (val) {bt['forecast_threshold']:4.0f} -> test cost {d['test_at_best_threshold']['total_cost']:8.0f} USD "
            f"(naive threshold {SURGE_THRESHOLD_RIDES}: {d['test_at_naive_threshold']['total_cost']:8.0f} USD)")
    perfect = surge_decision(counts["test"], counts["test"], SURGE_THRESHOLD_RIDES)
    always = surge_decision(counts["test"], np.full_like(counts["test"], 1e9), SURGE_THRESHOLD_RIDES)
    never = surge_decision(counts["test"], np.zeros_like(counts["test"]), SURGE_THRESHOLD_RIDES)
    decision["reference"] = {"perfect_foresight": perfect, "always_dispatch": always, "never_dispatch": never}
    log(f"   reference: always dispatch {always['total_cost']:.0f} USD | never dispatch {never['total_cost']:.0f} USD | perfect foresight {perfect['total_cost']:.0f} USD")

    # ------------------------------------------------------------ artifacts
    log("\n8. Saving artifacts")
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
    for name, m in models.items():
        m.save(ARTIFACTS / MODEL_FILES[name])

    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_set": FINAL_FEATURE_SET,
        "feature_names": feature_names,
        "n_rows": {s: int(len(f)) for s, f in frames.items()},
        "split_dates": SPLITS,
        "peak_cutoff": peak_cutoff,
        "hyperparameters": {
            "alpha": best_alpha,
            "torch_manual": {"lr": best_lr_manual, "n_steps": MANUAL_STEPS},
            "torch_standard": {"lr": best_lr_standard, "n_epochs": STANDARD_EPOCHS, "batch_size": BATCH_SIZE,
                               "weight_decay": 2.0 * best_alpha / len(y["train"])},
        },
        "results": results,
        "objective_train": objective,
        "agreement": agreement,
        "coefficients": {name: m.coef_.tolist() for name, m in models.items()},
        "intercepts": {name: m.intercept_ for name, m in models.items()},
        "decision": decision,
        "business": {"surge_threshold": SURGE_THRESHOLD_RIDES, "cost_missed_surge": COST_MISSED_SURGE,
                     "cost_unneeded_crew": COST_UNNEEDED_CREW},
    }
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ARTIFACTS / "loss_curves.json", "w") as f:
        json.dump({"torch_manual": manual_model.history, "torch_standard": standard_model.history,
                   "sklearn_train_mse": float(((sk_model.predict(X["train"]) - y["train"]) ** 2).mean()),
                   "sklearn_val_mse": float(((sk_model.predict(X["val"]) - y["val"]) ** 2).mean())}, f)

    keep_cols = ["dteday", "hr", "split", TARGET, LOG_TARGET, "season", "yr", "mnth", "holiday", "weekday",
                 "workingday", "weathersit", "temp_c", "hum", "windspeed", "windspeed_zero", "lag_level",
                 "casual", "registered"]
    table[keep_cols].to_csv(ARTIFACTS / "model_table.csv", index=False)

    pred_df = test[["dteday", "hr", "workingday", "weathersit", "temp_c", TARGET]].copy()
    for name, p in test_preds.items():
        pred_df[f"pred_{name}"] = to_counts(p)
    pred_df.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    (ARTIFACTS / "results_test.md").write_text(df_to_markdown(metrics_table(results, "test")))
    (ARTIFACTS / "results_val.md").write_text(df_to_markdown(metrics_table(results, "val")))
    (ARTIFACTS / "ablation.md").write_text(df_to_markdown(ablation))

    # --------------------------------------------------------------- figures
    make_figures(train, test, ablation, models, test_preds, counts["test"], manual_model, standard_model, feature_names, best_alpha,
                 sk_train_mse=float(((sk_model.predict(X["train"]) - y["train"]) ** 2).mean()))
    log(f"\nDone in {time.time() - t_start:.1f}s. Artifacts in {ARTIFACTS}")


def make_figures(train, test, ablation, models, test_preds, y_test_counts, manual_model, standard_model, feature_names, alpha,
                 sk_train_mse):
    # Hourly profile by day type (why hr x workingday exists)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for wd, label in ((1, "working day"), (0, "weekend / holiday")):
        prof = train[train["workingday"] == wd].groupby("hr")[TARGET].mean()
        ax.plot(prof.index, prof.values, marker="o", ms=3, label=label)
    ax.set_xlabel("hour of day"); ax.set_ylabel("mean rides / hour (train)"); ax.legend(); ax.set_title("Hourly demand profile by day type")
    fig.tight_layout(); fig.savefig(FIGURES / "hour_profile.png", dpi=120); plt.close(fig)

    # Target distribution, raw vs log
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(train[TARGET], bins=60, color="tab:blue"); axes[0].set_title("cnt (rides / hour)")
    axes[1].hist(train[LOG_TARGET], bins=60, color="tab:orange"); axes[1].set_title("log1p(cnt)")
    fig.tight_layout(); fig.savefig(FIGURES / "target_distribution.png", dpi=120); plt.close(fig)

    # Ablation
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = [f"{r.feature_set}\n[{r.target}]" for r in ablation.itertuples()]
    ax.barh(labels, ablation["rmse"], color="tab:green"); ax.invert_yaxis()
    ax.set_xlabel("validation RMSE (rides / hour)"); ax.set_title("Feature-set ablation (same ridge model)")
    fig.tight_layout(); fig.savefig(FIGURES / "ablation.png", dpi=120); plt.close(fig)

    # Predicted vs actual, three panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    lim = max(y_test_counts.max(), max(to_counts(p).max() for p in test_preds.values())) * 1.02
    for ax, name in zip(axes, models):
        ax.scatter(y_test_counts, to_counts(test_preds[name]), s=5, alpha=0.4)
        ax.plot([0, lim], [0, lim], "k--", lw=1); ax.set_title(MODEL_LABELS[name]); ax.set_xlabel("actual rides / hour")
    axes[0].set_ylabel("predicted rides / hour")
    fig.tight_layout(); fig.savefig(FIGURES / "pred_vs_actual.png", dpi=120); plt.close(fig)

    # Coefficient agreement
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    sk = models["sklearn"].coef_
    for ax, name in zip(axes, ("torch_manual", "torch_standard")):
        c = models[name].coef_
        ax.scatter(sk, c, s=12); lim2 = max(np.abs(sk).max(), np.abs(c).max()) * 1.05
        ax.plot([-lim2, lim2], [-lim2, lim2], "k--", lw=1)
        ax.set_xlabel("sklearn coefficient"); ax.set_ylabel(f"{MODEL_LABELS[name]} coefficient")
        ax.set_title(f"max |diff| = {np.abs(sk - c).max():.4f}")
    fig.tight_layout(); fig.savefig(FIGURES / "coefficient_agreement.png", dpi=120); plt.close(fig)

    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    h = pd.DataFrame(manual_model.history)
    axes[0].plot(h["step"], h["train_mse"], label="train"); axes[0].plot(h["step"], h["val_mse"], label="val")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("GD step"); axes[0].set_ylabel("MSE on log1p(cnt)"); axes[0].set_title("Manual loop"); axes[0].legend()
    h = pd.DataFrame(standard_model.history)
    axes[1].plot(h["epoch"], h["train_mse"], label="train"); axes[1].plot(h["epoch"], h["val_mse"], label="val")
    axes[1].set_yscale("log"); axes[1].set_xlabel("epoch"); axes[1].set_title("nn.Module + Adam"); axes[1].legend()
    for ax in axes:
        ax.axhline(sk_train_mse, color="tab:blue", ls="--", lw=1, label="sklearn closed form")
        ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "loss_curves.png", dpi=120); plt.close(fig)

    # Two test weeks: a normal one and the Hurricane Sandy one
    ts = test.assign(t=test["dteday"] + pd.to_timedelta(test["hr"], unit="h"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
    for ax, (start, end, title) in zip(axes, (("2012-11-05", "2012-11-11", "Normal week (5-11 Nov 2012)"),
                                              ("2012-10-27", "2012-11-02", "Hurricane Sandy week (27 Oct - 2 Nov 2012)"))):
        mask = (ts["dteday"] >= start) & (ts["dteday"] <= end)
        ax.plot(ts.loc[mask, "t"], y_test_counts[mask.to_numpy()], "k-", lw=1.5, label="actual")
        for name in models:
            ax.plot(ts.loc[mask, "t"], to_counts(test_preds[name])[mask.to_numpy()], lw=1, alpha=0.8, label=MODEL_LABELS[name])
        ax.set_title(title); ax.set_ylabel("rides / hour")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "test_weeks.png", dpi=120); plt.close(fig)

    # Decision cost curve on test
    fig, ax = plt.subplots(figsize=(8, 3.5))
    thresholds = np.arange(100, 801, 10)
    for name in models:
        curve = cost_curve(y_test_counts, to_counts(test_preds[name]), thresholds)
        ax.plot(thresholds, [c["total_cost"] for c in curve], label=MODEL_LABELS[name])
    ax.axvline(SURGE_THRESHOLD_RIDES, color="grey", ls=":", label=f"surge threshold ({SURGE_THRESHOLD_RIDES})")
    ax.set_xlabel("dispatch if forecast >= threshold (rides / hour)"); ax.set_ylabel("total cost on test quarter (USD)")
    ax.set_title("Decision cost vs forecast threshold"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "cost_curve.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
