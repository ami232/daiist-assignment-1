"""
Assignment 1 - training pipeline.  Run with:  uv run python main.py train

Business question: Chicago's health department can only make so many
routine restaurant inspections a month. Which restaurants should be
visited first, so that the ones that would fail are found sooner?

Stages, in order:
  1. Load the committed inspection data and build the features (features.py).
  2. Split by date: train on two years, validate on the next, test on the last.
  3. Score two baselines that need no model.
  4. Measure what each feature group adds.
  5. Pick the regularisation strength on the validation year.
  6. Train logistic regression three ways (models.py) and compare them.
  7. Pick the decision threshold from the cost of each mistake.
  8. Check how stable the model is from quarter to quarter.
  9. Save everything app.py needs. Nothing is trained outside this file.
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import features as F
import models as M

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# --- Business assumptions (REPORT.md, "Business framing") ----------------------
# A false positive spends a priority visit on a restaurant that would have
# passed. A false negative leaves a failing restaurant in the normal queue,
# so its violations go unnoticed for longer. Missing a fail is taken to be
# three times as costly as a wasted priority visit.
COST_FALSE_POSITIVE = 150  # dollars: roughly one inspector visit
COST_FALSE_NEGATIVE = 450  # dollars: 3 x a visit

# --- The only hyperparameters that are tuned -------------------------------------
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]      # regularisation strength (scikit-learn's C)
LEARNING_RATES = [0.01, 0.1, 1.0, 5.0]      # tried for both PyTorch versions
MANUAL_EPOCHS = 500
MODULE_EPOCHS = 60
MODULE_BATCH_SIZE = 256
MODULE_LR_DECAY = 0.9                        # learning rate x 0.9 after every epoch

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
TOP_SHARE = 0.20  # "if we could only prioritise 20% of restaurants"


# --- Metrics -----------------------------------------------------------------------
def capture_at_top(y: np.ndarray, p: np.ndarray, share: float = TOP_SHARE) -> float:
    """Share of all fails found among the `share` highest-scored restaurants.

    Restaurants with exactly the same score as the last one that fits are
    counted at their average fail rate, so the order of ties does not matter.
    A constant score therefore finds exactly `share` of the fails.
    """
    slots = int(round(share * len(y)))
    cutoff = np.sort(p)[::-1][slots - 1]
    above, tied = p > cutoff, p == cutoff
    found = y[above].sum() + (slots - above.sum()) * y[tied].mean()
    return float(found / y.sum())


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "capture_top20": capture_at_top(y, p),
    }


def decision_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Confusion matrix and cost when restaurants with p >= threshold are prioritised."""
    flagged = p >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    cost = COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "flagged_share": float(flagged.mean()),
        "cost": int(cost),
        "cost_per_1000": float(1000 * cost / len(y)),
    }


def cheapest_threshold(y: np.ndarray, p: np.ndarray) -> float:
    costs = [decision_metrics(y, p, t)["cost"] for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmin(costs))])


def finite(value: float):
    """JSON has no NaN/inf; a diverged run is stored as null."""
    return float(value) if np.isfinite(value) else None


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    results: dict = {}

    # ---------------------------------------------------------------------------
    heading("1. Data and features")
    inspections = F.load_inspections()
    # Chain sizes are counted on inspections before the validation year, so
    # the feature never sees names that only appear later.
    chain_sizes = F.count_chain_sizes(inspections, before=F.VAL_START)
    table = F.build_raw_table(inspections, chain_sizes)

    print(f"Inspections in file (all types, usable licence): {len(inspections):,}")
    print(f"Routine inspections with a result since {F.MODEL_START.date()}: {len(table):,}")

    # ---------------------------------------------------------------------------
    heading("2. Split by date")
    # The model will always be used on inspections that have not happened
    # yet, so it is tested the same way. A random split would let it train on
    # 2026 and be graded on 2023.
    split = {name: table[table["split"] == name] for name in ["train", "validation", "test"]}
    for name, rows in split.items():
        print(f"{name:<11}{rows['inspection_date'].min().date()} to {rows['inspection_date'].max().date()}"
              f"   {len(rows):>6,} inspections   fail rate {rows['failed'].mean():.1%}")

    constants = F.fit_constants(split["train"])
    features = {name: F.to_model_features(rows, constants) for name, rows in split.items()}
    # Standardise with training means and standard deviations only. Gradient
    # descent needs features on one scale, and it makes the L2 penalty treat
    # every feature evenly.
    scaler = StandardScaler().fit(features["train"])
    X = {name: scaler.transform(frame) for name, frame in features.items()}
    y = {name: rows["failed"].to_numpy() for name, rows in split.items()}
    n_train, n_features = X["train"].shape

    results["data"] = {
        "inspections_in_file": int(len(inspections)),
        "model_rows": int(len(table)),
        "splits": {
            name: {
                "start": str(rows["inspection_date"].min().date()),
                "end": str(rows["inspection_date"].max().date()),
                "rows": int(len(rows)),
                "fail_rate": float(rows["failed"].mean()),
                "no_history_share": float((rows["n_prior"] == 0).mean()),
            }
            for name, rows in split.items()
        },
    }

    # ---------------------------------------------------------------------------
    heading("3. Baselines (no model)")
    base_rate = constants["base_fail_rate"]
    baseline_scores = {
        # Naive: every restaurant gets the training fail rate.
        "Base rate": {name: np.full(len(rows), base_rate) for name, rows in split.items()},
        # What a manager could do with a spreadsheet: rank by past fail rate.
        "Own past fail rate": {name: frame["fail_rate_3y"].to_numpy() for name, frame in features.items()},
    }
    results["baselines"] = {}
    for label, scores in baseline_scores.items():
        results["baselines"][label] = {
            name: probability_metrics(y[name], scores[name]) for name in ["validation", "test"]
        }
        print(f"{label:<20} test AUC {results['baselines'][label]['test']['roc_auc']:.3f}"
              f"   log-loss {results['baselines'][label]['test']['log_loss']:.4f}")

    # ---------------------------------------------------------------------------
    heading("4. What each feature group adds (scikit-learn, validation year)")
    group_of = {name: group for name, group, _, _ in F.FEATURES}
    columns_in = lambda groups: [i for i, name in enumerate(F.FEATURE_NAMES) if group_of[name] in groups]

    def validation_score(column_indices: list[int]) -> dict:
        # scikit-learn's default C. Stage 5 shows the choice of C barely matters here.
        model = M.train_sklearn(X["train"][:, column_indices], y["train"], C=1.0)
        p = M.predict_sklearn(model, X["validation"][:, column_indices])
        return {"roc_auc": float(roc_auc_score(y["validation"], p)), "log_loss": float(log_loss(y["validation"], p))}

    ablation = {"cumulative": [], "leave_one_out": []}
    for i, group in enumerate(F.FEATURE_GROUPS):
        included = F.FEATURE_GROUPS[: i + 1]
        ablation["cumulative"].append({"step": group, "n_features": len(columns_in(included)), **validation_score(columns_in(included))})
    full = ablation["cumulative"][-1]
    for group in F.FEATURE_GROUPS:
        others = [g for g in F.FEATURE_GROUPS if g != group]
        score = validation_score(columns_in(others))
        ablation["leave_one_out"].append({"removed": group, **score, "auc_lost": full["roc_auc"] - score["roc_auc"]})
    results["ablation"] = ablation

    print("Adding groups one at a time:")
    for row in ablation["cumulative"]:
        print(f"  + {row['step']:<20} {row['n_features']:>2} features   AUC {row['roc_auc']:.3f}   log-loss {row['log_loss']:.4f}")
    print("Removing one group from the full model:")
    for row in ablation["leave_one_out"]:
        print(f"  - {row['removed']:<20} AUC {row['roc_auc']:.3f}   (loses {row['auc_lost']:+.3f})")

    # ---------------------------------------------------------------------------
    heading("5. Regularisation strength (validation log-loss)")
    c_search = []
    for C in C_GRID:
        p = M.predict_sklearn(M.train_sklearn(X["train"], y["train"], C), X["validation"])
        c_search.append({"C": C, "log_loss": float(log_loss(y["validation"], p)), "roc_auc": float(roc_auc_score(y["validation"], p))})
        print(f"  C = {C:<6}  log-loss {c_search[-1]['log_loss']:.4f}   AUC {c_search[-1]['roc_auc']:.4f}")
    # Differences in the fourth decimal are noise. Among the values that tie
    # at three decimals, take the strongest regularisation (smallest C).
    best = min(round(row["log_loss"], 3) for row in c_search)
    C = min(row["C"] for row in c_search if round(row["log_loss"], 3) == best)
    l2 = M.l2_from_C(C, n_train)
    print(f"Chosen C = {C}  ->  l2 = 1 / (C * n) = {l2:.6f} for the PyTorch versions")

    # ---------------------------------------------------------------------------
    heading("6. The same model, three ways")
    X_train_t, y_train_t = M.as_tensors(X["train"], y["train"])

    def gap_to(reference_weights, reference_bias, weights, bias) -> float:
        """Largest absolute difference in any coefficient, intercept included."""
        return float(max(np.abs(weights - reference_weights).max(), abs(bias - reference_bias)))

    # 6a. scikit-learn: the reference solution.
    sklearn_model = M.train_sklearn(X["train"], y["train"], C)
    sk_weights, sk_bias = M.coefficients("sklearn", sklearn_model)
    sklearn_objective = M.objective(
        torch.tensor(sk_weights), torch.tensor([sk_bias]), X_train_t, y_train_t, l2
    ).item()
    print(f"scikit-learn (L-BFGS) reached training objective {sklearn_objective:.8f}")

    # 6b. Manual loop. The learning rate does not change which solution is
    # best, only whether we reach it, so it is judged on the training
    # objective rather than on validation.
    print(f"\nManual loop, full-batch gradient descent, {MANUAL_EPOCHS} epochs:")
    manual_runs = {}
    manual_search = []
    for lr in LEARNING_RATES:
        weights, bias, history = M.train_manual(X["train"], y["train"], l2, lr, MANUAL_EPOCHS)
        manual_runs[lr] = (weights, bias, history)
        w, b = M.coefficients("manual", (weights, bias))
        manual_search.append({
            "lr": lr, "final_objective": finite(history[-1]),
            "objective_gap": finite(history[-1] - sklearn_objective),
            "max_coef_gap": finite(gap_to(sk_weights, sk_bias, w, b)),
            "history": [finite(value) for value in history],
        })
        print(f"  lr = {lr:<5} objective {history[-1]:.8f}   largest coefficient gap to scikit-learn {manual_search[-1]['max_coef_gap']:.2e}")
    manual_lr = min(manual_search, key=lambda row: row["final_objective"] if row["final_objective"] is not None else np.inf)["lr"]
    manual_model = manual_runs[manual_lr][:2]
    print(f"Chosen lr = {manual_lr}")

    # 6c. Standard workflow.
    print(f"\nnn.Module + torch.optim.SGD, mini-batches of {MODULE_BATCH_SIZE}, {MODULE_EPOCHS} epochs, lr x {MODULE_LR_DECAY} per epoch:")
    module_runs = {}
    module_search = []
    for lr in LEARNING_RATES:
        model, history = M.train_module(
            X["train"], y["train"], l2, lr, MODULE_EPOCHS, MODULE_BATCH_SIZE, MODULE_LR_DECAY,
        )
        module_runs[lr] = (model, history)
        w, b = M.coefficients("module", model)
        module_search.append({
            "lr": lr, "final_objective": finite(history[-1]),
            "objective_gap": finite(history[-1] - sklearn_objective),
            "max_coef_gap": finite(gap_to(sk_weights, sk_bias, w, b)),
            "history": [finite(value) for value in history],
        })
        print(f"  lr = {lr:<5} objective {history[-1]:.8f}   largest coefficient gap to scikit-learn {module_search[-1]['max_coef_gap']:.2e}")
    module_lr = min(module_search, key=lambda row: row["final_objective"] if row["final_objective"] is not None else np.inf)["lr"]
    module_model = module_runs[module_lr][0]
    print(f"Chosen lr = {module_lr}")

    # 6d. Why is the standard workflow not exactly on scikit-learn's solution?
    # The only algorithmic difference from the manual loop is mini-batching.
    # Two extra runs: one without the learning-rate decay, and one with the
    # manual loop's whole recipe (full batch, same number of epochs). The
    # last one changes batch size and epochs together, so it shows the recipe
    # works inside nn.Module rather than isolating batch size.
    print("\nInvestigating the remaining gap of the standard workflow:")
    investigation = []
    variants = [
        ("Mini-batches, constant learning rate", dict(batch_size=MODULE_BATCH_SIZE, lr_decay=1.0, epochs=MODULE_EPOCHS)),
        ("Mini-batches, decaying learning rate (the saved model)", dict(batch_size=MODULE_BATCH_SIZE, lr_decay=MODULE_LR_DECAY, epochs=MODULE_EPOCHS)),
        ("Full batch, constant learning rate (same recipe as the manual loop)", dict(batch_size=n_train, lr_decay=1.0, epochs=MANUAL_EPOCHS)),
    ]
    for label, settings in variants:
        model, history = M.train_module(X["train"], y["train"], l2, module_lr, **settings)
        w, b = M.coefficients("module", model)
        investigation.append({"variant": label, "max_coef_gap": gap_to(sk_weights, sk_bias, w, b), "objective_gap": history[-1] - sklearn_objective})
        print(f"  {label:<68} gap {investigation[-1]['max_coef_gap']:.2e}")

    trained = {"sklearn": sklearn_model, "manual": manual_model, "module": module_model}
    display_names = {
        "sklearn": "scikit-learn", "manual": "PyTorch manual loop", "module": "PyTorch nn.Module + optim",
    }
    predictions = {kind: {name: M.predict(kind, model, X[name]) for name in X} for kind, model in trained.items()}

    results["tuning"] = {
        "C_grid": c_search, "C": C, "l2": l2,
        "manual": {"epochs": MANUAL_EPOCHS, "lr": manual_lr, "search": manual_search},
        "module": {
            "epochs": MODULE_EPOCHS, "batch_size": MODULE_BATCH_SIZE, "lr_decay": MODULE_LR_DECAY,
            "lr": module_lr, "search": module_search, "investigation": investigation,
        },
        "sklearn_objective": sklearn_objective,
    }

    # ---------------------------------------------------------------------------
    heading("7. Decision threshold from the cost of each mistake")
    # For a well-calibrated model the cheapest threshold is
    # cost_FP / (cost_FP + cost_FN). The empirical one is chosen on the
    # validation year and then applied, untouched, to the test year.
    theoretical = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)
    threshold = cheapest_threshold(y["validation"], predictions["sklearn"]["validation"])
    print(f"Costs: missed fail ${COST_FALSE_NEGATIVE}, wasted priority visit ${COST_FALSE_POSITIVE}")
    print(f"Theoretical threshold {theoretical:.2f}; cheapest on validation {threshold:.2f} (used from here on)")

    results["models"] = {}
    for kind, model in trained.items():
        weights, bias = M.coefficients(kind, model)
        results["models"][kind] = {
            "name": display_names[kind],
            "coefficients": dict(zip(F.FEATURE_NAMES, map(float, weights))),
            "intercept": bias,
            "max_coef_gap_to_sklearn": gap_to(sk_weights, sk_bias, weights, bias),
            "max_probability_gap_to_sklearn": float(np.abs(predictions[kind]["test"] - predictions["sklearn"]["test"]).max()),
            "metrics": {
                name: {**probability_metrics(y[name], predictions[kind][name]),
                       **decision_metrics(y[name], predictions[kind][name], threshold)}
                for name in ["validation", "test"]
            },
        }
    for label, scores in baseline_scores.items():
        # Each baseline gets its own cheapest validation threshold, so the
        # comparison with the model is fair.
        own_threshold = cheapest_threshold(y["validation"], scores["validation"])
        for name in ["validation", "test"]:
            results["baselines"][label][name].update(decision_metrics(y[name], scores[name], own_threshold))

    results["decision"] = {
        "cost_false_negative": COST_FALSE_NEGATIVE,
        "cost_false_positive": COST_FALSE_POSITIVE,
        "theoretical_threshold": theoretical,
        "threshold": threshold,
        "top_share": TOP_SHARE,
    }

    comparison = pd.DataFrame(
        [{"model": label, **results["baselines"][label]["test"]} for label in baseline_scores]
        + [{"model": entry["name"], **entry["metrics"]["test"]} for entry in results["models"].values()]
    ).set_index("model")
    print(f"\nTest year ({results['data']['splits']['test']['start']} to {results['data']['splits']['test']['end']}):")
    print(comparison[["roc_auc", "pr_auc", "log_loss", "brier", "capture_top20", "precision", "recall", "flagged_share", "cost_per_1000"]]
          .round(4).to_string())
    print("\nAgreement with scikit-learn (largest absolute difference):")
    for kind in ["manual", "module"]:
        entry = results["models"][kind]
        print(f"  {entry['name']:<28} coefficients {entry['max_coef_gap_to_sklearn']:.2e}   test probabilities {entry['max_probability_gap_to_sklearn']:.2e}")

    # ---------------------------------------------------------------------------
    heading("8. Stability over time")
    # One AUC per calendar quarter. If the test year is weaker than the
    # validation year, this shows whether the model decays steadily (it is
    # going stale) or simply swings from quarter to quarter.
    scored = pd.concat([rows.assign(p=predictions["sklearn"][name]) for name, rows in split.items()])
    stability = []
    for quarter, rows in scored.groupby(scored["inspection_date"].dt.to_period("Q")):
        stability.append({
            "quarter": str(quarter), "split": rows["split"].iloc[0], "rows": int(len(rows)),
            "fail_rate": float(rows["failed"].mean()), "mean_predicted": float(rows["p"].mean()),
            "roc_auc": float(roc_auc_score(rows["failed"], rows["p"])),
        })
        print(f"  {quarter}  {rows['split'].iloc[0]:<11} AUC {stability[-1]['roc_auc']:.3f}"
              f"   actual fail rate {stability[-1]['fail_rate']:.1%}   predicted {stability[-1]['mean_predicted']:.1%}")
    results["stability"] = stability

    # Two follow-up checks on why the test year is weaker than validation.
    # (a) Has the model gone stale? Refit the same model on the 24 months
    #     before each test quarter and compare it with the frozen one.
    #     (Chain sizes are left as counted before the validation year.)
    print("\nFrozen model vs the same model refitted on the 24 months before each test quarter:")
    refits = []
    for start in pd.date_range(F.TEST_START, periods=4, freq="QS"):
        dates = table["inspection_date"]
        quarter = table[(dates >= start) & (dates < start + pd.DateOffset(months=3))]
        recent = table[(dates < start) & (dates >= start - pd.DateOffset(months=24))]
        recent_constants = F.fit_constants(recent)
        recent_scaler = StandardScaler().fit(F.to_model_features(recent, recent_constants))
        refitted = M.train_sklearn(
            recent_scaler.transform(F.to_model_features(recent, recent_constants)), recent["failed"].to_numpy(), C,
        )
        p_refitted = M.predict_sklearn(refitted, recent_scaler.transform(F.to_model_features(quarter, recent_constants)))
        refits.append({
            "quarter": str(start.to_period("Q")),
            "frozen_auc": float(roc_auc_score(quarter["failed"], scored.loc[quarter.index, "p"])),
            "refitted_auc": float(roc_auc_score(quarter["failed"], p_refitted)),
        })
        print(f"  {refits[-1]['quarter']}  frozen {refits[-1]['frozen_auc']:.3f}   refitted {refits[-1]['refitted_auc']:.3f}")

    # (b) Are the same ZIP codes the worst ones year after year? Correlate
    #     each ZIP's fail rate between periods, for ZIPs with at least 60
    #     routine inspections in every period.
    zip_rate = table.pivot_table(index="zip", columns="split", values="failed", aggfunc="mean")
    zip_size = table.pivot_table(index="zip", columns="split", values="failed", aggfunc="size").fillna(0)
    busy = zip_size.min(axis=1) >= 60
    correlation = zip_rate[busy].corr()
    zip_stability = {
        "zips": int(busy.sum()),
        "train_vs_validation": float(correlation.loc["train", "validation"]),
        "validation_vs_test": float(correlation.loc["validation", "test"]),
        "train_vs_test": float(correlation.loc["train", "test"]),
    }
    print(f"\nCorrelation of ZIP fail rates between periods ({zip_stability['zips']} ZIPs with 60+ inspections in each):")
    print(f"  train vs validation {zip_stability['train_vs_validation']:+.2f}   validation vs test "
          f"{zip_stability['validation_vs_test']:+.2f}   train vs test {zip_stability['train_vs_test']:+.2f}")
    results["stability_checks"] = {"refit": refits, "zip_fail_rate_correlation": zip_stability}

    # ---------------------------------------------------------------------------
    heading("9. Save artifacts for the dashboard")
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir()
    M.save_models(ARTIFACTS, sklearn_model, manual_model, module_model)
    joblib.dump({"scaler": scaler, "constants": constants, "chain_sizes": chain_sizes}, ARTIFACTS / "preprocessing.joblib")
    keep = ["inspection_id", "inspection_date", "split", "dba_name", "zip", "latitude", "longitude", "results", "failed"] + F.RAW_COLUMNS
    table[keep].to_csv(ARTIFACTS / "model_table.csv", index=False)
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=1))
    for path in sorted(ARTIFACTS.iterdir()):
        print(f"  {path.name:<28}{path.stat().st_size / 1024:>10,.1f} KB")
    print("\nDone. Start the dashboard with:  uv run python main.py app")


if __name__ == "__main__":
    main()
