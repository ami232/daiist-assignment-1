"""Reproduce three linear models and save every dashboard artifact."""
import hashlib
import json
import time
from itertools import combinations
import joblib
import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from features import METHODS, ROOT, calendar_features, date_splits, load_data, rental_units
from models import DTYPE, train_manual, train_standard

# Fixed modest settings. There is no hyperparameter search.
REGULARIZATION = 0.01
LEARNING_RATE = 0.05

def metrics(actual, predicted):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
        "mean_error": float(np.mean(predicted - actual)),
        "negative_predictions": int(np.sum(predicted < 0)),
    }

def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    data = load_data()
    if data.isna().any().any() or data["timestamp"].duplicated().any():
        raise ValueError("Inspect unexpected missing cells or duplicate hours.")
    if not (data["cnt"] == data["casual"] + data["registered"]).all():
        raise ValueError("Target does not agree with its component counts.")
    split = date_splits(data)
    assert data.loc[split["train"], "timestamp"].max() < data.loc[split["validation"], "timestamp"].min()
    assert data.loc[split["validation"], "timestamp"].max() < data.loc[split["test"], "timestamp"].min()
    features = calendar_features(data)
    scaler = StandardScaler().fit(features.loc[split["train"]])
    x = {name: scaler.transform(features.loc[mask]) for name, mask in split.items()}
    y = {name: data.loc[mask, "cnt"].to_numpy(dtype=float) for name, mask in split.items()}
    target_mean, target_scale = float(y["train"].mean()), float(y["train"].std())
    y_scaled = (y["train"] - target_mean) / target_scale
    bundle = {
        "scaler": scaler, "feature_names": list(features.columns), "engineered": True,
        "target_mean": target_mean, "target_scale": target_scale,
        "regularization": REGULARIZATION, "learning_rate": LEARNING_RATE,
    }
    # sklearn: SSE + alpha * sum(w²). PyTorch: MSE + lambda * sum(w²).
    # alpha = n_train * lambda matches objectives. Neither penalizes the bias.
    ridge = Ridge(alpha=len(y["train"]) * REGULARIZATION, solver="cholesky")
    started = time.perf_counter()
    ridge.fit(x["train"], y_scaled)
    training_seconds = {METHODS[0]: time.perf_counter() - started}
    tx = torch.tensor(x["train"], dtype=DTYPE)
    ty = torch.tensor(y_scaled, dtype=DTYPE)
    print(f"Training on {len(ty):,} hours and {tx.shape[1]} calendar features.", flush=True)
    started = time.perf_counter()
    manual, manual_history = train_manual(tx, ty, REGULARIZATION, LEARNING_RATE)
    training_seconds[METHODS[1]] = time.perf_counter() - started
    print(f"Manual PyTorch finished: epoch {manual_history[-1]['epoch']}.", flush=True)
    started = time.perf_counter()
    standard, standard_history = train_standard(tx, ty, REGULARIZATION, LEARNING_RATE)
    training_seconds[METHODS[2]] = time.perf_counter() - started
    print(f"Standard PyTorch finished: epoch {standard_history[-1]['epoch']}.", flush=True)
    baseline = DummyRegressor(strategy="mean").fit(x["train"], y["train"])
    predictions = {}
    with torch.no_grad():
        for name in split:
            tensor_x = torch.tensor(x[name], dtype=DTYPE)
            predictions[name] = {
                METHODS[0]: rental_units(ridge.predict(x[name]), bundle),
                METHODS[1]: rental_units((tensor_x @ manual["weights"] + manual["bias"]).numpy(), bundle),
                METHODS[2]: rental_units(standard(tensor_x).numpy(), bundle),
                "Training-mean baseline": baseline.predict(x[name]),
            }
    evaluation = {
        name: {method: metrics(y[name], values) for method, values in methods.items()}
        for name, methods in predictions.items()
    }
    # Save and report results below.
    # Test the engineered features on validation, keeping the test set untouched.
    basic = calendar_features(data, engineered=False)
    basic_scaler = StandardScaler().fit(basic.loc[split["train"]])
    basic_model = Ridge(alpha=len(y["train"]) * REGULARIZATION, solver="cholesky")
    basic_model.fit(basic_scaler.transform(basic.loc[split["train"]]), y_scaled)
    basic_validation = rental_units(basic_model.predict(basic_scaler.transform(basic.loc[split["validation"]])), bundle)
    experiment = {
        "Separate hour and season": {"features": basic.shape[1], **metrics(y["validation"], basic_validation)},
        "Season × hour": {"features": features.shape[1], **evaluation["validation"][METHODS[0]]},
    }
    agreement = {
        f"{a} / {b}": float(np.max(np.abs(predictions["test"][a] - predictions["test"][b])))
        for a, b in combinations(METHODS, 2)
    }
    if max(agreement.values()) > 0.01:
        raise RuntimeError(f"Models differ by more than 0.01 rental; inspect convergence: {agreement}")
    objective = {}
    for method in METHODS:
        scaled_prediction = (predictions["train"][method] - target_mean) / target_scale
        if method == METHODS[0]:
            weights = ridge.coef_
        elif method == METHODS[1]:
            weights = manual["weights"].numpy()
        else:
            weights = standard.linear.weight.detach().numpy().reshape(-1)
        objective[method] = float(np.mean((scaled_prediction-y_scaled)**2) + REGULARIZATION*np.sum(weights**2))
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    joblib.dump(bundle, out / "preprocessing.joblib")
    joblib.dump(ridge, out / "sklearn.joblib")
    joblib.dump(baseline, out / "baseline.joblib")
    torch.save(manual, out / "manual.pt")
    torch.save(standard.state_dict(), out / "standard.pt")
    saved_predictions = data.loc[split["test"], ["timestamp", "dteday", "hr", "season", "cnt"]].copy()
    for method, values in predictions["test"].items():
        saved_predictions[method] = values
    saved_predictions.to_csv(out / "test_predictions.csv", index=False)
    summary = {
        "evaluation": evaluation, "feature_experiment_validation": experiment,
        "test_prediction_max_difference": agreement, "training_objective": objective,
        "training_seconds": training_seconds,
        "split": {name: {"rows": int(mask.sum()), "start": str(data.loc[mask, "timestamp"].min()), "end": str(data.loc[mask, "timestamp"].max())} for name, mask in split.items()},
        "data_sha256": hashlib.sha256((ROOT / "data" / "hour.csv").read_bytes()).hexdigest(),
        "settings": {"regularization_lambda": REGULARIZATION, "sklearn_alpha": ridge.alpha, "learning_rate": LEARNING_RATE, "feature_count": features.shape[1]},
        "data_quality": {"rows": len(data), "missing_cells": 0, "duplicate_hours": 0, "unrecorded_hours": int((data["timestamp"].max()-data["timestamp"].min()).total_seconds()/3600)+1-len(data)},
    }
    import sklearn, pandas, gradio
    summary["versions"] = {"numpy": np.__version__, "pandas": pandas.__version__, "torch": torch.__version__, "scikit-learn": sklearn.__version__, "gradio": gradio.__version__}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "learning_curves.json").write_text(json.dumps({METHODS[1]: manual_history, METHODS[2]: standard_history}, indent=2) + "\n")
    table = ["| Method | Test RMSE | Test MAE | Test R² | Negative predictions |", "|---|---:|---:|---:|---:|"]
    for method, values in evaluation["test"].items():
        table.append(f"| {method} | {values['RMSE']:.3f} | {values['MAE']:.3f} | {values['R2']:.4f} | {values['negative_predictions']} |")
    (out / "results.md").write_text("# Automatically measured results\n\n" + "\n".join(table) + "\n\nThese are measurements. The student supplies the interpretation in REPORT.md.\n")
    print("\n".join(table), flush=True)
    print("Validation feature experiment:", json.dumps(experiment, indent=2), flush=True)
    print("Maximum test-prediction differences:", agreement, flush=True)

if __name__ == "__main__":
    main()
