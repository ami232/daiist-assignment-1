"""
Assignment 1 — training pipeline (Ames Housing, log sale price regression).

Run with:  uv run python main.py train

Pipeline:
  1. load -> filter population -> engineer features -> temporal split
  2. fit preprocessing on train (2006-08), apply to val (2009) / test (2010)
  3. train ridge regression three ways: sklearn, manual PyTorch loop,
     nn.Module + torch.optim (alpha and learning rate chosen on 2009)
  4. compare the three against each other and a naive baseline; the test
     year (2010) is only used here, once, for the final numbers
  5. save everything app.py needs to artifacts/, then re-load it as a check

This is the ONLY stage that trains anything; app.py just loads artifacts/.
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from models import (
    ARTIFACTS_DIR, MODEL_NAMES,
    load_models, predict_log_price, ridge_loss, save_models, train_manual, train_module,
)
from features import (
    FEATURES, TARGET,
    build_preprocessor, engineer_features, filter_population,
    fit_lot_frontage_medians, impute_lot_frontage, load_data, temporal_split,
)

SEED = 42
# Tuning is deliberately small (assignment brief): only the ridge strength and,
# for the two PyTorch versions, the learning rate.
ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100]
# Largest stable lr for full-batch gradient descent here is ~0.117 (2 / largest
# eigenvalue of the loss curvature, computed when exploring the data), so 0.3
# is expected to diverge: it is in the grid to show that, not to be chosen.
LEARNING_RATES = [0.003, 0.01, 0.03, 0.1, 0.3]
# Full-batch gradient descent needs many steps here because several engineered
# features are strongly correlated (e.g. log_TotalSF / log_GrLivArea /
# Qual_x_logSF): the loss surface is a long narrow valley. With 5,000 epochs the
# PyTorch predictions were still up to ~4.7% away from sklearn's; with 50,000
# they match to ~0.01% (REPORT.md, three-method comparison). ~4 s per run.
EPOCHS = 50_000


def prepare_data() -> dict:
    df = load_data()
    print(f"Raw data: {df.shape[0]} sales x {df.shape[1]} columns")

    df = filter_population(df)
    print(f"After population filter (Normal sales, residential zoning): {len(df)} sales")

    df = engineer_features(df)
    train, val, test = temporal_split(df)

    # Learned on train only, then applied to all three splits.
    frontage_medians = fit_lot_frontage_medians(train)
    train, val, test = (impute_lot_frontage(d, frontage_medians) for d in (train, val, test))

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train[FEATURES])  # fit: learns medians/means/stds/categories
    X_val = preprocessor.transform(val[FEATURES])          # transform: reuses what train learned
    X_test = preprocessor.transform(test[FEATURES])

    # Target on the log scale (REPORT.md: errors matter in relative terms).
    y_train, y_val, y_test = (np.log(d[TARGET].to_numpy()) for d in (train, val, test))

    return {
        "train": train, "val": val, "test": test,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "preprocessor": preprocessor, "frontage_medians": frontage_medians,
        "feature_names": list(preprocessor.get_feature_names_out()),
    }


def mae_dollars(y_log_pred: np.ndarray, price_true: np.ndarray) -> float:
    """Business metric: mean absolute error in dollars.
    Predictions are on the log scale, so exp() brings them back to dollars."""
    return float(np.mean(np.abs(np.exp(y_log_pred) - price_true)))


def all_metrics(y_log_pred: np.ndarray, price_true: np.ndarray) -> dict:
    """Headline metric (MAE $) plus secondary ones, all in dollar terms."""
    price_pred = np.exp(y_log_pred)
    err = price_pred - price_true
    pct_err = np.abs(err) / price_true
    return {
        "MAE ($)": float(np.mean(np.abs(err))),            # decision metric (REPORT.md)
        "RMSE ($)": float(np.sqrt(np.mean(err ** 2))),     # punishes big misses more
        "MAPE (%)": float(100 * np.mean(pct_err)),          # error relative to price
        "Within 10% (%)": float(100 * np.mean(pct_err <= 0.10)),  # share of usable estimates
        "Bias ($)": float(np.mean(err)),                    # >0 over-, <0 under-valuation
    }


def tune_alpha(data: dict) -> tuple[float, dict]:
    """Pick the ridge strength alpha on the 2009 validation year (sklearn only:
    it solves the problem exactly, so the choice does not depend on how well
    gradient descent converged). The PyTorch versions then reuse this alpha."""
    print("\nRidge alpha (validation MAE, $):")
    scores = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(data["X_train"], data["y_train"])
        scores[alpha] = mae_dollars(model.predict(data["X_val"]), data["val"][TARGET].to_numpy())
        print(f"  alpha={alpha:<6} {scores[alpha]:>10,.0f}")
    best = min(scores, key=scores.get)
    print(f"  -> chosen alpha = {best}")
    return best, scores


def tune_lr(train_fn, predict_fn, data: dict, lam: float, label: str) -> tuple[float, dict, dict]:
    """Pick the learning rate on the validation year for one PyTorch version.
    Same alpha, same epochs, same starting point: only lr changes."""
    print(f"\n{label}: learning rate (validation MAE, $), {EPOCHS} full-batch epochs:")
    best_lr, best_result, best_score = None, None, np.inf
    scores = {}
    for lr in LEARNING_RATES:
        result = train_fn(data["X_train"], data["y_train"], lam=lam, lr=lr, epochs=EPOCHS)
        final_loss = result["history"][-1]
        if not np.isfinite(final_loss):
            # lr too large: each step overshoots the minimum and the loss explodes.
            print(f"  lr={lr:<6} diverged (final loss = {final_loss})")
            scores[lr] = None
            continue
        score = mae_dollars(predict_fn(result, data["X_val"]), data["val"][TARGET].to_numpy())
        scores[lr] = score
        print(f"  lr={lr:<6} {score:>10,.0f}   (final train loss {final_loss:.5f})")
        if score < best_score:
            best_lr, best_result, best_score = lr, result, score
    print(f"  -> chosen lr = {best_lr}")
    return best_lr, best_result, scores


def predict_manual(result: dict, X: np.ndarray) -> np.ndarray:
    return X @ result["w"] + result["b"]


def predict_module(result: dict, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return result["model"](torch.tensor(X, dtype=torch.float32)).numpy()


def main() -> None:
    torch.manual_seed(SEED)  # training is deterministic anyway (zero init, full batch)
    data = prepare_data()

    print("\nTemporal split:")
    for name in ("train", "val", "test"):
        years = sorted(int(y) for y in data[name]["Yr Sold"].unique())
        print(f"  {name:5s}: {data['X_' + name].shape[0]:4d} sales, years {years}")
    print(f"Model inputs after preprocessing: {data['X_train'].shape[1]} columns")

    # Sanity check: preprocessing left no NaN behind.
    for name in ("X_train", "X_val", "X_test"):
        assert not np.isnan(data[name]).any(), f"NaN left in {name}"

    # =========================================================================
    # Training (train years only; choices made on the validation year)
    # =========================================================================
    # --- Method 1: scikit-learn ---------------------------------------------
    # Ridge solves the regularized least-squares problem in closed form
    # (linear algebra, no iterations), so it is the exact reference answer.
    alpha, alpha_scores = tune_alpha(data)
    sk_model = Ridge(alpha=alpha).fit(data["X_train"], data["y_train"])

    # Same objective for PyTorch, in "mean" form (see models.py).
    lam = alpha / len(data["y_train"])

    # --- Method 2: manual PyTorch loop ---------------------------------------
    manual_lr, manual, manual_lr_scores = tune_lr(
        train_manual, predict_manual, data, lam, "Manual PyTorch loop")

    # --- Method 3: nn.Module + torch.optim -----------------------------------
    module_lr, module, module_lr_scores = tune_lr(
        train_module, predict_module, data, lam, "nn.Module + optim.SGD")

    # =========================================================================
    # Save, then evaluate FROM THE SAVED FILES (the same path app.py uses)
    # =========================================================================
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    save_models(sk_model, manual, module["model"])
    joblib.dump(data["preprocessor"], ARTIFACTS_DIR / "preprocessor.joblib")
    with open(ARTIFACTS_DIR / "frontage_medians.json", "w") as f:
        json.dump(data["frontage_medians"], f, indent=2)

    n_features = data["X_train"].shape[1]
    loaded = load_models(n_features)

    # Check that the saved files reproduce the in-memory models exactly.
    in_memory = {
        "sklearn": sk_model.predict(data["X_test"]),
        "manual_torch": predict_manual(manual, data["X_test"]),
        "module_torch": predict_module(module, data["X_test"]),
    }
    reloaded = predict_log_price(loaded, data["X_test"])
    for name in MODEL_NAMES:
        assert np.allclose(in_memory[name], reloaded[name], atol=1e-6), f"{name} reload mismatch"
    print("\nSaved models re-loaded from artifacts/ and reproduce their predictions.")

    # --- Naive baseline -------------------------------------------------------
    # "Price every house at the median sale price of the training years".
    # It uses no features: any useful model must clearly beat it.
    baseline_price = float(np.median(data["train"][TARGET]))

    # --- Metrics on validation (2009) and test (2010) -------------------------
    metrics = {}
    for split in ("val", "test"):
        price_true = data[split][TARGET].to_numpy()
        preds = predict_log_price(loaded, data[f"X_{split}"])
        preds["baseline"] = np.full(len(price_true), np.log(baseline_price))
        metrics[split] = {name: all_metrics(p, price_true) for name, p in preds.items()}

    print(f"\nNaive baseline: every house priced at the training median, ${baseline_price:,.0f}")
    for split, title in (("val", "Validation 2009"), ("test", "TEST 2010 (final, used once)")):
        table = pd.DataFrame(metrics[split]).T
        print(f"\n{title}:\n{table.round(1).to_string()}")

    # --- Do the three implementations agree? ----------------------------------
    test_preds = predict_log_price(loaded, data["X_test"])
    coefs = {
        "sklearn": np.r_[sk_model.coef_, sk_model.intercept_],
        "manual_torch": np.r_[manual["w"], manual["b"]],
        "module_torch": np.r_[module["model"].linear.weight.detach().numpy().ravel(),
                              module["model"].linear.bias.item()],
    }
    # Training objective each one reached (sklearn's is the exact minimum).
    X_tr = torch.tensor(data["X_train"], dtype=torch.float64)
    y_tr = torch.tensor(data["y_train"], dtype=torch.float64)
    objective = {}
    for name, c in coefs.items():
        w_c = torch.tensor(c[:-1], dtype=torch.float64)  # PyTorch weights are float32
        objective[name] = float(ridge_loss(X_tr @ w_c + float(c[-1]), y_tr, w_c, lam))
    agreement = {}
    for a, b in (("manual_torch", "sklearn"), ("module_torch", "sklearn"), ("manual_torch", "module_torch")):
        agreement[f"{a} vs {b}"] = {
            "max prediction gap (%)": float(100 * np.max(np.abs(np.exp(test_preds[a] - test_preds[b]) - 1))),
            "max coefficient gap": float(np.max(np.abs(coefs[a] - coefs[b]))),
        }
    print("\nAgreement between implementations (test year):")
    print(pd.DataFrame(agreement).T.to_string(float_format=lambda v: f"{v:.2e}"))
    print("Training objective reached: " + ", ".join(f"{k} {v:.7f}" for k, v in objective.items()))

    # =========================================================================
    # Everything else the dashboard needs
    # =========================================================================
    # One table with every sale, its split, the engineered features (for the
    # distribution plots) and each model's predicted price in dollars.
    frames = []
    for split in ("train", "val", "test"):
        df = data[split][["PID", "Yr Sold", TARGET] + FEATURES].copy()
        df.insert(0, "split", split)
        for name, p in predict_log_price(loaded, data[f"X_{split}"]).items():
            df[f"pred_{name}"] = np.exp(p)
        df["pred_baseline"] = baseline_price
        frames.append(df)
    pd.concat(frames).to_csv(ARTIFACTS_DIR / "predictions.csv", index=False)

    # Loss curves of the chosen runs (every 10th epoch keeps the file small).
    pd.DataFrame({
        "epoch": np.arange(1, EPOCHS + 1),
        "manual_torch": manual["history"],
        "module_torch": module["history"],
    }).iloc[::10].to_csv(ARTIFACTS_DIR / "loss_history.csv", index=False)

    # Coefficients side by side (on standardized features, so comparable).
    pd.DataFrame(coefs, index=data["feature_names"] + ["intercept"]).rename_axis("feature") \
        .to_csv(ARTIFACTS_DIR / "coefficients.csv")

    summary = {
        "n_features": n_features,
        "feature_names": data["feature_names"],
        "split_sizes": {s: int(len(data[s])) for s in ("train", "val", "test")},
        "alpha": alpha, "lambda": lam, "epochs": EPOCHS,
        "lr": {"manual_torch": manual_lr, "module_torch": module_lr},
        "tuning": {
            "alpha_val_mae": {str(k): v for k, v in alpha_scores.items()},
            "manual_lr_val_mae": {str(k): v for k, v in manual_lr_scores.items()},
            "module_lr_val_mae": {str(k): v for k, v in module_lr_scores.items()},
        },
        "baseline_price": baseline_price,
        "metrics": metrics,
        "agreement_test": agreement,
        "train_objective": objective,
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nArtifacts written to {ARTIFACTS_DIR}/: "
          + ", ".join(sorted(p.name for p in ARTIFACTS_DIR.iterdir())))


if __name__ == "__main__":
    main()
