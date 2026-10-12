"""Train every model once and save the artefacts the dashboard loads.

This is the ONLY script that fits anything. It:
  1. loads the data and makes THE train/test split (src.data)
  2. builds features via the owner's build_features (src.features)
  3. fits two baselines, LogisticRegression, the manual PyTorch loop, and an nn.Linear model
  4. picks the decision threshold that minimises expected business cost on the validation slice
  5. writes everything under models/ that the dashboard and REPORT.md rely on

Run from the repo root:  uv run python main.py train

Hyperparameters (C / lr / L2) are chosen on a validation split carved out of TRAIN by
ROC-AUC; the threshold is chosen on the same slice by expected cost; the test set is scored
exactly once, at the end. The single train/test split in src.data stays the authority for
the final comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import data
from src.data import FN_COST, FP_MARGIN_MULTIPLIER, fp_cost_per_sku
from src.features import build_features
from src.train_manual import fit_manual, predict_manual

# Reproducibility: the data split already uses random_state=42; pin torch too so the
# manual loop and the nn.Linear model give the same numbers on every run.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# Hyperparameter grids. Tuning is limited to these and nothing else (see REPORT.md).
# C stops at 1: with {0.01, 0.1, 1, 10}, C=10 won validation by 0.0006 AUC over C=1 (noise),
# and at that near-zero L2 the manual gradient-descent loop stopped 1.85 short of sklearn on
# its worst coefficient (2000 epochs; still 0.91 at 50k) along the correlated/heavy-tailed
# empaque, log1p_empaque and size_cm columns (since dropped / log-transformed), with test AUC
# unchanged. Written up in REPORT.md.
LOGREG_CS = [0.01, 0.1, 1.0]
MANUAL_LRS = [0.1, 0.01, 0.001]
MANUAL_EPOCHS = 2000
# nn.Linear + Adam: lr is fixed (Adam is forgiving); weight_decay is matched to the
# chosen C below so all three models land on ~the same coefficients.
TORCH_LR = 0.05
TORCH_EPOCHS = 3000

# Thresholds scanned when picking the cost-minimising one (0.01 .. 0.99).
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


# ----------------------------------------------------------------------------- metrics
def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, fp_costs: np.ndarray) -> float:
    """Business cost of a set of decisions, in DOP. Each false positive (a live item flagged)
    costs that SKU's own fp_costs entry (a multiple of its margin, see src.data); each false
    negative (a dead item kept) costs the flat FN_COST."""
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return float(fp_costs[fp_mask].sum() + FN_COST * fn)


def classification_metrics(
    y_true: np.ndarray, prob: np.ndarray, threshold: float, fp_costs: np.ndarray
) -> dict:
    """Everything the report and dashboard need at ONE threshold.

    - accuracy / precision / recall / f1: on the hard decision prob >= threshold.
    - roc_auc: threshold-free ranking quality (does the model put dead items above live ones?).
    - expected_cost: the business number, in DOP, for this threshold.
    Computed by hand from the confusion counts so every number can be explained from them.
    """
    y_pred = (prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "expected_cost": expected_cost(y_true, y_pred, fp_costs),
    }


def test_block(y_true: np.ndarray, prob: np.ndarray, threshold: float, fp_costs: np.ndarray) -> dict:
    """Test metrics at the default 0.5 and at the chosen (cost-minimising) threshold."""
    return {
        "at_0.5": classification_metrics(y_true, prob, 0.5, fp_costs),
        "at_threshold": classification_metrics(y_true, prob, threshold, fp_costs),
    }


# ----------------------------------------------------------------------------- torch model
def train_torch_module(
    X: np.ndarray, y: np.ndarray, weight_decay: float, lr: float, epochs: int
) -> tuple[nn.Linear, list[float]]:
    """A single nn.Linear trained with Adam + BCEWithLogitsLoss.

    nn.Linear gives the logit z = Xw + b; BCEWithLogitsLoss applies the sigmoid and the
    cross-entropy together in a numerically stable way (same reason as in src.train_manual).
    weight_decay is L2 on the WEIGHT only (not the bias), to match sklearn's unpenalised
    intercept and the manual loop.
    """
    torch.manual_seed(SEED)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    model = nn.Linear(X.shape[1], 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": [model.weight], "weight_decay": weight_decay},
            {"params": [model.bias], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    loss_curve: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.item()))
    return model, loss_curve


def torch_predict_proba(model: nn.Linear, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).squeeze(1).numpy()


# ----------------------------------------------------------------------------- main
def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Data + THE split (identical for every model, defined once in src.data).
    df = data.load_raw()
    train_df, test_df = data.split(df)
    y_train = data.make_target(train_df).to_numpy(dtype=float)
    y_test = data.make_target(test_df).to_numpy(dtype=float)
    # Per-SKU cost of a false positive. Read from the price columns, which are NOT features:
    # they only price the mistakes (tests/test_leakage.py checks build_features ignores them).
    fp_train = fp_cost_per_sku(train_df).to_numpy(dtype=float)
    fp_test = fp_cost_per_sku(test_df).to_numpy(dtype=float)

    # 2. Features. fit is learned on TRAIN and reused (learns nothing) on TEST.
    X_train_df, fit = build_features(train_df, fit=None)
    X_test_df, _ = build_features(test_df, fit=fit)
    feature_names = list(fit["columns"])
    # build_features promises stable column order; make the arrays follow it explicitly.
    X_train = X_train_df[feature_names].to_numpy(dtype=float)
    X_test = X_test_df[feature_names].to_numpy(dtype=float)
    n_train = X_train.shape[0]
    print(f"features: {X_train.shape[1]}  |  train {n_train}  test {X_test.shape[0]}  "
          f"|  dead rate train {y_train.mean():.3f}")

    # Validation split carved out of TRAIN, used only to pick hyperparameters + threshold.
    # Stratified like the main split, so the slice has the same dead rate as train.
    # The per-SKU FP costs travel with the rows so the threshold scan can price mistakes.
    Xt, Xv, yt, yv, _, fpv = train_test_split(
        X_train, y_train, fp_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    metrics: dict = {
        "n_train": n_train,
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "dead_rate_train": float(y_train.mean()),
        "costs": {
            "fp_rule": f"{FP_MARGIN_MULTIPLIER:g} x max(0, Precio detalle - Costo) per SKU",
            "fp_margin_multiplier": FP_MARGIN_MULTIPLIER,
            "fn_cost": FN_COST,
            "fp_cost_test_mean": float(fp_test.mean()),
            "fp_cost_test_median": float(np.median(fp_test)),
        },
    }

    # 3a. LogisticRegression: choose C on the validation slice by AUC, refit on full train.
    logreg_candidates = {}
    logreg_val_models = {}
    for C in LOGREG_CS:
        val_model = LogisticRegression(C=C, max_iter=5000).fit(Xt, yt)
        logreg_val_models[C] = val_model
        logreg_candidates[str(C)] = float(roc_auc_score(yv, val_model.predict_proba(Xv)[:, 1]))
    best_C = max(LOGREG_CS, key=lambda c: logreg_candidates[str(c)])
    logreg = LogisticRegression(C=best_C, max_iter=5000).fit(X_train, y_train)
    logreg_prob = logreg.predict_proba(X_test)[:, 1]

    # 3b. Threshold: scan 0.01..0.99 on the validation slice with the chosen-C validation
    # model's probabilities, keep the one with the lowest expected cost. Test is untouched.
    val_prob = logreg_val_models[best_C].predict_proba(Xv)[:, 1]
    val_costs = {float(t): expected_cost(yv, (val_prob >= t).astype(int), fpv) for t in THRESHOLD_GRID}
    threshold = min(val_costs, key=val_costs.get)
    metrics["threshold"] = {
        "chosen": threshold,
        "picked_on": f"validation slice, LogisticRegression(C={best_C}) probabilities",
        "val_cost_at_chosen": val_costs[threshold],
        "val_cost_at_0.5": val_costs[0.5],
        "val_cost_curve": {str(t): c for t, c in val_costs.items()},
    }
    metrics["logreg"] = {
        "chosen_C": best_C,
        "val_candidates_auc": logreg_candidates,
        "test": test_block(y_test, logreg_prob, threshold, fp_test),
    }

    # 3c. Baseline A: majority class. Train is 60% dead, so "everything is dead": probability 1
    # for every row, so the decision is the same at any threshold. AUC is 0.5 by construction.
    majority_prob = np.ones_like(y_test)
    metrics["baseline_majority"] = {
        "predicts": int(round(y_train.mean())),
        "test": test_block(y_test, majority_prob, threshold, fp_test),
    }

    # 3d. Baseline B: dead if Costo > train median. A one-feature rule of thumb (expensive
    # items turn over slowly). Hard 0/1 "probabilities", so again threshold-independent.
    median_costo = float(train_df["Costo"].median())
    cost_prob = (test_df["Costo"].to_numpy(dtype=float) > median_costo).astype(float)
    metrics["baseline_cost_median"] = {
        "median_costo_train": median_costo,
        "test": test_block(y_test, cost_prob, threshold, fp_test),
    }

    # Match the manual loop and Adam to sklearn's regularisation. sklearn minimises
    #   C * sum_i BCE_i + 1/2 * ||w||^2
    # while the manual loss uses the MEAN cross-entropy plus an explicit penalty:
    #   mean_i BCE_i + l2 * ||w||^2.
    # Dividing sklearn's objective by C*n gives  mean_i BCE_i + 1/(2*C*n) * ||w||^2,
    # so the equivalent penalty is  l2 = 1 / (2 * C * n).
    # Adam's weight_decay adds  wd * w  to the gradient, and the gradient of l2*||w||^2 is
    # 2*l2*w, so to apply the SAME penalty the nn.Linear model needs  wd = 2 * l2 = 1/(C*n).
    l2 = 1.0 / (2.0 * best_C * n_train)
    l2_val = 1.0 / (2.0 * best_C * Xt.shape[0])  # same idea, sized to the smaller val-search fit
    weight_decay = 2.0 * l2

    # 3e. Manual PyTorch loop: choose lr on validation by AUC, refit on full train.
    manual_candidates = {}
    for lr in MANUAL_LRS:
        params_val = fit_manual(Xt, yt, lr=lr, l2=l2_val, epochs=MANUAL_EPOCHS)
        manual_candidates[str(lr)] = float(roc_auc_score(yv, predict_manual(params_val, Xv)))
    best_lr = max(MANUAL_LRS, key=lambda r: manual_candidates[str(r)])
    manual_params = fit_manual(X_train, y_train, lr=best_lr, l2=l2, epochs=MANUAL_EPOCHS)
    manual_prob = predict_manual(manual_params, X_test)
    metrics["manual"] = {
        "chosen_lr": best_lr,
        "l2": l2,
        "epochs": MANUAL_EPOCHS,
        "val_candidates_auc": manual_candidates,
        "final_train_loss": manual_params["loss_curve"][-1],
        "test": test_block(y_test, manual_prob, threshold, fp_test),
    }

    # 3f. nn.Linear + Adam + BCEWithLogitsLoss, weight_decay matched to sklearn (weight only).
    torch_model, torch_loss_curve = train_torch_module(
        X_train, y_train, weight_decay=weight_decay, lr=TORCH_LR, epochs=TORCH_EPOCHS
    )
    torch_prob = torch_predict_proba(torch_model, X_test)
    metrics["torch_module"] = {
        "lr": TORCH_LR,
        "weight_decay": weight_decay,
        "epochs": TORCH_EPOCHS,
        "final_train_loss": torch_loss_curve[-1],
        "test": test_block(y_test, torch_prob, threshold, fp_test),
    }

    # How far apart did the three fits land? Saved so the report can quote it.
    w_sk = logreg.coef_.ravel()
    w_manual = manual_params["w"].numpy()
    w_torch = torch_model.weight.detach().numpy().ravel()
    metrics["coefficient_agreement"] = {
        "max_abs_diff_manual_vs_logreg": float(np.max(np.abs(w_manual - w_sk))),
        "max_abs_diff_torch_vs_logreg": float(np.max(np.abs(w_torch - w_sk))),
        "intercepts": {
            "logreg": float(logreg.intercept_[0]),
            "manual": float(manual_params["b"]),
            "torch_module": float(torch_model.bias.detach()),
        },
    }

    # 4. Save every artefact the dashboard loads (listed in REPORT.md, How to run).
    scaler = fit.get("scaler")
    if scaler is None:
        raise KeyError(
            "fit['scaler'] not found — build_features must put the trained StandardScaler "
            "under fit['scaler'] (see train.py save step)."
        )
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    # The whole fit dict (frequency tables, top-N lists, scaler, medians, column order):
    # app.py needs it to featurize a single SKU typed into the dashboard.
    joblib.dump(fit, MODELS_DIR / "feature_fit.joblib")
    joblib.dump(logreg, MODELS_DIR / "logreg.joblib")
    torch.save({"w": manual_params["w"], "b": manual_params["b"]}, MODELS_DIR / "manual.pt")
    torch.save(torch_model.state_dict(), MODELS_DIR / "torch_module.pt")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Loss curves live in their own file: thousands of floats per model would otherwise
    # bury the handful of numbers metrics.json is meant to be read for.
    loss_curves = {
        "manual": [float(v) for v in manual_params["loss_curve"]],
        "torch_module": torch_loss_curve,
    }
    (MODELS_DIR / "loss_curves.json").write_text(json.dumps(loss_curves, indent=2))
    # The old regression artefact must not linger next to the new ones.
    (MODELS_DIR / "ridge.joblib").unlink(missing_ok=True)

    # test_predictions.csv: one row per test SKU, the true label and every model's probability,
    # for the dashboard's ROC curves, probability histograms and live confusion matrix.
    # Precio detalle and Costo are saved so the dashboard can price each false positive.
    preds = pd.DataFrame(
        {
            "Código": test_df["Código"].to_numpy(),
            "Descripción": test_df["Descripción"].to_numpy(),
            "Costo": test_df["Costo"].to_numpy(),
            "Precio detalle": test_df["Precio detalle"].to_numpy(),
            "y_true": y_test.astype(int),
            "prob_baseline_majority": majority_prob,
            "prob_baseline_cost": cost_prob,
            "prob_logreg": logreg_prob,
            "prob_manual": manual_prob,
            "prob_torch": torch_prob,
        }
    )
    preds.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    # 5. Console summary so the owner can eyeball convergence and agreement.
    print(f"\nthreshold chosen on validation: {threshold:.2f}  "
          f"(val cost {val_costs[threshold]:.0f} DOP vs {val_costs[0.5]:.0f} at 0.5; "
          f"FP = {FP_MARGIN_MULTIPLIER:g} x margin per SKU capped, FN = {FN_COST:.0f} flat)")
    rows = [
        ("baseline majority", metrics["baseline_majority"]["test"]),
        ("baseline cost>median", metrics["baseline_cost_median"]["test"]),
        (f"logreg (C={best_C})", metrics["logreg"]["test"]),
        (f"manual (lr={best_lr})", metrics["manual"]["test"]),
        ("nn.Linear+Adam", metrics["torch_module"]["test"]),
    ]
    for tag in ("at_0.5", "at_threshold"):
        print(f"\n=== test-set metrics {tag} ===")
        print(f"{'model':<22}{'acc':>7}{'prec':>7}{'rec':>7}{'F1':>7}{'AUC':>7}{'cost DOP':>10}")
        for name, m in rows:
            r = m[tag]
            print(f"{name:<22}{r['accuracy']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}"
                  f"{r['f1']:>7.3f}{r['roc_auc']:>7.3f}{r['expected_cost']:>10.0f}")
    agree = metrics["coefficient_agreement"]
    print(f"\nmax |coef diff| vs logreg: manual {agree['max_abs_diff_manual_vs_logreg']:.4f}, "
          f"nn.Linear {agree['max_abs_diff_torch_vs_logreg']:.4f}")
    print(f"final train loss: manual {manual_params['loss_curve'][-1]:.4f}, "
          f"nn.Linear {torch_loss_curve[-1]:.4f}")
    print(f"\nsaved artefacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
