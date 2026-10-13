"""
evaluate.py 
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from business import best_threshold, business_cost
from data_prep import clean, load_raw
from features import add_features, get_X_y
from preprocessing import time_split

MODELS_DIR = Path("models")
MODEL_NAMES = ["sklearn", "manual_torch", "standard_torch"]


def load_models():
    """Load the preprocessor and the three trained models from disk."""
    from train import LogisticRegressionNet  # class needed to rebuild the model

    pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
    sk = joblib.load(MODELS_DIR / "sklearn_logreg.joblib")

    manual = torch.load(MODELS_DIR / "torch_manual.pt")

    std_saved = torch.load(MODELS_DIR / "torch_standard.pt")
    std = LogisticRegressionNet(std_saved["n_features"])
    std.load_state_dict(std_saved["state_dict"])
    std.eval()

    return pre, {"sklearn": sk, "manual_torch": manual, "standard_torch": std}


def predict_all(pre, models, X):
    """P(cancel) from each of the three models for a raw feature table X."""
    Xn = pre.transform(X)
    Xt = torch.tensor(Xn, dtype=torch.float32)
    with torch.no_grad():
        p_manual = torch.sigmoid(Xt @ models["manual_torch"]["w"]
                                 + models["manual_torch"]["b"]).numpy()
        p_std = torch.sigmoid(models["standard_torch"](Xt)).numpy()
    return {
        "sklearn": models["sklearn"].predict_proba(Xn)[:, 1],
        "manual_torch": p_manual,
        "standard_torch": p_std,
    }


def load_splits():
    """Rebuild exactly the same train / validation / test split as training."""
    data = add_features(clean(load_raw(), verbose=False))
    return time_split(data)


def run_evaluation():
    train, val, test = load_splits()
    X_val, y_val = get_X_y(val)
    X_test, y_test = get_X_y(test)
    base_rate = float(get_X_y(train)[1].mean())

    pre, models = load_models()
    p_val = predict_all(pre, models, X_val)
    p_test = predict_all(pre, models, X_test)
    p_val["baseline"] = np.full(len(y_val), base_rate)
    p_test["baseline"] = np.full(len(y_test), base_rate)

    # Two simple "policies" without any model, for business context
    never = business_cost(y_test, np.zeros(len(y_test)), 0.5)   # never ask for a deposit
    always = business_cost(y_test, np.ones(len(y_test)), 0.5)   # always ask

    print("\n" + "=" * 70)
    print("STEP 9 - FINAL COMPARISON ON THE TEST SET "
          f"({len(y_test):,} bookings, {test['arrival_date'].min().date()} -> "
          f"{test['arrival_date'].max().date()})")
    print("=" * 70)
    print("Threshold chosen on VALIDATION (cheapest business cost), then applied to test")

    results = []
    for name in ["baseline"] + MODEL_NAMES:
        thr, _ = best_threshold(y_val, p_val[name])
        prob = p_test[name]
        row = {
            "model": name,
            "accuracy@0.5": float(accuracy_score(y_test, prob >= 0.5)),
            "roc_auc": float(roc_auc_score(y_test, prob)) if name != "baseline" else 0.5,
            "log_loss": float(log_loss(y_test, np.clip(prob, 1e-7, 1 - 1e-7))),
            "threshold": thr,
            "cost@0.5": int(business_cost(y_test, prob, 0.5)),
            "cost@threshold": int(business_cost(y_test, prob, thr)),
        }
        results.append(row)

    header = f"{'model':16s} {'acc@0.5':>8s} {'ROC-AUC':>8s} {'logloss':>8s} " \
             f"{'thresh':>7s} {'EUR@0.5':>9s} {'EUR@thr':>9s}"
    print(header)
    for r in results:
        print(f"{r['model']:16s} {r['accuracy@0.5']:8.3f} {r['roc_auc']:8.3f} "
              f"{r['log_loss']:8.3f} {r['threshold']:7.2f} {r['cost@0.5']:9,d} "
              f"{r['cost@threshold']:9,d}")

    print(f"\nBusiness context on the same {len(y_test):,} test bookings:")
    print(f"  never ask for a deposit : EUR {never:,}")
    print(f"  always ask for a deposit: EUR {always:,}")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"test_results": results, "never_ask_cost": int(never),
                   "always_ask_cost": int(always)}, f, indent=2)
    print(f"\nSaved results to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    run_evaluation()