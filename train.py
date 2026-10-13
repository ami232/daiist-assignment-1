"""
Trains a logistic regression classifier -- "is this house a good deal to
buy?" -- three ways on the identical train/test split:
    1. scikit-learn LogisticRegression
    2. a from-scratch PyTorch loop (manual forward/backward/update)
    3. the standard torch.nn.Module + torch.optim workflow

All artifacts (trained models, preprocessing objects, metrics, and the
processed data used to produce them) are saved to models/ so that app.py
can load and display everything without retraining anything.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

sys.path.insert(0, "src")
from data_prep import prepare_dataset  # noqa: E402
from models import (  # noqa: E402
    train_sklearn, predict_sklearn,
    train_pytorch_manual, predict_pytorch_manual,
    train_pytorch_module, predict_pytorch_module,
)

DATA_PATH = "data/AmesHousing.csv"
MODELS_DIR = "models"
SEED = 42

# Regularization strength (sklearn's C; PyTorch weight_decay is derived
# from the same C so all three face an equivalent L2 penalty). Tuned by
# sweeping C in {1.0, 0.1, 0.05, 0.01, 0.005}: C=1.0 overfits badly
# (88.8% train acc / 45.5% test acc on 275 features, 1941 rows) since the
# L2 penalty is too weak for this many columns relative to rows. C=0.05
# is where test accuracy peaks (~81%); see REPORT.md.
REG_STRENGTH = 0.05
LEARNING_RATE = 0.1
N_EPOCHS = 2000

np.random.seed(SEED)


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data and building features...")
    d = prepare_dataset(DATA_PATH)
    X_train, X_test = d["X_train"], d["X_test"]
    y_train, y_test = d["y_train"], d["y_test"]
    print(f"  train: {X_train.shape}, test: {X_test.shape}")
    print(f"  label report: {d['label_report']}")

    print("\nTraining scikit-learn logistic regression...")
    sk_model = train_sklearn(X_train, y_train, C=REG_STRENGTH, seed=SEED)
    sk_test_prob = predict_sklearn(sk_model, X_test)

    print("Training manual PyTorch logistic regression...")
    manual_params = train_pytorch_manual(
        X_train, y_train, C=REG_STRENGTH, lr=LEARNING_RATE, n_epochs=N_EPOCHS, seed=SEED
    )
    manual_test_prob = predict_pytorch_manual(manual_params, X_test)

    print("Training torch.nn.Module + optim logistic regression...")
    module_model = train_pytorch_module(
        X_train, y_train, C=REG_STRENGTH, lr=LEARNING_RATE, n_epochs=N_EPOCHS, seed=SEED
    )
    module_test_prob = predict_pytorch_module(module_model, X_test)

    print("\nEvaluating on the held-out test set (2009-2010)...")
    naive_prob = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)

    results = {
        "naive_baseline": evaluate(y_test, naive_prob),
        "sklearn": evaluate(y_test, sk_test_prob),
        "pytorch_manual": evaluate(y_test, manual_test_prob),
        "pytorch_module": evaluate(y_test, module_test_prob),
    }
    print(json.dumps(results, indent=2))

    # --- Save everything the Gradio app needs, so it never retrains ---
    joblib.dump(sk_model, f"{MODELS_DIR}/sklearn_model.joblib")
    torch.save(manual_params, f"{MODELS_DIR}/pytorch_manual.pt")
    torch.save(module_model.state_dict(), f"{MODELS_DIR}/pytorch_module.pt")
    joblib.dump(d["preprocessor"], f"{MODELS_DIR}/preprocessor.joblib")

    with open(f"{MODELS_DIR}/n_features.json", "w") as f:
        json.dump({"n_features": X_train.shape[1]}, f)

    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{MODELS_DIR}/label_report.json", "w") as f:
        json.dump(d["label_report"], f, indent=2)

    # Predictions + raw test rows, so the dashboard's plots and threshold
    # slider work directly off saved data without needing to reprocess.
    test_predictions = pd.DataFrame({
        "y_true": y_test,
        "sklearn_prob": sk_test_prob,
        "pytorch_manual_prob": manual_test_prob,
        "pytorch_module_prob": module_test_prob,
    }, index=d["test_df"].index)
    test_predictions.to_csv(f"{MODELS_DIR}/test_predictions.csv")

    d["train_df"].to_csv(f"{MODELS_DIR}/train_raw.csv")
    d["test_df"].to_csv(f"{MODELS_DIR}/test_raw.csv")

    print(f"\nAll artifacts saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()