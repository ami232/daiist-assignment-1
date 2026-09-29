"""
Assignment 1 — Gradio dashboard.

Loads the models saved by train.py (nothing is retrained here) and shows:
  1. Model comparison: predicted probability vs actual share of good wines
  2. Feature and target distributions
  3. A decision-threshold slider with confusion matrix and business cost

Run with:  uv run python main.py app
"""

import gradio as gr
import joblib
import matplotlib

matplotlib.use("Agg")  # render plots off-screen (no GUI window)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Reuse train.py's data code so the app sees exactly the same test split.
from train import (
    ART_DIR,
    MODEL_NAMES,
    LogisticRegressionNet,
    engineer,
    load_data,
    predict_manual,
    predict_nn,
    split,
    to_tensor,
    transform,
)

# Business cost per mistake (set these to match the framing in REPORT.md).
COST_FALSE_POSITIVE = 30.0  # ordinary wine sold as premium
COST_FALSE_NEGATIVE = 10.0  # good wine sold as ordinary

# ---- Load data and saved models ------------------------------------------
raw = load_data()
_, _, X_test, _, _, y_test = split(engineer(raw))
y_test = y_test.to_numpy()

prep = joblib.load(ART_DIR / "preprocessing.joblib")
Xte = transform(X_test, prep)
Xte_t = to_tensor(Xte)

baseline = joblib.load(ART_DIR / "baseline.joblib")
sk_model = joblib.load(ART_DIR / "sklearn_model.joblib")
manual = torch.load(ART_DIR / "manual_torch.pt")
nn_model = LogisticRegressionNet(Xte.shape[1])
nn_model.load_state_dict(torch.load(ART_DIR / "nn_module.pt"))

proba = dict(zip(MODEL_NAMES, [
    baseline.predict_proba(Xte)[:, 1],
    sk_model.predict_proba(Xte)[:, 1],
    predict_manual(manual["w"], manual["b"], Xte_t),
    predict_nn(nn_model, Xte_t),
]))
results = pd.read_csv(ART_DIR / "results.csv", index_col=0).reset_index(names="Model")


# ---- Plots ---------------------------------------------------------------
def prediction_vs_actual_plot():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for name in MODEL_NAMES[1:]:
        frac_good, mean_pred = calibration_curve(y_test, proba[name], n_bins=10)
        ax.plot(mean_pred, frac_good, marker="o", label=name)
    ax.axhline(y_test.mean(), color="grey", linestyle=":", label="Baseline (base rate)")
    ax.set(xlabel="Predicted probability of good wine",
           ylabel="Actual share of good wines",
           title="Predicted vs actual (test set)")
    ax.legend()
    return fig


def distribution_plot(feature):
    fig, ax = plt.subplots(figsize=(6, 4))
    if feature == "quality (target)":
        raw["quality"].value_counts().sort_index().plot.bar(ax=ax)
        ax.set(xlabel="Quality score", ylabel="Wines", title="Target: quality score")
    else:
        good = raw["quality"] >= 7
        ax.hist(raw.loc[~good, feature], bins=40, alpha=0.6, density=True, label="Not good (< 7)")
        ax.hist(raw.loc[good, feature], bins=40, alpha=0.6, density=True, label="Good (>= 7)")
        ax.set(xlabel=feature, ylabel="Density", title=f"{feature} by class")
        ax.legend()
    return fig


def threshold_view(model_name, threshold):
    pred = (proba[model_name] >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"],
           xlabel="Predicted", ylabel="Actual", title=f"{model_name} @ {threshold:.2f}")

    cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
    summary = (
        f"**Business cost: €{cost:,.0f}**  "
        f"({fp} false positives × €{COST_FALSE_POSITIVE:.0f} + "
        f"{fn} false negatives × €{COST_FALSE_NEGATIVE:.0f})\n\n"
        f"Precision: {precision_score(y_test, pred, zero_division=0):.3f} · "
        f"Recall: {recall_score(y_test, pred):.3f} · "
        f"Wines flagged as good: {tp + fp} of {len(pred)}"
    )
    return fig, summary


# ---- Layout --------------------------------------------------------------
features = ["quality (target)"] + [c for c in raw.columns if c not in ("quality", "color")]

with gr.Blocks(title="Wine Quality — Logistic Regression") as demo:
    gr.Markdown("# Wine Quality: is this a good wine (quality ≥ 7)?")

    with gr.Tab("Model comparison"):
        gr.Dataframe(results, label="Test-set metrics (threshold 0.5)")
        gr.Plot(prediction_vs_actual_plot())

    with gr.Tab("Distributions"):
        feature = gr.Dropdown(features, value="quality (target)", label="Variable")
        dist_plot = gr.Plot(distribution_plot("quality (target)"))
        feature.change(distribution_plot, feature, dist_plot)

    with gr.Tab("Threshold & cost"):
        model = gr.Dropdown(MODEL_NAMES[1:], value="scikit-learn", label="Model")
        threshold = gr.Slider(0.01, 0.99, value=0.5, step=0.01, label="Decision threshold")
        cm_plot = gr.Plot()
        cost_text = gr.Markdown()
        for control in (model, threshold):
            control.change(threshold_view, [model, threshold], [cm_plot, cost_text])
        demo.load(threshold_view, [model, threshold], [cm_plot, cost_text])

if __name__ == "__main__":
    demo.launch()
