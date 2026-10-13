"""
Dashboard stage: `uv run python main.py app`

Loads the models saved by train.py and only makes predictions. Nothing is trained here.
Tabs:
  1. Compare models   - metrics table, predicted probability vs actual class, coefficients
  2. Distributions    - target balance and any feature split by class
  3. Threshold & cost - slider -> confusion matrix and business cost
"""
import json

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from train import COST_FN, COST_FP, FEATURES, OUT, LogReg  # importing does not retrain (main() is not called)

# ---------------------------------------------------------------- load saved artifacts
scaler = joblib.load(OUT / "scaler.joblib")
sk_model = joblib.load(OUT / "sklearn.joblib")
manual = torch.load(OUT / "manual.pt")
nn_model = LogReg(len(FEATURES))
nn_model.load_state_dict(torch.load(OUT / "nn.pt"))
nn_model.eval()

meta = json.load(open(OUT / "meta.json"))
test = pd.read_csv(OUT / "test.csv")
train = pd.read_csv(OUT / "train.csv")
results = pd.read_csv(OUT / "results.csv", index_col=0)
coefs = pd.read_csv(OUT / "coefficients.csv", index_col=0)

# ---------------------------------------------------------------- predict on the test set
X_test = scaler.transform(test[FEATURES])
y_test = test["y"].values
X_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    probs = {
        "baseline": np.full(len(y_test), meta["prior"]),
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "manual": torch.sigmoid(X_t @ manual["w"] + manual["b"]).numpy(),
        "nn": torch.sigmoid(nn_model(X_t)).numpy(),
    }
MODELS = ["sklearn", "manual", "nn"]


# ---------------------------------------------------------------- plots
def plot_pred_vs_actual():
    """For each model: histogram of predicted P(planet), coloured by the true class."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, name in zip(axes, MODELS):
        ax.hist(probs[name][y_test == 0], bins=25, range=(0, 1), alpha=0.6, label="actual: false positive")
        ax.hist(probs[name][y_test == 1], bins=25, range=(0, 1), alpha=0.6, label="actual: confirmed")
        ax.axvline(meta["thresholds"][name], color="black", linestyle="--", label="threshold")
        ax.set_title(name)
        ax.set_xlabel("predicted P(confirmed)")
    axes[0].set_ylabel("number of KOIs")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_agreement():
    """sklearn's probability vs each PyTorch version's, per test KOI. On the diagonal = same model."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(probs["sklearn"], probs["manual"], s=6, alpha=0.5, label="manual PyTorch")
    ax.scatter(probs["sklearn"], probs["nn"], s=6, alpha=0.5, label="nn.Module PyTorch")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("sklearn P(confirmed)")
    ax.set_ylabel("PyTorch P(confirmed)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_target():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = train["y"].value_counts().sort_index()
    ax.bar(["false positive", "confirmed"], counts.values)
    ax.set_title("Target in the training set")
    fig.tight_layout()
    return fig


def plot_feature(feature):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(train.loc[train["y"] == 0, feature], bins=40, alpha=0.6, density=True, label="false positive")
    ax.hist(train.loc[train["y"] == 1, feature], bins=40, alpha=0.6, density=True, label="confirmed")
    ax.set_title(f"{feature} (training set)")
    ax.legend()
    fig.tight_layout()
    return fig


def threshold_view(model, threshold):
    pred = (probs[model] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    cost = COST_FP * fp + COST_FN * fn

    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array([[tn, fp], [fn, tp]])
    ax.imshow(matrix, cmap="Blues")
    names = [["TN", "FP\n(wasted follow-up)"], ["FN\n(missed planet)", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{matrix[i, j]}", ha="center", va="center")
    ax.set_xticks([0, 1], ["skip", "follow up"])
    ax.set_yticks([0, 1], ["false positive", "confirmed"])
    ax.set_xlabel("decision")
    ax.set_ylabel("actual")
    fig.tight_layout()

    text = (f"### Business cost: {cost}  ({100 * cost / len(y_test):.1f} per 100 KOIs)\n"
            f"= {fp} wasted follow-ups × {COST_FP} + {fn} missed planets × {COST_FN}  \n"
            f"Threshold chosen on validation for this model: {meta['thresholds'][model]:.2f}")
    return fig, text


# ---------------------------------------------------------------- layout
with gr.Blocks(title="Kepler KOI vetting") as demo:
    gr.Markdown("# Kepler KOI vetting: which signals deserve telescope follow-up?\n"
                "Logistic regression trained three ways. All numbers are on the test set.")

    with gr.Tab("Compare models"):
        gr.Dataframe(results.reset_index().rename(columns={"index": "model"}))
        gr.Plot(plot_pred_vs_actual())
        with gr.Row():
            gr.Plot(plot_agreement())
            gr.Dataframe(coefs.reset_index().rename(columns={"index": "feature"}))

    with gr.Tab("Distributions"):
        gr.Plot(plot_target())
        feature = gr.Dropdown(FEATURES, value="log_centroid_offset", label="feature")
        feature_plot = gr.Plot(plot_feature("log_centroid_offset"))
        feature.change(plot_feature, feature, feature_plot)

    with gr.Tab("Threshold & cost"):
        model = gr.Radio(MODELS, value="sklearn", label="model")
        threshold = gr.Slider(0.01, 0.99, value=meta["thresholds"]["sklearn"], step=0.01, label="threshold")
        fig0, text0 = threshold_view("sklearn", meta["thresholds"]["sklearn"])
        cost_text = gr.Markdown(text0)
        matrix_plot = gr.Plot(fig0)
        model.change(threshold_view, [model, threshold], [matrix_plot, cost_text])
        threshold.change(threshold_view, [model, threshold], [matrix_plot, cost_text])

demo.launch()
