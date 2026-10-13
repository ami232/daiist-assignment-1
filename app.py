"""Gradio dashboard. Loads the artifacts produced by train.ipynb; never trains anything."""
import json
from pathlib import Path

import gradio as gr
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ART = Path(__file__).resolve().parent / "artifacts"


class LogisticModel(nn.Module):
    """Same architecture as in train.ipynb (needed to load the saved weights)."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, X):
        return self.linear(X).squeeze(-1)


# ---------- load saved artifacts ----------
cfg = json.loads((ART / "config.json").read_text())
feature_cols = cfg["feature_cols"]

scaler = joblib.load(ART / "scaler.joblib")
sk_model = joblib.load(ART / "sklearn_model.joblib")
manual = torch.load(ART / "manual_torch.pt")
nn_ckpt = torch.load(ART / "nn_module.pt")
nn_model = LogisticModel(nn_ckpt["n_features"])
nn_model.load_state_dict(nn_ckpt["state_dict"])
nn_model.eval()

train_df = pd.read_csv(ART / "train_features.csv")
test_df = pd.read_csv(ART / "test_features.csv")
y_test = test_df["OUTCOME"].values.astype(int)
X_test = scaler.transform(test_df[feature_cols]).astype("float32")

with torch.no_grad():
    xt = torch.tensor(X_test)
    PROBS = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "manual PyTorch": torch.sigmoid(xt @ manual["w"] + manual["b"]).numpy(),
        "nn.Module + optim": torch.sigmoid(nn_model(xt)).numpy(),
    }

COST_FN = cfg["cost_fn"]
COST_FP = cfg["cost_fp"]


# ---------- tab 1: predictions vs actual ----------
def predictions_view():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, p in PROBS.items():
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
        ax1.plot(mean_pred, frac_pos, "o-", label=name)
    ax1.set(xlabel="predicted claim probability", ylabel="actual claim rate",
            title="Predicted vs actual (test set)")
    ax1.legend()

    p = PROBS["sklearn"]
    ax2.hist(p[y_test == 0], bins=30, range=(0, 1), alpha=0.6, label="no claim (actual)")
    ax2.hist(p[y_test == 1], bins=30, range=(0, 1), alpha=0.6, label="claim (actual)")
    ax2.set(xlabel="predicted claim probability (sklearn)", ylabel="customers",
            title="Predicted probabilities by actual outcome")
    ax2.legend()
    fig.tight_layout()

    rows = [{"model": n, "AUC": roc_auc_score(y_test, p), "log loss": log_loss(y_test, p),
             "Brier": brier_score_loss(y_test, p)} for n, p in PROBS.items()]
    table = pd.DataFrame(rows).round(4)
    plt.close(fig)
    return fig, table


# ---------- tab 2: distributions ----------
def distribution_view(feature):
    fig, ax = plt.subplots(figsize=(7, 4))
    if feature == "OUTCOME":
        counts = train_df["OUTCOME"].value_counts().sort_index()
        ax.bar(["no claim (0)", "claim (1)"], counts.values)
        ax.set(ylabel="customers (train)", title="Target distribution")
    else:
        bins = min(20, max(2, train_df[feature].nunique()))
        for outcome, label in [(0, "no claim"), (1, "claim")]:
            ax.hist(train_df.loc[train_df["OUTCOME"] == outcome, feature], bins=bins,
                    alpha=0.6, density=True, label=label)
        ax.set(xlabel=feature, ylabel="density", title=f"{feature} by outcome (train)")
        ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------- tab 3: threshold, confusion matrix, cost ----------
def threshold_view(model_name, t, cost_fn, cost_fp):
    p = PROBS[model_name]
    flag = p >= t
    tp = int((flag & (y_test == 1)).sum())
    fp = int((flag & (y_test == 0)).sum())
    fn = int((~flag & (y_test == 1)).sum())
    tn = int((~flag & (y_test == 0)).sum())

    cost = fn * cost_fn + fp * cost_fp
    cost_nobody = int((y_test == 1).sum()) * cost_fn
    cost_everybody = int((y_test == 0).sum()) * cost_fp
    t_theory = cost_fp / (cost_fp + cost_fn)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["no flag", "flagged"])
    ax.set_yticks([0, 1], ["no claim", "claim"])
    ax.set(xlabel="model decision", ylabel="actual", title=f"Confusion matrix @ {t:.2f}")
    fig.tight_layout()
    plt.close(fig)

    text = (
        f"### Total business cost: **{cost:,.0f} EUR**\n"
        f"- Missed claims (FN): {fn:,} x {cost_fn:,} EUR\n"
        f"- False alarms (FP): {fp:,} x {cost_fp:,} EUR\n"
        f"- Customers flagged: {flag.mean():.1%}\n\n"
        f"For comparison: flag nobody = {cost_nobody:,.0f} EUR, flag everybody = {cost_everybody:,.0f} EUR.\n\n"
        f"Cost-based threshold from the formula: **{t_theory:.2f}**"
    )
    return fig, text


# ---------- layout ----------
with gr.Blocks(title="Car insurance claims: model comparison") as demo:
    gr.Markdown("# Car insurance claims: model comparison\n"
                "Three implementations of the same logistic regression. "
                "All numbers come from the saved artifacts of the training stage.")
    with gr.Tabs():
        with gr.Tab("Predictions vs actual"):
            pred_plot = gr.Plot()
            pred_table = gr.Dataframe()
        with gr.Tab("Distributions"):
            feat = gr.Dropdown(["OUTCOME"] + feature_cols, value="OUTCOME", label="Feature or target")
            dist_plot = gr.Plot()
        with gr.Tab("Threshold and cost"):
            with gr.Row():
                model_dd = gr.Dropdown(list(PROBS), value="sklearn", label="Model")
                thr = gr.Slider(0, 1, value=round(cfg["threshold_theory"], 2), step=0.01, label="Decision threshold")
            with gr.Row():
                c_fn = gr.Number(value=COST_FN, label="Cost of a missed claim (EUR)")
                c_fp = gr.Number(value=COST_FP, label="Cost of a false alarm (EUR)")
            with gr.Row():
                cm_plot = gr.Plot()
                cost_md = gr.Markdown()

    demo.load(predictions_view, None, [pred_plot, pred_table])
    demo.load(distribution_view, feat, dist_plot)
    demo.load(threshold_view, [model_dd, thr, c_fn, c_fp], [cm_plot, cost_md])
    feat.change(distribution_view, feat, dist_plot)
    for control in (model_dd, thr, c_fn, c_fp):
        control.change(threshold_view, [model_dd, thr, c_fn, c_fp], [cm_plot, cost_md])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
