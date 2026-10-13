"""
Assignment 1 — Gradio dashboard.

Loads the artifacts written by train.ipynb (preprocessor, the three trained
models, test-set rows, metrics) and never retrains anything. Three tabs:

1. Compare models   — results table and predicted vs actual slow share.
2. Distributions    — finish time or any feature, split by actual outcome.
3. Threshold & cost — threshold slider driving the confusion matrix and the
                      business cost.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

torch.set_default_dtype(torch.float64)

MODELS = Path("models")
CUTOFF = 360  # 6 hours, in minutes
LABELS = {"sklearn": "scikit-learn", "manual": "Manual PyTorch loop", "module": "nn.Module + optim"}
COLORS = {"sklearn": "#0072B2", "manual": "#E69F00", "module": "#009E73"}
OUTCOME_COLORS = {"Finished < 6h": "#0072B2", "Slow (≥ 6h)": "#D55E00"}


#  loading
class LogReg(torch.nn.Module):
    """Same architecture as in train.ipynb, needed to load its state_dict."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


metrics = json.loads((MODELS / "metrics.json").read_text())
COST_FN, COST_FP = metrics["cost_fn"], metrics["cost_fp"]

art = joblib.load(MODELS / "preprocessor.joblib")
prep, FEATURES = art["prep"], art["features"]
sk = joblib.load(MODELS / "sklearn.joblib")
manual = torch.load(MODELS / "torch_manual.pt")
module = LogReg(len(FEATURES))
module.load_state_dict(torch.load(MODELS / "torch_module.pt"))

test = pd.read_csv(MODELS / "test_predictions.csv")
test["is_female"] = (test["Sex"] == "F").astype(int)
test["Outcome"] = np.where(test["Slow"] == 1, "Slow (≥ 6h)", "Finished < 6h")
y = test["Slow"].to_numpy()

# Score the test set with the saved models (inference only)
X = prep.transform(test[FEATURES])
Xt = torch.tensor(X)
with torch.no_grad():
    PROBS = {
        "sklearn": sk.predict_proba(X)[:, 1],
        "manual": torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy(),
        "module": torch.sigmoid(module(Xt)).numpy(),
    }


# ---------------------------------------------------------------- tab 1
def results_table():
    cols = ["model", "roc_auc", "log_loss", "threshold", "recall", "precision", "cost"]
    return pd.DataFrame(metrics["results"])[cols].round(4)


def calibration_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(color="black", dash="dash", width=1)))
    for name, p in PROBS.items():
        bins = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", name=LABELS[name],
                                 line=dict(color=COLORS[name])))
    fig.update_layout(
        title="Predicted vs actual share of slow runners (test set, 10 groups)",
        xaxis_title="Average predicted probability of being slow",
        yaxis_title="Actual share of slow runners",
        xaxis_range=[0, 1], yaxis_range=[0, 1], height=500,
    )
    return fig


#  tab 2
def distribution_plot(column):
    fig = px.histogram(test, x=column, color="Outcome", barmode="overlay", histnorm="percent",
                       opacity=0.6, nbins=40, color_discrete_map=OUTCOME_COLORS,
                       title=f"{column} on the test set, by actual outcome")
    if column == "Time":
        fig.add_vline(x=CUTOFF, line_dash="dash", annotation_text="6h cutoff")
    fig.update_layout(yaxis_title="% of runners in each group", height=450)
    return fig


#  tab 3
def threshold_view(model_label, t):
    name = next(k for k, v in LABELS.items() if v == model_label)
    pred = PROBS[name] >= t
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    total = COST_FN * fn + COST_FP * fp

    summary = (
        f"### Business cost: **{total:,}**\n"
        f"{fn} slow runners missed × {COST_FN} + {fp} false alarms × {COST_FP}  \n"
        f"Slow runners caught (recall): **{tp / (tp + fn):.0%}** · "
        f"Runners flagged: **{tp + fp:,} / {len(y):,}**"
    )
    cm = px.imshow(
        [[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
        x=["Predicted: finishes < 6h", "Predicted: slow"],
        y=["Actual: finished < 6h", "Actual: slow"],
        title=f"Confusion matrix at threshold {t:.2f}",
    )
    cm.update_layout(coloraxis_showscale=False, height=400)
    return summary, cm


#  layout
with gr.Blocks(title="Marine Corps Marathon — slow finisher model") as demo:
    gr.Markdown(
        "# Who will still be on the course after 6 hours?\n"
        "Logistic regression trained three ways, evaluated on the **test set "
        f"(2016-2018, {len(y):,} runners)** with the saved models. Nothing is retrained."
    )

    with gr.Tab("1 · Compare models"):
        gr.Dataframe(results_table(), interactive=False)
        gr.Plot(calibration_plot())

    with gr.Tab("2 · Distributions"):
        column = gr.Dropdown(["Time"] + FEATURES, value="Time", label="Variable")
        dist = gr.Plot(distribution_plot("Time"))
        column.change(distribution_plot, column, dist)

    with gr.Tab("3 · Threshold & cost"):
        gr.Markdown(
            f"A runner is **flagged** when their predicted probability is above the threshold. "
            f"A missed slow runner costs {COST_FN}, a false alarm costs {COST_FP}."
        )
        with gr.Row():
            model = gr.Dropdown(list(LABELS.values()), value=LABELS["sklearn"], label="Model")
            thr = gr.Slider(0.01, 0.99, value=metrics["thresholds"]["sklearn"], step=0.01,
                            label="Decision threshold")
        summary = gr.Markdown()
        cm_plot = gr.Plot()
        for comp in (model, thr):
            comp.change(threshold_view, [model, thr], [summary, cm_plot])
        demo.load(threshold_view, [model, thr], [summary, cm_plot])


if __name__ == "__main__":
    demo.launch()
