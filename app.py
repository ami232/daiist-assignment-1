"""
Gradio dashboard for the inflation-spike early-warning models.

Loads only what train.ipynb saved under artifacts/ (scaler, three trained
models, config, engineered features, metrics) and never retrains anything.
Run with:

    uv run python main.py app
"""

import json
import os
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
RAW_PATH = ROOT / "data" / "raw" / "bcb_monthly.csv"

MODELS = ["scikit-learn", "manual PyTorch", "nn.Module + optim"]
LINE_STYLES = {"scikit-learn": "solid", "manual PyTorch": "dash", "nn.Module + optim": "dot"}
SPLIT_CHOICES = ["train", "val", "test", "all"]
CLASS_COLORS = {0: "#4c78a8", 1: "#d62728"}


class LogisticRegressionModel(nn.Module):
    """Same definition as in train.ipynb, needed to load the saved state_dict."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Load artifacts and compute predictions once at startup (inference only)
# ---------------------------------------------------------------------------

if not (MODEL_DIR / "config.json").exists():
    raise SystemExit("No trained models in artifacts/models. Run `uv run python main.py train` first.")

CONFIG = json.loads((MODEL_DIR / "config.json").read_text())
NUMERIC, SEASONAL = CONFIG["numeric_features"], CONFIG["seasonal_features"]
COST_FP, COST_FN = CONFIG["cost_fp"], CONFIG["cost_fn"]
DEFAULT_THRESHOLD = CONFIG["threshold"]
SPIKE_PCT = CONFIG["high_inflation_threshold_pct"]

data = pd.read_csv(ARTIFACTS / "features.csv", parse_dates=["date"], index_col="date")
metrics = pd.read_csv(ARTIFACTS / "metrics.csv")

scaler = joblib.load(MODEL_DIR / "scaler.joblib")
sk_model = joblib.load(MODEL_DIR / "sklearn_logreg.joblib")
manual = torch.load(MODEL_DIR / "torch_manual.pt")
module_model = LogisticRegressionModel(len(NUMERIC) + len(SEASONAL))
module_model.load_state_dict(torch.load(MODEL_DIR / "torch_module.pt"))
module_model.eval()

X = np.hstack([scaler.transform(data[NUMERIC]), data[SEASONAL].to_numpy()])
Xt = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    PROBA = pd.DataFrame({
        "scikit-learn": sk_model.predict_proba(X)[:, 1],
        "manual PyTorch": torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy(),
        "nn.Module + optim": torch.sigmoid(module_model(Xt)).numpy(),
    }, index=data.index)

# Persistence baseline: hedge if the last *published* month (t-1) was a spike.
raw_ipca = pd.read_csv(RAW_PATH, parse_dates=["date"], index_col="date")["ipca"]
PERSISTENCE = (raw_ipca.shift(1) > SPIKE_PCT).astype(int).reindex(data.index)


def rows_for(split):
    return data if split == "all" else data[data["split"] == split]


def decision_counts(y_true, pred):
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    return {
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
    }


def cost_of(counts, cost_fn, cost_fp):
    return cost_fn * counts["fn"] + cost_fp * counts["fp"]


def shade_splits(fig):
    for split, color in [("val", "orange"), ("test", "red")]:
        idx = data.index[data["split"] == split]
        fig.add_vrect(x0=idx.min(), x1=idx.max(), fillcolor=color, opacity=0.08, line_width=0,
                      annotation_text=split, annotation_position="top left")


# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------

def prediction_timeline(split):
    d = rows_for(split)
    fig = go.Figure()
    for name in MODELS:
        fig.add_scatter(x=d.index, y=PROBA.loc[d.index, name], mode="lines", name=name,
                        line=dict(dash=LINE_STYLES[name], width=2))
    for cls, label in [(1, f"actual: IPCA(t+1) > {SPIKE_PCT}%"), (0, "actual: normal month")]:
        dc = d[d["target"] == cls]
        fig.add_scatter(x=dc.index, y=dc["target"], mode="markers", name=label,
                        marker=dict(color=CLASS_COLORS[cls], size=7),
                        customdata=dc["next_ipca"], hovertemplate="%{x|%Y-%m}<br>IPCA(t+1) = %{customdata:.2f}%")
    fig.add_hline(y=DEFAULT_THRESHOLD, line_dash="dash", line_color="black",
                  annotation_text=f"cost-based threshold {DEFAULT_THRESHOLD:.2f}")
    if split == "all":
        shade_splits(fig)
    fig.update_layout(title="Predicted P(spike next month) vs. actual outcome",
                      xaxis_title="forecast month t", yaxis_title="probability / actual (0 or 1)",
                      yaxis_range=[-0.05, 1.05], hovermode="x unified", height=450)
    return fig


def prediction_vs_actual_scatter(split, model):
    d = rows_for(split)
    fig = go.Figure()
    for cls, label in [(1, "spike"), (0, "normal")]:
        dc = d[d["target"] == cls]
        fig.add_scatter(x=dc["next_ipca"], y=PROBA.loc[dc.index, model], mode="markers", name=label,
                        marker=dict(color=CLASS_COLORS[cls], size=8, opacity=0.8),
                        text=dc.index.strftime("%Y-%m"), hovertemplate="%{text}<br>IPCA(t+1) = %{x:.2f}%<br>p = %{y:.3f}")
    fig.add_vline(x=SPIKE_PCT, line_dash="dash", annotation_text=f"spike line {SPIKE_PCT}%")
    fig.add_hline(y=DEFAULT_THRESHOLD, line_dash="dash", annotation_text="hedge threshold")
    fig.update_layout(title=f"{model}: predicted probability vs. actual next-month IPCA",
                      xaxis_title="actual IPCA in month t+1 (%)", yaxis_title="predicted P(spike)", height=450)
    return fig


def agreement_table():
    rows = []
    for split in ["train", "val", "test"]:
        idx = data.index[data["split"] == split]
        ref = PROBA.loc[idx, "scikit-learn"]
        for name in MODELS[1:]:
            p = PROBA.loc[idx, name]
            rows.append({
                "split": split,
                "model vs scikit-learn": name,
                "max |probability diff|": f"{(p - ref).abs().max():.1e}",
                "same hedge decisions": f"{((p >= DEFAULT_THRESHOLD) == (ref >= DEFAULT_THRESHOLD)).mean():.0%}",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tab 2: distributions
# ---------------------------------------------------------------------------

def target_distribution():
    fig = go.Figure()
    for split in ["train", "val", "test"]:
        fig.add_histogram(x=data.loc[data["split"] == split, "next_ipca"], name=split,
                          opacity=0.6, xbins=dict(size=0.1), histnorm="probability")
    fig.add_vline(x=SPIKE_PCT, line_dash="dash", annotation_text=f"spike line {SPIKE_PCT}%")
    fig.update_layout(barmode="overlay", title="Next-month IPCA by split (share of months)",
                      xaxis_title="IPCA in month t+1 (%)", yaxis_title="share of months", height=400)
    return fig


def class_balance_table():
    g = data.groupby("split")["target"]
    return (pd.DataFrame({"months": g.size(), "spike months": g.sum().astype(int), "spike rate": g.mean().round(3)})
            .loc[["train", "val", "test"]].reset_index())


def feature_distribution(feature, split):
    d = rows_for(split)
    fig = go.Figure()
    for cls, label in [(0, "normal next month"), (1, "spike next month")]:
        fig.add_histogram(x=d.loc[d["target"] == cls, feature], name=label, opacity=0.6,
                          marker_color=CLASS_COLORS[cls], nbinsx=25)
    fig.update_layout(barmode="overlay", title=f"{feature} by outcome ({split})",
                      xaxis_title=feature, yaxis_title="months", height=400)
    return fig


# ---------------------------------------------------------------------------
# Tab 3: threshold and business cost
# ---------------------------------------------------------------------------

def confusion_figure(c):
    z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    labels = [[f"TN<br>{c['tn']}<br>no hedge, calm", f"FP<br>{c['fp']}<br>fee paid for nothing"],
              [f"FN<br>{c['fn']}<br>missed spike", f"TP<br>{c['tp']}<br>hedged a spike"]]
    fig = go.Figure(go.Heatmap(z=z, x=["no hedge (pred 0)", "hedge (pred 1)"], y=["normal (actual 0)", "spike (actual 1)"],
                               text=labels, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion matrix", yaxis_autorange="reversed", height=380)
    return fig


def cost_curve(p, y_true, threshold, cost_fn, cost_fp, persistence_cost):
    grid = np.linspace(0, 1, 101)
    costs = [cost_of(decision_counts(y_true, (p >= t).astype(int)), cost_fn, cost_fp) for t in grid]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=costs, mode="lines", name="model cost")
    fig.add_hline(y=persistence_cost, line_dash="dot", line_color="grey", annotation_text="persistence baseline")
    fig.add_vline(x=threshold, line_color="crimson", annotation_text="selected")
    fig.add_vline(x=cost_fp / (cost_fp + cost_fn), line_dash="dash", line_color="black",
                  annotation_text="cost-based", annotation_position="bottom right")
    fig.update_layout(title="Business cost vs. threshold", xaxis_title="decision threshold",
                      yaxis_title="cost (units of one hedging fee)", height=380)
    return fig


def threshold_view(model, split, threshold, cost_fn, cost_fp):
    d = rows_for(split)
    y_true = d["target"].to_numpy().astype(int)
    p = PROBA.loc[d.index, model].to_numpy()

    c = decision_counts(y_true, (p >= threshold).astype(int))
    model_cost = cost_of(c, cost_fn, cost_fp)
    persistence_cost = cost_of(decision_counts(y_true, PERSISTENCE.loc[d.index]), cost_fn, cost_fp)
    always_cost = cost_of(decision_counts(y_true, np.ones_like(y_true)), cost_fn, cost_fp)
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")

    summary = (
        f"### Business cost: **{model_cost:g}** (units of one month's hedging fee)\n"
        f"= {cost_fn:g} × {c['fn']} missed spikes + {cost_fp:g} × {c['fp']} false alarms\n\n"
        f"| | cost |\n|---|---|\n"
        f"| **{model}** at threshold {threshold:.2f} | **{model_cost:g}** |\n"
        f"| persistence baseline (hedge if last published month was a spike) | {persistence_cost:g} |\n"
        f"| always hedge (reference) | {always_cost:g} |\n\n"
        f"Recall {recall:.0%} · precision {precision:.0%} · hedged in {c['tp'] + c['fp']} of {len(y_true)} months"
    )
    return confusion_figure(c), summary, cost_curve(p, y_true, threshold, cost_fn, cost_fp, persistence_cost)


def cost_based_threshold(cost_fn, cost_fp):
    return round(cost_fp / (cost_fp + cost_fn), 2)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Inflation Spike Early Warning") as demo:
    gr.Markdown(
        "# Inflation spike early warning\n"
        f"At the end of month *t*, estimate the probability that IPCA in month *t+1* exceeds **{SPIKE_PCT}%**, "
        "and decide whether to switch on currency and interest-rate hedges. "
        "The same logistic regression was trained three ways; all models are loaded from `artifacts/`, nothing is retrained."
    )

    with gr.Tab("Model comparison"):
        split_1 = gr.Radio(SPLIT_CHOICES, value="test", label="Period")
        timeline = gr.Plot()
        gr.Markdown("The three lines overlap because all three implementations minimise the same "
                    "regularised objective and converge to the same coefficients (table below).")
        model_1 = gr.Dropdown(MODELS, value="scikit-learn", label="Model for the scatter plot")
        scatter = gr.Plot()
        gr.Markdown("### Agreement between implementations")
        gr.Dataframe(agreement_table(), interactive=False)
        gr.Markdown(f"### Saved metrics (threshold {DEFAULT_THRESHOLD:.2f}, costs FN = {COST_FN:g}, FP = {COST_FP:g})")
        gr.Dataframe(metrics.round(3), interactive=False)

        split_1.change(prediction_timeline, split_1, timeline)
        split_1.change(prediction_vs_actual_scatter, [split_1, model_1], scatter)
        model_1.change(prediction_vs_actual_scatter, [split_1, model_1], scatter)
        demo.load(prediction_timeline, split_1, timeline)
        demo.load(prediction_vs_actual_scatter, [split_1, model_1], scatter)

    with gr.Tab("Distributions"):
        gr.Plot(target_distribution())
        gr.Dataframe(class_balance_table(), interactive=False)
        with gr.Row():
            feature = gr.Dropdown(NUMERIC, value=NUMERIC[0], label="Feature")
            split_2 = gr.Radio(SPLIT_CHOICES, value="train", label="Period")
        feature_plot = gr.Plot()
        for control in (feature, split_2):
            control.change(feature_distribution, [feature, split_2], feature_plot)
        demo.load(feature_distribution, [feature, split_2], feature_plot)

    with gr.Tab("Threshold & business cost"):
        gr.Markdown("Hedge when P(spike) ≥ threshold. The cost-based threshold FP / (FP + FN) is where hedging "
                    "and not hedging have the same expected cost.")
        with gr.Row():
            model_3 = gr.Dropdown(MODELS, value="scikit-learn", label="Model")
            split_3 = gr.Radio(["train", "val", "test"], value="test", label="Period")
        with gr.Row():
            cost_fn_in = gr.Number(value=COST_FN, label="Cost of a missed spike (FN)", minimum=0)
            cost_fp_in = gr.Number(value=COST_FP, label="Cost of a false alarm (FP)", minimum=0)
        threshold = gr.Slider(0, 1, value=round(DEFAULT_THRESHOLD, 2), step=0.01, label="Decision threshold")
        reset = gr.Button("Set threshold to cost-based value")
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        summary = gr.Markdown()

        inputs = [model_3, split_3, threshold, cost_fn_in, cost_fp_in]
        outputs = [cm_plot, summary, cost_plot]
        for control in inputs:
            control.change(threshold_view, inputs, outputs)
        reset.click(cost_based_threshold, [cost_fn_in, cost_fp_in], threshold)
        demo.load(threshold_view, inputs, outputs)


if __name__ == "__main__":
    demo.launch(inbrowser=os.environ.get("CI") != "true")
