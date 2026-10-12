"""
Assignment 1: Gradio dashboard for the ATP value-betting model.

Loads ONLY the artifacts saved by train.ipynb (models/ folder) and never retrains anything.
Tabs:
  1. Model comparison: predicted vs actual (calibration), agreement between the three implementations, metrics
  2. Data: feature and target distributions
  3. Betting decision: edge-threshold slider -> confusion matrix, profit and ROI
"""
import json
import sys

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss

# ---------- 1) Load the artifacts produced by the training stage ----------
with open("models/config.json") as f:
    CFG = json.load(f)
FEATURES = CFG["features"]

data = pd.read_csv("models/eval_data.csv", parse_dates=["Date"])
scaler = joblib.load("models/scaler.joblib")
sk_model = joblib.load("models/sklearn_logreg.joblib")
manual = torch.load("models/manual_torch.pt")


class LogReg(nn.Module):
    """Same architecture as in training, needed to load the saved nn.Module weights."""
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(n, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


nn_model = LogReg(len(FEATURES))
nn_model.load_state_dict(torch.load("models/nn_module.pt"))
nn_model.eval()

# ---------- 2) Inference only: predicted probabilities from each saved model ----------
X = scaler.transform(data[FEATURES])          # same scaling as training (scaler fitted on train only)
X_t = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    data["p_sklearn"] = sk_model.predict_proba(X)[:, 1]
    data["p_manual"] = torch.sigmoid(X_t @ manual["w"] + manual["b"]).numpy()
    data["p_nn"] = torch.sigmoid(nn_model(X_t)).numpy()
inv_a, inv_b = 1 / data["A_odds"], 1 / data["B_odds"]
data["p_book"] = inv_a / (inv_a + inv_b)      # bookmaker probability with margin removed

MODELS = {"scikit-learn": "p_sklearn", "Manual PyTorch": "p_manual", "nn.Module": "p_nn"}
ALL_PROBS = {**MODELS, "Bookmaker (margin removed)": "p_book"}
SPLITS = {
    "Test 2025": data["season"] == 2025,
    "Validation 2024": data["season"] == 2024,
    "Train 2019-2023": data["season"].between(2019, 2023),
}


# ---------- 3) Tab 1: model comparison ----------
def metrics_table(split):
    d = data[SPLITS[split]]
    rows = [{"model": "Naive: higher-ranked player",
             "accuracy": round(accuracy_score(d["y"], (d["A_rank"] < d["B_rank"]).astype(int)), 4),
             "log_loss": "n/a"}]
    for name, col in ALL_PROBS.items():
        rows.append({"model": name,
                     "accuracy": round(accuracy_score(d["y"], (d[col] > 0.5).astype(int)), 4),
                     "log_loss": round(log_loss(d["y"], d[col]), 4)})
    return pd.DataFrame(rows)


def calibration_fig(split):
    """Prediction vs actual for a classifier: within each predicted-probability bin, how often did A actually win?"""
    d = data[SPLITS[split]]
    bins = np.linspace(0, 1, 11)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(dash="dash", color="grey")))
    for name, col in ALL_PROBS.items():
        g = (d.groupby(pd.cut(d[col], bins, include_lowest=True), observed=True)
               .agg(predicted=(col, "mean"), actual=("y", "mean")))
        fig.add_trace(go.Scatter(x=g["predicted"].values, y=g["actual"].values,
                                 mode="lines+markers", name=name))
    fig.update_layout(title=f"Predicted vs actual win rate ({split})",
                      xaxis_title="Predicted P(A wins), binned", yaxis_title="Actual share of matches A won",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], height=450)
    return fig


def agreement_fig(split):
    """Each dot is a match: sklearn's probability vs the PyTorch versions'. On the diagonal = identical."""
    d = data[SPLITS[split]]
    max_diff = max((d["p_sklearn"] - d["p_manual"]).abs().max(), (d["p_sklearn"] - d["p_nn"]).abs().max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["p_sklearn"].values, y=d["p_manual"].values, mode="markers",
                             name="Manual PyTorch", marker=dict(size=4, opacity=0.5)))
    fig.add_trace(go.Scatter(x=d["p_sklearn"].values, y=d["p_nn"].values, mode="markers",
                             name="nn.Module", marker=dict(size=4, opacity=0.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect agreement",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title=f"Agreement with scikit-learn (max difference {max_diff:.5f})",
                      xaxis_title="scikit-learn P(A wins)", yaxis_title="PyTorch P(A wins)", height=450)
    return fig


def comparison(split):
    return metrics_table(split), calibration_fig(split), agreement_fig(split)


# ---------- 4) Tab 2: data distributions ----------
def feature_fig(feature, split):
    d = data.loc[SPLITS[split], [feature, "y"]]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=d.loc[d["y"] == 1, feature].values, name="A won", opacity=0.6,
                               nbinsx=60, bingroup=1))
    fig.add_trace(go.Histogram(x=d.loc[d["y"] == 0, feature].values, name="A lost", opacity=0.6,
                               nbinsx=60, bingroup=1))   # bingroup: both traces share the same bins
    fig.update_layout(barmode="overlay", title=f"Distribution of {feature} by outcome ({split})",
                      xaxis_title=feature, yaxis_title="count", height=420)
    return fig


def target_fig():
    counts = data.groupby(["season", "y"]).size().unstack(fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts.index.values, y=counts[1].values, name="A won"))
    fig.add_trace(go.Bar(x=counts.index.values, y=counts[0].values, name="A lost"))
    fig.update_layout(barmode="group", title="Target per season: balanced by the random A/B assignment",
                      xaxis_title="season", yaxis_title="count", height=380)
    return fig


# ---------- 5) Tab 3: betting decision ----------
def candidate_bets(d, p):
    """Each match gives two candidate bets: back A, back B."""
    a = pd.DataFrame({"prob": p, "odds": d["A_odds"].values, "won": d["y"].values})
    b = pd.DataFrame({"prob": 1 - p, "odds": d["B_odds"].values, "won": 1 - d["y"].values})
    c = pd.concat([a, b], ignore_index=True)
    c["ev"] = c["prob"] * c["odds"] - 1
    c["pnl"] = np.where(c["won"] == 1, c["odds"] - 1, -1.0)
    return c


def favourite_roi(d):
    fav_is_a = d["A_odds"].values < d["B_odds"].values
    fav_odds = np.where(fav_is_a, d["A_odds"], d["B_odds"])
    fav_won = np.where(fav_is_a, d["y"] == 1, d["y"] == 0)
    return np.where(fav_won, fav_odds - 1, -1.0).mean()


def betting_view(model, split, threshold):
    d = data[SPLITS[split]]
    c = candidate_bets(d, d[MODELS[model]].values)
    placed = (c["ev"] > threshold).values
    won = (c["won"] == 1).values

    # Confusion matrix of the betting decision
    cm = np.array([[(placed & won).sum(), (placed & ~won).sum()],
                   [(~placed & won).sum(), (~placed & ~won).sum()]])
    cm_fig = go.Figure(go.Heatmap(z=cm, x=["Bet would win", "Bet would lose"], y=["Bet placed", "Not placed"],
                                  text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm_fig.update_layout(title="Confusion matrix of the betting decision (candidate bets)",
                         yaxis_autorange="reversed", height=380)

    # Business number: profit and ROI
    fav = favourite_roi(d)
    n = int(placed.sum())
    if n:
        profit = c.loc[placed, "pnl"].sum()
        summary = (f"### {model} | {split} | minimum edge {threshold:.2f}\n"
                   f"| Bets placed | Profit (1 unit per bet) | ROI | Win rate | Avg odds |\n"
                   f"|---|---|---|---|---|\n"
                   f"| {n} | {profit:+.1f} units | {profit / n:+.1%} | {c.loc[placed, 'won'].mean():.1%} "
                   f"| {c.loc[placed, 'odds'].mean():.2f} |\n\n"
                   f"Benchmark, backing every bookmaker favourite in the same season: **ROI {fav:+.1%}**")
    else:
        summary = f"### No bets placed at minimum edge {threshold:.2f}"

    # ROI across all thresholds, with the current one marked
    ts = np.round(np.arange(0, 0.31, 0.01), 2)
    ev, pnl = c["ev"].values, c["pnl"].values
    rois = [pnl[ev > t].mean() if (ev > t).any() else np.nan for t in ts]
    curve = go.Figure(go.Scatter(x=ts, y=rois, mode="lines+markers", name="Model strategy"))
    curve.add_hline(y=fav, line_dash="dash", annotation_text="Back every favourite")
    curve.add_hline(y=0, line_color="grey")
    curve.add_vline(x=threshold, line_color="red", annotation_text="current threshold")
    curve.update_layout(title="ROI vs minimum edge threshold", xaxis_title="Minimum edge (expected value)",
                        yaxis_title="ROI per unit staked", yaxis_tickformat=".0%", height=380)
    return cm_fig, summary, curve


# ---------- 6) Layout ----------
with gr.Blocks(title="ATP Value Betting") as demo:
    gr.Markdown("# ATP Value Betting: Logistic Regression Three Ways\n"
                "Models trained on 2019-2023, tuned on 2024, tested on 2025. "
                "Everything below is computed from the saved models; nothing is retrained.")

    with gr.Tab("1. Model comparison"):
        split1 = gr.Dropdown(list(SPLITS), value="Test 2025", label="Season")
        table = gr.Dataframe(label="Accuracy and log loss vs baselines")
        with gr.Row():
            cal_plot = gr.Plot()
            agree_plot = gr.Plot()
        gr.Markdown("The three implementation lines overlap almost perfectly in the calibration chart: "
                    "they are the same model trained three ways.")

    with gr.Tab("2. Data"):
        with gr.Row():
            feat = gr.Dropdown(FEATURES, value=FEATURES[0], label="Feature", filterable=False)
            split2 = gr.Dropdown(list(SPLITS), value="Train 2019-2023", label="Season")
        feat_plot = gr.Plot()
        target_plot = gr.Plot()

    with gr.Tab("3. Betting decision"):
        with gr.Row():
            model3 = gr.Dropdown(list(MODELS), value="scikit-learn", label="Model")
            split3 = gr.Dropdown(["Test 2025", "Validation 2024"], value="Test 2025", label="Season")
        thr = gr.Slider(0, 0.3, value=CFG["edge_threshold"], step=0.01,
                        label="Minimum edge to place a bet (model probability x odds - 1)")
        summary = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            roi_plot = gr.Plot()

    # Events
    split1.change(comparison, split1, [table, cal_plot, agree_plot])
    for comp in (feat, split2):
        comp.change(feature_fig, [feat, split2], feat_plot)
    for comp in (model3, split3, thr):
        comp.change(betting_view, [model3, split3, thr], [cm_plot, summary, roi_plot])

    # Initial render
    demo.load(comparison, split1, [table, cal_plot, agree_plot])
    demo.load(feature_fig, [feat, split2], feat_plot)
    demo.load(target_fig, None, target_plot)
    demo.load(betting_view, [model3, split3, thr], [cm_plot, summary, roi_plot])

if __name__ == "__main__":
    demo.launch(share="google.colab" in sys.modules)   # plain launch locally (127.0.0.1:7860), share link only in Colab
