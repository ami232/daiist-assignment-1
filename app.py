"""Assignment 1 - Gradio dashboard.

Loads what train.py saved in artifacts/ (models, preprocessor, matrices) and
never retrains anything.
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

from train import RidgeRegression  # only the class definition; importing it trains nothing

ART = Path("artifacts")
if not (ART / "metadata.json").exists():
    raise SystemExit("artifacts/ not found: run `uv run python main.py train` first.")

# ---- Load the saved artifacts -------------------------------------------------
meta = json.loads((ART / "metadata.json").read_text())
Y_MEAN, Y_STD = meta["y_mean"], meta["y_std"]
pre = joblib.load(ART / "preprocessor.joblib")
sk_model = joblib.load(ART / "sklearn_ridge.joblib")
manual = torch.load(ART / "manual_pytorch.pt")
nn_model = RidgeRegression(meta["n_features"])
nn_model.load_state_dict(torch.load(ART / "nn_module.pt"))
nn_model.eval()
mats = np.load(ART / "matrices.npz")
results = pd.read_csv(ART / "results.csv")
raw = pd.read_csv("data/ames.csv")
raw["split"] = np.where(raw["YrSold"] <= 2008, "train (2006-2008)", np.where(raw["YrSold"] == 2009, "val (2009)", "test (2010)"))


def predict_all(A):
    """Run the three SAVED models on a model matrix and return prices in dollars."""
    t = torch.tensor(A, dtype=torch.float32)
    with torch.no_grad():
        return {
            "scikit-learn": sk_model.predict(A) * Y_STD + Y_MEAN,
            "manual PyTorch": (t @ manual["w"] + manual["b"]).numpy().ravel() * Y_STD + Y_MEAN,
            "nn.Module": nn_model(t).numpy().ravel() * Y_STD + Y_MEAN,
        }


PRED = {s: predict_all(mats[f"A_{s}"]) for s in ("val", "test")}
ACTUAL = {s: mats[f"y_{s}"] for s in ("val", "test")}
MODELS = list(PRED["val"])


# ---- Tab 1: predicted vs actual ----------------------------------------------
def pred_vs_actual(split, models):
    y = ACTUAL[split]
    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Scatter(x=y, y=PRED[split][m], mode="markers", name=m, opacity=0.6))
    lo, hi = float(y.min()), float(y.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction", line=dict(dash="dash", color="black")))
    fig.update_layout(xaxis_title="Actual sale price ($)", yaxis_title="Predicted sale price ($)",
                      title="Predicted vs actual (%s set)" % split, height=520)
    return fig


# ---- Tab 2: distributions ------------------------------------------------------
def target_plot():
    fig = px.histogram(raw, x="SalePrice", color="split", nbins=40, barmode="overlay", opacity=0.6,
                       title="Distribution of SalePrice by split")
    fig.update_layout(height=450)
    return fig


def feature_plot(col):
    if pd.api.types.is_numeric_dtype(raw[col]):
        fig = px.histogram(raw, x=col, color="split", nbins=40, barmode="overlay", opacity=0.6)
    else:
        counts = raw.groupby([col, "split"]).size().reset_index(name="houses")
        fig = px.bar(counts, x=col, y="houses", color="split", barmode="group")
    fig.update_layout(title="Distribution of %s" % col, height=450)
    return fig


# ---- Tab 3: what drives the price ----------------------------------------------
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
effect = pd.Series(sk_model.coef_ * Y_STD, index=names)


def drivers(top_n):
    top = effect.reindex(effect.abs().sort_values(ascending=False).index).head(int(top_n)).sort_values()
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                           marker_color=["crimson" if v < 0 else "seagreen" for v in top.values]))
    fig.update_layout(title="Largest effects in the scikit-learn Ridge model (USD)",
                      xaxis_title="Price change ($) per +1 std of the feature (numeric) or vs. the reference category (one-hot)",
                      height=120 + 28 * int(top_n))
    return fig


# ---- Layout -------------------------------------------------------------------
with gr.Blocks(title="Ames house price model") as demo:
    gr.Markdown("# Ames house prices: three ways to train the same Ridge model\n"
                "Scenario: a bank uses the model as a second opinion on property valuations. "
                "Trained on sales up to 2008, validated on 2009, tested on 2010.")
    with gr.Tab("Model comparison"):
        gr.Dataframe(value=results.round(0), label="RMSE / MAE in dollars (baseline = always predict the mean price)", interactive=False)
        with gr.Row():
            split_in = gr.Radio(["val", "test"], value="test", label="Split")
            models_in = gr.CheckboxGroup(MODELS, value=MODELS, label="Models")
        plot1 = gr.Plot(value=pred_vs_actual("test", MODELS))
        split_in.change(pred_vs_actual, [split_in, models_in], plot1)
        models_in.change(pred_vs_actual, [split_in, models_in], plot1)
    with gr.Tab("Distributions"):
        gr.Plot(value=target_plot())
        feat_in = gr.Dropdown([c for c in raw.columns if c not in ("Id", "SalePrice", "split")], value="OverallQual", label="Feature")
        plot2 = gr.Plot(value=feature_plot("OverallQual"))
        feat_in.change(feature_plot, feat_in, plot2)
    with gr.Tab("What drives the price"):
        top_in = gr.Slider(5, 30, value=15, step=1, label="Number of features")
        plot3 = gr.Plot(value=drivers(15))
        top_in.change(drivers, top_in, plot3)

demo.launch()