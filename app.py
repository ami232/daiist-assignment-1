"""
Assignment 1 — Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file — main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py app`, this file must build and launch a Gradio
app (a `demo` that calls `.launch()`) using only what your training stage
already produced — it must never retrain anything itself. At minimum, the
dashboard must let you:

- Compare your three trained models via a prediction-vs-actual plot.
- See feature and/or target distributions.
- For classification: move a decision-threshold slider and watch the
  confusion matrix, and a business-cost number tied to your REPORT.md's
  framing, change with it.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""

"""Gradio dashboard. Loads saved artifacts from models/. Never trains anything."""
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from train import LinReg, TEST_YEAR, TRAIN_YEARS, VAL_YEAR, engineer, load, make_xy

MODELS = Path("models")
MODEL_NAMES = ["scikit-learn", "manual PyTorch", "nn.Module"]

# ---------- load saved artifacts ----------
pre = joblib.load(MODELS / "preprocessor.joblib")
sk_model = joblib.load(MODELS / "sklearn_model.joblib")
manual = torch.load(MODELS / "manual_pytorch.pt")
w, b = manual["w"].numpy(), manual["b"].numpy()
nn_model = LinReg(len(w))
nn_model.load_state_dict(torch.load(MODELS / "nn_module.pt"))
nn_model.eval()
saved = json.loads((MODELS / "results.json").read_text())
test_preds = np.load(MODELS / "test_predictions.npz")

# ---------- data for plots and form defaults ----------
raw = load()
raw_train = raw[raw["Yr Sold"].isin(TRAIN_YEARS)]
df_all = engineer(raw)
df_all["Split"] = np.where(
    df_all["Yr Sold"].isin(TRAIN_YEARS), "train (2006-08)",
    np.where(df_all["Yr Sold"] == VAL_YEAR, "val (2009)", "test (2010)"))
df_all["Log SalePrice"] = np.log(df_all["SalePrice"])

# Defaults for every house feature the agent does NOT enter:
# the training median (numeric) or most common value (categorical).
defaults = {}
for c in raw.columns:
    s = raw_train[c]
    if c == "MS SubClass" or not pd.api.types.is_numeric_dtype(s):
        defaults[c] = s.mode().iloc[0]
    else:
        defaults[c] = float(s.median())
defaults["Yr Sold"] = TEST_YEAR   # used only to compute house age
defaults["SalePrice"] = 1.0       # dummy; make_xy needs the column, prediction ignores it


# ---------- view 1: model comparison ----------
def results_table():
    rows = []
    for name in ["baseline"] + MODEL_NAMES:
        row = {"Model": name}
        for sp in ["val", "test"]:
            m = saved["results"][sp][name]
            row[f"{sp} median APE %"] = m["median_APE_%"]
            row[f"{sp} within 10% (%)"] = m["within_10%"]
            row[f"{sp} RMSE $"] = m["RMSE_$"]
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_fig():
    names = ["baseline"] + MODEL_NAMES
    fig = go.Figure()
    for sp, label in [("val", "Validation (2009)"), ("test", "Test (2010)")]:
        fig.add_bar(name=label, x=names,
                    y=[saved["results"][sp][n]["median_APE_%"] for n in names])
    fig.update_layout(title="Typical error: median absolute % error (lower is better)",
                      yaxis_title="%", barmode="group")
    return fig


# ---------- view 2: predicted vs actual ----------
def pred_vs_actual(model):
    y, p = test_preds["y_true"], test_preds[model]
    fig = px.scatter(x=y, y=p, opacity=0.6,
                     labels={"x": "Actual sale price ($)", "y": "Predicted price ($)"},
                     title=f"{model}: predicted vs actual (test set, 2010)")
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect prediction",
                             line=dict(dash="dash")))
    return fig


# ---------- view 3: distributions ----------
DIST_CHOICES = ["SalePrice", "Log SalePrice", "Gr Liv Area", "Total SF", "House Age",
                "Lot Area", "Overall Qual", "Total Baths", "Neighborhood"]


def distribution(col):
    fig = px.histogram(df_all, x=col, color="Split", barmode="overlay",
                       opacity=0.6, histnorm="probability density",
                       title=f"Distribution of {col}, by split")
    if col == "Neighborhood":
        fig.update_xaxes(categoryorder="total descending")
    return fig

   


# ---------- view 4: agent form ----------
NEIGHBORHOODS = sorted(raw_train["Neighborhood"].unique())


def suggest_price(neigh, area, lot, year_built, qual, bsmt, garage, baths):
    row = dict(defaults)
    row.update({"Neighborhood": neigh, "Gr Liv Area": float(area), "Lot Area": float(lot),
                "Year Built": float(year_built), "Year Remod/Add": float(year_built),
                "Overall Qual": float(qual), "Total Bsmt SF": float(bsmt),
                "Garage Area": float(garage), "Full Bath": float(baths)})
    X, _ = make_xy(engineer(pd.DataFrame([row])))
    Xp = pre.transform(X).astype(np.float32)
    with torch.no_grad():
        preds = {
            "scikit-learn": sk_model.predict(Xp),
            "manual PyTorch": Xp @ w + b,
            "nn.Module": nn_model(torch.tensor(Xp)).numpy(),
        }
    return [[n, f"${float(np.exp(p[0])):,.0f}"] for n, p in preds.items()]


with gr.Blocks(title="Ames listing-price assistant") as demo:
    gr.Markdown("# Ames listing-price assistant\n"
                "Three versions of one linear model (scikit-learn, manual PyTorch, "
                "nn.Module), trained on 2006-2008 sales, validated on 2009, tested on 2010.")
    with gr.Tab("Model comparison"):
        gr.Dataframe(results_table(), label="Metrics (dollar scale)")
        gr.Plot(comparison_fig())
    with gr.Tab("Predicted vs actual"):
        dd = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
        plot = gr.Plot(pred_vs_actual(MODEL_NAMES[0]))
        dd.change(pred_vs_actual, dd, plot)
    with gr.Tab("Distributions"):
        dd2 = gr.Dropdown(DIST_CHOICES, value="SalePrice", label="Feature or target")
        plot2 = gr.Plot(distribution("SalePrice"))
        dd2.change(distribution, dd2, plot2)
    with gr.Tab("Price a house"):
        gr.Markdown("Enter what the agent knows. Every other feature is set to the "
                    "training-set typical value (median or most common).")
        neigh = gr.Dropdown(NEIGHBORHOODS, value=defaults["Neighborhood"], label="Neighborhood")
        area = gr.Number(value=defaults["Gr Liv Area"], label="Above-ground living area (sq ft)")
        lot = gr.Number(value=defaults["Lot Area"], label="Lot area (sq ft)")
        year = gr.Number(value=defaults["Year Built"], label="Year built")
        qual = gr.Slider(1, 10, value=defaults["Overall Qual"], step=1, label="Overall quality (1-10)")
        bsmt = gr.Number(value=defaults["Total Bsmt SF"], label="Basement area (sq ft)")
        garage = gr.Number(value=defaults["Garage Area"], label="Garage area (sq ft)")
        baths = gr.Number(value=defaults["Full Bath"], label="Full bathrooms")
        btn = gr.Button("Suggest asking price")
        out = gr.Dataframe(headers=["Model", "Suggested price"], label="Suggested asking price")
        btn.click(suggest_price, [neigh, area, lot, year, qual, bsmt, garage, baths], out)

demo.launch()
