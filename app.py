"""
app.py - Gradio dashboard for the Telco churn models.
Loads the artifacts train.py saved in models/. Nothing is trained here.
"""
from pathlib import Path
import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from model_def import LogisticRegressionModel

MODELS_DIR = Path("models")
MODEL_NAMES = ["sklearn", "manual_pytorch", "nn_pytorch"]

# ---------------------------------------------------------------
# 1. Load saved artifacts (inference only, no .fit anywhere)
# ---------------------------------------------------------------
config = json.loads((MODELS_DIR / "config.json").read_text())
run = json.loads((MODELS_DIR / "results.json").read_text())
COST_FP, COST_FN = config["cost_fp"], config["cost_fn"]

preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
sk_model = joblib.load(MODELS_DIR / "sklearn_model.joblib")
manual = torch.load(MODELS_DIR / "manual_pytorch.pt")
nn_model = LogisticRegressionModel(run["n_features"])
nn_model.load_state_dict(torch.load(MODELS_DIR / "nn_pytorch.pt"))
nn_model.eval()

test_df = pd.read_csv(MODELS_DIR / "test_set.csv")
y_test = test_df["Churn"].to_numpy()
X_test = preprocessor.transform(test_df.drop(columns=["Churn"]))
X_t = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    PROBS = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "manual_pytorch": torch.sigmoid(X_t @ manual["w"] + manual["b"]).numpy(),
        "nn_pytorch": torch.sigmoid(nn_model(X_t)).numpy(),
    }

coefs = pd.read_csv(MODELS_DIR / "coefficients.csv", index_col=0)
results_table = (pd.DataFrame(run["results"]).T
                 .reset_index().rename(columns={"index": "model"}))

# Full dataset, only for the distribution plots
raw = pd.read_csv("data/telco_churn.csv")
raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)
DIST_COLS = [c for c in raw.columns if c not in ("customerID", "Churn")]


# ---------------------------------------------------------------
# 2. Plot helpers
# ---------------------------------------------------------------
def calibration_plot():
    """Predicted vs actual: mean predicted churn vs real churn rate, in 10 bins."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="grey"),
                             name="perfect calibration"))
    for name in MODEL_NAMES:
        bins = pd.qcut(PROBS[name], 10, duplicates="drop")
        g = pd.DataFrame({"p": PROBS[name], "y": y_test, "bin": bins}) \
              .groupby("bin", observed=True).mean()
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers",
                                 name=name))
    fig.update_layout(title="Predicted vs actual churn rate (test set, 10 bins)",
                      xaxis_title="Mean predicted churn probability",
                      yaxis_title="Actual churn rate")
    return fig


def prob_distribution(model):
    """Predicted probabilities, split by what actually happened."""
    df = pd.DataFrame({"predicted probability": PROBS[model],
                       "actual": np.where(y_test == 1, "churned", "stayed")})
    return px.histogram(df, x="predicted probability", color="actual",
                        barmode="overlay", nbins=40, opacity=0.6,
                        title=f"{model}: predicted probability by actual outcome")


def coef_plot():
    """The three models' coefficients side by side: they should overlap."""
    long = coefs.reset_index().melt(id_vars="index", var_name="model",
                                    value_name="coefficient")
    fig = px.bar(long, x="coefficient", y="index", color="model",
                 barmode="group", orientation="h", height=750,
                 title="Learned coefficients (standardized features)")
    fig.update_layout(yaxis_title=None)
    return fig


def feature_plot(col):
    if not pd.api.types.is_numeric_dtype(raw[col]):
        df = (raw.groupby(col)["Churn"].value_counts().rename("count")
                 .reset_index())
        return px.bar(df, x=col, y="count", color="Churn", barmode="group",
                      title=f"{col} by churn")
    return px.histogram(raw, x=col, color="Churn", barmode="overlay",
                        nbins=40, opacity=0.6, title=f"{col} by churn")


def target_plot():
    counts = raw["Churn"].value_counts().reset_index()
    return px.bar(counts, x="Churn", y="count",
                  title="Target: churned vs stayed (full dataset)")


def threshold_view(model, threshold):
    p = PROBS[model]
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    cost = fp * COST_FP + fn * COST_FN

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted: stay", "Predicted: churn"],
        y=["Actual: stayed", "Actual: churned"],
        text=[[f"TN {tn}", f"FP {fp}"], [f"FN {fn}", f"TP {tp}"]],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix at threshold {threshold:.2f}",
                     yaxis=dict(autorange="reversed"))

    grid = np.linspace(0.01, 0.99, 99)
    costs = [((p >= t) & (y_test == 0)).sum() * COST_FP
             + ((p < t) & (y_test == 1)).sum() * COST_FN for t in grid]
    curve = px.line(x=grid, y=costs, labels={"x": "threshold", "y": "cost (EUR)"},
                    title="Business cost vs threshold (test set)")
    curve.add_vline(x=threshold, line_dash="dash")

    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    text = (f"### Business cost: EUR {cost:,}\n"
            f"Baseline (contact nobody): EUR {config['baseline_cost']:,}  \n"
            f"Customers contacted: **{tp + fp}** of {len(p)}  \n"
            f"Churners caught (recall): **{recall:.0%}**  \n"
            f"Contacts that were real churners (precision): **{precision:.0%}**  \n"
            f"Costs: wasted offer EUR {COST_FP}, missed churner EUR {COST_FN}")
    return text, cm, curve


# ---------------------------------------------------------------
# 3. Layout
# ---------------------------------------------------------------
default_t = run["results"]["sklearn"]["threshold"]

with gr.Blocks(title="Telco churn dashboard") as demo:
    gr.Markdown("# Telco churn: one logistic regression, three implementations\n"
                "All models were trained by `train.py` and are only loaded here.")

    with gr.Tab("Model comparison"):
        gr.Dataframe(value=results_table, label="Test-set results")
        gr.Plot(value=calibration_plot())
        model_dd = gr.Dropdown(MODEL_NAMES, value="sklearn", label="Model")
        dist_plot = gr.Plot(value=prob_distribution("sklearn"))
        model_dd.change(prob_distribution, model_dd, dist_plot)
        gr.Plot(value=coef_plot())

    with gr.Tab("Data distributions"):
        gr.Plot(value=target_plot())
        feat_dd = gr.Dropdown(DIST_COLS, value="tenure", label="Feature")
        feat_plot = gr.Plot(value=feature_plot("tenure"))
        feat_dd.change(feature_plot, feat_dd, feat_plot)

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            t_model = gr.Dropdown(MODEL_NAMES, value="sklearn", label="Model")
            t_slider = gr.Slider(0.01, 0.99, value=default_t, step=0.01,
                                 label="Decision threshold")
        cost_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        for comp in (t_model, t_slider):
            comp.change(threshold_view, [t_model, t_slider],
                        [cost_md, cm_plot, cost_plot])
        demo.load(threshold_view, [t_model, t_slider],
                  [cost_md, cm_plot, cost_plot])

demo.launch()
