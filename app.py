"""Assignment 1 - Gradio dashboard. Loads the saved artifacts; never retrains."""
import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ART = "artifacts"
MODELS = {"scikit-learn": "sklearn", "PyTorch manual": "manual_torch", "PyTorch nn.Module": "nn_torch",
          "Baseline (majority class)": "baseline"}

data = np.load(f"{ART}/data.npz")
meta = json.load(open(f"{ART}/meta.json"))
y_val, y_te = data["y_val"], data["y_te"]
raw = pd.read_csv("data/telco_churn.csv")
raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)


def confusion(y, p, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def total_cost(y, p, thr, c_fn, c_fp):
    tp, fp, fn, tn = confusion(y, p, thr)
    return fn * c_fn + fp * c_fp


# ---------- Tab 1: model comparison ----------
def comparison_table():
    df = pd.DataFrame(meta["metrics"]).T.round(4)
    df.insert(0, "model", [k for k in df.index])
    return df.reset_index(drop=True)


def pred_vs_actual(model_name):
    p = data[f"p_test_{MODELS[model_name]}"]
    df = pd.DataFrame({"predicted churn probability": p,
                       "actual": np.where(y_te == 1, "Churned", "Stayed")})
    fig = px.histogram(df, x="predicted churn probability", color="actual", nbins=40, barmode="overlay",
                       opacity=0.6, title=f"Predicted probability vs actual outcome (test set) - {model_name}")
    return fig


def calibration_plot():
    fig = go.Figure()
    for name, key in MODELS.items():
        if key == "baseline":
            continue
        p = data[f"p_test_{key}"]
        bins = pd.qcut(p, 8, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y_te, "bin": bins}).groupby("bin", observed=True).mean()
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash")))
    fig.update_layout(title="Mean predicted probability vs actual churn rate (test set)",
                      xaxis_title="mean predicted probability", yaxis_title="actual churn rate")
    return fig


# ---------- Tab 2: distributions ----------
def target_dist():
    return px.histogram(raw, x="Churn", color="Churn", title="Target distribution (Churn)")


def feature_dist(feature):
    if raw[feature].dtype == object:
        return px.histogram(raw, x=feature, color="Churn", barmode="group", title=f"{feature} by churn")
    return px.histogram(raw, x=feature, color="Churn", barmode="overlay", opacity=0.6, nbins=40,
                        title=f"{feature} by churn")


FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]


# ---------- Tab 3: threshold, confusion matrix and business cost ----------
def threshold_view(model_name, thr, c_fn, c_fp):
    p = data[f"p_test_{MODELS[model_name]}"]
    tp, fp, fn, tn = confusion(y_te, p, thr)
    cm = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Pred: stay", "Pred: churn"], y=["Actual: stay", "Actual: churn"],
                              text=[[tn, fp], [fn, tp]], texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(title="Confusion matrix (test set)", yaxis_autorange="reversed")
    cost = fn * c_fn + fp * c_fp
    # cost curve over all thresholds + optimal threshold chosen on the validation set
    grid = np.linspace(0.01, 0.99, 99)
    p_val = data[f"p_val_{MODELS[model_name]}"]
    best_thr = float(grid[np.argmin([total_cost(y_val, p_val, t, c_fn, c_fp) for t in grid])])
    curve = go.Figure(go.Scatter(x=grid, y=[total_cost(y_te, p, t, c_fn, c_fp) for t in grid], mode="lines"))
    curve.add_vline(x=thr, line_dash="dot", annotation_text="current")
    curve.update_layout(title="Total business cost vs threshold (test set)", xaxis_title="threshold", yaxis_title="cost (EUR)")
    base_cost = int(y_te.sum()) * c_fn  # baseline: never flag anyone
    summary = (f"**Total cost at threshold {thr:.2f}: {cost:,.0f} EUR** "
               f"({fn} missed churners x {c_fn:,.0f} + {fp} unnecessary offers x {c_fp:,.0f}).  \n"
               f"Doing nothing (never flagging) would cost {base_cost:,.0f} EUR.  \n"
               f"Cost-minimising threshold chosen on the validation set: **{best_thr:.2f}**.")
    return cm, curve, summary


with gr.Blocks(title="Telco churn - model dashboard") as demo:
    gr.Markdown("# Telco churn: three implementations of one logistic regression")
    with gr.Tab("Model comparison"):
        gr.Dataframe(comparison_table(), label="Test-set metrics (threshold 0.5 for accuracy/F1)")
        m1 = gr.Dropdown(list(MODELS), value="scikit-learn", label="Model")
        plot1 = gr.Plot(pred_vs_actual("scikit-learn"))
        m1.change(pred_vs_actual, m1, plot1)
        gr.Plot(calibration_plot())
    with gr.Tab("Distributions"):
        gr.Plot(target_dist())
        f = gr.Dropdown(FEATURES, value="tenure", label="Feature")
        plot2 = gr.Plot(feature_dist("tenure"))
        f.change(feature_dist, f, plot2)
    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            m3 = gr.Dropdown([k for k in MODELS if "Baseline" not in k], value="scikit-learn", label="Model")
            thr = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Decision threshold")
        with gr.Row():
            c_fn = gr.Number(value=500, label="Cost of a missed churner (EUR)")
            c_fp = gr.Number(value=50, label="Cost of an unnecessary offer (EUR)")
        txt = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        ins = [m3, thr, c_fn, c_fp]
        for c in ins:
            c.change(threshold_view, ins, [cm_plot, curve_plot, txt])
        demo.load(threshold_view, ins, [cm_plot, curve_plot, txt])

demo.launch()