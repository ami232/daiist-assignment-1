#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 — Gradio dashboard
# 
# You're using a notebook for this stage. If you'd rather use a script,
# write `app.py` instead and delete this file — `main.py` refuses to run if
# it finds both `app.py` and `app.ipynb`, so exactly one of them must exist.
# 
# Replace this cell and everything below it with your own code. When run via
# `uv run python main.py app`, this notebook must build and launch a Gradio
# app (a `demo` that calls `.launch()`) using only what your training stage
# already produced — it must never retrain anything itself. At minimum, the
# dashboard must let you:
# 
# - Compare your three trained models via a prediction-vs-actual plot.
# - See feature and/or target distributions.
# - For classification: move a decision-threshold slider and watch the
#   confusion matrix, and a business-cost number tied to your `REPORT.md`'s
#   framing, change with it.
# 
# How you structure the code beyond that — cell layout, function
# boundaries, naming — is your call to make and be able to defend.

# In[1]:


import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

preds = pd.read_csv("models/test_predictions.csv")
clean = pd.read_csv("models/clean_data.csv")
results = pd.read_csv("models/comparison_table.csv", index_col=0)

MODELS = {"scikit-learn": "sklearn", "Manual PyTorch": "manual", "nn.Module": "nn_module"}
y_true = preds["actual"].to_numpy()

clean["Churned"] = (clean["Attrition_Flag"] == "Attrited Customer").astype(int)
num_cols = [c for c in clean.select_dtypes("number").columns if c not in ("CLIENTNUM", "Churned")]

print(preds.shape, clean.shape)
print(results)


# In[2]:


def metrics_plot():
    fig = go.Figure()
    for name in results.index:
        fig.add_bar(x=list(results.columns), y=results.loc[name].tolist(), name=name)
    fig.update_layout(barmode="group", title="Test-set metrics at threshold 0.5")
    return fig

def pred_vs_actual(n_bins=10):
    fig = go.Figure()
    for label, col in MODELS.items():
        p = preds[col]
        bins = pd.qcut(p, q=n_bins, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y_true}).groupby(bins, observed=True).mean()
        fig.add_scatter(x=g["p"], y=g["y"], mode="lines+markers", name=label)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                    line=dict(dash="dash", color="gray"))
    fig.update_layout(title="Predicted churn probability vs actual churn rate (test set)",
                      xaxis_title="mean predicted probability", yaxis_title="actual churn rate")
    return fig

pred_vs_actual()


# In[3]:


def target_plot():
    counts = clean["Churned"].map({0: "Stayed", 1: "Churned"}).value_counts()
    fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.values.tolist()))
    fig.update_layout(title="Target: customers who stayed vs churned", yaxis_title="customers")
    return fig

def feature_plot(col):
    fig = go.Figure()
    for flag, name in [(0, "Stayed"), (1, "Churned")]:
        fig.add_histogram(x=clean.loc[clean["Churned"] == flag, col], name=name,
                          opacity=0.6, histnorm="probability")
    fig.update_layout(barmode="overlay", title=f"{col} by outcome",
                      xaxis_title=col, yaxis_title="share of group")
    return fig

target_plot()


# In[4]:


def threshold_view(model_label, threshold, cost_miss, cost_offer):
    cost_miss, cost_offer = cost_miss or 0, cost_offer or 0
    p = preds[MODELS[model_label]].to_numpy()
    pred = (p >= threshold).astype(int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted stay", "Predicted churn"],
        y=["Actually stayed", "Actually churned"],
        text=[[tn, fp], [fn, tp]], texttemplate="%{text}",
        colorscale="Blues", showscale=False))
    cm.update_layout(title="Confusion matrix (test set)")
    cm.update_yaxes(autorange="reversed")

    ts = np.linspace(0.01, 0.99, 99)
    costs = [int(((p < t) & (y_true == 1)).sum()) * cost_miss +
             int(((p >= t) & (y_true == 0)).sum()) * cost_offer for t in ts]
    curve = go.Figure(go.Scatter(x=ts, y=costs, mode="lines"))
    curve.add_vline(x=threshold, line_dash="dash")
    curve.update_layout(title="Total cost by threshold", xaxis_title="threshold", yaxis_title="cost")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    total = fn * cost_miss + fp * cost_offer
    no_model = (fn + tp) * cost_miss
    text = (f"**Threshold {threshold:.2f}**: precision {precision:.3f}, recall {recall:.3f}\n\n"
            f"Missed churners: {fn} | Wasted offers: {fp}\n\n"
            f"**Total cost: {total:,.0f}** (doing nothing would cost {no_model:,.0f})")
    return cm, curve, text

cm, curve, text = threshold_view("scikit-learn", 0.5, 500, 50)
print(text)
cm


# In[5]:


with gr.Blocks(title="Customer churn dashboard") as demo:
    gr.Markdown("# Customer churn dashboard\nAll results come from the models saved during training.")

    with gr.Tab("Compare models"):
        gr.Plot(value=pred_vs_actual())
        gr.Plot(value=metrics_plot())
        gr.Dataframe(value=results.reset_index().rename(columns={"index": "model"}))

    with gr.Tab("Distributions"):
        gr.Plot(value=target_plot())
        first = "Total_Trans_Ct" if "Total_Trans_Ct" in num_cols else num_cols[0]
        feat = gr.Dropdown(num_cols, value=first, label="Feature")
        feat_plot = gr.Plot(value=feature_plot(first))
        feat.change(feature_plot, feat, feat_plot)

    with gr.Tab("Threshold and cost"):
        with gr.Row():
            model_dd = gr.Dropdown(list(MODELS), value="scikit-learn", label="Model")
            thr = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Decision threshold")
        with gr.Row():
            c_miss = gr.Number(value=500, label="Cost of a missed churner")
            c_offer = gr.Number(value=50, label="Cost of a retention offer")
        summary = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        controls = [model_dd, thr, c_miss, c_offer]
        outputs = [cm_plot, cost_plot, summary]
        for c in controls:
            c.change(threshold_view, controls, outputs)
        demo.load(threshold_view, controls, outputs)

demo.launch()

