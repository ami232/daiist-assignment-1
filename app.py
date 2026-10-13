"""
Assignment 1 — Gradio dashboard for the dropout early-warning model.

Only loads what train.py saved (preprocessor, three models, test/train sets,
metrics). Nothing is fitted here: the models are used for inference only.
"""

import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_curve

from pipeline import (
    ARTIFACTS_DIR, BINARY, CATEGORICAL, FEATURES, MODELS_DIR, NUMERIC,
    LogisticRegressionNet, business_cost,
)

LABELS = {
    "sklearn": "scikit-learn",
    "torch_manual": "PyTorch manual loop",
    "torch_module": "PyTorch nn.Module + optim",
    "baseline": "Naive baseline (train dropout rate)",
}
MODELS = ["sklearn", "torch_manual", "torch_module"]


# ---------------------------------------------------------------- load artifacts
def read_set(name):
    df = pd.read_csv(ARTIFACTS_DIR / name)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    return df


meta = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
loss_curves = json.loads((ARTIFACTS_DIR / "loss_curves.json").read_text())
coefs = pd.read_csv(ARTIFACTS_DIR / "coefficients.csv")
train_df, test_df = read_set("train_set.csv"), read_set("test_set.csv")
y_test = test_df["dropout"].to_numpy()

preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
sk_model = joblib.load(MODELS_DIR / "sklearn_logreg.joblib")
manual_params = torch.load(MODELS_DIR / "torch_manual.pt")
ckpt = torch.load(MODELS_DIR / "torch_module.pt")
module_model = LogisticRegressionNet(ckpt["n_features"])
module_model.load_state_dict(ckpt["state_dict"])
module_model.eval()

X_test = preprocessor.transform(test_df[FEATURES])
X_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    PROBAS = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "torch_manual": torch.sigmoid(X_t @ manual_params["w"] + manual_params["b"]).numpy(),
        "torch_module": torch.sigmoid(module_model(X_t)).numpy(),
        "baseline": np.full(len(y_test), meta["baseline_prevalence"]),
    }


# ---------------------------------------------------------------- tab 1: comparison
def metrics_table():
    cols = ["roc_auc", "pr_auc", "log_loss", "threshold", "recall", "precision",
            "flagged_share", "cost_eur", "cost_per_student_eur"]
    t = pd.DataFrame(meta["results_test"]).T[cols].astype(float).round(3)
    t.index = [LABELS.get(i, i.replace("_", " ")) for i in t.index]
    return t.reset_index(names="model")


def prediction_vs_actual_plot():
    """Reliability plot: mean predicted probability vs observed dropout rate, by decile."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="grey")))
    for m in MODELS:
        d = pd.DataFrame({"p": PROBAS[m], "y": y_test})
        d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(x=g["pred"], y=g["actual"], mode="lines+markers", name=LABELS[m],
                                 text=g["n"], hovertemplate="pred %{x:.2f}<br>actual %{y:.2f}<br>n=%{text}"))
    fig.update_layout(title="Predicted vs actual dropout rate (test cohorts, deciles)",
                      xaxis_title="mean predicted probability", yaxis_title="actual dropout rate")
    return fig


def proba_by_outcome_plot():
    d = pd.concat([pd.DataFrame({"model": LABELS[m], "probability": PROBAS[m],
                                 "actual": np.where(y_test == 1, "Dropout", "Graduate")}) for m in MODELS])
    return px.box(d, x="model", y="probability", color="actual",
                  title="Predicted probability by actual outcome (test cohorts)")


def agreement_plot():
    d = pd.DataFrame({"sklearn": PROBAS["sklearn"], "PyTorch manual": PROBAS["torch_manual"],
                      "PyTorch nn.Module": PROBAS["torch_module"]})
    d = d.melt(id_vars="sklearn", var_name="torch version", value_name="torch probability")
    fig = px.scatter(d, x="sklearn", y="torch probability", color="torch version", opacity=0.5,
                     title="Do the three implementations agree? (each dot = one test student)")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    return fig


def roc_plot():
    fig = go.Figure()
    for m in MODELS + ["baseline"]:
        fpr, tpr, _ = roc_curve(y_test, PROBAS[m])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=LABELS[m]))
    fig.update_layout(title="ROC curves (test cohorts)", xaxis_title="false positive rate",
                      yaxis_title="true positive rate")
    return fig


def coef_plot():
    d = coefs.melt(id_vars="feature", var_name="model", value_name="coefficient")
    d["model"] = d["model"].map(LABELS)
    order = coefs.reindex(coefs["sklearn"].abs().sort_values().index)["feature"]
    fig = px.bar(d, y="feature", x="coefficient", color="model", barmode="group", orientation="h",
                 category_orders={"feature": list(order)}, height=1100,
                 title="Coefficients (standardised numeric features, log-odds)")
    return fig


def loss_plot():
    fig = go.Figure()
    for m in ("torch_manual", "torch_module"):
        fig.add_trace(go.Scatter(y=loss_curves[m], mode="lines", name=LABELS[m]))
    fig.add_hline(y=loss_curves["sklearn_objective"], line_dash="dash",
                  annotation_text="sklearn optimum (same objective)")
    fig.update_layout(title="Training objective per epoch (BCE + L2)", xaxis_title="epoch",
                      yaxis_title="loss", yaxis_type="log")
    return fig


# ---------------------------------------------------------------- tab 2: distributions
def feature_plot(feature, split):
    df = train_df if split == "train" else test_df
    d = df.assign(outcome=np.where(df["dropout"] == 1, "Dropout", "Graduate"))
    if feature in NUMERIC:
        return px.histogram(d, x=feature, color="outcome", barmode="overlay", histnorm="percent",
                            nbins=40, title=f"{feature} by outcome ({split} cohorts)")
    rate = d.groupby(feature)["dropout"].agg(["mean", "size"]).reset_index()
    rate.columns = [feature, "dropout_rate", "students"]
    return px.bar(rate, x=feature, y="dropout_rate", hover_data=["students"], text="students",
                  title=f"Dropout rate by {feature} ({split} cohorts, bar label = students)")


def target_plot():
    d = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")])
    g = d.groupby(["cohort", "split"])["dropout"].mean().reset_index()
    fig = px.bar(g, x="cohort", y="dropout", color="split",
                 title="Target: dropout rate per enrollment cohort (val cohorts 2014-2015 not shown)")
    fig.update_yaxes(title="dropout rate")
    return fig


# ---------------------------------------------------------------- tab 3: threshold & cost
def threshold_view(model, threshold, cost_fn, cost_fp):
    p = PROBAS[model]
    c = business_cost(y_test, p, threshold, cost_fn, cost_fp)
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       x=["predicted: stays", "predicted: drops out"],
                       y=["actual: graduates", "actual: drops out"],
                       title=f"Confusion matrix @ threshold {threshold:.2f}")
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [business_cost(y_test, p, t, cost_fn, cost_fp)["cost"] for t in grid]
    everybody = (y_test == 0).sum() * cost_fp
    fig_cost = px.line(x=grid, y=costs, labels={"x": "threshold", "y": "total cost (€)"},
                       title="Business cost vs threshold (test cohorts)")
    fig_cost.add_vline(x=threshold, line_dash="dot", annotation_text="current")
    fig_cost.add_hline(y=everybody, line_dash="dash", annotation_text="mentor everybody")
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    best_t = grid[int(np.argmin(costs))]
    summary = (
        f"### Total cost: €{c['cost']:,.0f}  (€{c['cost'] / len(y_test):,.0f} per student)\n"
        f"- FN {c['fn']} × €{cost_fn:,.0f} + FP {c['fp']} × €{cost_fp:,.0f}\n"
        f"- Recall {recall:.1%} · precision {precision:.1%} · "
        f"students flagged {(c['tp'] + c['fp']) / len(y_test):.1%}\n"
        f"- Reference: mentor everybody = €{everybody:,.0f}; mentor nobody = "
        f"€{(y_test == 1).sum() * cost_fn:,.0f}\n"
        f"- Threshold chosen on validation cohorts: **{meta['results_test'][model]['threshold']:.2f}** "
        f"(cost-minimising on test, for comparison only: {best_t:.2f}; "
        f"theoretical FP/(FP+FN) = {cost_fp / (cost_fp + cost_fn):.2f})"
    )
    return fig_cm, summary, fig_cost


# ---------------------------------------------------------------- layout
with gr.Blocks(title="Dropout early-warning — 3 implementations") as demo:
    gr.Markdown(
        "# Student dropout early-warning model\n"
        "Logistic regression trained three ways on cohorts 2008-2013, threshold tuned on "
        "2014-2015, evaluated here on **2016-2017**. Uses only information available at the end "
        f"of the 1st semester. C = {meta['best_C']}, learning rate = {meta['best_lr']}, "
        f"{meta['epochs']} full-batch epochs. Models are loaded from disk, nothing is retrained."
    )
    with gr.Tab("Model comparison"):
        gr.Dataframe(metrics_table(), label="Test-cohort metrics (threshold chosen on validation)")
        gr.JSON(meta["agreement"], label="Agreement with scikit-learn")
        gr.Plot(prediction_vs_actual_plot())
        gr.Plot(proba_by_outcome_plot())
        gr.Plot(agreement_plot())
        gr.Plot(roc_plot())
        gr.Plot(loss_plot())
        gr.Plot(coef_plot())

    with gr.Tab("Distributions"):
        gr.Plot(target_plot())
        with gr.Row():
            feat = gr.Dropdown(NUMERIC + BINARY + CATEGORICAL, value="s1_approval_rate", label="feature")
            split = gr.Radio(["train", "test"], value="train", label="cohorts")
        feat_plot = gr.Plot(feature_plot("s1_approval_rate", "train"))
        feat.change(feature_plot, [feat, split], feat_plot)
        split.change(feature_plot, [feat, split], feat_plot)

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            model_dd = gr.Dropdown([(LABELS[m], m) for m in MODELS + ["baseline"]], value="sklearn", label="model")
            thr = gr.Slider(0.01, 0.99, value=meta["results_test"]["sklearn"]["threshold"], step=0.01,
                            label="decision threshold")
        with gr.Row():
            cfn = gr.Number(value=meta["cost_fn"], label="cost of a missed dropout (FN, €)")
            cfp = gr.Number(value=meta["cost_fp"], label="cost of unneeded mentoring (FP, €)")
        summary_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        inputs, outputs = [model_dd, thr, cfn, cfp], [cm_plot, summary_md, cost_plot]
        for comp in inputs:
            comp.change(threshold_view, inputs, outputs)
        demo.load(threshold_view, inputs, outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=not os.environ.get("CI"))
