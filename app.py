"""
Assignment 1 — Gradio dashboard (Telco churn, logistic regression x3).

    uv run python main.py app

Loads the artifacts written by train.py (preprocessor, scikit-learn model,
manual-PyTorch weights, nn.Module state_dict, split assignment) and scores
the held-out test set with them. Nothing is trained here.
"""

import json
import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from features import (ARTIFACTS, CHURN_COST, OFFER_COST, business_cost,
                      engineer, load_raw, target)
from train import LogisticModel, predict_manual, predict_standard

MODEL_LABELS = {
    "sklearn": "scikit-learn",
    "manual_torch": "Manual PyTorch loop",
    "standard_torch": "nn.Module + torch.optim",
}
BASELINE_LABELS = {
    "baseline": "Naive baseline (train churn rate)",
    "rule_month_to_month": "Rule: contact all month-to-month",
}
MODEL_CHOICES = [(label, key) for key, label in MODEL_LABELS.items()]
COLORS = {"sklearn": "#2a6fdb", "manual_torch": "#e07b00", "standard_torch": "#2a9d55"}

if not (ARTIFACTS / "metrics.json").exists():
    raise SystemExit("artifacts/ not found: run `uv run python main.py train` first.")

# --------------------------------------------------------------------------
# Load artifacts (no training)
# --------------------------------------------------------------------------
with open(ARTIFACTS / "preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
with open(ARTIFACTS / "sklearn_model.pkl", "rb") as f:
    sk_model = pickle.load(f)
manual_params = torch.load(ARTIFACTS / "manual_torch.pt")
summary = json.loads((ARTIFACTS / "metrics.json").read_text())
std_model = LogisticModel(summary["n_features"])
std_model.load_state_dict(torch.load(ARTIFACTS / "standard_torch.pt"))
coefs = pd.read_csv(ARTIFACTS / "coefficients.csv")

raw = load_raw()
split = pd.read_csv(ARTIFACTS / "split.csv")
assert (split["customerID"] == raw["customerID"]).all()
feats = engineer(raw)
y_all = target(raw)
test_mask = (split["split"] == "test").to_numpy()

X_test = preprocessor.transform(feats[test_mask]).astype(np.float64)
y_test = y_all[test_mask]
PROBS = {
    "sklearn": sk_model.predict_proba(X_test)[:, 1],
    "manual_torch": predict_manual(manual_params, X_test),
    "standard_torch": predict_standard(std_model, X_test),
}
VAL_THRESHOLDS = {k: summary["test_metrics"][k]["threshold"] for k in MODEL_LABELS}

# Columns available for distribution plots: raw + engineered.
dist_df = pd.concat([raw.drop(columns=["customerID"]), feats.add_prefix("fe_")], axis=1)
dist_df["TotalCharges"] = pd.to_numeric(dist_df["TotalCharges"], errors="coerce").fillna(0)
dist_df["split"] = split["split"]
DIST_COLUMNS = [c for c in dist_df.columns if c not in ("Churn", "split")]


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def comparison_table():
    rows = []
    for key in list(MODEL_LABELS) + ["baseline", "rule_month_to_month"]:
        m = summary["test_metrics"][key]
        rows.append({
            "Model": {**MODEL_LABELS, **BASELINE_LABELS}[key],
            "ROC AUC": round(m["roc_auc"], 4), "PR AUC": round(m["pr_auc"], 4),
            "Log loss": round(m["log_loss"], 4), "Threshold (val)": m["threshold"],
            "Precision": round(m["precision"], 3), "Recall": round(m["recall"], 3),
            "Offers sent": m["offers_sent"], "Cost (€)": int(m["business_cost"]),
        })
    return pd.DataFrame(rows)


def pred_vs_actual_plot(n_bins):
    """Calibration-style plot: mean predicted probability vs observed churn rate."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(color="grey", dash="dash")))
    for key, p in PROBS.items():
        bins = pd.qcut(p, int(n_bins), duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y_test, "b": bins}).groupby("b", observed=True)
        fig.add_trace(go.Scatter(x=g["p"].mean(), y=g["y"].mean(), mode="lines+markers",
                                 name=MODEL_LABELS[key], line=dict(color=COLORS[key])))
    fig.update_layout(title="Predicted churn probability vs actual churn rate (test set, quantile bins)",
                      xaxis_title="Mean predicted probability", yaxis_title="Observed churn rate",
                      height=450)
    return fig


def prob_by_class_plot(model_key):
    df = pd.DataFrame({"Predicted probability": PROBS[model_key],
                       "Actual": np.where(y_test == 1, "Churned", "Stayed")})
    fig = px.histogram(df, x="Predicted probability", color="Actual", nbins=40, barmode="overlay",
                       opacity=0.65, color_discrete_map={"Churned": "#d1495b", "Stayed": "#2a6fdb"})
    fig.update_layout(title=f"{MODEL_LABELS[model_key]}: predicted probability by actual outcome",
                      height=400)
    return fig


def model_vs_model_plot(a, b):
    fig = px.scatter(x=PROBS[a], y=PROBS[b], color=np.where(y_test == 1, "Churned", "Stayed"),
                     opacity=0.5, labels={"x": MODEL_LABELS[a], "y": MODEL_LABELS[b], "color": "Actual"},
                     color_discrete_map={"Churned": "#d1495b", "Stayed": "#2a6fdb"})
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="grey", dash="dash"),
                             showlegend=False))
    diff = np.abs(PROBS[a] - PROBS[b])
    fig.update_layout(title=f"Per-customer probabilities (max |diff| = {diff.max():.4f}, mean = {diff.mean():.4f})",
                      height=450)
    return fig


def roc_plot():
    fig = go.Figure()
    for key, p in PROBS.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=COLORS[key]),
                                 name=f"{MODEL_LABELS[key]} (AUC {roc_auc_score(y_test, p):.4f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline (AUC 0.5)",
                             line=dict(color="grey", dash="dash")))
    fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", height=450)
    return fig


def coef_plot():
    df = coefs.sort_values("sklearn")
    fig = go.Figure()
    for key in MODEL_LABELS:
        fig.add_trace(go.Bar(y=df["feature"], x=df[key], name=MODEL_LABELS[key], orientation="h",
                             marker_color=COLORS[key]))
    fig.update_layout(barmode="group", height=800, title="Learned coefficients (standardised inputs)",
                      xaxis_title="Coefficient (log-odds)")
    return fig


def distribution_plot(column, which_split):
    df = dist_df if which_split == "all" else dist_df[dist_df["split"] == which_split]
    if df[column].dtype.kind in "if" and df[column].nunique() > 10:
        fig = px.histogram(df, x=column, color="Churn", nbins=40, barmode="overlay", opacity=0.65,
                           color_discrete_map={"Yes": "#d1495b", "No": "#2a6fdb"})
        title = f"Distribution of {column} by churn"
    else:
        rate = df.groupby(column)["Churn"].apply(lambda s: (s == "Yes").mean()).reset_index(name="churn_rate")
        count = df[column].value_counts().rename("customers").reset_index()
        rate = rate.merge(count, on=column)
        fig = px.bar(rate, x=column, y="churn_rate", text="customers",
                     labels={"churn_rate": "Churn rate"}, color_discrete_sequence=["#2a6fdb"])
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        title = f"Churn rate by {column}"
    fig.update_layout(title=f"{title} ({which_split})", height=450)
    return fig


def target_plot():
    df = dist_df.groupby("split")["Churn"].apply(lambda s: (s == "Yes").mean()).reset_index(name="churn_rate")
    fig = px.bar(df, x="split", y="churn_rate", text_auto=".3f", color_discrete_sequence=["#d1495b"])
    fig.update_layout(title="Target: churn rate per split (stratified)", height=350)
    return fig


def threshold_view(model_key, threshold, offer_cost, churn_cost):
    p = PROBS[model_key]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Predicted: stay", "Predicted: churn"], y=["Actual: stay", "Actual: churn"])
    fig.update_layout(title=f"Confusion matrix — {MODEL_LABELS[model_key]} @ threshold {threshold:.2f}",
                      height=400, coloraxis_showscale=False)

    cost = business_cost(y_test, pred, offer_cost, churn_cost)
    base = min(business_cost(y_test, np.ones_like(y_test), offer_cost, churn_cost),
               business_cost(y_test, np.zeros_like(y_test), offer_cost, churn_cost))
    # Cost curve over all thresholds for context.
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    curve = [business_cost(y_test, (p >= t).astype(int), offer_cost, churn_cost) for t in grid]
    cfig = go.Figure(go.Scatter(x=grid, y=curve, mode="lines", line=dict(color=COLORS[model_key]),
                                name="Model cost"))
    cfig.add_hline(y=base, line_dash="dash", line_color="grey", annotation_text="Best naive policy")
    cfig.add_vline(x=threshold, line_color="#d1495b", annotation_text="Current")
    cfig.update_layout(title="Business cost on the test set vs threshold", xaxis_title="Threshold",
                       yaxis_title="Total cost (€)", height=350)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    md = (
        f"### Business cost: **€{cost:,.0f}**  (€{cost / len(y_test):.2f} per customer)\n"
        f"- Offers sent (TP+FP): **{tp + fp}** × €{offer_cost:.0f} = €{(tp + fp) * offer_cost:,.0f}\n"
        f"- Churners missed (FN): **{fn}** × €{churn_cost:.0f} = €{fn * churn_cost:,.0f}\n"
        f"- Best naive policy (contact everyone or no one): €{base:,.0f} → "
        f"savings **€{base - cost:,.0f}** ({(base - cost) / base:.1%})\n"
        f"- Precision {precision:.3f} · Recall {recall:.3f} · "
        f"Threshold chosen on validation: {VAL_THRESHOLDS[model_key]:.2f}\n"
        f"- Break-even threshold from costs alone (offer / churn cost): {offer_cost / churn_cost:.2f}"
    )
    return fig, md, cfig


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
with gr.Blocks(title="Telco churn — three logistic regressions") as demo:
    gr.Markdown(
        "# Telco customer churn — one logistic regression, three implementations\n"
        f"Test set: **{len(y_test)} customers** (churn rate {y_test.mean():.1%}). "
        "All models are loaded from `artifacts/`; nothing is retrained. "
        f"Business framing: each retention offer costs €{OFFER_COST:.0f}; "
        f"each missed churner costs €{CHURN_COST:.0f}."
    )

    with gr.Tab("Model comparison"):
        gr.Dataframe(comparison_table(), label="Test-set metrics (threshold picked on validation)")
        n_bins = gr.Slider(5, 20, value=10, step=1, label="Number of quantile bins")
        pva = gr.Plot(pred_vs_actual_plot(10))
        n_bins.change(pred_vs_actual_plot, n_bins, pva)
        with gr.Row():
            a = gr.Dropdown(MODEL_CHOICES, value="sklearn", label="Model A")
            b = gr.Dropdown(MODEL_CHOICES, value="standard_torch", label="Model B")
        mvm = gr.Plot(model_vs_model_plot("sklearn", "standard_torch"))
        a.change(model_vs_model_plot, [a, b], mvm)
        b.change(model_vs_model_plot, [a, b], mvm)
        gr.Plot(roc_plot())
        pbc_model = gr.Dropdown(MODEL_CHOICES, value="sklearn", label="Model")
        pbc = gr.Plot(prob_by_class_plot("sklearn"))
        pbc_model.change(prob_by_class_plot, pbc_model, pbc)

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            t_model = gr.Dropdown(MODEL_CHOICES, value="sklearn", label="Model")
            t_slider = gr.Slider(0.01, 0.99, value=VAL_THRESHOLDS["sklearn"], step=0.01,
                                 label="Decision threshold")
        with gr.Row():
            offer = gr.Number(value=OFFER_COST, label="Cost per retention offer (€)")
            churn = gr.Number(value=CHURN_COST, label="Cost per missed churner (€)")
        init = threshold_view("sklearn", VAL_THRESHOLDS["sklearn"], OFFER_COST, CHURN_COST)
        with gr.Row():
            cm_plot = gr.Plot(init[0])
            cost_md = gr.Markdown(init[1])
        cost_curve = gr.Plot(init[2])
        inputs = [t_model, t_slider, offer, churn]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, cost_md, cost_curve])

    with gr.Tab("Distributions"):
        gr.Plot(target_plot())
        with gr.Row():
            col = gr.Dropdown(DIST_COLUMNS, value="tenure", label="Feature (fe_ = engineered)")
            which = gr.Radio(["all", "train", "val", "test"], value="all", label="Split")
        dplot = gr.Plot(distribution_plot("tenure", "all"))
        col.change(distribution_plot, [col, which], dplot)
        which.change(distribution_plot, [col, which], dplot)

    with gr.Tab("Coefficients"):
        gr.Markdown(
            "Coefficients from the three implementations. They should coincide; where they do not, "
            "the features involved are strongly correlated (e.g. MonthlyCharges vs. the services "
            "that make up the bill, or the tenure / log-tenure / tenure-band family), so the loss is "
            "almost flat along those directions and mini-batch Adam stops at a slightly different point."
        )
        gr.Plot(coef_plot())
        gr.Dataframe(coefs.round(4))


if __name__ == "__main__":
    demo.launch(inbrowser=not os.environ.get("CI"))
