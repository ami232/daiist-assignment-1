"""
Assignment 1 — Gradio dashboard.

Loads ONLY what train.py saved in artifacts/ (models, test data, settings)
and never retrains anything. Three tabs:
  1. Compare models  — prediction-vs-actual plot for the three implementations
  2. Distributions   — target and feature distributions on the test clients
  3. Threshold & cost — threshold slider -> confusion matrix + business cost
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch import nn

ARTIFACTS = Path("artifacts")

# ---------------------------------------------------------------------------
# 1. Load everything train.py saved
# ---------------------------------------------------------------------------
with open(ARTIFACTS / "results.json") as f:
    saved = json.load(f)

test = np.load(ARTIFACTS / "test_data.npz")
X_test, y_test, bill_test = test["X_test"], test["y_test"], test["bill_test"]
clients = pd.read_csv(ARTIFACTS / "test_clients.csv")  # readable test features

COSTS = saved["costs"]  # the same business assumptions train.py used
RESULTS = saved["results"]
MODEL_LABELS = {
    "sklearn": "scikit-learn",
    "torch_manual": "PyTorch manual loop",
    "torch_standard": "PyTorch nn.Module + optim",
}


# The standard PyTorch model must have the same structure as in train.py
# so its saved weights fit into it.
class LogisticModel(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


sk_model = joblib.load(ARTIFACTS / "model_sklearn.joblib")
manual = torch.load(ARTIFACTS / "model_torch_manual.pt")
std_model = LogisticModel(X_test.shape[1])
std_model.load_state_dict(torch.load(ARTIFACTS / "model_torch_standard.pt"))
std_model.eval()

# Predicted chance of default for every test client, from each model
Xt = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    PROBS = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "torch_manual": torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy(),
        "torch_standard": torch.sigmoid(std_model(Xt)).numpy(),
    }


# ---------------------------------------------------------------------------
# 2. Business cost (same formula as train.py, same saved assumptions)
# ---------------------------------------------------------------------------
def total_cost(y_true, flagged, bill):
    loss = np.maximum(bill, 0) * COSTS["loss_share"]
    call_costs = flagged * COSTS["call_cost"]
    defaulter_losses = y_true * loss * np.where(flagged == 1, 1 - COSTS["call_success"], 1.0)
    return float(np.sum(call_costs + defaulter_losses))


# Safety check: recomputing each model's test cost here must give exactly
# what train.py reported, otherwise the dashboard and the report disagree.
for name, probs in PROBS.items():
    flagged = (probs >= RESULTS[name]["threshold"]).astype(int)
    recomputed = total_cost(y_test, flagged, bill_test)
    assert abs(recomputed - RESULTS[name]["test_cost"]) < 1, f"{name}: cost mismatch"

COST_NOBODY = total_cost(y_test, np.zeros_like(y_test), bill_test)
COST_EVERYONE = total_cost(y_test, np.ones_like(y_test), bill_test)


# ---------------------------------------------------------------------------
# 3. Tab 1 — compare the three models
# ---------------------------------------------------------------------------
def prediction_vs_actual_plot():
    """Group clients into 10 equal-sized groups by predicted chance; for each
    group, plot the average predicted chance against the share that actually
    defaulted. A perfect model lies on the diagonal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect match",
        line=dict(color="#9AA5B1", dash="dash"),
    ))
    styles = {
        "sklearn": dict(color="#1F3A5F", width=4),
        "torch_manual": dict(color="#D08C00", width=2, dash="dot"),
        "torch_standard": dict(color="#2A9D8F", width=2, dash="dashdot"),
    }
    for name, probs in PROBS.items():
        groups = pd.qcut(probs, 10, labels=False, duplicates="drop")
        df = pd.DataFrame({"pred": probs, "actual": y_test, "group": groups})
        summary = df.groupby("group").mean()
        fig.add_trace(go.Scatter(
            x=summary["pred"], y=summary["actual"], mode="lines+markers",
            name=MODEL_LABELS[name], line=styles[name],
        ))
    fig.update_layout(
        xaxis_title="Average predicted chance of default",
        yaxis_title="Share that actually defaulted",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2), height=480,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def results_table():
    rows = []
    labels = {**MODEL_LABELS, "call_nobody": "Baseline: call nobody",
              "call_everyone": "Baseline: call everyone"}
    for name, r in RESULTS.items():
        rows.append({
            "Strategy": labels[name],
            "Threshold": "-" if r["threshold"] is None else f"{r['threshold']:.2f}",
            "Total cost (NT$)": f"{r['test_cost']:,.0f}",
            "Called": f"{r['called']:,}",
            "Defaulters caught": f"{r['defaulters_caught']:.1%}",
            "Accuracy": f"{r['accuracy']:.1%}",
            "AUC": "-" if r["auc"] is None else f"{r['auc']:.3f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Tab 2 — distributions
# ---------------------------------------------------------------------------
clients["Outcome"] = np.where(clients["default"] == 1, "Defaulted", "Paid")
OUTCOME_COLORS = {"Defaulted": "#B23A48", "Paid": "#6B7B8C"}

FEATURES = {
    "Credit limit (NT$)": "LIMIT_BAL",
    "Credit usage (September bill ÷ limit)": "credit_usage",
    "Payment ratio (paid ÷ owed)": "payment_ratio",
    "Debt trend ((Sept − Apr bill) ÷ limit)": "debt_trend",
    "Late in September (0 = no, 1 = yes)": "late_sep",
    "Late months, April–August": "late_months_apr_aug",
    "Months late in September": "months_late_sep",
    "September bill (NT$)": "bill_sep",
    "Paid in September (NT$)": "paid_sep",
    "Education": "EDUCATION",
}


def target_plot():
    counts = clients["Outcome"].value_counts().reindex(["Paid", "Defaulted"])
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[OUTCOME_COLORS[o] for o in counts.index],
        text=[f"{v:,} ({v / counts.sum():.1%})" for v in counts.values],
        textposition="outside",
    ))
    fig.update_layout(yaxis_title="Test clients", height=320,
                      margin=dict(l=40, r=20, t=30, b=40))
    return fig


def feature_plot(feature_label):
    col = FEATURES[feature_label]
    values = clients[col]
    fig = go.Figure()
    if values.dtype == object or values.nunique() <= 10:
        # Few distinct values: show the default rate for each value
        rate = clients.groupby(col)["default"].agg(["mean", "count"]).reset_index()
        fig.add_trace(go.Bar(
            x=rate[col].astype(str), y=rate["mean"], marker_color="#1F3A5F",
            text=[f"{m:.0%} of {c:,}" for m, c in zip(rate["mean"], rate["count"])],
            textposition="outside",
        ))
        fig.update_layout(yaxis_title="Share that defaulted", yaxis_tickformat=".0%")
    else:
        # Many values: overlaid histograms for paid vs defaulted clients.
        # Extreme outliers (top/bottom 1%) are left out so the shape is visible.
        low, high = values.quantile([0.01, 0.99])
        for outcome in ["Paid", "Defaulted"]:
            subset = values[(clients["Outcome"] == outcome) & values.between(low, high)]
            fig.add_trace(go.Histogram(
                x=subset, name=outcome, histnorm="percent", nbinsx=40,
                marker_color=OUTCOME_COLORS[outcome], opacity=0.6,
            ))
        fig.update_layout(barmode="overlay", yaxis_title="% of clients in group")
    fig.update_layout(xaxis_title=feature_label, height=400,
                      margin=dict(l=40, r=20, t=30, b=40))
    return fig


# ---------------------------------------------------------------------------
# 5. Tab 3 — threshold slider, confusion matrix, business cost
# ---------------------------------------------------------------------------
def threshold_view(model_label, threshold):
    name = {v: k for k, v in MODEL_LABELS.items()}[model_label]
    flagged = (PROBS[name] >= threshold).astype(int)

    called_payers = int(((flagged == 1) & (y_test == 0)).sum())
    called_defaulters = int(((flagged == 1) & (y_test == 1)).sum())
    missed_defaulters = int(((flagged == 0) & (y_test == 1)).sum())
    left_payers = int(((flagged == 0) & (y_test == 0)).sum())

    # Confusion matrix: rows = what the model decided, columns = what happened
    z = [[called_payers, called_defaulters], [left_payers, missed_defaulters]]
    text = [
        [f"Wasted call<br><b>{called_payers:,}</b>", f"Useful call<br><b>{called_defaulters:,}</b>"],
        [f"Correctly left alone<br><b>{left_payers:,}</b>", f"Missed defaulter<br><b>{missed_defaulters:,}</b>"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Would have paid", "Would have defaulted"],
        y=["Called", "Not called"], text=text, texttemplate="%{text}",
        textfont=dict(size=16, color="#1B2430"),
        # Light colour scale so the dark text stays readable in every cell
        colorscale=[[0, "#F3F6F9"], [1, "#9DB4CC"]], showscale=False,
    ))
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=20, b=40),
                      yaxis=dict(autorange="reversed"))

    cost = total_cost(y_test, flagged, bill_test)
    best_baseline = min(COST_NOBODY, COST_EVERYONE)
    diff = best_baseline - cost
    verdict = (f"saves NT$ {diff:,.0f} compared with the cheaper baseline" if diff >= 0
               else f"costs NT$ {-diff:,.0f} more than the cheaper baseline")
    summary = (
        f"### Total business cost: NT$ {cost:,.0f}\n"
        f"At a threshold of **{threshold:.2f}**, the model calls **{flagged.sum():,}** of "
        f"{len(y_test):,} test clients and catches **{called_defaulters / max(y_test.sum(), 1):.1%}** "
        f"of the defaulters. It {verdict}.\n\n"
        f"- Call nobody: NT$ {COST_NOBODY:,.0f}\n"
        f"- Call everyone: NT$ {COST_EVERYONE:,.0f}\n"
        f"- Threshold chosen on the validation set: **{RESULTS[name]['threshold']:.2f}**\n\n"
        f"Assumptions (see REPORT.md): NT$ {COSTS['call_cost']} per call, "
        f"{COSTS['loss_share']:.0%} of a defaulted bill never recovered, "
        f"a call prevents {COSTS['call_success']:.0%} of the loss."
    )
    return fig, summary


# ---------------------------------------------------------------------------
# 6. The dashboard layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Credit card default: early-warning outreach") as demo:
    gr.Markdown(
        "# Early-warning outreach for credit card defaults\n"
        "Which cardholders should the collections team call before the October "
        "due date? Logistic regression trained three ways, evaluated on "
        f"{len(y_test):,} test clients."
    )

    with gr.Tab("Compare models"):
        gr.Markdown(
            "Each point is a group of clients with similar predicted chances. "
            "The closer the points are to the dashed diagonal, the better the "
            "predicted chances match what actually happened. The three lines "
            "overlap because the three implementations agree."
        )
        gr.Plot(value=prediction_vs_actual_plot())
        gr.Dataframe(value=results_table(), interactive=False)

    with gr.Tab("Distributions"):
        gr.Markdown("How many test clients defaulted in October:")
        gr.Plot(value=target_plot())
        feature_choice = gr.Dropdown(
            choices=list(FEATURES), value="Credit usage (September bill ÷ limit)",
            label="Feature",
        )
        feature_out = gr.Plot(value=feature_plot("Credit usage (September bill ÷ limit)"))
        feature_choice.change(feature_plot, inputs=feature_choice, outputs=feature_out)

    with gr.Tab("Threshold & cost"):
        gr.Markdown(
            "Clients whose predicted chance of default is at or above the "
            "threshold get a call. Move the slider to see who gets called and "
            "what it costs the bank."
        )
        with gr.Row():
            model_choice = gr.Dropdown(
                choices=list(MODEL_LABELS.values()), value=MODEL_LABELS["sklearn"],
                label="Model",
            )
            threshold = gr.Slider(
                minimum=0.01, maximum=0.99, step=0.01,
                value=RESULTS["sklearn"]["threshold"], label="Decision threshold",
            )
        start_fig, start_text = threshold_view(MODEL_LABELS["sklearn"], RESULTS["sklearn"]["threshold"])
        matrix_out = gr.Plot(value=start_fig)
        cost_out = gr.Markdown(value=start_text)
        for control in (model_choice, threshold):
            control.change(threshold_view, inputs=[model_choice, threshold],
                           outputs=[matrix_out, cost_out])


if __name__ == "__main__":
    demo.launch()
