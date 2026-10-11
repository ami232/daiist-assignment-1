"""
Assignment 1 — Gradio dashboard for the Telco churn models.

Run with:  uv run python main.py app   (after main.py train)

Only LOADS what train.py saved in artifacts/ — nothing is retrained here.
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
from sklearn.metrics import confusion_matrix

ART = Path("artifacts")

# ---------------------------------------------------------------------------
# Load saved artifacts
# ---------------------------------------------------------------------------
config = json.load(open(ART / "config.json"))
results = pd.read_csv(ART / "results.csv")
coefs = pd.read_csv(ART / "coefficients.csv")
test = pd.read_csv(ART / "test_set_scaled.csv")
X_test = test.drop(columns="Churn").values.astype("float32")
y_test = test["Churn"].values
bill_test = pd.read_csv(ART / "test_set_raw.csv")["MonthlyCharges"].values  # for the € cost
raw = pd.read_csv("data/telco_churn.csv")  # original data, for distributions

# Model 1: scikit-learn
sk_model = joblib.load(ART / "sklearn_model.joblib")

# Model 2: manual PyTorch (just the weight vector and bias)
manual = torch.load(ART / "manual_pytorch.pt")


# Model 3: standard PyTorch (same class as in train.py, weights loaded from disk)
class LogReg(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = torch.nn.Linear(n, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


std_model = LogReg(config["n_features"])
std_model.load_state_dict(torch.load(ART / "standard_pytorch.pt"))
std_model.eval()

# Predicted churn probabilities on the test set, from each saved model
X_t = torch.tensor(X_test)
with torch.no_grad():
    probs = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "manual_pytorch": torch.sigmoid(X_t @ manual["w"] + manual["b"]).numpy(),
        "standard_pytorch": torch.sigmoid(std_model(X_t)).numpy(),
    }
MODELS = list(probs)

OFFER_COST, MONTHS_LOST, OFFER_SUCCESS = config["offer_cost"], config["months_lost"], config["offer_success"]


def business_cost(y_true, y_pred):
    """Same formula as train.py: offer cost + 12 months of each lost customer's bill."""
    y_pred = np.asarray(y_pred).astype(int)
    loss = MONTHS_LOST * bill_test
    contacted = y_pred * (OFFER_COST + y_true * (1 - OFFER_SUCCESS) * loss)
    not_contacted = (1 - y_pred) * y_true * loss
    return float(np.sum(contacted + not_contacted))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def prediction_vs_actual_plot():
    """Predicted churn probability split by what actually happened, per model."""
    long = pd.concat([
        pd.DataFrame({"model": m, "predicted probability": p,
                      "actual": np.where(y_test == 1, "Churned", "Stayed")})
        for m, p in probs.items()
    ])
    fig = px.box(long, x="actual", y="predicted probability", color="model",
                 title="Predicted churn probability vs actual outcome (test set)")
    return fig


def agreement_plot():
    """Each dot is a test customer: sklearn probability vs the PyTorch ones."""
    fig = go.Figure()
    for m in ["manual_pytorch", "standard_pytorch"]:
        fig.add_scatter(x=probs["sklearn"], y=probs[m], mode="markers", name=m, opacity=0.5)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect agreement", line=dict(dash="dash"))
    fig.update_layout(title="Do the three implementations agree?",
                      xaxis_title="sklearn probability", yaxis_title="PyTorch probability")
    return fig


def coefficient_plot():
    long = coefs.melt(id_vars="feature", var_name="model", value_name="coefficient")
    order = coefs.sort_values("sklearn")["feature"]
    fig = px.bar(long, y="feature", x="coefficient", color="model", barmode="group",
                 category_orders={"feature": list(order)}, height=700,
                 title="Coefficients (standardised features): + raises churn risk, − lowers it")
    return fig


def distribution_plot(column):
    fig = px.histogram(raw, x=column, color="Churn", barmode="group",
                       title=f"{column} by churn (all 7,043 customers)")
    if column == "Churn":
        fig = px.histogram(raw, x="Churn", title="Target distribution: 26.5% churn")
    return fig


def threshold_view(threshold, model):
    pred = (probs[model] >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       x=["Predicted: stays", "Predicted: churns"],
                       y=["Actually stayed", "Actually churned"],
                       title=f"Confusion matrix — {model} @ threshold {threshold:.2f}")

    cost = business_cost(y_test, pred)
    nobody = business_cost(y_test, np.zeros_like(y_test))
    everyone = business_cost(y_test, np.ones_like(y_test))
    text = (
        f"### Business cost on the test set ({len(y_test)} customers): **€{cost:,.0f}**\n"
        f"- Customers contacted: **{pred.sum()}** · churners caught: **{cm[1, 1]} / {cm[1].sum()}**\n"
        f"- Contact nobody (naive baseline): €{nobody:,.0f} → model saves **€{nobody - cost:,.0f}**\n"
        f"- Contact everyone: €{everyone:,.0f}\n"
        f"- Threshold chosen on validation in train.py: **{config['best_threshold']}**\n\n"
        f"*Costs: offer €{OFFER_COST}, lost customer = {MONTHS_LOST} × their monthly bill, "
        f"offer keeps {OFFER_SUCCESS:.0%} of contacted churners.*"
    )

    ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = px.line(x=ts, y=[business_cost(y_test, (probs[model] >= t).astype(int)) for t in ts],
                    labels={"x": "threshold", "y": "total cost (€)"},
                    title="Business cost vs threshold (test set)")
    curve.add_vline(x=threshold, line_dash="dash")
    return cm_fig, text, curve


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Telco churn — 3 logistic regressions") as demo:
    gr.Markdown("# Telco customer churn: who should get a retention offer?\n"
                "Same logistic regression trained three ways (scikit-learn, manual PyTorch, "
                "nn.Module + torch.optim). Everything below is loaded from `artifacts/`; nothing is retrained.")

    with gr.Tab("Model comparison"):
        gr.Dataframe(results, label="Test-set results (baseline = everyone gets the training churn rate)")
        gr.Markdown(f"Max coefficient difference vs sklearn: **{config['max_coef_diff']:.4f}** · "
                    f"max probability difference: **{config['max_prob_diff']:.4f}**")
        gr.Plot(prediction_vs_actual_plot())
        gr.Plot(agreement_plot())
        gr.Plot(coefficient_plot())

    with gr.Tab("Distributions"):
        col = gr.Dropdown(list(raw.columns.drop("customerID")), value="Churn", label="Column")
        dist = gr.Plot(distribution_plot("Churn"))
        col.change(distribution_plot, col, dist)

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            thr = gr.Slider(0.05, 0.95, value=config["best_threshold"], step=0.01, label="Decision threshold")
            model = gr.Dropdown(MODELS, value="sklearn", label="Model")
        cost_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        for c in (thr, model):
            c.change(threshold_view, [thr, model], [cm_plot, cost_md, curve_plot])
        demo.load(threshold_view, [thr, model], [cm_plot, cost_md, curve_plot])

if __name__ == "__main__":
    demo.launch()
