"""
Assignment 1 — Gradio dashboard.

Loads the models and data saved by train.ipynb (in artifacts/) and shows:

1. Model comparison: predicted vs. actual on the test set for the three
   models, the results table, and the PyTorch loss curves.
2. Distributions: histogram of any column, optionally split by smoker.
3. Simulator: type in a new member and get the three models' predictions.

Nothing is trained here: every model is loaded from disk.
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
import torch.nn as nn
from plotly.subplots import make_subplots

ARTIFACTS_DIR = Path("artifacts")
if not (ARTIFACTS_DIR / "metrics.json").exists():
    raise SystemExit("No trained models found in artifacts/. Run `uv run python main.py train` first.")

# ---------------------------------------------------------------------------
# Load everything the training stage saved
# ---------------------------------------------------------------------------
with open(ARTIFACTS_DIR / "metrics.json") as f:
    metrics = json.load(f)

FEATURES = metrics["features"]
TARGET = metrics["target"]
BASELINE_VALUE = metrics["baseline_value"]

scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
sk_model = joblib.load(ARTIFACTS_DIR / "sklearn_model.joblib")

manual_params = torch.load(ARTIFACTS_DIR / "manual_params.pt")
w_manual, b_manual = manual_params["w"], manual_params["b"]

torch_model = nn.Linear(len(FEATURES), 1)
torch_model.load_state_dict(torch.load(ARTIFACTS_DIR / "torch_model.pt"))
torch_model.eval()

data = pd.read_csv(ARTIFACTS_DIR / "dataset_with_split.csv")
test_data = data[data["split"] == "test"]

MODEL_NAMES = ["scikit-learn", "PyTorch manual", "PyTorch standard"]


def predict_all(features_df):
    """Predictions in $ of the three saved models (plus the baseline) for some rows."""
    X = scaler.transform(features_df[FEATURES])
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return {
            "scikit-learn": sk_model.predict(X),
            "PyTorch manual": (X_t @ w_manual + b_manual).numpy(),
            "PyTorch standard": torch_model(X_t).squeeze(-1).numpy(),
            "baseline (train mean)": np.full(len(X), BASELINE_VALUE),
        }


test_preds = predict_all(test_data)
y_test = test_data[TARGET].to_numpy()


# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------
def prediction_vs_actual_figure():
    fig = make_subplots(rows=1, cols=3, subplot_titles=MODEL_NAMES, shared_yaxes=True)
    lims = [0, max(y_test.max(), max(p.max() for p in test_preds.values())) * 1.05]
    smoker = test_data["smoker"].map({0: "non-smoker", 1: "smoker"}).to_numpy()

    for i, name in enumerate(MODEL_NAMES, start=1):
        mae = np.abs(test_preds[name] - y_test).mean()
        for group, color in [("non-smoker", "#1f77b4"), ("smoker", "#d62728")]:
            mask = smoker == group
            fig.add_trace(go.Scatter(
                x=y_test[mask], y=test_preds[name][mask], mode="markers",
                marker=dict(color=color, size=6, opacity=0.6),
                name=group, legendgroup=group, showlegend=(i == 1),
            ), row=1, col=i)
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines", line=dict(color="gray", dash="dash"),
            name="perfect prediction", showlegend=(i == 1),
        ), row=1, col=i)
        fig.layout.annotations[i - 1].text = f"{name}<br>test MAE = {mae:,.0f} $"
        fig.update_xaxes(title_text="actual charges ($)", range=lims, row=1, col=i)

    fig.update_yaxes(title_text="predicted charges ($)", range=lims, row=1, col=1)
    fig.update_layout(height=450, margin=dict(t=70))
    return fig


def results_table():
    df = pd.DataFrame(metrics["results"])
    money_cols = [c for c in df.columns if "MAE" in c or "RMSE" in c]
    df[money_cols] = df[money_cols].round(0)
    df[[c for c in df.columns if "R2" in c]] = df[[c for c in df.columns if "R2" in c]].round(3)
    return df


def loss_curve_figure():
    fig = go.Figure()
    for name, history in metrics["loss_history"].items():
        fig.add_trace(go.Scatter(y=history, mode="lines", name=name))
    fig.update_layout(
        title=f"Training loss of the two PyTorch models (lr = {metrics['best_lr']})",
        xaxis_title="epoch", yaxis_title="train MSE (log scale)", yaxis_type="log", height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 2: distributions
# ---------------------------------------------------------------------------
DIST_COLUMNS = [TARGET, "age", "bmi", "children", "smoker", "smoker_obese"]


def distribution_figure(column, by_smoker):
    plot_df = data.assign(group=data["smoker"].map({0: "non-smoker", 1: "smoker"}))
    fig = px.histogram(
        plot_df, x=column, color="group" if by_smoker else None,
        barmode="overlay", opacity=0.7, nbins=40,
        color_discrete_map={"non-smoker": "#1f77b4", "smoker": "#d62728"},
    )
    fig.update_layout(title=f"Distribution of {column} (all {len(data)} members)", height=420)
    return fig


# ---------------------------------------------------------------------------
# Tab 3: simulator for a new member
# ---------------------------------------------------------------------------
REGIONS = ["northeast", "northwest", "southeast", "southwest"]


def simulate(age, bmi, children, smoker, region):
    # Same feature engineering as in train.ipynb
    row = {"age": age, "bmi": bmi, "children": children, "smoker": int(smoker == "yes")}
    row["smoker_obese"] = int(row["smoker"] == 1 and bmi >= 30)
    for r in REGIONS[1:]:  # northeast is the reference region (dropped in training)
        row[f"region_{r}"] = int(region == r)

    preds = predict_all(pd.DataFrame([row]))
    return pd.DataFrame({
        "model": list(preds.keys()),
        "predicted annual cost ($)": [round(float(p[0])) for p in preds.values()],
    })


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        "# Medical cost prediction: one linear regression, three implementations\n"
        "Expected annual medical cost of an insurance member. Models are loaded from "
        "`artifacts/` (trained by `train.ipynb`); nothing is retrained here."
    )

    with gr.Tab("Model comparison"):
        gr.Markdown("**Predicted vs. actual on the test set** (201 members never used for any decision). "
                    "Points on the dashed line are perfect predictions.")
        gr.Plot(prediction_vs_actual_figure())
        gr.Markdown("**Results** (validation and test)")
        gr.Dataframe(results_table(), interactive=False)
        gr.Plot(loss_curve_figure())

    with gr.Tab("Distributions"):
        with gr.Row():
            column_input = gr.Dropdown(DIST_COLUMNS, value=TARGET, label="Column")
            by_smoker_input = gr.Checkbox(value=True, label="Split by smoker")
        dist_plot = gr.Plot(distribution_figure(TARGET, True))
        column_input.change(distribution_figure, [column_input, by_smoker_input], dist_plot)
        by_smoker_input.change(distribution_figure, [column_input, by_smoker_input], dist_plot)

    with gr.Tab("Simulator"):
        gr.Markdown("Describe a new member and compare the three models' predictions.")
        with gr.Row():
            age_input = gr.Slider(18, 64, value=40, step=1, label="Age")
            bmi_input = gr.Slider(16, 53, value=30, step=0.1, label="BMI (≥ 30 = obese)")
            children_input = gr.Slider(0, 5, value=1, step=1, label="Children")
        with gr.Row():
            smoker_input = gr.Radio(["no", "yes"], value="no", label="Smoker")
            region_input = gr.Dropdown(REGIONS, value="southeast", label="Region")
        inputs = [age_input, bmi_input, children_input, smoker_input, region_input]
        sim_output = gr.Dataframe(simulate(40, 30, 1, "no", "southeast"), interactive=False)
        for component in inputs:
            component.change(simulate, inputs, sim_output)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
