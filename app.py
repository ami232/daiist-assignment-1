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

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr

#LOAD AND TEST
OUT = Path(__file__).resolve().parent / "artifacts"
meta = json.loads((OUT / "metrics.json").read_text())
df = pd.read_csv(OUT / "prepared_data.csv", parse_dates=["Date"])
test = df[df["split"] == "test"].copy()
features, y = meta["features"], test["target"].to_numpy()

scaler = joblib.load(OUT / "scaler.joblib")
sklearn_model = joblib.load(OUT / "sklearn_model.joblib")
manual = np.load(OUT / "manual_model.npz")
standard = nn.Linear(len(features), 1).double()
standard.load_state_dict(torch.load(OUT / "standard_model.pt", map_location="cpu", weights_only=True))
standard.eval()

X = scaler.transform(test[features])

with torch.no_grad():
    xt = torch.tensor(X, dtype=torch.float64)
    weights = torch.tensor(manual["weights"], dtype=torch.float64)
    bias = torch.tensor(manual["bias"], dtype=torch.float64)
    predictions = {
        "scikit-learn": sklearn_model.predict_proba(X)[:, 1],
        "manual PyTorch": torch.sigmoid(xt @ weights + bias).numpy().ravel(),
        "standard PyTorch": torch.sigmoid(standard(xt)).numpy().ravel()}

#PREDICTION VS ACTUAL
def compare_plot(threshold):
    fig = go.Figure()
    dates = test["Date"].dt.strftime("%Y-%m-%d")
    for i, (name, p) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            x=y + (i - 1) * 0.13, y=p, mode="markers", name=name,
            customdata=dates, marker=dict(size=5, opacity=0.45),
            hovertemplate="Date: %{customdata}<br>P(UP): %{y:.3f}<extra>%{fullData.name}</extra>"))

    fig.add_hline(y=threshold, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Predicted Probability vs Actual Outcome - Test Set",
        xaxis=dict(tickvals=[0, 1], ticktext=["Actual DOWN", "Actual UP"], range=[-0.4, 1.4]),
        yaxis=dict(title="Predicted Probability of UP", range=[0, 1]),
        height=420)
    return fig

#FEATURE DISTRIBUTIONS
def distribute(feature):
    data = df.copy()
    if feature == "target":
        data["target"] = data["target"].map({0: "DOWN", 1: "UP"})
        fig = px.histogram(data, x=feature, color="split",
                           barmode="group", histnorm="probability")
    else:
        fig = px.histogram(data, x=feature, color="split", nbins=45,
                           barmode="overlay", histnorm="probability", opacity=0.65)
    fig.update_layout(title=f"Distribution: {feature}", height=360)
    return fig

#CONFUSION MATRIX AND COST
def refresh(name, threshold):
    predicted = (predictions[name] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    cm = np.array([[tn, fp], [fn, tp]])

    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred DOWN", "Pred UP"], y=["Actual DOWN", "Actual UP"],
        text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion Matrix", height=360,
                      yaxis=dict(autorange="reversed"))
    cost = (3 * fp + fn) * 100 / len(y)
    always_down = y.sum() * 100 / len(y)

    summary = (
        f"**Accuracy:** {(tp + tn) / len(y):.2%} | "
        f"**False positives:** {fp} | **False negatives:** {fn}\n\n"
        f"**Business cost per 100 sessions: {cost:.2f}**\n\n"
        f"Always predicting DOWN costs {always_down:.2f} per 100 sessions.\n\n"
        "Costs are hypothetical penalty units, not actual investment returns.")
    return compare_plot(threshold), fig, summary

#DASHBOARD
scores = pd.DataFrame.from_dict(meta["test_metrics"], orient="index").reset_index(names="Model")
initial_plot, initial_cm, initial_cost = refresh("scikit-learn", meta["threshold"])

with gr.Blocks(title="SPY - Machine Learning Dashboard") as demo:
    gr.Markdown("# SPY Stock Market Prediction Dashboard")
    gr.Markdown("Comparing Scikit-learn, Manual PyTorch and Standard PyTorch Logistic Regression.")
    gr.Markdown("## Model Comparison")
    gr.Dataframe(value=scores, interactive=False, label="Test Set Results")
    gr.Markdown("## Interactive Predictions")
    gr.Markdown("The initial threshold was selected on validation data. "
                "The slider explores test results but should not be used to retune the model.")

    with gr.Row():
        model_choice = gr.Dropdown(choices=list(predictions), value="scikit-learn", label="Model")
        threshold = gr.Slider(0.01, 0.99, value=meta["threshold"],
                              step=0.01, label="Decision Threshold")
    comparison = gr.Plot(value=initial_plot)

    with gr.Row():
        confusion = gr.Plot(value=initial_cm)
        business_cost = gr.Markdown(value=initial_cost)

    gr.Markdown("## Feature and Target Distributions")
    feature = gr.Dropdown(choices=["target"] + features, value="target", label="Select Variable")
    distribution = gr.Plot(value=distribute("target"))

    model_choice.change(refresh, [model_choice, threshold],
                        [comparison, confusion, business_cost])
    threshold.change(refresh, [model_choice, threshold],
                     [comparison, confusion, business_cost])
    feature.change(distribute, feature, distribution)

#LAUNCH
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
