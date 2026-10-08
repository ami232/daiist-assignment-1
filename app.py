"""Gradio dashboard for Assignment 1.

Loads the artifacts written by train.ipynb and trains nothing.
Run with:  uv run python main.py app
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch import nn
import gradio as gr

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

CFG = json.loads((ART / "config.json").read_text())
COUPON_COST = CFG["coupon_cost"]
RECOVERY_GAIN = CFG["recovery_gain"]
DEFAULT_THRESHOLD = CFG["default_threshold"]

# The four ID-like columns were strings when the preprocessor was fitted, so they
# have to come back as strings or the one-hot encoder silently drops them.
ID_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType"]
X_test = pd.read_csv(ART / "X_test.csv", dtype={c: str for c in ID_COLS})
y_true = np.load(ART / "test_predictions.npz")["y_true"]

preprocessor = joblib.load(ART / "preprocessor.joblib")
A_test = preprocessor.transform(X_test).astype(np.float32)
COMPARISON = pd.read_csv(ART / "comparison.csv")


class LogisticRegressionNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def load_predictions() -> dict:
    """Load the three saved models and score the held-out test sessions."""
    out = {}
    sk_model = joblib.load(ART / "model_sklearn.joblib")
    out["scikit-learn"] = sk_model.predict_proba(A_test)[:, 1]

    manual = torch.load(ART / "model_manual.pt")
    with torch.no_grad():
        logits = torch.from_numpy(A_test) @ manual["W"] + manual["b"]
        out["manual PyTorch"] = torch.sigmoid(logits).squeeze(1).numpy()

    net = LogisticRegressionNet(A_test.shape[1])
    net.load_state_dict(torch.load(ART / "model_standard.pt"))
    net.eval()
    with torch.no_grad():
        out["standard PyTorch"] = torch.sigmoid(net(torch.from_numpy(A_test))).numpy()
    return out


PREDS = load_predictions()
MODELS = list(PREDS)


def calibration_figure():
    """Predicted probability vs what actually happened, in deciles."""
    fig = go.Figure()
    bins = np.linspace(0, 1, 11)
    for name, p in PREDS.items():
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() >= 10:
                xs.append(float(p[mask].mean()))
                ys.append(float(y_true[mask].mean()))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title="Predicted probability vs actual conversion rate",
                      xaxis_title="mean predicted probability",
                      yaxis_title="observed conversion rate", height=420)
    return fig


def score_figure(model_name: str):
    p = PREDS[model_name]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=p[y_true == 0], nbinsx=50, name="did not buy", opacity=0.65))
    fig.add_trace(go.Histogram(x=p[y_true == 1], nbinsx=50, name="bought", opacity=0.65))
    fig.update_layout(barmode="overlay", height=420,
                      title=f"Predicted probabilities - {model_name}",
                      xaxis_title="predicted probability of purchase",
                      yaxis_title="sessions")
    return fig


def feature_figure(feature: str):
    col = X_test[feature]
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(col) and col.nunique() > 10:
        fig.add_trace(go.Histogram(x=col[y_true == 0], nbinsx=40, name="did not buy", opacity=0.65))
        fig.add_trace(go.Histogram(x=col[y_true == 1], nbinsx=40, name="bought", opacity=0.65))
        fig.update_layout(barmode="overlay", yaxis_title="sessions")
    else:
        rate = (pd.DataFrame({"v": col.astype(str), "y": y_true})
                  .groupby("v")["y"].agg(["mean", "count"]))
        rate = rate[rate["count"] >= 20].sort_values("mean", ascending=False)
        fig.add_trace(go.Bar(x=rate.index, y=rate["mean"], name="conversion rate"))
        fig.update_layout(yaxis_title="conversion rate")
    fig.update_layout(title=f"{feature} vs purchase", xaxis_title=feature, height=420)
    return fig


def net_value(p, threshold: float) -> float:
    fires = p < threshold
    return float(RECOVERY_GAIN * ((y_true == 0) & fires).sum()
                 - COUPON_COST * ((y_true == 1) & fires).sum())


def value_curve(model_name: str, threshold: float):
    p = PREDS[model_name]
    grid = np.linspace(0.01, 0.99, 99)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=[net_value(p, t) for t in grid],
                             mode="lines", name="net value"))
    fig.add_vline(x=threshold, line_dash="dash")
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(title="Net value across every threshold", height=380,
                      xaxis_title="threshold", yaxis_title="net value (EUR)")
    return fig


def threshold_view(model_name: str, threshold: float):
    p = PREDS[model_name]
    fires = p < threshold
    buyers_couponed = int(((y_true == 1) & fires).sum())
    buyers_left = int(((y_true == 1) & ~fires).sum())
    nonbuyers_couponed = int(((y_true == 0) & fires).sum())
    nonbuyers_left = int(((y_true == 0) & ~fires).sum())

    z = [[nonbuyers_couponed, nonbuyers_left], [buyers_couponed, buyers_left]]
    cm = go.Figure(go.Heatmap(z=z, x=["coupon sent", "no coupon"],
                              y=["did not buy", "bought"], text=z,
                              texttemplate="%{text}", colorscale="Blues",
                              showscale=False))
    cm.update_layout(title=f"What happens at threshold {threshold:.2f}", height=380)

    gained = RECOVERY_GAIN * nonbuyers_couponed
    wasted = COUPON_COST * buyers_couponed
    summary = (
        f"### Net value: EUR {gained - wasted:,.0f}\n"
        f"- Coupons sent: **{int(fires.sum()):,}** of {len(p):,} sessions\n"
        f"- Non-buyers reached, 5% of them recovered: **+EUR {gained:,.0f}**\n"
        f"- Margin given away to people who were buying anyway: **-EUR {wasted:,.0f}**\n"
        f"- Buyers correctly left alone: {buyers_left:,}\n"
    )
    return cm, summary, value_curve(model_name, threshold)


with gr.Blocks(title="Session scoring dashboard") as demo:
    gr.Markdown(
        "# Online shoppers - session scoring dashboard\n"
        "Three models trained in `train.ipynb` and loaded from `artifacts/`. "
        "Nothing is trained here. Test period: November and December."
    )

    with gr.Tab("Model comparison"):
        gr.Dataframe(value=COMPARISON, label="Test results (threshold picked on October)")
        gr.Plot(value=calibration_figure())
        model_a = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
        scores = gr.Plot(value=score_figure(MODELS[0]))
        model_a.change(score_figure, model_a, scores)

    with gr.Tab("Data explorer"):
        feature = gr.Dropdown(sorted(X_test.columns), value="product_page_share",
                              label="Feature")
        fplot = gr.Plot(value=feature_figure("product_page_share"))
        feature.change(feature_figure, feature, fplot)

    with gr.Tab("Threshold and business impact"):
        model_b = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
        thr = gr.Slider(0.01, 0.99, value=DEFAULT_THRESHOLD, step=0.01,
                        label="Decision threshold - the coupon fires below this")
        cm_plot = gr.Plot()
        summary_md = gr.Markdown()
        curve_plot = gr.Plot()
        for widget in (model_b, thr):
            widget.change(threshold_view, [model_b, thr], [cm_plot, summary_md, curve_plot])
        demo.load(threshold_view, [model_b, thr], [cm_plot, summary_md, curve_plot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
