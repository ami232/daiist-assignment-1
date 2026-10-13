"""
Assignment 1 -- Gradio dashboard (run with `uv run python main.py app`).

Loads the preprocessor and the three trained models saved by train.py and
only runs INFERENCE with them. Nothing is fitted here.
"""

import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_curve

from pipeline import (
    ARTIFACTS, THRESHOLD_GRID, LogisticRegressionNet, business_outcome, engineer,
    load_raw, predict_manual, predict_standard, profit_curve, temporal_split,
)

if not (ARTIFACTS / "meta.json").exists():
    raise SystemExit("No trained artifacts found. Run `uv run python main.py train` first.")

# ---------------------------------------------------------------------------
# Load artifacts (no training)
# ---------------------------------------------------------------------------
meta = json.loads((ARTIFACTS / "meta.json").read_text())
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_model.joblib")
manual_params = torch.load(ARTIFACTS / "manual_torch.pt", weights_only=True)
std_model = LogisticRegressionNet(meta["n_features"])
std_model.load_state_dict(torch.load(ARTIFACTS / "standard_torch.pt", weights_only=True))
std_model.eval()

metrics = pd.read_csv(ARTIFACTS / "metrics.csv")
coefs = pd.read_csv(ARTIFACTS / "coefficients.csv")
loss_hist = pd.read_csv(ARTIFACTS / "loss_history.csv")
tuning = pd.read_csv(ARTIFACTS / "tuning.csv")

# Same deterministic data + split as training, then inference only.
df = engineer(load_raw())
train_df, val_df, test_df = temporal_split(df)
SPLITS = {"train": train_df, "validation": val_df, "test": test_df}
y_test = test_df["y"].to_numpy()
X_test = preprocessor.transform(test_df).astype(np.float32)

MODEL_NAMES = ["scikit-learn", "Manual PyTorch", "Standard PyTorch"]
COLORS = {"scikit-learn": "#1f77b4", "Manual PyTorch": "#ff7f0e", "Standard PyTorch": "#2ca02c"}
PROBS = {
    "scikit-learn": sk_model.predict_proba(X_test)[:, 1],
    "Manual PyTorch": predict_manual(manual_params, X_test),
    "Standard PyTorch": predict_standard(std_model, X_test),
}


# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------
def calibration_fig():
    """Prediction vs actual for a classifier: bin clients by predicted
    probability and compare the mean prediction with the real subscription
    rate in each bin. Points on the diagonal = well calibrated."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="grey")))
    for name in MODEL_NAMES:
        p = PROBS[name]
        bins = pd.qcut(p, q=10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y_test, "bin": bins}).groupby("bin", observed=True)
        fig.add_trace(go.Scatter(x=g["p"].mean(), y=g["y"].mean(), mode="lines+markers",
                                 name=name, line=dict(color=COLORS[name])))
    fig.update_layout(title="Predicted vs actual subscription rate (test, 10 bins)",
                      xaxis_title="mean predicted probability", yaxis_title="actual subscription rate",
                      height=420)
    return fig


def agreement_fig():
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_test), size=min(3000, len(y_test)), replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x",
                             line=dict(dash="dash", color="grey")))
    for name in ["Manual PyTorch", "Standard PyTorch"]:
        fig.add_trace(go.Scatter(x=PROBS["scikit-learn"][idx], y=PROBS[name][idx], mode="markers",
                                 name=name, marker=dict(size=4, opacity=0.5, color=COLORS[name])))
    fig.update_layout(title="Do the three implementations agree? (test probabilities)",
                      xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability",
                      height=420)
    return fig


def roc_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="naive baseline",
                             line=dict(dash="dash", color="grey")))
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y_test, PROBS[name])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name, line=dict(color=COLORS[name])))
    fig.update_layout(title="ROC curve (test)", xaxis_title="false positive rate",
                      yaxis_title="true positive rate", height=420)
    return fig


def coef_fig(top_n: int = 20):
    top = coefs.reindex(coefs["scikit-learn"].abs().sort_values(ascending=False).index).head(top_n)
    top = top.iloc[::-1]
    fig = go.Figure()
    for name in MODEL_NAMES:
        fig.add_trace(go.Bar(y=top["feature"], x=top[name], name=name, orientation="h",
                             marker_color=COLORS[name]))
    fig.update_layout(barmode="group", title=f"Top {top_n} coefficients (standardised features)",
                      height=650, xaxis_title="coefficient (log-odds)")
    return fig


def loss_fig():
    fig = go.Figure()
    for name, col in [("Manual PyTorch", "manual"), ("Standard PyTorch", "standard")]:
        fig.add_trace(go.Scatter(x=loss_hist["epoch"], y=loss_hist[col], name=name,
                                 line=dict(color=COLORS[name])))
    # Drawn as a trace, not add_hline(annotation_text=...): on a log y-axis plotly
    # places shape annotations at 10**y, so the label would land off the chart.
    fig.add_trace(go.Scatter(x=loss_hist["epoch"], y=loss_hist["sklearn_optimum"],
                             name="scikit-learn optimum", mode="lines",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title="Training objective per epoch (log-loss + L2)", xaxis_title="epoch",
                      yaxis_title="objective", yaxis_type="log", height=420)
    return fig


# ---------------------------------------------------------------------------
# Tab 2: distributions
# ---------------------------------------------------------------------------
NUMERIC_VIEW = ["age", "prior_calls_log", "previous", "days_since_prev", "euribor3m",
                "nr_employed", "emp_var_rate", "cons_price_idx", "cons_conf_idx"]
CATEGORICAL_VIEW = ["job", "marital", "education", "default", "housing", "loan", "contact",
                    "month", "day_of_week", "poutcome", "never_contacted", "age_lt25", "age_ge60"]


def distribution_fig(feature: str, split: str):
    data = SPLITS[split]
    if feature in NUMERIC_VIEW:
        fig = go.Figure()
        for label, colour, name in [(0, "#999999", "did not subscribe"), (1, "#d62728", "subscribed")]:
            fig.add_trace(go.Histogram(x=data.loc[data["y"] == label, feature], name=name,
                                       histnorm="probability density", opacity=0.6,
                                       marker_color=colour, nbinsx=40))
        fig.update_layout(barmode="overlay", xaxis_title=feature, yaxis_title="density")
    else:
        g = data.groupby(feature, observed=True)["y"].agg(["mean", "size"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g[feature].astype(str), y=g["size"], name="rows", yaxis="y",
                             marker_color="#bbbbbb"))
        fig.add_trace(go.Scatter(x=g[feature].astype(str), y=g["mean"], name="subscription rate",
                                 yaxis="y2", mode="markers+lines", marker_color="#d62728"))
        fig.update_layout(yaxis=dict(title="rows"),
                          yaxis2=dict(title="subscription rate", overlaying="y", side="right",
                                      range=[0, 1]))
    fig.update_layout(title=f"{feature} -- {split} split", height=450)
    return fig


def target_over_time_fig():
    g = df.groupby("period")["y"].agg(["mean", "size"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["period"], y=g["size"], name="calls", marker_color="#cccccc"))
    fig.add_trace(go.Scatter(x=g["period"], y=g["mean"], name="subscription rate", yaxis="y2",
                             mode="lines+markers", marker_color="#d62728"))
    for split in ["validation", "test"]:
        start = SPLITS[split]["period"].iloc[0]
        fig.add_vline(x=start, line_dash="dot")
        fig.add_annotation(x=start, y=1.02, yref="paper", text=f"{split} starts", showarrow=False)
    fig.update_layout(title="Target over time: calls per month and subscription rate",
                      yaxis=dict(title="calls"),
                      yaxis2=dict(title="subscription rate", overlaying="y", side="right", range=[0, 1]),
                      height=450)
    return fig


# ---------------------------------------------------------------------------
# Tab 3: threshold & business cost
# ---------------------------------------------------------------------------
def threshold_view(model_name: str, threshold: float, value: float, cost: float):
    value, cost = float(value or 0), float(cost or 0)
    p = PROBS[model_name]
    out = business_outcome(y_test, p, threshold, value, cost)
    everyone = business_outcome(y_test, p, 0.0, value, cost)

    z = [[out["tn"], out["fp"]], [out["fn"], out["tp"]]]
    cm = go.Figure(go.Heatmap(
        z=z, x=["predicted: don't call", "predicted: call"],
        y=["actual: no", "actual: subscribed"], text=z, texttemplate="%{text}",
        colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix -- {model_name} @ {threshold:.2f}", height=380,
                     yaxis=dict(autorange="reversed"))

    profits = profit_curve(y_test, p, value, cost)
    curve = go.Figure(go.Scatter(x=THRESHOLD_GRID, y=profits, mode="lines", name="profit"))
    curve.add_vline(x=threshold, line_dash="dash", annotation_text="current")
    curve.add_hline(y=everyone["profit"], line_dash="dot", annotation_text="call everyone")
    curve.update_layout(title="Test-set profit vs threshold", xaxis_title="threshold",
                        yaxis_title="profit (EUR)", height=380)

    break_even = cost / value if value else float("nan")
    summary = (
        f"### Business outcome on the test period\n"
        f"- Calls made: **{out['calls']:,}** of {len(y_test):,} clients\n"
        f"- Subscriptions won: **{out['tp']:,}** (recall {out['recall']:.1%}, precision {out['precision']:.1%})\n"
        f"- Missed subscribers (not called): **{out['fn']:,}**\n"
        f"- Wasted calls: **{out['fp']:,}** x EUR {cost:.0f} = EUR {out['fp'] * cost:,.0f}\n"
        f"- **Profit: EUR {out['profit']:,.0f}** vs call-everyone EUR {everyone['profit']:,.0f} "
        f"(difference EUR {out['profit'] - everyone['profit']:+,.0f})\n"
        f"- Break-even probability for a calibrated model: cost / value = **{break_even:.3f}**\n"
        f"- Threshold chosen on validation for this model: **{meta['thresholds'][model_name]:.2f}**"
    )
    return cm, summary, curve


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
split_md = "\n".join(
    f"- **{k}**: {v['rows']:,} calls, {v['from']} to {v['to']}, subscription rate {v['positive_rate']:.1%}"
    for k, v in meta["splits"].items())

with gr.Blocks() as demo:
    gr.Markdown(
        "# Term-deposit campaign: whom should we call?\n"
        "Logistic regression trained three ways (scikit-learn, manual PyTorch, standard PyTorch). "
        f"Feature set: **{meta['feature_set']}**, C = {meta['C']}. "
        "All models are loaded from `artifacts/`; nothing is retrained here.\n\n" + split_md)

    with gr.Tab("Model comparison"):
        gr.Markdown("### Test-set metrics (thresholds chosen on validation)")
        gr.Dataframe(value=metrics.round(4), interactive=False)
        with gr.Row():
            gr.Plot(value=calibration_fig())
            gr.Plot(value=agreement_fig())
        with gr.Row():
            gr.Plot(value=roc_fig())
            gr.Plot(value=loss_fig())
        gr.Plot(value=coef_fig())
        gr.Markdown("### Validation tuning (feature-set ablation x C)")
        gr.Dataframe(value=tuning.round(4), interactive=False)

    with gr.Tab("Distributions"):
        gr.Plot(value=target_over_time_fig())
        with gr.Row():
            feature_dd = gr.Dropdown(NUMERIC_VIEW + CATEGORICAL_VIEW, value="age", label="Feature")
            split_radio = gr.Radio(list(SPLITS), value="train", label="Split")
        dist_plot = gr.Plot()
        feature_dd.change(distribution_fig, [feature_dd, split_radio], dist_plot)
        split_radio.change(distribution_fig, [feature_dd, split_radio], dist_plot)

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            model_dd = gr.Dropdown(MODEL_NAMES, value="scikit-learn", label="Model")
            thr = gr.Slider(0.01, 0.99, value=meta["thresholds"]["scikit-learn"], step=0.01,
                            label="Decision threshold (call if p >= threshold)")
        with gr.Row():
            value_in = gr.Number(value=meta["sub_value"], label="Value of one subscription (EUR)")
            cost_in = gr.Number(value=meta["call_cost"], label="Cost of one call (EUR)")
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        summary_md = gr.Markdown()
        th_inputs = [model_dd, thr, value_in, cost_in]
        th_outputs = [cm_plot, summary_md, curve_plot]
        for comp in th_inputs:
            comp.change(threshold_view, th_inputs, th_outputs)

    demo.load(distribution_fig, [feature_dd, split_radio], dist_plot)
    demo.load(threshold_view, th_inputs, th_outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=not os.environ.get("CI"))
