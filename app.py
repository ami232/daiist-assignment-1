"""
Assignment 1 - Gradio dashboard (run with `uv run python main.py app`).

Loads ONLY what train.py saved in artifacts/ and never trains anything.
The three saved models re-score the saved test set at startup (inference only),
so every plot below comes from the real saved models.

Tabs:
 1. Model comparison  - metrics table, calibration plot, P(churn) by true class
 2. Distributions     - any feature, coloured by churn
 3. Threshold & gain  - slider -> confusion matrix + net gain vs doing nothing
 4. Coefficients      - top features by |weight| for the three implementations
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from pipeline import ARTIFACTS, TARGET, LogisticRegressionNN, engineer_features, net_gain

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
if not (ARTIFACTS / "model_meta.json").exists():
    raise SystemExit("artifacts/ not found - run `uv run python main.py train` first.")

meta = json.loads((ARTIFACTS / "model_meta.json").read_text())
metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
metrics_table = pd.read_csv(ARTIFACTS / "metrics.csv", index_col=0)
dist = pd.read_csv(ARTIFACTS / "distributions.csv")
test_raw = pd.read_csv(ARTIFACTS / "test_set.csv")

preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_model.joblib")
manual = torch.load(ARTIFACTS / "torch_manual.pt")
nn_model = LogisticRegressionNN(meta["n_features"])
nn_model.load_state_dict(torch.load(ARTIFACTS / "torch_nn.pt"))
nn_model.eval()

# Re-score the test set with the saved preprocessor + models (inference only).
X_test = np.asarray(preprocessor.transform(engineer_features(test_raw)), dtype=np.float64)
y_test = test_raw[TARGET].to_numpy()
mc_test = test_raw["MonthlyCharges"].to_numpy()
with torch.no_grad():
    Xt = torch.tensor(X_test)
    proba = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "torch_manual": torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy(),
        "torch_nn": torch.sigmoid(nn_model(Xt)).numpy(),
    }
MODELS = list(proba)

# Sanity check: the app must reproduce the predictions saved at training time.
saved = pd.read_csv(ARTIFACTS / "test_predictions.csv")
for m in MODELS:
    assert np.allclose(saved[f"p_{m}"], proba[m], atol=1e-6), f"{m} predictions differ from training"

COLORS = {"sklearn": "#1f77b4", "torch_manual": "#ff7f0e", "torch_nn": "#2ca02c"}
CHURN_COLORS = {"No": "#1f77b4", "Yes": "#d62728"}
dist["Churn label"] = dist[TARGET].map({0: "No", 1: "Yes"})


# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------
def calibration_plot():
    """Predicted probability (binned into deciles) vs actual churn rate."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(dash="dash", color="grey")))
    for m in MODELS:
        df = pd.DataFrame({"p": proba[m], "y": y_test})
        df["bin"] = pd.qcut(df["p"], 10, duplicates="drop")
        g = df.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"))
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", name=m,
                                 line=dict(color=COLORS[m])))
    fig.update_layout(title="Calibration on the test set (deciles of predicted P(churn))",
                      xaxis_title="Mean predicted P(churn)", yaxis_title="Actual churn rate",
                      height=450)
    return fig


def proba_hist(model):
    df = pd.DataFrame({"P(churn)": proba[model], "Actual": np.where(y_test == 1, "Yes", "No")})
    fig = px.histogram(df, x="P(churn)", color="Actual", nbins=40, barmode="overlay",
                       opacity=0.6, color_discrete_map=CHURN_COLORS,
                       title=f"{model}: predicted probability by actual outcome (test set)")
    fig.add_vline(x=meta["thresholds"][model], line_dash="dash",
                  annotation_text=f"chosen threshold {meta['thresholds'][model]:.2f}")
    fig.update_layout(height=450)
    return fig


def agreement_table():
    rows = [{"pair": k, **v} for k, v in metrics["agreement"].items()]
    return pd.DataFrame(rows).round(8)


# ---------------------------------------------------------------------------
# Tab 2: distributions
# ---------------------------------------------------------------------------
FEATURE_CHOICES = [c for c in dist.columns if c not in (TARGET, "Churn label", "split")]


def feature_plot(feature):
    col = dist[feature]
    if pd.api.types.is_numeric_dtype(col) and col.nunique() > 10:
        fig = px.histogram(dist, x=feature, color="Churn label", barmode="overlay", opacity=0.6,
                           nbins=40, color_discrete_map=CHURN_COLORS,
                           title=f"{feature} by churn (all 7,043 customers)")
    else:
        rate = dist.groupby(feature)[TARGET].agg(["mean", "size"]).reset_index()
        fig = px.bar(dist, x=feature, color="Churn label", color_discrete_map=CHURN_COLORS,
                     title=f"{feature} by churn (all customers). Churn rate per level: "
                     + ", ".join(f"{r[feature]}={r['mean']:.0%}" for _, r in rate.iterrows()))
        fig.update_xaxes(type="category")
    fig.update_layout(height=450)
    return fig


def target_plot():
    counts = dist["Churn label"].value_counts().reindex(["No", "Yes"])
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                 color_discrete_map=CHURN_COLORS, labels={"x": "Churn", "y": "Customers"},
                 title=f"Target: churn rate {dist[TARGET].mean():.1%}")
    fig.update_layout(height=350, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Tab 3: threshold & net gain
# ---------------------------------------------------------------------------
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def threshold_view(model, threshold, call_cost, horizon, save_rate):
    costs = dict(call_cost=call_cost, horizon_months=horizon, save_rate=save_rate)
    p = proba[model]
    pred = p >= threshold
    tp = int((pred & (y_test == 1)).sum())
    fp = int((pred & (y_test == 0)).sum())
    fn = int((~pred & (y_test == 1)).sum())
    tn = int((~pred & (y_test == 0)).sum())

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["Predicted: stays", "Predicted: churns"],
        y=["Actual: stays", "Actual: churns"], colorscale="Blues", showscale=False,
        text=[[f"TN<br>{tn}", f"FP (wasted call)<br>{fp}"], [f"FN (missed churner)<br>{fn}", f"TP<br>{tp}"]],
        texttemplate="%{text}", textfont=dict(size=16)))
    cm.update_layout(title=f"Confusion matrix on test set, threshold {threshold:.2f}",
                     height=400, yaxis=dict(autorange="reversed"))

    curve = [net_gain(y_test, p >= t, mc_test, **costs) for t in GRID]
    best_t = float(GRID[int(np.argmax(curve))])
    everyone = net_gain(y_test, np.ones_like(pred), mc_test, **costs)
    fig = go.Figure(go.Scatter(x=GRID, y=curve, mode="lines", name=model, line=dict(color=COLORS[model])))
    fig.add_hline(y=0, line_dash="dot", annotation_text="call nobody ($0)")
    fig.add_hline(y=everyone, line_dash="dot", annotation_text="call everyone",
                  annotation_position="bottom right")
    fig.add_vline(x=threshold, line_color="black", annotation_text="current")
    fig.add_vline(x=meta["thresholds"][model], line_dash="dash", line_color="green",
                  annotation_text="chosen on validation", annotation_position="bottom left")
    fig.add_trace(go.Scatter(x=[best_t], y=[max(curve)], mode="markers", marker=dict(size=12, color="red"),
                             name=f"max on test curve ({best_t:.2f})"))
    fig.update_layout(title="Net gain vs doing nothing, test set", xaxis_title="Threshold",
                      yaxis_title="Net gain ($)", height=400)

    gain = net_gain(y_test, pred, mc_test, **costs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    summary = (
        f"### Net gain vs calling nobody: ${gain:,.0f}\n"
        f"- Customers contacted: **{tp + fp}** of {len(p)}\n"
        f"- Recall (churners caught): **{recall:.1%}**, precision (contacts who were churners): **{precision:.1%}**\n"
        f"- Baselines: call nobody $0, call everyone ${everyone:,.0f}\n"
        f"- Extra gain vs best baseline: **${gain - max(0.0, everyone):,.0f}**\n"
        f"- Threshold chosen on validation under the default assumptions: **{meta['thresholds'][model]:.2f}**; "
        f"gain-maximising threshold on this test curve with the assumptions above: **{best_t:.2f}**"
    )
    return cm, fig, summary


# ---------------------------------------------------------------------------
# Tab 4: coefficients
# ---------------------------------------------------------------------------
def coef_plot(top_n=15):
    coefs = pd.DataFrame(metrics["coefficients"]).drop(index="intercept")
    top = coefs.reindex(coefs["sklearn"].abs().sort_values(ascending=False).index)[:top_n]
    long = top.reset_index(names="feature").melt(id_vars="feature", var_name="model", value_name="weight")
    fig = px.bar(long, x="weight", y="feature", color="model", barmode="group", orientation="h",
                 color_discrete_map=COLORS,
                 title=f"Top {top_n} coefficients by |weight| (standardised / one-hot features; "
                       "positive = raises churn odds)")
    fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
c = meta["costs"]
with gr.Blocks(title="Telco churn: retention targeting") as demo:
    gr.Markdown(
        "# Telco churn: who should get a retention offer?\n"
        "Logistic regression trained three ways (scikit-learn, manual PyTorch, nn.Module + optim) "
        "on the same split. All numbers are on the held-out **test set** "
        f"({meta['split']['test']} customers). Default cost assumptions: call cost ${c['call_cost']:.0f}, "
        f"lost revenue = MonthlyCharges x {c['horizon_months']} months, offer saves "
        f"{c['save_rate']:.0%} of contacted churners."
    )
    with gr.Tab("Model comparison"):
        gr.Markdown("**Test metrics** (each model at its own gain-maximising threshold chosen on validation; net gain is vs calling nobody)")
        gr.Dataframe(metrics_table.reset_index(names="model").round(4))
        gr.Markdown("**Agreement between implementations** (max absolute differences)")
        gr.Dataframe(agreement_table())
        with gr.Row():
            gr.Plot(calibration_plot())
            with gr.Column():
                hist_model = gr.Dropdown(MODELS, value="sklearn", label="Model")
                hist_plot = gr.Plot(proba_hist("sklearn"))
        hist_model.change(proba_hist, hist_model, hist_plot)

    with gr.Tab("Distributions"):
        with gr.Row():
            feat = gr.Dropdown(FEATURE_CHOICES, value="log_tenure", label="Feature")
        feat_plot = gr.Plot(feature_plot("log_tenure"))
        gr.Plot(target_plot())
        feat.change(feature_plot, feat, feat_plot)

    with gr.Tab("Threshold & net gain"):
        with gr.Row():
            t_model = gr.Dropdown(MODELS, value="sklearn", label="Model")
            t_slider = gr.Slider(0.0, 1.0, value=meta["thresholds"]["sklearn"], step=0.01,
                                 label="Decision threshold: contact if P(churn) >= threshold")
        with gr.Row():
            in_call = gr.Number(c["call_cost"], label="Call cost ($)")
            in_horizon = gr.Number(c["horizon_months"], label="Revenue horizon (months)")
            in_save = gr.Slider(0.0, 1.0, value=c["save_rate"], step=0.05, label="Save rate of offer")
        t_summary = gr.Markdown()
        with gr.Row():
            t_cm = gr.Plot()
            t_curve = gr.Plot()
        inputs = [t_model, t_slider, in_call, in_horizon, in_save]
        for comp in inputs:
            comp.change(threshold_view, inputs, [t_cm, t_curve, t_summary])
        demo.load(threshold_view, inputs, [t_cm, t_curve, t_summary])

    with gr.Tab("Coefficients"):
        gr.Plot(coef_plot())

if __name__ == "__main__":
    # 127.0.0.1:7860 is what CI checks; inbrowser=False so it never blocks headless.
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
