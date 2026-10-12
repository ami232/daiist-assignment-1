"""
Assignment 1 — Gradio dashboard.

    uv run python main.py app

Loads only what train.py saved in artifacts/. Nothing is trained here: the
three models are loaded from disk and used for prediction on the saved test set.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from common import (ARTIFACTS, DATA_PATH, TARGET, LogisticRegressionNet,
                    engineer_features, make_xy, sigmoid)

LABELS = {
    "sklearn": "scikit-learn",
    "torch_manual": "PyTorch, manual loop",
    "torch_module": "PyTorch, nn.Module",
}
COLORS = {"sklearn": "#1f77b4", "torch_manual": "#ff7f0e", "torch_module": "#2ca02c"}

# ---------------------------------------------------------------------------
# Load artifacts (fail with a clear message if training hasn't run)
# ---------------------------------------------------------------------------
if not (ARTIFACTS / "meta.json").exists():
    raise SystemExit("No artifacts found. Run `uv run python main.py train` first.")

meta = json.loads((ARTIFACTS / "meta.json").read_text())
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
manual = torch.load(ARTIFACTS / "torch_manual.pt")
module = LogisticRegressionNet(meta["n_features"]).double()
module.load_state_dict(torch.load(ARTIFACTS / "torch_module.pt"))
module.eval()

# Rerun the same feature pipeline on the raw test rows, then predict.
test_raw = pd.read_csv(ARTIFACTS / "test_raw.csv")
X_test_df, y_test = make_xy(test_raw)
X_test = preprocessor.transform(X_test_df)
with torch.no_grad():
    PROBA = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "torch_manual": sigmoid(X_test @ manual["w"].numpy() + manual["b"].numpy()),
        "torch_module": torch.sigmoid(module(torch.tensor(X_test, dtype=torch.float64))).numpy(),
    }

DATA = engineer_features(pd.read_csv(DATA_PATH).dropna(subset=[TARGET]))
FEATURE_COLUMNS = [c for c in DATA.columns if c != TARGET]


# ---------------------------------------------------------------------------
# Model comparison views
# ---------------------------------------------------------------------------
def results_table() -> pd.DataFrame:
    df = pd.DataFrame(meta["results"])
    df["model"] = df["model"].map(lambda m: LABELS.get(m, m.replace("_", " ")))
    cols = ["model", "roc_auc", "pr_auc", "log_loss", "brier", "threshold",
            "precision", "recall", "flag_rate", "cost_per_1000"]
    return df[cols].round(4)


def calibration_plot() -> go.Figure:
    """Predicted vs actual: within each probability bin, the average predicted
    risk (x) against the share of people who actually had a heart attack (y).
    Points on the diagonal mean the probabilities can be taken at face value."""
    fig = go.Figure()
    edges = np.unique(np.quantile(PROBA["sklearn"], np.linspace(0, 1, 11)))
    for name, p in PROBA.items():
        bins = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        grouped = pd.DataFrame({"bin": bins, "p": p, "y": y_test}).groupby("bin")
        fig.add_scatter(x=grouped["p"].mean(), y=grouped["y"].mean(), mode="lines+markers",
                        name=LABELS[name], line=dict(color=COLORS[name]),
                        customdata=grouped.size(), hovertemplate=(
                            "predicted %{x:.3f}<br>actual %{y:.3f}<br>n=%{customdata}"))
    top = max(max(p.max() for p in PROBA.values()), y_test.mean()) * 1.05
    fig.add_scatter(x=[0, top], y=[0, top], mode="lines", name="perfect calibration",
                    line=dict(dash="dash", color="gray"))
    fig.update_layout(title="Predicted risk vs actual rate (test set, deciles)",
                      xaxis_title="Mean predicted probability",
                      yaxis_title="Actual share with heart attack", height=420)
    return fig


def probability_by_class_plot() -> go.Figure:
    long = pd.concat([pd.DataFrame({"model": LABELS[k], "probability": v,
                                    "actual": np.where(y_test == 1, "Yes", "No")})
                      for k, v in PROBA.items()])
    fig = px.histogram(long, x="probability", color="actual", facet_col="model",
                       histnorm="probability density", barmode="overlay", nbins=40,
                       opacity=0.6, color_discrete_map={"No": "#7f7f7f", "Yes": "#d62728"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(title="Predicted probability, split by what actually happened",
                      height=380)
    return fig


def agreement_plot() -> go.Figure:
    fig = go.Figure()
    for name in ("torch_manual", "torch_module"):
        fig.add_scatter(x=PROBA["sklearn"], y=PROBA[name], mode="markers",
                        name=LABELS[name], marker=dict(size=4, opacity=0.5,
                                                       color=COLORS[name]))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical",
                    line=dict(dash="dash", color="gray"))
    fig.update_layout(title="Do the three implementations predict the same thing?",
                      xaxis_title="scikit-learn probability",
                      yaxis_title="PyTorch probability", height=420)
    return fig


def coefficient_plot(top_k: int) -> go.Figure:
    names = np.array(meta["feature_names"])
    sk = np.array(meta["coefs"]["sklearn"])
    order = np.argsort(np.abs(sk))[::-1][: int(top_k)][::-1]
    fig = go.Figure()
    for name in LABELS:
        fig.add_bar(y=names[order], x=np.array(meta["coefs"][name])[order],
                    orientation="h", name=LABELS[name], marker_color=COLORS[name])
    fig.update_layout(barmode="group", height=max(400, 22 * int(top_k)),
                      title=f"Top {int(top_k)} coefficients (log-odds per 1 std. dev.)",
                      xaxis_title="Coefficient")
    return fig


def convergence_plot() -> go.Figure:
    target = meta["history"]["sklearn_final"]
    fig = go.Figure()
    for name in ("torch_manual", "torch_module"):
        gap = np.maximum(np.array(meta["history"][name]) - target, 1e-12)
        fig.add_scatter(y=gap, mode="lines", name=LABELS[name],
                        line=dict(color=COLORS[name],
                                  dash="dot" if name == "torch_module" else "solid"))
    fig.update_yaxes(type="log")
    fig.update_layout(title="Training objective minus scikit-learn's optimum",
                      xaxis_title="Epoch", yaxis_title="Gap (log scale)", height=380)
    return fig


def agreement_text() -> str:
    a = meta["agreement"]
    lines = ["| Pair | Max prob. difference | Max coef. difference |", "|---|---|---|"]
    for pair in a["max_abs_prob_diff"]:
        pretty = " vs ".join(LABELS[p] for p in pair.split(" vs "))
        lines.append(f"| {pretty} | {a['max_abs_prob_diff'][pair]:.2e} | "
                     f"{a['max_abs_coef_diff'][pair]:.2e} |")
    lines.append(f"\nC = {meta['C']}, learning rate = {meta['lr']}, "
                 f"epochs = {meta['epochs']}, features = {meta['n_features']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Threshold and business cost
# ---------------------------------------------------------------------------
def threshold_view(model: str, threshold: float, cost_fn: float, cost_fp: float):
    p = PROBA[model]
    flagged = p >= threshold
    tp = int((flagged & (y_test == 1)).sum())
    fp = int((flagged & (y_test == 0)).sum())
    fn = int((~flagged & (y_test == 1)).sum())
    tn = int((~flagged & (y_test == 0)).sum())
    cost = cost_fn * fn + cost_fp * fp

    cm = px.imshow([[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
                   x=["Predicted no", "Predicted yes"], y=["Actually no", "Actually yes"])
    cm.update_layout(title=f"Confusion matrix at threshold {threshold:.2f}",
                     coloraxis_showscale=False, height=360)

    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [cost_fn * ((p < t) & (y_test == 1)).sum() + cost_fp * ((p >= t) & (y_test == 0)).sum()
             for t in grid]
    curve = go.Figure(go.Scatter(x=grid, y=np.array(costs) * 1000 / len(y_test),
                                 mode="lines", name="cost"))
    curve.add_vline(x=threshold, line_dash="dash", line_color="red",
                    annotation_text="current")
    curve.add_vline(x=meta["thresholds"][model], line_dash="dot", line_color="gray",
                    annotation_text="chosen on validation", annotation_position="bottom right")
    curve.update_layout(title="Business cost per 1,000 people vs threshold (test set)",
                        xaxis_title="Threshold", yaxis_title="Cost per 1,000", height=360)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    summary = (
        f"**Total cost on the test set: {cost:,.0f}** "
        f"({cost * 1000 / len(y_test):,.1f} per 1,000 people)\n\n"
        f"Flagged {tp + fp:,} of {len(y_test):,} people ({(tp + fp) / len(y_test):.1%}). "
        f"Caught {tp:,} of {tp + fn:,} actual cases (recall {recall:.1%}); "
        f"{precision:.1%} of flagged people were actual cases (precision).\n\n"
        f"For comparison: flagging nobody costs {cost_fn * (tp + fn):,.0f}, "
        f"flagging everyone costs {cost_fp * (tn + fp):,.0f}."
    )
    return cm, curve, summary


def reset_threshold(model: str):
    return meta["thresholds"][model]


# ---------------------------------------------------------------------------
# Data distributions
# ---------------------------------------------------------------------------
def target_plot() -> go.Figure:
    counts = DATA[TARGET].value_counts()
    fig = px.bar(x=counts.index, y=counts.values, text=[f"{v / counts.sum():.1%}" for v in counts],
                 labels={"x": TARGET, "y": "People"})
    fig.update_layout(title=f"{TARGET}: how rare is 'Yes'?", height=360)
    return fig


def feature_plot(column: str) -> go.Figure:
    df = DATA[[column, TARGET]].dropna()
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(df, x=column, color=TARGET, barmode="overlay", histnorm="percent",
                           nbins=40, opacity=0.6,
                           color_discrete_map={"No": "#7f7f7f", "Yes": "#d62728"})
        fig.update_layout(title=f"{column}: distribution within each group "
                                f"(each colour sums to 100%)", height=420)
    else:
        rate = (df.groupby(column)[TARGET].agg(rate=lambda s: (s == "Yes").mean(), n="size")
                .reset_index().sort_values(column))
        fig = px.bar(rate, x=column, y="rate", hover_data=["n"],
                     labels={"rate": f"Share with {TARGET} = Yes"})
        fig.add_hline(y=(df[TARGET] == "Yes").mean(), line_dash="dash",
                      annotation_text="overall rate")
        fig.update_layout(title=f"{column}: heart-attack rate per category", height=420)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
default_model = "sklearn"
with gr.Blocks(title="Heart attack risk: three logistic regressions") as demo:
    gr.Markdown("# Heart attack risk: one model, three implementations\n"
                "Everything here comes from models saved by `train.py`, evaluated on the "
                "held-out test set. Nothing is retrained.")

    with gr.Tab("Compare models"):
        gr.Dataframe(value=results_table(), label="Test-set metrics", interactive=False)
        gr.Markdown(agreement_text())
        with gr.Row():
            gr.Plot(calibration_plot())
            gr.Plot(agreement_plot())
        gr.Plot(probability_by_class_plot())
        with gr.Row():
            gr.Plot(convergence_plot())
        top_k = gr.Slider(5, min(40, meta["n_features"]), value=15, step=1,
                          label="Number of coefficients to show")
        coef = gr.Plot(coefficient_plot(15))
        top_k.change(coefficient_plot, top_k, coef)

    with gr.Tab("Threshold and cost"):
        with gr.Row():
            model_in = gr.Dropdown([(v, k) for k, v in LABELS.items()], value=default_model,
                                   label="Model")
            thr_in = gr.Slider(0.01, 0.99, value=meta["thresholds"][default_model], step=0.01,
                               label="Decision threshold: flag if predicted risk ≥ this")
        with gr.Row():
            fn_in = gr.Number(value=meta["cost_fn"], label="Cost of a missed case (false negative)")
            fp_in = gr.Number(value=meta["cost_fp"], label="Cost of a false alarm (false positive)")
        summary_out = gr.Markdown()
        with gr.Row():
            cm_out = gr.Plot()
            curve_out = gr.Plot()
        inputs = [model_in, thr_in, fn_in, fp_in]
        outputs = [cm_out, curve_out, summary_out]
        for component in (thr_in, fn_in, fp_in):
            component.change(threshold_view, inputs, outputs)
        model_in.change(reset_threshold, model_in, thr_in).then(threshold_view, inputs, outputs)
        demo.load(threshold_view, inputs, outputs)

    with gr.Tab("Explore the data"):
        gr.Plot(target_plot())
        col_in = gr.Dropdown(FEATURE_COLUMNS, value=FEATURE_COLUMNS[0], label="Column")
        col_plot = gr.Plot(feature_plot(FEATURE_COLUMNS[0]))
        col_in.change(feature_plot, col_in, col_plot)

if __name__ == "__main__":
    demo.launch()
