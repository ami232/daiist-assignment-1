"""
Assignment 1 -- Gradio dashboard (run with: uv run python main.py app).

Loads what train.py saved in artifacts/ (preprocessor, the three models, the
engineered dataset with its split labels) and only ever does INFERENCE. Nothing
is trained or tuned here.

Tabs
  Models               metrics table, prediction-vs-actual (calibration), score distributions
  Do the three agree?  coefficient / probability / objective agreement between the models
  Data                 target and feature distributions (train split)
  Threshold and cost   threshold slider -> confusion matrix and business cost
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

import gradio as gr

import common as cm
from features import TARGET

# ----------------------------------------------------------- load artifacts ----
REQUIRED = [
    "config.json", "metrics.json", "coefficients.csv", "dataset_engineered.csv.gz",
    "preprocessor.joblib", "sklearn_logreg.joblib", "manual_torch.pt", "nn_torch.pt",
]
missing = [f for f in REQUIRED if not (cm.ARTIFACTS / f).exists()]
if missing:
    raise FileNotFoundError(
        f"Missing {missing} in {cm.ARTIFACTS}. Run `uv run python main.py train` first."
    )

CONFIG = json.loads((cm.ARTIFACTS / "config.json").read_text())
METRICS = json.loads((cm.ARTIFACTS / "metrics.json").read_text())
COEF = pd.read_csv(cm.ARTIFACTS / "coefficients.csv")
DATA = pd.read_csv(cm.ARTIFACTS / "dataset_engineered.csv.gz")
PRE = joblib.load(cm.ARTIFACTS / "preprocessor.joblib")
SK = joblib.load(cm.ARTIFACTS / "sklearn_logreg.joblib")
_man = torch.load(cm.ARTIFACTS / "manual_torch.pt", weights_only=True)
_nn_ckpt = torch.load(cm.ARTIFACTS / "nn_torch.pt", weights_only=True)
NN = cm.LogisticRegressionNN(_nn_ckpt["in_features"]).to(cm.DTYPE)
NN.load_state_dict(_nn_ckpt["state_dict"])
NN.eval()

MODELS = ["scikit-learn", "manual PyTorch", "nn.Module + optim"]
COST_FN, COST_FP = CONFIG["cost_fn"], CONFIG["cost_fp"]  # the costs the thresholds were chosen with

COLORS = {"scikit-learn": "#0072B2", "manual PyTorch": "#E69F00", "nn.Module + optim": "#009E73"}
GREY, PAYS, DEFAULTS = "#8D99AE", "#6C7A89", "#D55E00"

FEATURES = DATA.drop(columns=[TARGET, "split"])
TEST = DATA["split"] == "test"
X_TEST = PRE.transform(FEATURES[TEST])
Y_TEST = DATA.loc[TEST, TARGET].to_numpy().astype(int)


def _predict_all(X: np.ndarray) -> dict[str, np.ndarray]:
    Xt = torch.tensor(X, dtype=cm.DTYPE)
    with torch.no_grad():
        return {
            "scikit-learn": SK.predict_proba(X)[:, 1],
            "manual PyTorch": torch.sigmoid(Xt @ _man["w"] + _man["b"]).numpy().ravel(),
            "nn.Module + optim": torch.sigmoid(NN(Xt)).numpy().ravel(),
        }


PROBA = _predict_all(X_TEST)  # inference on the saved test split
THRESHOLDS = np.linspace(0.01, 0.99, 99)
COST_CURVES = {
    m: cm.cost_curve(Y_TEST, PROBA[m], THRESHOLDS, COST_FN, COST_FP)[1] / len(Y_TEST) for m in MODELS
}
BASELINE = METRICS["results"]["naive baseline"]["test"]


# ------------------------------------------------------------------ helpers ----
def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13),
    )
    return fig


# ------------------------------------------------------------- tab 1: models ----
def metrics_table() -> pd.DataFrame:
    rows = []
    for name in MODELS + ["naive baseline"]:
        r = METRICS["results"][name]["test"]
        rows.append({
            "Model": name,
            "ROC AUC": round(r["roc_auc"], 4),
            "PR AUC": round(r["pr_auc"], 4),
            "Log loss": round(r["log_loss"], 4),
            "Brier": round(r["brier"], 4),
            "Threshold": "-" if r["threshold"] is None else round(r["threshold"], 2),
            "Recall": "-" if "recall" not in r else round(r["recall"], 3),
            "Precision": "-" if "precision" not in r else round(r["precision"], 3),
            "Cost per client": round(r["cost_per_client"], 4),
        })
    return pd.DataFrame(rows)


def calibration_fig() -> go.Figure:
    """Prediction vs actual: mean predicted probability against observed default rate."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color=GREY, dash="dash")))
    for m in MODELS:
        bins = pd.qcut(PROBA[m], 10, duplicates="drop")
        g = pd.DataFrame({"p": PROBA[m], "y": Y_TEST}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", name=m,
                                 line=dict(color=COLORS[m]), marker=dict(size=8)))
    fig.update_layout(title="Predicted vs actual default rate (test set, 10 equal-size bins)",
                      xaxis_title="Mean predicted probability of default",
                      yaxis_title="Observed default rate",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    return _style(fig, 460)


def score_distribution_fig() -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=MODELS, shared_yaxes=True)
    for i, m in enumerate(MODELS, start=1):
        for label, val, color in [("actually pays", 0, PAYS), ("actually defaults", 1, DEFAULTS)]:
            fig.add_trace(
                go.Histogram(x=PROBA[m][Y_TEST == val], name=label, marker_color=color, opacity=0.65,
                             histnorm="probability", xbins=dict(start=0, end=1, size=0.025),
                             showlegend=(i == 1), legendgroup=label),
                row=1, col=i,
            )
        fig.add_vline(x=CONFIG["thresholds"][m], line_dash="dash", line_color="black", row=1, col=i)
        fig.update_xaxes(title_text="Predicted probability", range=[0, 1], row=1, col=i)
    fig.update_yaxes(title_text="Share of that group", row=1, col=1)
    fig.update_layout(barmode="overlay",
                      title="Predicted probability by actual outcome (dashed = chosen threshold)")
    return _style(fig, 400)


# ---------------------------------------------------------- tab 2: agreement ----
def agreement_table() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        a = METRICS["agreement"][m]
        rows.append({
            "Model": m,
            "Training objective": round(a["objective"], 6),
            "Gap vs scikit-learn": f"{a['objective_gap_vs_sklearn']:+.2e}",
            "Max |coef diff|": round(a["max_abs_coef_diff_vs_sklearn"], 4),
            "Max |proba diff| (test)": round(a["max_abs_test_proba_diff_vs_sklearn"], 4),
            "Proba correlation": round(a["test_proba_correlation_vs_sklearn"], 5),
        })
    return pd.DataFrame(rows)


def agreement_scatter_fig() -> go.Figure:
    rng = np.random.default_rng(cm.SEED)
    idx = rng.choice(len(Y_TEST), size=min(3000, len(Y_TEST)), replace=False)
    others = [m for m in MODELS if m != "scikit-learn"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{m} vs scikit-learn" for m in others])
    for i, m in enumerate(others, start=1):
        fig.add_trace(go.Scattergl(x=PROBA["scikit-learn"][idx], y=PROBA[m][idx], mode="markers",
                                   marker=dict(size=4, color=COLORS[m], opacity=0.5),
                                   showlegend=False), row=1, col=i)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash="dash"),
                                 showlegend=False), row=1, col=i)
        fig.update_xaxes(title_text="scikit-learn probability", range=[0, 1], row=1, col=i)
        fig.update_yaxes(title_text=f"{m} probability", range=[0, 1], row=1, col=i)
    fig.update_layout(title="Same test clients, probability from each model (3,000 sampled)")
    return _style(fig, 420)


def coefficient_fig(top: int = 15) -> go.Figure:
    c = COEF.set_index("feature")
    order = c.abs().max(axis=1).sort_values(ascending=False).head(top).index[::-1]
    fig = go.Figure()
    for m in MODELS:
        fig.add_trace(go.Bar(y=list(order), x=c.loc[order, m], name=m, orientation="h",
                             marker_color=COLORS[m]))
    fig.update_layout(barmode="group",
                      title=f"Coefficients on standardised inputs (top {top} by size)",
                      xaxis_title="Coefficient")
    return _style(fig, 140 + 34 * top)


# ----------------------------------------------------------------- tab 3: data ----
TRAIN = DATA[DATA["split"] == "train"]
FEATURE_COLS = list(FEATURES.columns)


def target_fig() -> go.Figure:
    g = DATA.groupby("split")[TARGET].agg(["mean", "size"]).reindex(["train", "val", "test"])
    fig = go.Figure(go.Bar(x=g.index, y=g["mean"] * 100, marker_color=DEFAULTS,
                           text=[f"{r['mean']*100:.1f}%  (n={int(r['size']):,})" for _, r in g.iterrows()],
                           textposition="outside"))
    fig.update_layout(title="Target: share of clients who default, by split",
                      yaxis_title="Default rate (%)", yaxis=dict(range=[0, 30]))
    return _style(fig, 340)


def feature_fig(feature: str) -> go.Figure:
    s, y = TRAIN[feature], TRAIN[TARGET]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{feature}: distribution", f"{feature}: default rate"])
    if s.nunique() <= 15:  # codes / categories: one bar per value
        counts = pd.crosstab(s, y, normalize="columns") * 100
        for val, label, color in [(0, "pays", PAYS), (1, "defaults", DEFAULTS)]:
            fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts[val], name=label,
                                 marker_color=color, opacity=0.8), row=1, col=1)
        rate = y.groupby(s).agg(["mean", "size"])
        x_rate = rate.index.astype(str)
    else:  # continuous: clip to 1st-99th percentile for the histogram, decile bins for the rate
        lo, hi = s.quantile([0.01, 0.99])
        clipped = s.clip(lo, hi)
        for val, label, color in [(0, "pays", PAYS), (1, "defaults", DEFAULTS)]:
            fig.add_trace(go.Histogram(x=clipped[y == val], name=label, marker_color=color, opacity=0.65,
                                       histnorm="percent", nbinsx=40), row=1, col=1)
        bins = pd.qcut(s, 10, duplicates="drop")
        rate = y.groupby(bins, observed=True).agg(["mean", "size"])
        x_rate = [f"{iv.left:,.0f} to {iv.right:,.0f}" for iv in rate.index]
    fig.add_trace(go.Bar(x=list(x_rate), y=rate["mean"] * 100, marker_color=DEFAULTS, showlegend=False,
                         hovertext=[f"n = {n:,}" for n in rate["size"]]), row=1, col=2)
    fig.add_hline(y=y.mean() * 100, line_dash="dash", line_color="black", row=1, col=2)
    fig.update_yaxes(title_text="% of that group", row=1, col=1)
    fig.update_yaxes(title_text="Default rate (%)", row=1, col=2)
    fig.update_layout(barmode="overlay" if s.nunique() > 15 else "group")
    return _style(fig, 400)


# ------------------------------------------------------ tab 4: threshold/cost ----
def threshold_view(model: str, threshold: float):
    proba = PROBA[model]
    pred = (proba >= threshold).astype(int)  # 1 = predicted to default -> reject
    tn, fp, fn, tp = cm.confusion_counts(Y_TEST, pred)
    n = len(Y_TEST)
    cost = fn * COST_FN + fp * COST_FP
    base = BASELINE["cost_total"]

    z = [[tn, fp], [fn, tp]]
    cm_fig = go.Figure(go.Heatmap(
        z=z, x=["Approve (predicted pays)", "Reject (predicted defaults)"],
        y=["Actually pays", "Actually defaults"],
        text=[[f"{tn:,}", f"{fp:,}<br>rejected a good client"],
              [f"{fn:,}<br>approved a defaulter", f"{tp:,}"]],
        texttemplate="%{text}", colorscale=[[0, "#F3F5F8"], [1, COLORS[model]]],
        showscale=False, hoverinfo="skip", xgap=3, ygap=3,
    ))
    cm_fig.update_layout(title=f"Confusion matrix at threshold {threshold:.2f} ({model}, test set)")
    cm_fig.update_yaxes(autorange="reversed")
    _style(cm_fig, 380)

    curve = go.Figure()
    curve.add_trace(go.Scatter(x=THRESHOLDS, y=COST_CURVES[model], mode="lines", name="cost per client",
                               line=dict(color=COLORS[model], width=3)))
    curve.add_hline(y=BASELINE["cost_per_client"], line_dash="dot", line_color=GREY,
                    annotation_text=f"naive baseline ({BASELINE['policy']})", annotation_position="top left")
    curve.add_vline(x=CONFIG["thresholds"][model], line_dash="dash", line_color="black",
                    annotation_text="chosen on validation", annotation_position="bottom right")
    curve.add_trace(go.Scatter(x=[threshold], y=[cost / n], mode="markers", name="current threshold",
                               marker=dict(size=14, color=DEFAULTS, line=dict(width=2, color="white"))))
    curve.update_layout(title="Business cost per client as the threshold moves (test set)",
                        xaxis_title="Threshold (reject if predicted default probability is at least this)",
                        yaxis_title="Cost per client")
    _style(curve, 380)

    saving = (1 - cost / base) * 100 if base else 0.0
    vs_base = f"{abs(saving):.1f}% {'cheaper' if saving >= 0 else 'more expensive'} than the baseline"
    summary = (
        f"**Business cost: {cost:,.0f}** in total, **{cost / n:.4f} per client**, {vs_base} "
        f"(naive baseline, {BASELINE['policy']}: {base:,.0f})\n\n"
        f"Cost model: approving a client who defaults costs **{COST_FN:g}**, rejecting a client who "
        f"would have paid costs **{COST_FP:g}** (from `common.py`).\n\n"
        f"Rejected {tp + fp:,} of {n:,} clients ({(tp + fp) / n:.1%}). "
        f"Recall {tp / max(tp + fn, 1):.1%}, precision {tp / max(tp + fp, 1):.1%}, "
        f"accuracy {(tp + tn) / n:.1%}."
    )
    return cm_fig, curve, summary


def optimal_threshold(model: str) -> float:
    return CONFIG["thresholds"][model]


# ----------------------------------------------------------------------- UI ----
with gr.Blocks(title="Credit default: one model, three implementations") as demo:
    gr.Markdown(
        "# Credit card default: one model, three implementations\n"
        "L2-regularised logistic regression trained with scikit-learn, a manual PyTorch loop and "
        "`nn.Module` + `optim`, on the same split. This dashboard only loads the saved models; "
        "it never trains."
    )

    with gr.Tab("Models"):
        gr.Markdown("Test-set results. The threshold for each model was chosen on the validation set "
                    "to minimise business cost.")
        gr.Dataframe(value=metrics_table(), interactive=False, wrap=True)
        gr.Plot(value=calibration_fig())
        gr.Plot(value=score_distribution_fig())

    with gr.Tab("Do the three agree?"):
        gr.Markdown("All three minimise the same objective, so they should land on (almost) the same "
                    "solution. Gaps here are differences in optimisation, not in the model.")
        gr.Dataframe(value=agreement_table(), interactive=False, wrap=True)
        gr.Plot(value=agreement_scatter_fig())
        gr.Plot(value=coefficient_fig())

    with gr.Tab("Data"):
        gr.Markdown("Distributions use the training split only, so nothing is learned from validation "
                    "or test data.")
        gr.Plot(value=target_fig())
        feature_dd = gr.Dropdown(FEATURE_COLS, value=FEATURE_COLS[0], label="Feature")
        feature_plot = gr.Plot(value=feature_fig(FEATURE_COLS[0]))
        feature_dd.change(feature_fig, feature_dd, feature_plot)

    with gr.Tab("Threshold and cost"):
        with gr.Row():
            model_dd = gr.Dropdown(MODELS, value=MODELS[0], label="Model", scale=1)
            thr = gr.Slider(0.01, 0.99, step=0.01, value=optimal_threshold(MODELS[0]),
                            label="Decision threshold", scale=3)
            reset_btn = gr.Button("Use threshold chosen on validation", scale=1)
        summary_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()

        outputs = [cm_plot, cost_plot, summary_md]
        thr.input(threshold_view, [model_dd, thr], outputs)
        model_dd.change(optimal_threshold, model_dd, thr).then(threshold_view, [model_dd, thr], outputs)
        reset_btn.click(optimal_threshold, model_dd, thr).then(threshold_view, [model_dd, thr], outputs)
        demo.load(threshold_view, [model_dd, thr], outputs)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
