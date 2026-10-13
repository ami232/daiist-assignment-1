"""
Assignment 1 — Gradio dashboard for the speed-dating match model.

Loads the artifacts written by train.ipynb (three fitted logistic regressions,
the fitted scaler, the engineered feature table and the training metrics) and
only *scores* with them: nothing is trained or refitted here.

Run with:  uv run python main.py app
"""

import json
import threading
from functools import wraps
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ARTIFACTS = Path("artifacts")
TARGET = "match"
MODEL_NAMES = ["scikit-learn", "Manual PyTorch", "Standard PyTorch"]
# Colour-blind-safe (Okabe-Ito) colours, one per model.
COLORS = {"Naive baseline": "#999999", "scikit-learn": "#0072B2",
          "Manual PyTorch": "#E69F00", "Standard PyTorch": "#009E73"}

if not (ARTIFACTS / "metrics.json").exists():
    raise SystemExit("No artifacts found. Run `uv run python main.py train` first.")


# ---------------------------------------------------------------------------
# Load artifacts (no training)
# ---------------------------------------------------------------------------
class LogisticRegressionModel(nn.Module):
    """Same architecture as in train.ipynb, so the saved state_dict fits it."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
FEATURES = metrics["features"]
COST_FP, COST_FN = metrics["cost_fp"], metrics["cost_fn"]

table = pd.read_csv(ARTIFACTS / "features_table.csv")
scaler = joblib.load(ARTIFACTS / "scaler.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_model.joblib")
manual = torch.load(ARTIFACTS / "manual_torch.pt")
std_model = LogisticRegressionModel(len(FEATURES))
std_model.load_state_dict(torch.load(ARTIFACTS / "standard_torch.pt"))
std_model.eval()

coefs = pd.read_csv(ARTIFACTS / "coefficients.csv")
loss_curves = pd.read_csv(ARTIFACTS / "loss_curves.csv")
ablation = pd.read_csv(ARTIFACTS / "feature_ablation.csv")

# Score every row once with each saved model.
X = torch.from_numpy(scaler.transform(table[FEATURES]).astype(np.float32))
base_rate = table.loc[table["split"] == "train", TARGET].mean()
with torch.no_grad():
    table["p_scikit-learn"] = sk_model.predict_proba(X.numpy())[:, 1]
    table["p_Manual PyTorch"] = torch.sigmoid(X @ manual["w"] + manual["b"]).numpy()
    table["p_Standard PyTorch"] = torch.sigmoid(std_model(X)).numpy()
table["p_Naive baseline"] = base_rate


def rows(split):
    return table[table["split"] == split]


# Plotly Express isn't thread-safe and Gradio runs events in parallel threads,
# so figure-building functions take turns.
_plot_lock = threading.Lock()


def one_at_a_time(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with _plot_lock:
            return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Figures and callbacks
# ---------------------------------------------------------------------------
def results_table():
    df = pd.DataFrame(metrics["results"])
    df = df[["model", "roc_auc", "pr_auc", "threshold", "tp", "fp", "fn", "tn",
             "business_cost", "cost_per_date"]]
    return df.round(3)


@one_at_a_time
def calibration_plot(split, n_bins):
    """Predicted probability (binned) vs. the actual match rate in each bin."""
    data = rows(split)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(dash="dash", color="#777777")))
    for name in MODEL_NAMES:
        p = data[f"p_{name}"]
        bins = pd.qcut(p, q=int(n_bins), duplicates="drop")
        grouped = data.groupby(bins, observed=True).agg(
            predicted=(f"p_{name}", "mean"), actual=(TARGET, "mean"), n=(TARGET, "size"))
        fig.add_trace(go.Scatter(
            x=grouped["predicted"], y=grouped["actual"], mode="lines+markers", name=name,
            line=dict(color=COLORS[name]), customdata=grouped["n"],
            hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>n=%{customdata}"))
    fig.update_layout(title=f"Predicted vs. actual match rate ({split} set)",
                      xaxis_title="Mean predicted probability (bin)",
                      yaxis_title="Actual match rate (bin)",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], legend_title="Model")
    return fig


@one_at_a_time
def probability_by_outcome(split):
    """Distribution of predicted probabilities, split by the actual outcome."""
    data = rows(split)
    long = data.melt(id_vars=[TARGET], value_vars=[f"p_{n}" for n in MODEL_NAMES],
                     var_name="model", value_name="predicted probability")
    long["model"] = long["model"].str.removeprefix("p_")
    long["actual"] = long[TARGET].map({0: "No match", 1: "Match"})
    fig = px.box(long, x="model", y="predicted probability", color="actual",
                 color_discrete_map={"No match": "#999999", "Match": "#D55E00"},
                 title=f"Predicted probability by actual outcome ({split} set)")
    return fig


@one_at_a_time
def agreement_plot(split):
    """Each PyTorch model's probabilities against scikit-learn's."""
    data = rows(split)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Identical",
                             line=dict(dash="dash", color="#777777")))
    for name in ["Manual PyTorch", "Standard PyTorch"]:
        fig.add_trace(go.Scatter(x=data["p_scikit-learn"], y=data[f"p_{name}"],
                                 mode="markers", name=name, opacity=0.5,
                                 marker=dict(color=COLORS[name], size=5)))
    fig.update_layout(title=f"Do the implementations agree? ({split} set)",
                      xaxis_title="scikit-learn probability",
                      yaxis_title="PyTorch probability")
    return fig


@one_at_a_time
def coefficient_plot(top_n):
    c = coefs.reindex(coefs["sklearn"].abs().sort_values(ascending=False).index).head(int(top_n))
    long = c.melt(id_vars="feature", var_name="model", value_name="coefficient")
    long["model"] = long["model"].map({"sklearn": "scikit-learn", "manual_torch": "Manual PyTorch",
                                       "standard_torch": "Standard PyTorch"})
    fig = px.bar(long, x="coefficient", y="feature", color="model", barmode="group",
                 orientation="h", color_discrete_map=COLORS,
                 title=f"Top {int(top_n)} coefficients (standardised features)")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=250 + 28 * int(top_n))
    return fig


@one_at_a_time
def loss_plot():
    long = loss_curves.melt(id_vars="epoch", var_name="model", value_name="training objective")
    long["model"] = long["model"].map({"manual_torch": "Manual PyTorch",
                                       "standard_torch": "Standard PyTorch"})
    fig = px.line(long, x="epoch", y="training objective", color="model",
                  color_discrete_map=COLORS, log_x=True,
                  title="PyTorch training loss (log-loss + L2 penalty)")
    sk_obj = metrics["training_objective"]["sklearn"]
    fig.add_hline(y=sk_obj, line_dash="dash", line_color=COLORS["scikit-learn"],
                  annotation_text=f"scikit-learn optimum {sk_obj:.4f}")
    return fig


@one_at_a_time
def feature_distribution(feature):
    data = table.copy()
    data["outcome"] = data[TARGET].map({0: "No match", 1: "Match"})
    fig = px.histogram(data, x=feature, color="outcome", barmode="overlay",
                       histnorm="probability density", opacity=0.6, nbins=40,
                       color_discrete_map={"No match": "#999999", "Match": "#D55E00"},
                       title=f"Distribution of {feature} by outcome (all splits)")
    return fig


@one_at_a_time
def target_distribution():
    summary = (table.groupby("split")[TARGET].agg(["mean", "size"])
               .reindex(["train", "val", "test"]).reset_index())
    fig = px.bar(summary, x="split", y="mean", text=summary["mean"].map("{:.1%}".format),
                 hover_data={"size": True}, title="Match rate (target) by split",
                 color_discrete_sequence=["#D55E00"])
    fig.update_layout(yaxis_title="Share of dates that matched", yaxis_tickformat=".0%")
    return fig


@one_at_a_time
def threshold_view(model, split, threshold, cost_fp, cost_fn):
    data = rows(split)
    y, p = data[TARGET].to_numpy(), data[f"p_{model}"].to_numpy()
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    cost = cost_fp * fp + cost_fn * fn

    # Baseline: best of "nudge everyone" vs. "nudge no one" under the same costs.
    baseline_cost = min(cost_fp * (y == 0).sum(), cost_fn * (y == 1).sum())

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted: no match", "Predicted: match"],
        y=["Actual: no match", "Actual: match"],
        text=[[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]],
        texttemplate="%{text}", textfont=dict(size=18),
        colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix (threshold {threshold:.2f})",
                     yaxis=dict(autorange="reversed"))

    grid = np.linspace(0, 1, 101)
    costs = [cost_fp * ((p >= t) & (y == 0)).sum() + cost_fn * ((p < t) & (y == 1)).sum()
             for t in grid]
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name=model,
                               line=dict(color=COLORS[model])))
    curve.add_hline(y=baseline_cost, line_dash="dash", line_color="#999999",
                    annotation_text="Naive baseline")
    curve.add_vline(x=threshold, line_color="#D55E00")
    curve.update_layout(title="Business cost across thresholds",
                        xaxis_title="Threshold", yaxis_title="Business cost")

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    saving = 1 - cost / baseline_cost if baseline_cost else float("nan")
    summary = (
        f"### Business cost: **{cost:,}**  ({cost / len(y):.3f} per date)\n"
        f"= {cost_fp} × {fp} false positives (wasted nudges) "
        f"+ {cost_fn} × {fn} false negatives (missed matches)\n\n"
        f"Naive baseline cost: **{baseline_cost:,}** → this threshold "
        f"{'saves' if saving >= 0 else 'costs'} **{abs(saving):.1%}** "
        f"{'vs.' if saving >= 0 else 'more than'} the baseline.\n\n"
        f"Nudges sent: {tp + fp:,} of {len(y):,} pairs · precision {precision:.1%} · "
        f"recall {recall:.1%} · ROC-AUC {roc_auc_score(y, p):.3f} · "
        f"PR-AUC {average_precision_score(y, p):.3f}\n\n"
        f"*Threshold chosen in training (cost-minimising on validation): "
        f"{metrics['thresholds'][model]:.2f}*"
    )
    return cm, curve, summary


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Speed-dating match predictor") as demo:
    gr.Markdown(
        "# Speed-dating mutual-match predictor\n"
        "An organiser predicts, the morning after an event, which pairs will match "
        "mutually and nudges them. One logistic regression, trained three ways "
        "(scikit-learn, a manual PyTorch loop, the standard PyTorch workflow), "
        f"loaded from saved artifacts. Business cost = {COST_FP} × FP + {COST_FN} × FN. "
        "*Data is synthetic.*"
    )

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            model_in = gr.Dropdown(MODEL_NAMES, value="scikit-learn", label="Model")
            split_in = gr.Radio(["val", "test"], value="test", label="Data split")
        threshold_in = gr.Slider(0, 1, value=metrics["thresholds"]["scikit-learn"], step=0.01,
                                 label="Decision threshold (nudge if probability ≥ threshold)")
        with gr.Row():
            fp_in = gr.Number(value=COST_FP, label="Cost of a false positive (wasted nudge)",
                              minimum=0)
            fn_in = gr.Number(value=COST_FN, label="Cost of a false negative (missed match)",
                              minimum=0)
        summary_out = gr.Markdown()
        with gr.Row():
            cm_out = gr.Plot(label="Confusion matrix")
            curve_out = gr.Plot(label="Cost curve")
        inputs = [model_in, split_in, threshold_in, fp_in, fn_in]
        for component in inputs:
            component.change(threshold_view, inputs, [cm_out, curve_out, summary_out])
        model_in.change(lambda m: metrics["thresholds"][m], model_in, threshold_in)

    with gr.Tab("Model comparison"):
        gr.Markdown("### Test-set results (thresholds chosen on validation)")
        gr.Dataframe(results_table(), interactive=False)
        with gr.Row():
            split_cmp = gr.Radio(["val", "test"], value="test", label="Data split")
            bins_in = gr.Slider(5, 20, value=10, step=1, label="Calibration bins")
        calib_out = gr.Plot()
        with gr.Row():
            box_out = gr.Plot()
            agree_out = gr.Plot()
        split_cmp.change(calibration_plot, [split_cmp, bins_in], calib_out)
        bins_in.change(calibration_plot, [split_cmp, bins_in], calib_out)
        split_cmp.change(probability_by_outcome, split_cmp, box_out)
        split_cmp.change(agreement_plot, split_cmp, agree_out)
        top_in = gr.Slider(5, len(FEATURES), value=15, step=1, label="Coefficients to show")
        coef_out = gr.Plot()
        top_in.change(coefficient_plot, top_in, coef_out)
        loss_out = gr.Plot()

    with gr.Tab("Feature & target distributions"):
        target_out = gr.Plot()
        feature_in = gr.Dropdown(FEATURES, value="shared_interests", label="Feature")
        feature_out = gr.Plot()
        feature_in.change(feature_distribution, feature_in, feature_out)
        gr.Markdown("### Feature-set experiments (grouped 5-fold CV on training events)")
        gr.Dataframe(ablation.round(4), interactive=False)

    demo.load(threshold_view, inputs, [cm_out, curve_out, summary_out])
    demo.load(calibration_plot, [split_cmp, bins_in], calib_out)
    demo.load(probability_by_outcome, split_cmp, box_out)
    demo.load(agreement_plot, split_cmp, agree_out)
    demo.load(coefficient_plot, top_in, coef_out)
    demo.load(loss_plot, None, loss_out)
    demo.load(target_distribution, None, target_out)
    demo.load(feature_distribution, feature_in, feature_out)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
