"""
Assignment 1 - Gradio dashboard for the hotel-cancellation models.

Run with:  uv run python main.py app      (run `main.py train` first)

This file NEVER trains anything. It only loads what train.py saved in
artifacts/ (the preprocessor, the three models, the engineered data and
the metrics) and draws charts from them.

Tabs:
  1. Compare models    - scores table, predicted vs actual plots, how the
                         three implementations agree, most important features
  2. Threshold & cost  - slider -> confusion matrix + euro cost
  3. Data              - feature and target distributions
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

from train import CATEGORICAL, NUMERIC, LogReg  # same model class & feature lists

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
if not (ART / "metrics.json").exists():
    raise SystemExit("No trained models found. Run `uv run python main.py train` first.")

MODEL_NAMES = {
    "sklearn": "scikit-learn",
    "torch_manual": "PyTorch (manual loop)",
    "torch_standard": "PyTorch (nn.Module + optim)",
}
COLORS = {"sklearn": "#1f77b4", "torch_manual": "#ff7f0e",
          "torch_standard": "#2ca02c", "baseline": "#7f7f7f"}

# ---------------------------------------------------------------------------
# Load saved artifacts (no training here)
# ---------------------------------------------------------------------------
metrics = json.loads((ART / "metrics.json").read_text())
data = pd.read_csv(ART / "engineered_data.csv", parse_dates=["arrival_date"])
coefs = pd.read_csv(ART / "coefficients.csv")
loss_hist = pd.read_csv(ART / "training_loss.csv")

preprocessor = joblib.load(ART / "preprocessor.joblib")
sk_model = joblib.load(ART / "model_sklearn.joblib")
manual = torch.load(ART / "model_torch_manual.pt")
std_model = LogReg(len(coefs))
std_model.load_state_dict(torch.load(ART / "model_torch_standard.pt"))
std_model.eval()

# Predict the test set with the three loaded models
test = data[data["split"] == "test"].reset_index(drop=True)
X_test = torch.tensor(preprocessor.transform(test[NUMERIC + CATEGORICAL]), dtype=torch.float32)
y_test = test["is_canceled"].to_numpy()
with torch.no_grad():
    PROBA = {
        "sklearn": sk_model.predict_proba(X_test.numpy())[:, 1],
        "torch_manual": torch.sigmoid(X_test @ manual["w"] + manual["b"]).numpy(),
        "torch_standard": torch.sigmoid(std_model(X_test)).numpy(),
    }
BASE_RATE = float(data.loc[data["split"] == "train", "is_canceled"].mean())
PROBA_BASE = np.full(len(y_test), BASE_RATE)


# ---------------------------------------------------------------------------
# Tab 1: compare models
# ---------------------------------------------------------------------------
def results_table():
    rows = []
    for key, r in metrics["results_test"].items():
        rows.append({
            "Model": MODEL_NAMES.get(key, "Naive baseline (always train cancel rate)"),
            "ROC AUC": r["roc_auc"], "Log loss": r["log_loss"],
            "Accuracy @0.5": r["accuracy@0.5"], "Threshold (from val)": r["threshold"],
            "Cost on test (€)": f"{r['cost_eur']:,.0f}",
            "Cost per booking (€)": r["cost_per_booking_eur"],
        })
    return pd.DataFrame(rows)


def calibration_plot():
    """Predicted probability (binned) vs actual cancel rate.
    A perfect model sits on the diagonal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="lightgray")))
    bins = np.linspace(0, 1, 11)
    for key, p in PROBA.items():
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        df = pd.DataFrame({"bin": idx, "p": p, "y": y_test}).groupby("bin").agg(
            pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
        fig.add_trace(go.Scatter(x=df.pred, y=df.actual, mode="lines+markers",
                                 name=MODEL_NAMES[key], line=dict(color=COLORS[key]),
                                 customdata=df.n, hovertemplate="predicted %{x:.2f}<br>"
                                 "actual %{y:.2f}<br>%{customdata} bookings"))
    fig.update_layout(title="Predicted vs actual cancel rate (test set, 10 bins)",
                      xaxis_title="Predicted probability of cancel",
                      yaxis_title="Actual share cancelled", height=420)
    return fig


def over_time_plot():
    df = test[["arrival_date", "is_canceled"]].copy()
    for key, p in PROBA.items():
        df[key] = p
    weekly = df.resample("W", on="arrival_date").mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly.is_canceled, name="Actual",
                             line=dict(color="black", width=3)))
    for key in PROBA:
        fig.add_trace(go.Scatter(x=weekly.index, y=weekly[key], name=MODEL_NAMES[key],
                                 line=dict(color=COLORS[key])))
    fig.add_hline(y=BASE_RATE, line_dash="dot", annotation_text="baseline")
    fig.update_layout(title="Weekly cancel rate: predicted vs actual (test, May-Aug 2017)",
                      yaxis_title="Cancel rate", height=380)
    return fig


def agreement_plot():
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=PROBA["sklearn"], y=PROBA["torch_manual"], mode="markers",
                               name="manual loop", marker=dict(size=4, color=COLORS["torch_manual"])))
    fig.add_trace(go.Scattergl(x=PROBA["sklearn"], y=PROBA["torch_standard"], mode="markers",
                               name="nn.Module", marker=dict(size=4, color=COLORS["torch_standard"], opacity=0.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title="Same booking, three models: do they agree?",
                      xaxis_title="scikit-learn probability",
                      yaxis_title="PyTorch probability", height=420)
    return fig


def loss_plot():
    fig = go.Figure()
    for key in ["torch_manual", "torch_standard"]:
        fig.add_trace(go.Scatter(x=loss_hist.epoch + 1, y=loss_hist[key], name=MODEL_NAMES[key],
                                 line=dict(color=COLORS[key])))
    fig.add_hline(y=metrics["agreement"]["sklearn_train_objective"], line_dash="dash",
                  annotation_text="scikit-learn optimum")
    fig.update_layout(title="PyTorch training loss slowly reaching the scikit-learn optimum",
                      xaxis_title="Epoch (log scale)", xaxis_type="log",
                      yaxis_title="Train loss (log loss + L2)", yaxis_range=[0.35, 0.40],
                      height=380)
    return fig


def convergence_table():
    rows = metrics["agreement"]["convergence"]
    df = pd.DataFrame(rows).rename(columns={
        "epochs": "Epochs", "train_objective": "Train loss",
        "max_abs_prob_diff_vs_sklearn": "Biggest gap vs sklearn (prob.)",
        "intercept": "Intercept"})
    df.loc[len(df)] = ["scikit-learn", metrics["agreement"]["sklearn_train_objective"], 0.0,
                       metrics["agreement"]["sklearn_intercept"]]
    return df


def coef_plot():
    df = coefs.assign(size=coefs.sklearn.abs()).nlargest(15, "size").sort_values("sklearn")
    fig = go.Figure(go.Bar(x=df.sklearn, y=df.feature, orientation="h",
                           marker_color=np.where(df.sklearn > 0, "#d62728", "#2ca02c")))
    fig.update_layout(title="15 strongest effects (red = more likely to cancel)",
                      xaxis_title="Coefficient (on scaled / one-hot inputs)", height=480)
    return fig


# ---------------------------------------------------------------------------
# Tab 2: threshold & cost
# ---------------------------------------------------------------------------
def threshold_view(model_label, threshold, cost_fn, cost_fp):
    key = {v: k for k, v in MODEL_NAMES.items()}[model_label]
    pred = (PROBA[key] >= threshold).astype(int)
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tp = int(((pred == 1) & (y_test == 1)).sum())

    z = [[tn, fp], [fn, tp]]
    text = [[f"Came & not flagged<br><b>{tn:,}</b>", f"False alarm<br><b>{fp:,}</b><br>€{fp * cost_fp:,.0f}"],
            [f"Missed cancel<br><b>{fn:,}</b><br>€{fn * cost_fn:,.0f}", f"Caught cancel<br><b>{tp:,}</b>"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted: will come", "Predicted: will cancel"],
                               y=["Actually came", "Actually cancelled"], text=text,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title=f"Confusion matrix at threshold {threshold:.2f} (test set)",
                      yaxis_autorange="reversed", height=420)

    cost = fn * cost_fn + fp * cost_fp
    flag_nobody = int(y_test.sum()) * cost_fn
    flag_everyone = int((y_test == 0).sum()) * cost_fp
    baseline = min(flag_nobody, flag_everyone)
    n = len(y_test)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    summary = (
        f"### Business cost: **€{cost:,.0f}**  (€{cost / n:.2f} per booking)\n"
        f"- Missed cancellations: {fn:,} × €{cost_fn:.0f} = €{fn * cost_fn:,.0f}\n"
        f"- False alarms: {fp:,} × €{cost_fp:.0f} = €{fp * cost_fp:,.0f}\n\n"
        f"**Compare:** flag nobody = €{flag_nobody:,.0f} · flag everyone = €{flag_everyone:,.0f}. "
        f"The model saves **€{baseline - cost:,.0f}** vs the cheaper of the two.\n\n"
        f"Of the bookings we flag, {precision:.0%} really cancel (precision). "
        f"We catch {recall:.0%} of all cancellations (recall).\n\n"
        f"*Rule of thumb: if probabilities were perfectly calibrated, the cheapest threshold "
        f"would be cost_fp / (cost_fp + cost_fn) = {cost_fp / (cost_fp + cost_fn):.2f}.*"
    )
    return fig, summary


def cost_curve(model_label, cost_fn, cost_fp):
    key = {v: k for k, v in MODEL_NAMES.items()}[model_label]
    ts = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [((PROBA[key] < t) & (y_test == 1)).sum() * cost_fn +
             ((PROBA[key] >= t) & (y_test == 0)).sum() * cost_fp for t in ts]
    best = ts[int(np.argmin(costs))]
    fig = go.Figure(go.Scatter(x=ts, y=costs, line=dict(color=COLORS[key])))
    fig.add_vline(x=best, line_dash="dash", annotation_text=f"cheapest on test: {best:.2f}")
    fig.add_vline(x=metrics["results_test"][key]["threshold"], line_dash="dot", line_color="red",
                  annotation_text="chosen on validation", annotation_position="bottom right")
    fig.update_layout(title="Total cost for every threshold (test set)",
                      xaxis_title="Threshold", yaxis_title="Cost (€)", height=380)
    return fig


# ---------------------------------------------------------------------------
# Tab 3: data
# ---------------------------------------------------------------------------
def feature_plot(feature, split):
    df = data if split == "all" else data[data["split"] == split]
    df = df.assign(outcome=np.where(df.is_canceled == 1, "Cancelled", "Came"))
    if feature in CATEGORICAL:
        rate = df.groupby(feature).agg(cancel_rate=("is_canceled", "mean"),
                                       bookings=("is_canceled", "size")).reset_index()
        fig = px.bar(rate.sort_values("bookings", ascending=False), x=feature, y="bookings",
                     color="cancel_rate", color_continuous_scale="Reds",
                     title=f"{feature}: number of bookings (colour = cancel rate)")
    else:
        fig = px.histogram(df, x=feature, color="outcome", barmode="overlay", nbins=50,
                           histnorm="percent", opacity=0.6,
                           color_discrete_map={"Came": "#2ca02c", "Cancelled": "#d62728"},
                           title=f"{feature}: distribution for cancelled vs not cancelled")
    fig.update_layout(height=420)
    return fig


def target_over_time():
    monthly = data.resample("MS", on="arrival_date").agg(
        cancel_rate=("is_canceled", "mean"), split=("split", "first")).reset_index()
    fig = px.bar(monthly, x="arrival_date", y="cancel_rate", color="split",
                 color_discrete_map={"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"},
                 title="Target over time: monthly cancel rate, coloured by split")
    fig.update_layout(yaxis_title="Cancel rate", xaxis_title="Arrival month", height=380)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
default_model = MODEL_NAMES["sklearn"]
default_t = metrics["results_test"]["sklearn"]["threshold"]

with gr.Blocks(title="Hotel cancellation models") as demo:
    gr.Markdown(
        "# Will this Resort Hotel booking be cancelled?\n"
        "Logistic regression trained three ways on bookings arriving Jul 2015 - Dec 2016, "
        "tuned on Jan - Apr 2017, and tested on **May - Aug 2017** "
        f"({len(y_test):,} bookings). Everything here is loaded from `artifacts/`; nothing is retrained."
    )
    with gr.Tab("1 · Compare models"):
        gr.Dataframe(results_table(), label="Test-set results", interactive=False)
        with gr.Row():
            gr.Plot(calibration_plot())
            gr.Plot(agreement_plot())
        gr.Plot(over_time_plot())
        gr.Markdown("### Why the PyTorch versions needed so many epochs")
        with gr.Row():
            gr.Plot(loss_plot())
            gr.Dataframe(convergence_table(), interactive=False,
                         label="Manual PyTorch loop vs scikit-learn as training goes on")
        gr.Plot(coef_plot())

    with gr.Tab("2 · Threshold & cost"):
        gr.Markdown("A booking is **flagged as 'will cancel'** when its predicted probability "
                    "is above the threshold. Move the slider and watch the mistakes and the cost change.")
        with gr.Row():
            model_dd = gr.Dropdown(list(MODEL_NAMES.values()), value=default_model, label="Model")
            slider = gr.Slider(0.01, 0.99, value=default_t, step=0.01, label="Decision threshold")
        with gr.Row():
            fn_cost = gr.Number(value=metrics["cost_missed_cancel"], label="Cost of a missed cancellation (€)")
            fp_cost = gr.Number(value=metrics["cost_false_alarm"], label="Cost of a false alarm (€)")
        with gr.Row():
            cm_plot = gr.Plot()
            cost_md = gr.Markdown()
        curve = gr.Plot()

        inputs = [model_dd, slider, fn_cost, fp_cost]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, cost_md])
        for comp in [model_dd, fn_cost, fp_cost]:
            comp.change(cost_curve, [model_dd, fn_cost, fp_cost], curve)
        demo.load(threshold_view, inputs, [cm_plot, cost_md])
        demo.load(cost_curve, [model_dd, fn_cost, fp_cost], curve)

    with gr.Tab("3 · Data"):
        with gr.Row():
            feat_dd = gr.Dropdown(NUMERIC + CATEGORICAL, value="log_lead_time", label="Feature")
            split_dd = gr.Radio(["all", "train", "val", "test"], value="all", label="Which rows")
        feat_plot = gr.Plot()
        feat_dd.change(feature_plot, [feat_dd, split_dd], feat_plot)
        split_dd.change(feature_plot, [feat_dd, split_dd], feat_plot)
        demo.load(feature_plot, [feat_dd, split_dd], feat_plot)
        gr.Plot(target_over_time())


if __name__ == "__main__":
    demo.launch(inbrowser=True)
