"""
Assignment 1 — Gradio dashboard.

Run with `uv run python main.py app` after `uv run python main.py train`.

The dashboard only LOADS what train.py saved in artifacts/: the three trained models,
the fitted scaler, the feature table and the metadata. Nothing is fitted or retrained
here. The test-set predictions are recomputed from the loaded models (and checked
against the ones train.py saved), so every chart reflects the real saved models.

Tabs:
    1. Pitch list & threshold   move the cut-off, watch the confusion matrix and cost
    2. Predicted vs actual      the three models side by side on the test set
    3. Distributions            any feature or the target, by split and outcome
    4. Model comparison         results table, agreement, coefficients, loss curves
"""

# %% Setup
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

if not (ARTIFACTS / "metadata.json").exists():
    raise SystemExit("artifacts/ not found: run `uv run python main.py train` first.")

MODELS = ["scikit-learn", "PyTorch (manual)", "PyTorch (nn + optim)"]
BASELINES = ["baseline: base rate", "baseline: funding only"]
COLORS = {"scikit-learn": "#1f77b4", "PyTorch (manual)": "#ff7f0e",
          "PyTorch (nn + optim)": "#2ca02c", "baseline: base rate": "#7f7f7f",
          "baseline: funding only": "#9467bd"}
# The three implementations give identical predictions, so their lines would sit
# exactly on top of each other: a wide pale line, a medium one and a thin dashed one
# keep all three visible. Baselines are dotted.
LINE_STYLE = {"scikit-learn": dict(width=8, color="rgba(31,119,180,0.35)"),
              "PyTorch (manual)": dict(width=4, color="#ff7f0e"),
              "PyTorch (nn + optim)": dict(width=2, color="#2ca02c", dash="dash"),
              "baseline: base rate": dict(width=2, color="#7f7f7f", dash="dot"),
              "baseline: funding only": dict(width=2, color="#9467bd", dash="dot")}

# %% Load everything train.py saved
with open(ARTIFACTS / "metadata.json") as f:
    meta = json.load(f)
FEATURES = meta["features"]

feature_table = pd.read_csv(ARTIFACTS / "features.csv")
saved_predictions = pd.read_csv(ARTIFACTS / "test_predictions.csv")
results = pd.read_csv(ARTIFACTS / "results.csv")
coefficients = pd.read_csv(ARTIFACTS / "coefficients.csv", index_col=0)

scaler = joblib.load(ARTIFACTS / "scaler.joblib")
sk_model = joblib.load(ARTIFACTS / "model_sklearn.joblib")
manual = torch.load(ARTIFACTS / "model_torch_manual.pt")
nn_model = torch.nn.Linear(len(FEATURES), 1, dtype=torch.float64)
nn_model.load_state_dict(torch.load(ARTIFACTS / "model_torch_nn.pt"))
nn_model.eval()

# %% Score the test set with the loaded models (inference only)
test = feature_table[feature_table["split"] == "test"].reset_index(drop=True)
y_test = test["acquired"].to_numpy()
X_test = scaler.transform(test[FEATURES])
X_test_t = torch.tensor(X_test, dtype=torch.float64)
with torch.no_grad():
    scores = {
        "scikit-learn": sk_model.predict_proba(X_test)[:, 1],
        "PyTorch (manual)": torch.sigmoid(X_test_t @ manual["w"] + manual["b"]).numpy(),
        "PyTorch (nn + optim)": torch.sigmoid(nn_model(X_test_t).squeeze(1)).numpy(),
    }
for name in BASELINES:
    scores[name] = saved_predictions[name].to_numpy()

# The recomputed predictions must match the ones train.py saved. Reading the features
# back from CSV loses a little float precision (~1e-9), so allow 1e-6: a wrong or
# mismatched model would be off by orders of magnitude more.
for name in MODELS:
    gap = np.abs(scores[name] - saved_predictions[name].to_numpy()).max()
    assert gap < 1e-6, f"{name}: loaded model disagrees with saved predictions ({gap:.1e})"
print("Loaded 3 models; test predictions reproduced. Starting the dashboard...")

N_TEST, N_ACQUIRED = len(y_test), int(y_test.sum())
DEFAULT_SHARE = meta["top_share"] * 100


# =================================================================================
# Callbacks
# =================================================================================

def pitch_view(model, mode, share_pct, threshold, cost_fp, cost_fn):
    """Confusion matrix and business cost for one model at the chosen cut-off."""
    model_scores = scores[model]
    if mode == "Top-k share of the list":
        k = int(round(share_pct / 100 * N_TEST))
        order = np.argsort(-model_scores, kind="stable")
        pitched = np.zeros(N_TEST, dtype=bool)
        pitched[order[:k]] = True
        rule = f"pitch the top {share_pct:.1f}% of the ranked list ({k} companies)"
        if k > 0 and model in MODELS:
            rule += f", i.e. probability ≥ {model_scores[order[k - 1]]:.3f}"
    else:
        pitched = model_scores >= threshold
        rule = f"pitch every company with predicted probability ≥ {threshold:.3f}"
        if model in BASELINES:
            rule += " (baselines are not probabilities: use the top-k mode)"

    tp = int((pitched & (y_test == 1)).sum())
    fp = int((pitched & (y_test == 0)).sum())
    fn = int((~pitched & (y_test == 1)).sum())
    tn = int((~pitched & (y_test == 0)).sum())
    cost = fp * cost_fp + fn * cost_fn
    cost_nothing = N_ACQUIRED * cost_fn  # pitch nobody: every acquirer is missed

    matrix = go.Figure(go.Heatmap(
        z=[[tp, fn], [fp, tn]],
        x=["pitched", "not pitched"], y=["acquired", "not acquired"],
        text=[[f"TP<br>{tp}", f"FN<br>{fn}"], [f"FP<br>{fp}", f"TN<br>{tn}"]],
        texttemplate="%{text}", textfont={"size": 18}, colorscale="Blues", showscale=False,
    ))
    matrix.update_layout(title=f"Confusion matrix: {model}", height=380,
                         xaxis_title="decision", yaxis_title="actual outcome",
                         yaxis_autorange="reversed", margin=dict(t=60, b=40))

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / N_ACQUIRED
    summary = (
        f"### Business cost: **€{cost:,.0f}**\n"
        f"Rule: {rule}.\n\n"
        f"| | |\n|---|---|\n"
        f"| Pitches | {tp + fp:,} |\n"
        f"| Acquirers pitched (TP) | {tp} of {N_ACQUIRED} |\n"
        f"| Wasted pitches (FP) | {fp:,} × €{cost_fp:,.0f} = €{fp * cost_fp:,.0f} |\n"
        f"| Acquirers missed (FN) | {fn} × €{cost_fn:,.0f} = €{fn * cost_fn:,.0f} |\n"
        f"| Precision | {precision:.1%} (base rate {N_ACQUIRED / N_TEST:.1%}) |\n"
        f"| Recall | {recall:.1%} |\n"
        f"| Cost of pitching nobody | €{cost_nothing:,.0f} |\n\n"
        f"Test set: {N_TEST:,} startups founded 2007–2008, {N_ACQUIRED} acquired."
    )
    return matrix, summary


def cost_curve(cost_fp, cost_fn):
    """Business cost and precision across every pitch-list size, for all rankings."""
    fig = go.Figure()
    shares = np.arange(1, N_TEST + 1) / N_TEST * 100
    for name in MODELS + BASELINES:
        if name == "baseline: base rate":
            hits = shares / 100 * N_ACQUIRED  # a constant score is a random ranking
        else:
            hits = np.cumsum(y_test[np.argsort(-scores[name], kind="stable")])
        pitched = np.arange(1, N_TEST + 1)
        cost = (pitched - hits) * cost_fp + (N_ACQUIRED - hits) * cost_fn
        fig.add_trace(go.Scatter(x=shares, y=cost, name=name, line=LINE_STYLE[name]))
    fig.add_vline(x=DEFAULT_SHARE, line_dash="dash", line_color="black",
                  annotation_text=f"capacity: top {DEFAULT_SHARE:.0f}%")
    fig.update_layout(title="Business cost by size of the pitch list (test set)",
                      xaxis_title="share of the ranked list pitched (%)",
                      yaxis_title="cost (EUR)", height=420)
    return fig


def predicted_vs_actual(n_bins):
    """Reliability diagram and cumulative gains for the three models on the test set."""
    calibration = go.Figure()
    calibration.add_trace(go.Scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", name="perfect",
                                     line=dict(color="black", dash="dash")))
    for name in MODELS:
        bins = pd.qcut(scores[name], q=int(n_bins), duplicates="drop")
        grouped = pd.DataFrame({"p": scores[name], "y": y_test}).groupby(bins, observed=True)
        calibration.add_trace(go.Scatter(
            x=grouped["p"].mean(), y=grouped["y"].mean(), mode="lines+markers", name=name,
            line=LINE_STYLE[name], marker=dict(size=12 if name == "scikit-learn" else 6),
        ))
    calibration.update_layout(
        title=f"Predicted vs actual acquisition rate ({int(n_bins)} equal-size bins)",
        xaxis_title="mean predicted probability in bin",
        yaxis_title="actual acquired rate in bin", height=450)

    gains = go.Figure()
    shares = np.arange(0, N_TEST + 1) / N_TEST * 100
    for name in MODELS + BASELINES:
        if name == "baseline: base rate":
            captured = shares
        else:
            hits = np.cumsum(y_test[np.argsort(-scores[name], kind="stable")])
            captured = np.concatenate([[0], hits]) / N_ACQUIRED * 100
        gains.add_trace(go.Scatter(x=shares, y=captured, name=name, line=LINE_STYLE[name]))
    gains.add_vline(x=DEFAULT_SHARE, line_dash="dash", line_color="black",
                    annotation_text=f"top {DEFAULT_SHARE:.0f}%")
    gains.update_layout(title="Cumulative gains: share of acquirers captured by the pitch list",
                        xaxis_title="share of the ranked list pitched (%)",
                        yaxis_title="share of actual acquirers pitched (%)", height=450)

    by_class = go.Figure()
    for outcome, label in ((1, "acquired"), (0, "not acquired")):
        by_class.add_trace(go.Histogram(
            x=scores["scikit-learn"][y_test == outcome], name=label, opacity=0.6,
            histnorm="probability", nbinsx=40))
    by_class.update_layout(barmode="overlay", height=380,
                           title="Predicted probability by actual outcome (scikit-learn; "
                                 "the PyTorch versions are identical to 8e-6)",
                           xaxis_title="predicted probability", yaxis_title="share of group")
    return calibration, gains, by_class


def distribution(variable, split):
    """Distribution of one feature by outcome, or the target by founding year."""
    rows = feature_table if split == "all" else feature_table[feature_table["split"] == split]
    if variable == "acquired (target)":
        years = rows.assign(founded_year=2014 - rows["company_age"])
        by_year = years.groupby(["founded_year", "split"])["acquired"].agg(["mean", "size"]).reset_index()
        fig = go.Figure()
        for split_name, color in (("train", "#1f77b4"), ("val", "#ff7f0e"), ("test", "#2ca02c")):
            part = by_year[by_year["split"] == split_name]
            fig.add_trace(go.Bar(x=part["founded_year"], y=part["mean"], name=split_name,
                                 marker_color=color, text=part["size"],
                                 hovertemplate="founded %{x}<br>acquired %{y:.1%}<br>n=%{text}"))
        fig.update_layout(title="Acquisition rate by founding year and split",
                          xaxis_title="founding year", yaxis_title="share acquired",
                          yaxis_tickformat=".0%", height=450)
        note = (f"{len(rows):,} companies, {rows['acquired'].mean():.1%} acquired. Younger "
                "cohorts have had less time to be acquired, so the rate drifts down "
                "from train to test.")
        return fig, note

    fig = go.Figure()
    for outcome, label in ((1, "acquired"), (0, "not acquired")):
        values = rows.loc[rows["acquired"] == outcome, variable]
        fig.add_trace(go.Histogram(x=values, name=label, opacity=0.6,
                                   histnorm="probability", nbinsx=40))
    fig.update_layout(barmode="overlay", height=450,
                      title=f"{variable} by outcome ({split}, unscaled)",
                      xaxis_title=variable, yaxis_title="share of group")
    means = rows.groupby("acquired")[variable].mean()
    note = (f"{len(rows):,} companies. Mean {variable}: **{means.get(1, float('nan')):.3f}** "
            f"for acquired vs **{means.get(0, float('nan')):.3f}** for not acquired.")
    return fig, note


def loss_curves():
    """Left: the objective itself. Right: how far it still is from sklearn's optimum.

    Step 0 (all weights zero, loss = ln 2 = 0.693) is drawn at x = 1, because a log axis
    cannot show 0. Almost all of the drop happens in the first 100 steps; the slow tail
    after that, caused by the badly conditioned features (5.7), only shows on a log scale.
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14, subplot_titles=(
        "Objective (mean BCE + L2)", "Gap to sklearn's optimum (log scale)"))
    optimum = meta["sklearn_objective"]
    for name in ("PyTorch (manual)", "PyTorch (nn + optim)"):
        epochs, losses = zip(*meta["loss_history"][name])
        steps = np.array(epochs) + 1
        gap = np.maximum(np.array(losses) - optimum, 1e-12)
        fig.add_trace(go.Scatter(x=steps, y=losses, name=name, line=LINE_STYLE[name],
                                 legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=gap, name=name, line=LINE_STYLE[name],
                                 legendgroup=name, showlegend=False), row=1, col=2)
    fig.update_xaxes(type="log", title_text="step + 1")
    fig.update_yaxes(title_text="objective", row=1, col=1)
    fig.update_yaxes(type="log", title_text="objective - sklearn optimum", row=1, col=2)
    fig.update_layout(height=460, title=f"Full-batch gradient descent, {meta['epochs']:,} steps "
                                        f"(sklearn optimum {optimum:.8f})",
                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    return fig


agreement = pd.DataFrame({
    "comparison": ["PyTorch (manual) vs scikit-learn", "PyTorch (nn + optim) vs scikit-learn",
                   "PyTorch (manual) vs PyTorch (nn + optim)"],
    "max |Δ weight|": [
        f"{(coefficients['PyTorch (manual)'] - coefficients['scikit-learn']).abs().max():.1e}",
        f"{(coefficients['PyTorch (nn + optim)'] - coefficients['scikit-learn']).abs().max():.1e}",
        f"{(coefficients['PyTorch (manual)'] - coefficients['PyTorch (nn + optim)']).abs().max():.1e}",
    ],
    "max |Δ test probability|": [
        f"{np.abs(scores['PyTorch (manual)'] - scores['scikit-learn']).max():.1e}",
        f"{np.abs(scores['PyTorch (nn + optim)'] - scores['scikit-learn']).max():.1e}",
        f"{np.abs(scores['PyTorch (manual)'] - scores['PyTorch (nn + optim)']).max():.1e}",
    ],
})
coefficient_table = (coefficients.reindex(coefficients["scikit-learn"].abs()
                                          .sort_values(ascending=False).index)
                     .round(4).reset_index(names="feature"))


# =================================================================================
# Layout
# =================================================================================
with gr.Blocks(title="Startup acquisition ranking") as demo:
    gr.Markdown(
        "# Which startups should the boutique pitch?\n"
        "Logistic regression trained three ways (scikit-learn, a manual PyTorch loop, "
        "`nn.Module` + `torch.optim`) to rank venture-backed startups by how likely they "
        "are to be acquired. Test set: startups founded 2007–2008. Everything here is "
        "loaded from `artifacts/`; nothing is retrained."
    )

    with gr.Tab("1. Pitch list & threshold"):
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(MODELS + BASELINES, value="scikit-learn", label="Ranking")
                mode = gr.Radio(["Top-k share of the list", "Probability threshold"],
                                value="Top-k share of the list", label="Cut-off rule")
                share = gr.Slider(0.5, 50, value=DEFAULT_SHARE, step=0.5,
                                  label="Pitch the top X% of the ranked list")
                threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.005,
                                      label="Probability threshold (pitch if p ≥ threshold)")
                cost_fp = gr.Number(value=meta["cost_fp"], label="Cost of a wasted pitch, FP (€)")
                cost_fn = gr.Number(value=meta["cost_fn"], label="Cost of a missed acquirer, FN (€)")
            with gr.Column(scale=2):
                matrix_plot = gr.Plot()
                summary_md = gr.Markdown()
        curve_plot = gr.Plot()
        gr.Markdown(
            f"The operating rule from REPORT.md is **pitch the top {DEFAULT_SHARE:.0f}%**: "
            "capacity, not the cost ratio, sets the list size. The costs score each choice. "
            "Change them to see how sensitive the conclusion is."
        )
        pitch_inputs = [model_choice, mode, share, threshold, cost_fp, cost_fn]
        for control in pitch_inputs:
            control.change(pitch_view, pitch_inputs, [matrix_plot, summary_md])
        for control in (cost_fp, cost_fn):
            control.change(cost_curve, [cost_fp, cost_fn], curve_plot)

    with gr.Tab("2. Predicted vs actual"):
        n_bins = gr.Slider(5, 20, value=10, step=1, label="Number of bins")
        calibration_plot = gr.Plot()
        gains_plot = gr.Plot()
        by_class_plot = gr.Plot()
        gr.Markdown(
            "The three models overlap almost exactly: they minimise the same objective. "
            "Points above the diagonal mean the actual rate is higher than predicted, i.e. "
            "the model under-predicts: on test the mean prediction (0.086) is below the "
            "actual rate (0.097), because the age coefficient is extrapolated to younger "
            "cohorts (see REPORT.md). The ranking, and so the pitch list, is unaffected."
        )
        n_bins.change(predicted_vs_actual, n_bins, [calibration_plot, gains_plot, by_class_plot])

    with gr.Tab("3. Distributions"):
        with gr.Row():
            variable = gr.Dropdown(["acquired (target)"] + FEATURES, value="acquired (target)",
                                   label="Variable")
            split_choice = gr.Radio(["train", "val", "test", "all"], value="train", label="Split")
        dist_plot = gr.Plot()
        dist_note = gr.Markdown()
        for control in (variable, split_choice):
            control.change(distribution, [variable, split_choice], [dist_plot, dist_note])

    with gr.Tab("4. Model comparison"):
        gr.Markdown("### Test-set results (top 5% pitch list)")
        gr.Dataframe(results.assign(**{"cost (EUR)": results["cost (EUR)"].map("€{:,.0f}".format)}),
                     interactive=False)
        gr.Markdown("### Do the three implementations agree?")
        gr.Dataframe(agreement, interactive=False)
        gr.Markdown(f"Shared settings: C = {meta['C']}, λ = 1/(C·n) = {meta['lambda']:.2e}, "
                    f"learning rate {meta['learning_rate']}, {meta['epochs']:,} full-batch steps.")
        loss_plot = gr.Plot()
        gr.Markdown("### Coefficients (standardised features). Correlated features split "
                    "their effect, so read them in families, not one by one.")
        gr.Dataframe(coefficient_table, interactive=False)

    demo.load(pitch_view, pitch_inputs, [matrix_plot, summary_md])
    demo.load(cost_curve, [cost_fp, cost_fn], curve_plot)
    demo.load(predicted_vs_actual, n_bins, [calibration_plot, gains_plot, by_class_plot])
    demo.load(distribution, [variable, split_choice], [dist_plot, dist_note])
    demo.load(loss_curves, None, loss_plot)

demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
