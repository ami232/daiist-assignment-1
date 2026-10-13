"""
Assignment 1 — Gradio dashboard.

This file TRAINS NOTHING. It only loads what `train.py` saved in `artifacts/`
(the three fitted models, the scaler, the dataset with features, and the metrics)
and runs inference. Run `uv run python main.py train` first.

Views:
  1. Model comparison: calibration plot (predicted probability vs actual win rate),
     agreement between the three models, and the metrics table.
  2. Distributions: each feature split by outcome, and the target by season.
  3. Threshold & cost: slider -> confusion matrix and business cost (framing in REPORT.md).
  4. Try a game: enter the four features and see each model's probability.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

MODEL_LABELS = {
    "sklearn": "scikit-learn",
    "manual_torch": "Manual PyTorch loop",
    "nn_module": "nn.Module + torch.optim",
}
COLORS = {"sklearn": "#1f77b4", "manual_torch": "#ff7f0e", "nn_module": "#2ca02c"}


# ----------------------------------------------------------------------------
# Load saved artifacts (no training anywhere in this file)
# ----------------------------------------------------------------------------
if not (ART / "metrics.json").exists():
    raise SystemExit("artifacts/ not found. Run `uv run python main.py train` first.")

METRICS = json.loads((ART / "metrics.json").read_text())
FEATURES = METRICS["features"]
COST_FP, COST_FN = METRICS["cost_fp"], METRICS["cost_fn"]
DEFAULT_THRESHOLD = METRICS["decision_threshold"]

SCALER = joblib.load(ART / "scaler.joblib")
SK_MODEL = joblib.load(ART / "sklearn_model.joblib")
_manual = torch.load(ART / "manual_model.pt", weights_only=True)
W_MANUAL, B_MANUAL = _manual["w"], _manual["b"]
NET = nn.Sequential(nn.Linear(len(FEATURES), 1), nn.Sigmoid())
NET.load_state_dict(torch.load(ART / "torch_model.pt", weights_only=True))
NET.eval()

DATA = pd.read_csv(ART / "dataset.csv", parse_dates=["gameday"])
TEST = DATA[DATA["split"] == "test"].reset_index(drop=True)
Y_TEST = TEST["home_win"].to_numpy()


def predict_all(raw_features: pd.DataFrame) -> dict:
    """Probabilities from the three loaded models, for unscaled feature rows."""
    X = SCALER.transform(raw_features[FEATURES]).astype(np.float32)
    with torch.no_grad():
        p_manual = torch.sigmoid(torch.tensor(X) @ W_MANUAL + B_MANUAL).numpy()
        p_net = NET(torch.tensor(X)).squeeze(-1).numpy()
    return {
        "sklearn": SK_MODEL.predict_proba(X.astype(np.float64))[:, 1],
        "manual_torch": p_manual,
        "nn_module": p_net,
    }


TEST_PROBS = predict_all(TEST)


# ----------------------------------------------------------------------------
# View 1: model comparison
# ----------------------------------------------------------------------------
def calibration_figure() -> go.Figure:
    """Prediction vs actual for a classifier: bin games by predicted probability and
    compare to the share of games the home team actually won in that bin."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color="gray", dash="dash")))
    for key, p in TEST_PROBS.items():
        bins = pd.qcut(p, 8, duplicates="drop")
        grouped = pd.DataFrame({"p": p, "y": Y_TEST}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(x=grouped["p"], y=grouped["y"], mode="lines+markers",
                                 name=MODEL_LABELS[key], line=dict(color=COLORS[key])))
    fig.update_layout(
        title="Predicted probability vs actual home-win rate (test set, 8 equal-size bins)",
        xaxis_title="Mean predicted probability of home win",
        yaxis_title="Actual share of home wins",
        height=420, margin=dict(t=60, b=40),
    )
    return fig


def agreement_figure() -> go.Figure:
    """Each PyTorch model's probability against scikit-learn's, game by game."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical",
                             line=dict(color="gray", dash="dash")))
    for key in ("manual_torch", "nn_module"):
        fig.add_trace(go.Scatter(x=TEST_PROBS["sklearn"], y=TEST_PROBS[key], mode="markers",
                                 name=MODEL_LABELS[key], marker=dict(size=5, opacity=0.5, color=COLORS[key])))
    worst = max(np.abs(TEST_PROBS["sklearn"] - TEST_PROBS[k]).max() for k in ("manual_torch", "nn_module"))
    fig.update_layout(
        title=f"Do the three models agree? Largest gap vs scikit-learn: {worst:.1e}",
        xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability",
        height=420, margin=dict(t=60, b=40),
    )
    return fig


def metrics_table() -> pd.DataFrame:
    rows = []
    for key, label in MODEL_LABELS.items():
        m = METRICS["test"][key]
        rows.append({"Model": label, "Accuracy @0.5": round(m["accuracy@0.5"], 4),
                     "Log loss": round(m["log_loss"], 4), "AUC": round(m["auc"], 4),
                     f"Cost @{DEFAULT_THRESHOLD:.2f}": m["cost@0.75"],
                     "Games rested": m["games_rested@0.75"]})
    b = METRICS["baselines"]
    rows.append({"Model": "Baseline: always home win", "Accuracy @0.5": round(b["accuracy_always_home_win"], 4),
                 "Log loss": round(b["log_loss_train_rate"], 4), "AUC": None,
                 f"Cost @{DEFAULT_THRESHOLD:.2f}": b["cost_always_rest"], "Games rested": len(Y_TEST)})
    rows.append({"Model": "Baseline: never rest starters", "Accuracy @0.5": None, "Log loss": None,
                 "AUC": None, f"Cost @{DEFAULT_THRESHOLD:.2f}": b["cost_never_rest"], "Games rested": 0})
    return pd.DataFrame(rows)


def coefficient_table() -> pd.DataFrame:
    cols = FEATURES + ["bias"]
    rows = [{"Model": MODEL_LABELS[k], **{c: round(v, 5) for c, v in zip(cols, vals)}}
            for k, vals in METRICS["coefficients"].items()]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# View 2: distributions
# ----------------------------------------------------------------------------
TARGET_CHOICE = "home_win (target) by season"
FEATURE_HELP = {
    "pd5_diff": "Home minus away average point margin over the last 5 games (recent form).",
    "pd16_diff": "Home minus away average point margin over the last 16 games (underlying strength).",
    "rest_diff": "Home minus away days of rest.",
    "div_game": "1 if both teams are in the same division.",
}


def distribution_figure(choice: str, split: str) -> go.Figure:
    data = DATA if split == "all" else DATA[DATA["split"] == split]
    fig = go.Figure()
    if choice == TARGET_CHOICE:
        by_season = data.groupby("season").agg(rate=("home_win", "mean"), n=("home_win", "size")).reset_index()
        fig.add_trace(go.Bar(x=by_season["season"], y=by_season["rate"], name="home win rate",
                             hovertext=[f"{n} games" for n in by_season["n"]]))
        fig.add_hline(y=data["home_win"].mean(), line_dash="dash", annotation_text="average")
        fig.update_layout(title=f"Home win rate by season ({split})", yaxis_title="Share of home wins",
                          xaxis_title="Season", yaxis_range=[0, 1])
    else:
        for outcome, name in [(1, "home team won"), (0, "home team lost")]:
            fig.add_trace(go.Histogram(x=data.loc[data["home_win"] == outcome, choice], name=name,
                                       opacity=0.6, histnorm="probability"))
        fig.update_layout(barmode="overlay", title=f"{choice} by outcome ({split})",
                          xaxis_title=choice, yaxis_title="Share of games")
    fig.update_layout(height=420, margin=dict(t=60, b=40))
    return fig


def feature_help(choice: str) -> str:
    return FEATURE_HELP.get(choice, "Share of games won by the home team in each season.")


# ----------------------------------------------------------------------------
# View 3: threshold slider -> confusion matrix and business cost
# ----------------------------------------------------------------------------
def confusion_counts(y, p, threshold):
    say_win = p >= threshold          # "win" => rest starters
    tp = int((say_win & (y == 1)).sum())
    fp = int((say_win & (y == 0)).sum())     # rested starters and lost
    fn = int((~say_win & (y == 1)).sum())    # played starters, would have won
    tn = int((~say_win & (y == 0)).sum())
    return tp, fp, fn, tn


def threshold_view(model_key: str, threshold: float, cost_fp: float, cost_fn: float):
    p = TEST_PROBS[model_key]
    cost_fp = cost_fp or 0.0           # an emptied number box arrives as None
    cost_fn = cost_fn or 0.0
    tp, fp, fn, tn = confusion_counts(Y_TEST, p, threshold)
    cost = cost_fp * fp + cost_fn * fn
    never_rest = cost_fn * int((Y_TEST == 1).sum())
    always_rest = cost_fp * int((Y_TEST == 0).sum())
    optimal = cost_fp / (cost_fp + cost_fn) if (cost_fp + cost_fn) > 0 else float("nan")

    z = [[tp, fn], [fp, tn]]
    cm = go.Figure(go.Heatmap(
        z=z, x=["Model says WIN (rest starters)", "Model says NO WIN (play starters)"],
        y=["Home team WON", "Home team LOST"], text=z, texttemplate="%{text}",
        textfont=dict(size=22), colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix at threshold {threshold:.2f} (test set)",
                     height=380, margin=dict(t=60, b=40), yaxis_autorange="reversed")

    grid = np.linspace(0.30, 0.95, 66)
    costs = [cost_fp * confusion_counts(Y_TEST, p, t)[1] + cost_fn * confusion_counts(Y_TEST, p, t)[2]
             for t in grid]
    curve = go.Figure(go.Scatter(x=grid, y=costs, mode="lines", name="total cost"))
    curve.add_vline(x=threshold, line_color="crimson", annotation_text="your threshold")
    if not np.isnan(optimal):
        curve.add_vline(x=optimal, line_dash="dash", line_color="gray", annotation_text="cost-based optimum")
    curve.add_hline(y=never_rest, line_dash="dot", annotation_text="never rest baseline")
    curve.update_layout(title="Total business cost as the threshold moves", xaxis_title="Decision threshold",
                        yaxis_title="Cost", height=380, margin=dict(t=60, b=40))

    if never_rest > 0:
        change = (cost - never_rest) / never_rest
        vs_baseline = (f"Cost is **{abs(change):.1%} {'lower' if change < 0 else 'higher'}** "
                       f"than the *never rest* baseline ({never_rest:,.0f}).")
    else:
        vs_baseline = "The *never rest* baseline costs 0 with these costs, so there is nothing to beat."
    summary = (
        f"### Business cost: **{cost:,.0f}**\n"
        f"- Games where starters were rested: **{tp + fp}** of {len(Y_TEST)}\n"
        f"- False positives (rested, then lost): **{fp}** x {cost_fp:g} = {cost_fp * fp:,.0f}\n"
        f"- False negatives (played starters, would have won): **{fn}** x {cost_fn:g} = {cost_fn * fn:,.0f}\n"
        f"- Baseline *never rest*: {never_rest:,.0f} | Baseline *always rest*: {always_rest:,.0f}\n"
        f"- Cost-based optimal threshold for these costs: "
        f"`{cost_fp:g} / ({cost_fp:g} + {cost_fn:g}) = {optimal:.2f}`\n\n"
        f"{vs_baseline}"
    )
    return cm, curve, summary


# ----------------------------------------------------------------------------
# View 4: try a game
# ----------------------------------------------------------------------------
def try_game(pd5_diff, pd16_diff, rest_diff, div_game, threshold):
    row = pd.DataFrame([{"pd5_diff": pd5_diff, "pd16_diff": pd16_diff,
                         "rest_diff": rest_diff, "div_game": int(div_game)}])
    probs = predict_all(row)
    out = pd.DataFrame([{"Model": MODEL_LABELS[k], "P(home win)": round(float(v[0]), 4),
                         "Decision": "rest starters" if v[0] >= threshold else "play starters"}
                        for k, v in probs.items()])
    return out


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
with gr.Blocks(title="NFL home-win model: three implementations") as demo:
    gr.Markdown(
        "# Will the home team win? Same logistic regression, three implementations\n"
        "Decision supported: **rest starters only when the model is confident of a win.** "
        f"A false positive (rest, then lose) costs {COST_FP:g}; a false negative costs {COST_FN:g} "
        "(invented numbers, see REPORT.md). All models below were trained by `train.py`; "
        "this app only loads them."
    )
    with gr.Tab("1. Model comparison"):
        with gr.Row():
            gr.Plot(calibration_figure(), label="Prediction vs actual")
            gr.Plot(agreement_figure(), label="Agreement")
        gr.Markdown("### Test set results (2023-2025)")
        gr.Dataframe(metrics_table(), interactive=False)
        gr.Markdown("### Learned coefficients (standardized features)")
        gr.Dataframe(coefficient_table(), interactive=False)

    with gr.Tab("2. Distributions"):
        with gr.Row():
            choice = gr.Dropdown(FEATURES + [TARGET_CHOICE], value=FEATURES[0], label="What to plot")
            split = gr.Dropdown(["all", "train", "val", "test"], value="all", label="Data split")
        help_text = gr.Markdown(feature_help(FEATURES[0]))
        dist_plot = gr.Plot(distribution_figure(FEATURES[0], "all"))
        for ctrl in (choice, split):
            ctrl.change(distribution_figure, [choice, split], dist_plot)
        choice.change(feature_help, choice, help_text)

    with gr.Tab("3. Threshold & cost"):
        with gr.Row():
            model_pick = gr.Dropdown([(v, k) for k, v in MODEL_LABELS.items()], value="sklearn", label="Model")
            thr = gr.Slider(0.30, 0.95, value=DEFAULT_THRESHOLD, step=0.01, label="Decision threshold")
            c_fp = gr.Number(value=COST_FP, label="Cost of a false positive (rest, then lose)")
            c_fn = gr.Number(value=COST_FN, label="Cost of a false negative (play, would have won)")
        summary_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        inputs = [model_pick, thr, c_fp, c_fn]
        outputs = [cm_plot, curve_plot, summary_md]
        for ctrl in inputs:
            ctrl.change(threshold_view, inputs, outputs)
        demo.load(threshold_view, inputs, outputs)

    with gr.Tab("4. Try a game"):
        gr.Markdown("Enter the four (unscaled) features for a game. All three loaded models answer.")
        with gr.Row():
            i1 = gr.Slider(-20, 20, value=0, step=0.5, label="pd5_diff (recent margin, home minus away)")
            i2 = gr.Slider(-15, 15, value=0, step=0.5, label="pd16_diff (season margin, home minus away)")
            i3 = gr.Slider(-10, 10, value=0, step=1, label="rest_diff (rest days, home minus away)")
            i4 = gr.Checkbox(value=False, label="Division game")
        i5 = gr.Slider(0.30, 0.95, value=DEFAULT_THRESHOLD, step=0.01, label="Decision threshold")
        game_out = gr.Dataframe(try_game(0, 0, 0, False, DEFAULT_THRESHOLD), interactive=False)
        for ctrl in (i1, i2, i3, i4, i5):
            ctrl.change(try_game, [i1, i2, i3, i4, i5], game_out)


if __name__ == "__main__":
    demo.launch()
