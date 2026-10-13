"""
Assignment 1 — Gradio dashboard.

Loads the models and data that `main.py train` saved in ./artifacts and
never trains anything. Run with:  uv run python main.py app
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_curve

from models import LogisticRegressionModule, predict_manual, predict_standard

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
if not (ART / "metrics.json").exists():
    raise SystemExit("No artifacts found — run `uv run python main.py train` first.")

# --- Load everything the training stage produced ---------------------------
METRICS = json.loads((ART / "metrics.json").read_text())
CFG = METRICS["config"]
FEATURES = pd.read_csv(ART / "features.csv")
COEFS = pd.read_csv(ART / "coefficients.csv")
HISTORY = pd.read_csv(ART / "training_history.csv")
PRE = joblib.load(ART / "preprocessor.joblib")
SK = joblib.load(ART / "sklearn_logreg.joblib")
MANUAL = torch.load(ART / "manual_torch.pt")
STANDARD = LogisticRegressionModule(METRICS["n_features"])
STANDARD.load_state_dict(torch.load(ART / "standard_torch.pt"))

# Re-score every row through the saved preprocessor + saved models.
MODEL_COLS = list(PRE.feature_names_in_)
X_ALL = PRE.transform(FEATURES[MODEL_COLS])
Y = FEATURES["acquired"].to_numpy()
SPLIT = FEATURES["split"].to_numpy()
TRAIN_RATE = Y[SPLIT == "train"].mean()
PROBS = {
    "sklearn": SK.predict_proba(X_ALL)[:, 1],
    "manual_torch": predict_manual(MANUAL, X_ALL),
    "standard_torch": predict_standard(STANDARD, X_ALL),
    "baseline": np.full(len(Y), TRAIN_RATE),
}
MODELS = ["sklearn", "manual_torch", "standard_torch"]

# Reference palette (dataviz skill): fixed categorical order, gray for baseline.
COLOR = {"sklearn": "#2a78d6", "manual_torch": "#eb6834",
         "standard_torch": "#1baf7a", "baseline": "#8a8985"}
OUTCOME_COLOR = {1: "#2a78d6", 0: "#eb6834"}
OUTCOME_NAME = {1: "Acquired", 0: "Closed"}
LAYOUT = dict(template="plotly_white", margin=dict(l=50, r=20, t=50, b=45),
              font=dict(size=13), hovermode="closest",
              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def mask(split):
    return SPLIT == split


# --- Tab 1: model comparison -----------------------------------------------
def metrics_table():
    rows = []
    for name, r in METRICS["results"].items():
        rows.append({"model": name, "ROC AUC": r["roc_auc"], "PR AUC": r["pr_auc"],
                     "log loss": r["log_loss"], "Brier": r["brier"],
                     "threshold (val)": r["threshold"], "precision": r["precision"],
                     "recall": r["recall"], "meetings": r["meetings_taken"],
                     "cost / 100 startups": r["cost_per_100"]})
    return pd.DataFrame(rows).round(4)


def pred_vs_actual(split):
    """Predicted P(acquired) for each startup, grouped by what actually happened."""
    m = mask(split)
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for i, name in enumerate(MODELS):
        for outcome in (1, 0):
            sel = m & (Y == outcome)
            x = i + (0.18 if outcome else -0.18) + rng.uniform(-0.08, 0.08, sel.sum())
            fig.add_trace(go.Scatter(
                x=x, y=PROBS[name][sel], mode="markers",
                marker=dict(size=8, color=OUTCOME_COLOR[outcome], opacity=0.55,
                            line=dict(width=1, color="white")),
                name=OUTCOME_NAME[outcome], legendgroup=str(outcome),
                showlegend=(i == 0), text=FEATURES["name"][sel],
                hovertemplate="%{text}<br>P(acquired)=%{y:.3f}<extra>" + name + "</extra>"))
    fig.add_hline(y=TRAIN_RATE, line_dash="dot", line_color=COLOR["baseline"],
                  annotation_text=f"baseline {TRAIN_RATE:.2f}", annotation_position="top left")
    fig.update_layout(**LAYOUT, title=f"Predicted P(acquired) vs actual outcome — {split} set",
                      xaxis=dict(tickvals=[0, 1, 2], ticktext=MODELS, range=[-0.6, 2.6]),
                      yaxis=dict(title="predicted P(acquired)", range=[0, 1]))
    return fig


def calibration(split):
    """Binned predicted probability vs observed acquisition rate."""
    m = mask(split)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color=COLOR["baseline"], dash="dot", width=1)))
    bins = np.linspace(0, 1, 11)
    for name in MODELS:
        p, y = PROBS[name][m], Y[m]
        b = np.clip(np.digitize(p, bins) - 1, 0, 9)
        df = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(
            p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(
            x=df.p, y=df.y, mode="lines+markers", name=name, customdata=df.n,
            line=dict(color=COLOR[name], width=2), marker=dict(size=8),
            hovertemplate="mean predicted %{x:.2f}<br>observed %{y:.2f}<br>n=%{customdata}"))
    fig.update_layout(**LAYOUT, title=f"Calibration — {split} set",
                      xaxis=dict(title="mean predicted P(acquired)", range=[0, 1]),
                      yaxis=dict(title="observed acquisition rate", range=[0, 1]))
    return fig


def roc(split):
    m = mask(split)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="baseline (AUC 0.5)",
                             line=dict(color=COLOR["baseline"], dash="dot", width=1)))
    for name in MODELS:
        fpr, tpr, _ = roc_curve(Y[m], PROBS[name][m])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name,
                                 line=dict(color=COLOR[name], width=2)))
    fig.update_layout(**LAYOUT, title=f"ROC — {split} set",
                      xaxis=dict(title="false positive rate"),
                      yaxis=dict(title="true positive rate"))
    return fig


def agreement_plot(split):
    m = mask(split)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x",
                             line=dict(color=COLOR["baseline"], dash="dot", width=1)))
    for name in ["manual_torch", "standard_torch"]:
        fig.add_trace(go.Scatter(x=PROBS["sklearn"][m], y=PROBS[name][m], mode="markers",
                                 name=name, marker=dict(size=8, color=COLOR[name], opacity=0.6)))
    fig.update_layout(**LAYOUT, title="Do the three agree? PyTorch vs sklearn probabilities "
                      "(points on y = x mean identical; the two PyTorch series overlap)",
                      xaxis=dict(title="sklearn P(acquired)"),
                      yaxis=dict(title="PyTorch P(acquired)"))
    return fig


def coef_plot():
    c = COEFS[COEFS.feature != "intercept"].copy()
    c["label"] = c.feature.str.replace(r"^\w+__", "", regex=True)
    c = c.reindex(c.sklearn.abs().sort_values().index)
    fig = go.Figure()
    for name in MODELS:
        fig.add_trace(go.Bar(y=c.label, x=c[name], orientation="h", name=name,
                             marker_color=COLOR[name]))
    fig.update_layout(**LAYOUT, barmode="group", bargap=0.25,
                      title="Coefficients (log-odds per unit of scaled feature)",
                      height=max(400, 28 * len(c)), xaxis=dict(title="coefficient"))
    return fig


def loss_plot():
    fig = go.Figure()
    for name in ["manual_torch", "standard_torch"]:
        fig.add_trace(go.Scatter(y=HISTORY[name], mode="lines", name=name,
                                 line=dict(color=COLOR[name], width=2)))
    fig.update_layout(**LAYOUT, title="PyTorch training objective by epoch",
                      xaxis=dict(title="epoch"), yaxis=dict(title="BCE + L2 penalty"))
    return fig


# --- Tab 2: distributions --------------------------------------------------
EXCLUDE = {"acquired", "split", "name", "founded_year_raw"}
FEATURE_CHOICES = [c for c in FEATURES.columns if c not in EXCLUDE] + ["founded_year_raw"]


def feature_dist(col, split):
    df = FEATURES if split == "all" else FEATURES[FEATURES.split == split]
    fig = go.Figure()
    categorical = (not pd.api.types.is_numeric_dtype(df[col])) or df[col].nunique() <= 2
    for outcome in (1, 0):
        sub = df[df.acquired == outcome][col]
        if categorical:
            vc = sub.value_counts()
            fig.add_trace(go.Bar(x=vc.index.astype(str), y=vc.values,
                                 name=OUTCOME_NAME[outcome], marker_color=OUTCOME_COLOR[outcome]))
        else:
            fig.add_trace(go.Histogram(x=sub, name=OUTCOME_NAME[outcome], opacity=0.6,
                                       marker_color=OUTCOME_COLOR[outcome], nbinsx=30))
    fig.update_layout(**LAYOUT, barmode="group" if categorical else "overlay",
                      title=f"{col} by outcome ({split})", xaxis=dict(title=col),
                      yaxis=dict(title="startups"))
    return fig


def target_by_year():
    g = FEATURES.groupby("founded_year_raw").agg(rate=("acquired", "mean"), n=("acquired", "size"))
    g = g[g.n >= 5]
    fig = go.Figure(go.Bar(x=g.index, y=g.rate, customdata=g.n, marker_color=COLOR["sklearn"],
                           hovertemplate="founded %{x}<br>acquired %{y:.0%}<br>n=%{customdata}"))
    fig.update_layout(**LAYOUT, title="Acquisition rate by founding year (years with ≥5 startups)",
                      xaxis=dict(title="founded"), yaxis=dict(title="share acquired", range=[0, 1]))
    return fig


def target_by_split():
    g = FEATURES.groupby("split").agg(rate=("acquired", "mean"), n=("acquired", "size"))
    g = g.reindex(["train", "val", "test"])
    fig = go.Figure(go.Bar(x=g.index, y=g.rate, customdata=g.n, marker_color=COLOR["sklearn"],
                           text=[f"{r:.0%} (n={n})" for r, n in zip(g.rate, g.n)],
                           textposition="outside",
                           hovertemplate="%{x}: %{y:.1%} acquired, n=%{customdata}"))
    fig.update_layout(**LAYOUT, title="Target: share acquired in each split",
                      yaxis=dict(title="share acquired", range=[0, 1]))
    return fig


# --- Tab 3: threshold, confusion matrix, business cost ---------------------
def confusion(y, p, t):
    meet = p >= t
    return (int((meet & (y == 1)).sum()), int((meet & (y == 0)).sum()),
            int((~meet & (y == 1)).sum()), int((~meet & (y == 0)).sum()))


def threshold_view(model, split, t, c_fp, c_fn):
    m = mask(split)
    y, p = Y[m], PROBS[model][m]
    tp, fp, fn, tn = confusion(y, p, t)
    cost = c_fp * fp + c_fn * fn

    z = [[tp, fn], [fp, tn]]
    labels = [[f"Met & acquired<br><b>{tp}</b>", f"Passed, but acquired<br><b>{fn}</b><br>(missed winner)"],
              [f"Met, but closed<br><b>{fp}</b><br>(wasted meeting)", f"Passed & closed<br><b>{tn}</b>"]]
    cm = go.Figure(go.Heatmap(z=z, x=["Meet", "Pass"], y=["Acquired", "Closed"],
                              text=labels, texttemplate="%{text}", showscale=False,
                              colorscale=[[0, "#eaf1fb"], [1, "#2a78d6"]], xgap=2, ygap=2,
                              hovertemplate="%{text}<extra></extra>"))
    cm.update_layout(**LAYOUT, title=f"Confusion matrix — {model}, {split}, threshold {t:.2f}",
                     xaxis=dict(title="analyst decision", side="top"),
                     yaxis=dict(title="actual outcome", autorange="reversed"), height=400)
    cm.update_layout(margin=dict(l=50, r=20, t=110, b=20))

    grid = np.linspace(0, 1, 101)
    costs = [c_fp * confusion(y, p, g)[1] + c_fn * confusion(y, p, g)[2] for g in grid]
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name=model,
                               line=dict(color=COLOR[model], width=2)))
    curve.add_trace(go.Scatter(x=[t], y=[cost], mode="markers", name="current threshold",
                               marker=dict(size=12, color=COLOR[model], line=dict(width=2, color="white"))))
    curve.add_hline(y=c_fp * (y == 0).sum(), line_dash="dot", line_color=COLOR["baseline"],
                    annotation_text="meet everyone", annotation_position="bottom right")
    curve.add_hline(y=c_fn * (y == 1).sum(), line_dash="dash", line_color=COLOR["baseline"],
                    annotation_text="meet no one", annotation_position="top right")
    curve.update_layout(**LAYOUT, title="Total cost vs threshold",
                        xaxis=dict(title="threshold"), yaxis=dict(title="total cost"), height=380)

    n = len(y)
    meet_all = c_fp * (y == 0).sum()
    summary = (
        f"### Cost: **{cost:.1f}**  ({100 * cost / n:.1f} per 100 startups)\n"
        f"- Meetings taken: **{tp + fp}** of {n} · precision {tp / (tp + fp) if tp + fp else float('nan'):.2f}"
        f" · recall {tp / (tp + fn) if tp + fn else float('nan'):.2f}\n"
        f"- 'Meet everyone' would cost **{meet_all:.1f}**; 'meet no one' **{c_fn * (y == 1).sum():.1f}**\n"
        f"- Saving vs meeting everyone: **{meet_all - cost:+.1f}**")
    return cm, curve, summary


# --- Layout ----------------------------------------------------------------
HEADER = (
    "# Inbound startup screening — logistic regression, three ways\n"
    f"Meeting stage **{CFG['meeting_stage']}** · split **{CFG['split']}** · "
    f"founded ≤ **{CFG['max_founded_year']}** · negative ages **{CFG['negative_age']}** · "
    f"C = **{METRICS['C']}** · lr = **{METRICS['lr']}** · rows {METRICS['n']}\n\n"
    "Models are loaded from `artifacts/`; nothing is retrained here.")

with gr.Blocks(title="Startup screening dashboard") as demo:
    gr.Markdown(HEADER)
    with gr.Tab("Model comparison"):
        gr.Markdown("Test-set metrics; each model's threshold was chosen on the validation set by cost.")
        gr.Dataframe(metrics_table(), interactive=False)
        gr.Markdown("**Agreement (test set):** " + " · ".join(
            f"{k.replace('_', ' ')}: `{v:.1e}`" for k, v in METRICS["agreement"].items()))
        split_cmp = gr.Radio(["test", "val", "train"], value="test", label="Data split")
        p1 = gr.Plot(pred_vs_actual("test"))
        with gr.Row():
            p2 = gr.Plot(calibration("test"))
            p3 = gr.Plot(roc("test"))
        p4 = gr.Plot(agreement_plot("test"))
        split_cmp.change(lambda s: (pred_vs_actual(s), calibration(s), roc(s), agreement_plot(s)),
                         split_cmp, [p1, p2, p3, p4])
        with gr.Row():
            gr.Plot(coef_plot())
            gr.Plot(loss_plot())

    with gr.Tab("Distributions"):
        with gr.Row():
            feat = gr.Dropdown(FEATURE_CHOICES, value=FEATURE_CHOICES[0], label="Feature")
            split_d = gr.Radio(["all", "train", "val", "test"], value="all", label="Split")
        fd = gr.Plot(feature_dist(FEATURE_CHOICES[0], "all"))
        feat.change(feature_dist, [feat, split_d], fd)
        split_d.change(feature_dist, [feat, split_d], fd)
        with gr.Row():
            gr.Plot(target_by_split())
            gr.Plot(target_by_year())

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            model = gr.Dropdown(MODELS + ["baseline"], value="sklearn", label="Model")
            split_t = gr.Radio(["test", "val"], value="test", label="Split")
        thr = gr.Slider(0, 1, value=METRICS["results"]["sklearn"]["threshold"], step=0.01,
                        label="Decision threshold: meet if P(acquired) ≥ threshold")
        with gr.Row():
            c_fp = gr.Number(CFG["cost_wasted_meeting"], label="Cost of a wasted meeting (met, closed)")
            c_fn = gr.Number(CFG["cost_missed_winner"], label="Cost of a missed winner (passed, acquired)")
        summary = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        inputs = [model, split_t, thr, c_fp, c_fn]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, cost_plot, summary])
        demo.load(threshold_view, inputs, [cm_plot, cost_plot, summary])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
