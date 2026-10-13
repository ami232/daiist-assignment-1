"""
Gradio dashboard for the SNL cast-renewal models.

Loads the models and artifacts written by train.py. Nothing is trained here:
predictions are recomputed from the saved models on the saved dataset.
"""

import json
import pickle
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from train import (FEATURE_GROUPS, FEATURES, TARGET, LogReg, predict_manual,
                   predict_module)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

MODEL_NAMES = ["sklearn", "manual_torch", "torch_module"]

# ---- Look & feel: late-night TV studio ------------------------------------
# Data colors were checked for colour-blind separation on the dark surface.
# Neon red / electric blue are only used for page chrome, never for data.
SURFACE = "#0d1020"
INK = "#eceef6"
INK_MUTED = "#9aa0b8"
GRID = "rgba(255,255,255,0.07)"
COLORS = {"baseline": "#8a8fa3", "sklearn": "#3987e5",
          "manual_torch": "#d95926", "torch_module": "#199e70"}
OUTCOME_COLORS = {"renewed": "#3987e5", "not renewed": "#d95926"}
SPLIT_COLORS = {"train": "#3987e5", "val": "#d95926", "test": "#199e70"}

LATE_NIGHT = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
    title=dict(font=dict(family="Oswald, Inter, sans-serif", size=19, color=INK),
               x=0.01, xanchor="left"),
    xaxis=dict(automargin=True, gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
               tickfont=dict(color=INK_MUTED), title_font=dict(color=INK_MUTED)),
    yaxis=dict(automargin=True, gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
               tickfont=dict(color=INK_MUTED), title_font=dict(color=INK_MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_MUTED),
                orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    hoverlabel=dict(bgcolor="#1a1f36", bordercolor="#2b3150",
                    font=dict(color=INK, family="Inter, sans-serif")),
    margin=dict(l=60, r=24, t=72, b=52),
    bargap=0.25,
))


def styled(fig, height=420):
    fig.update_layout(template=LATE_NIGHT, height=height)
    return fig


# --------------------------------------------------------------------------
# Load saved artifacts
# --------------------------------------------------------------------------

def load_everything():
    with open(MODELS / "sklearn_logreg.pkl", "rb") as f:
        sk = pickle.load(f)
    with open(MODELS / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    manual = torch.load(MODELS / "manual_torch.pt")
    module = LogReg(len(FEATURES))
    module.load_state_dict(torch.load(MODELS / "torch_module.pt"))
    module.eval()

    with open(ARTIFACTS / "metrics.json") as f:
        info = json.load(f)
    if info["features"] != FEATURES:
        raise SystemExit("Saved models were trained with different features. "
                         "Run `main.py train` again.")

    data = pd.read_csv(ARTIFACTS / "dataset.csv")
    X = scaler.transform(data[FEATURES].to_numpy(dtype=float))
    data["p_baseline"] = info["baseline_rate"]
    data["p_sklearn"] = sk.predict_proba(X)[:, 1]
    data["p_manual_torch"] = predict_manual(manual, X)
    data["p_torch_module"] = predict_module(module, X)

    coefs = pd.read_csv(ARTIFACTS / "coefficients.csv")
    losses = pd.read_csv(ARTIFACTS / "loss_curves.csv")
    ablation = pd.read_csv(ARTIFACTS / "ablation.csv")
    return data, info, coefs, losses, ablation


DATA, INFO, COEFS, LOSSES, ABLATION = load_everything()
THRESHOLD = INFO["chosen_threshold"]


def subset(split):
    return DATA if split == "all" else DATA[DATA["split"] == split]


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def prediction_vs_actual(split):
    d = subset(split)
    long = d.melt(id_vars=["aid", "sid", TARGET],
                  value_vars=[f"p_{m}" for m in MODEL_NAMES],
                  var_name="model", value_name="p_renewed")
    long["model"] = long["model"].str.removeprefix("p_")
    long["actual"] = long[TARGET].map({1: "renewed", 0: "not renewed"})
    fig = px.box(long, x="actual", y="p_renewed", color="model", points="all",
                 hover_data=["aid", "sid"], color_discrete_map=COLORS,
                 category_orders={"actual": ["not renewed", "renewed"]})
    fig.add_hline(y=INFO["baseline_rate"], line_dash="dot",
                  line_color=COLORS["baseline"], annotation_text="baseline",
                  annotation_font_color=INK_MUTED)
    fig.update_layout(title="Predicted probability of renewal vs. what happened",
                      yaxis_title="P(renewed)", xaxis_title="Actual outcome")
    return styled(fig)


def agreement_plot(split):
    d = subset(split)
    fig = go.Figure()
    for m in ["manual_torch", "torch_module"]:
        fig.add_trace(go.Scatter(x=d["p_sklearn"], y=d[f"p_{m}"], mode="markers",
                                 name=m, marker_color=COLORS[m], opacity=0.6))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x",
                             line=dict(dash="dash", color=INK_MUTED, width=1.5)))
    fig.update_layout(title="PyTorch predictions vs. scikit-learn predictions",
                      xaxis_title="sklearn P(renewed)",
                      yaxis_title="PyTorch P(renewed)")
    return styled(fig)


def calibration_plot(split, n_bins=8):
    d = subset(split)
    fig = go.Figure()
    for m in MODEL_NAMES:
        bins = pd.qcut(d[f"p_{m}"], q=n_bins, duplicates="drop")
        g = d.groupby(bins, observed=True).agg(pred=(f"p_{m}", "mean"),
                                               actual=(TARGET, "mean"))
        fig.add_trace(go.Scatter(x=g["pred"], y=g["actual"], mode="lines+markers",
                                 name=m, line_color=COLORS[m]))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color=INK_MUTED, width=1.5)))
    fig.update_layout(title="Calibration: mean predicted vs. observed renewal rate",
                      xaxis_title="Mean predicted P(renewed)",
                      yaxis_title="Observed renewal rate")
    return styled(fig)


def feature_distribution(feature, split):
    d = subset(split).copy()
    d["actual"] = d[TARGET].map({1: "renewed", 0: "not renewed"})
    fig = px.histogram(d, x=feature, color="actual", barmode="overlay",
                       opacity=0.65, nbins=30,
                       color_discrete_map=OUTCOME_COLORS)
    fig.update_layout(title=f"Distribution of {feature} by outcome")
    return styled(fig)


def target_over_time():
    g = DATA.groupby(["sid", "split"], as_index=False)[TARGET].mean()
    fig = px.bar(g, x="sid", y=TARGET, color="split",
                 color_discrete_map=SPLIT_COLORS)
    fig.update_layout(title="Renewal rate per season (and where the split falls)",
                      xaxis_title="Season", yaxis_title="Share renewed")
    return styled(fig)


def coefficient_plot():
    c = COEFS[COEFS["feature"] != "(intercept)"].melt(
        id_vars="feature", var_name="model", value_name="coefficient")
    fig = px.bar(c, x="coefficient", y="feature", color="model", barmode="group",
                 orientation="h", color_discrete_map=COLORS)
    fig.update_layout(title="Coefficients on standardized features")
    return styled(fig)


def loss_plot():
    long = LOSSES.melt(id_vars="epoch", var_name="model", value_name="loss")
    fig = px.line(long, x="epoch", y="loss", color="model",
                  color_discrete_map=COLORS, log_x=True)
    fig.update_layout(title="Training loss (PyTorch versions)")
    return styled(fig)


def metrics_table(split):
    if split not in INFO["metrics"]:
        return pd.DataFrame({"note": [
            "The final models were refit on train + val (seasons 1-37), so "
            "only test gives an honest score. Plots for train/val show "
            "in-sample fit."]})
    parts = []
    for key, label in [("threshold_0.5", "0.5"),
                       ("chosen_threshold", f"{THRESHOLD} (chosen on val)")]:
        t = pd.DataFrame(INFO["metrics"][split][key]).T.reset_index(names="model")
        t.insert(1, "threshold", label)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Three answers instead of two: auto-flag, auto-renew, or send to a person
# --------------------------------------------------------------------------

def review_band(model, split, low, high):
    d = subset(split)
    p = d[f"p_{model}"].to_numpy()
    y = d[TARGET].to_numpy()
    low, high = min(low, high), max(low, high)

    flag = p < low                # model fairly sure the person leaves
    renew = p >= high             # model fairly sure the person stays
    review = ~flag & ~renew       # the producer looks at these by hand

    auto = flag | renew
    auto_correct = (flag & (y == 0)).sum() + (renew & (y == 1)).sum()
    departures = max(int((y == 0).sum()), 1)

    table = pd.DataFrame({
        "group": ["auto: flagged as leaving", "sent to review",
                  "auto: renewed"],
        "people": [int(flag.sum()), int(review.sum()), int(renew.sum())],
        "of which actually left": [int((flag & (y == 0)).sum()),
                                   int((review & (y == 0)).sum()),
                                   int((renew & (y == 0)).sum())],
    })
    text = tiles([
        ("Coverage", f"{auto.mean():.0%}", "decided without a person"),
        ("Auto accuracy", f"{auto_correct / max(auto.sum(), 1):.0%}",
         "on the automatic decisions"),
        ("To review", f"{int(review.sum())}", "people for the producer"),
        ("Slipped through", f"{int((renew & (y == 0)).sum())} / {departures}",
         "departures auto-renewed"),
    ])
    return table, text


# --------------------------------------------------------------------------
# Threshold, confusion matrix and cost
# --------------------------------------------------------------------------

def threshold_view(model, split, threshold, cost_renew_but_left,
                   cost_cut_but_stayed):
    d = subset(split)
    y = d[TARGET].to_numpy()
    pred = (d[f"p_{model}"].to_numpy() >= threshold).astype(int)

    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())  # predicted renewed, actually left
    fn = int(((pred == 0) & (y == 1)).sum())  # predicted left, actually renewed
    tp = int(((pred == 1) & (y == 1)).sum())

    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["predicted: leaves", "predicted: renewed"],
        y=["actually left", "actually renewed"],
        text=z, texttemplate="%{text}", textfont=dict(size=22, color=INK),
        colorscale=[[0, "#141a33"], [1, "#2a78d6"]], showscale=False,
        xgap=3, ygap=3))
    fig.update_layout(title=f"Confusion matrix ({model}, {split}, t = {threshold:.2f})",
                      yaxis_autorange="reversed")

    cost = fp * cost_renew_but_left + fn * cost_cut_but_stayed
    base_cost = int((y == 0).sum()) * cost_renew_but_left  # "everyone renewed"
    summary = tiles([
        ("Business cost", f"{cost:,.0f}", f"baseline: {base_cost:,.0f}"),
        ("Departures caught", f"{tn} / {tn + fp}", "recall on the people who left"),
        ("False alarms", f"{fn}", "flagged, but they stayed"),
        ("Accuracy", f"{(tp + tn) / max(len(y), 1):.0%}", f"on {len(y)} rows"),
    ])

    # Cost across all thresholds, to see where the minimum is.
    ts = np.linspace(0, 1, 101)
    p = d[f"p_{model}"].to_numpy()
    costs = [(((p >= t) & (y == 0)).sum() * cost_renew_but_left
              + ((p < t) & (y == 1)).sum() * cost_cut_but_stayed) for t in ts]
    cfig = go.Figure(go.Scatter(x=ts, y=costs, mode="lines", name="cost",
                                line=dict(color=COLORS["sklearn"], width=2)))
    cfig.add_vline(x=threshold, line_dash="dash", line_color=INK_MUTED)
    best_t = ts[int(np.argmin(costs))]
    cfig.update_layout(title=f"Cost vs. threshold (minimum at t = {best_t:.2f})",
                       xaxis_title="Threshold on P(renewed)",
                       yaxis_title="Total cost")
    return styled(fig, 380), summary, styled(cfig, 380)


# --------------------------------------------------------------------------
# Page chrome
# --------------------------------------------------------------------------

def tiles(items):
    """A row of stat tiles: (label, big value, small note)."""
    cards = "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div>'
        f'<div class="tile-note">{note}</div></div>'
        for label, value, note in items)
    return f'<div class="tiles">{cards}</div>'


def hero():
    test = INFO["metrics"]["test"]
    chosen = test["chosen_threshold"]["sklearn"]
    base = test["chosen_threshold"]["baseline"]
    max_diff = (COEFS[["manual_torch", "torch_module"]]
                .sub(COEFS["sklearn"], axis=0).abs().max().max())
    return f"""
    <div class="hero">
      <div class="streaks"></div>
      <div class="hero-inner">
        <div class="eyebrow"><span class="live-dot"></span>SNL &middot; cast renewal
          &middot; seasons 1&ndash;45</div>
        <h1>Who's back<br>next season?</h1>
        <p class="lede">One logistic regression, trained three ways
          (scikit-learn, a hand-written PyTorch loop and <code>nn.Module</code>),
          on a time-based split. Everything below is loaded from the saved
          models &mdash; nothing is retrained.</p>
      </div>
    </div>
    {tiles([
        ("Test AUC", f"{test['threshold_0.5']['sklearn']['roc_auc']:.2f}",
         "0.50 = coin flip"),
        ("Departures caught", f"{chosen['recall_departures']:.0%}",
         f"at threshold {THRESHOLD}"),
        ("Cost vs. baseline", f"{chosen['cost']:.0f} vs {base['cost']:.0f}",
         "lower is better"),
        ("Max coef. gap", f"{max_diff:.4f}", "PyTorch vs. sklearn"),
    ])}
    """


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Inter:wght@400;500;600&display=swap');

:root, .dark {
  --snl-surface: #0d1020;
  --snl-panel: rgba(22, 27, 52, 0.72);
  --snl-line: rgba(255, 255, 255, 0.08);
  --snl-ink: #eceef6;
  --snl-muted: #9aa0b8;
  --snl-red: #ff3b4e;
  --snl-blue: #3d8bff;
}
body, gradio-app, .gradio-container {
  background: var(--snl-surface) !important;
  color: var(--snl-ink);
  font-family: Inter, system-ui, sans-serif;
}
.gradio-container { max-width: 1280px !important; }

/* Hero: blurred city-light streaks behind the title */
.hero {
  position: relative; overflow: hidden; border-radius: 18px;
  padding: 48px 40px 40px; margin-bottom: 14px;
  background: linear-gradient(120deg, #120a1c 0%, #0d1020 55%, #08142b 100%);
  border: 1px solid var(--snl-line);
}
.streaks {
  position: absolute; inset: -40% -10%;
  background:
    linear-gradient(100deg, transparent 20%, rgba(255,59,78,0.55) 32%, transparent 44%),
    linear-gradient(100deg, transparent 48%, rgba(61,139,255,0.45) 58%, transparent 68%),
    linear-gradient(100deg, transparent 62%, rgba(255,176,32,0.25) 70%, transparent 78%);
  filter: blur(38px); opacity: .85;
  animation: drift 14s ease-in-out infinite alternate;
}
@keyframes drift { from { transform: translateX(-6%); } to { transform: translateX(6%); } }
@media (prefers-reduced-motion: reduce) { .streaks { animation: none; } }
.hero-inner { position: relative; }
.eyebrow {
  font: 600 12px/1 Inter, sans-serif; letter-spacing: .18em;
  text-transform: uppercase; color: var(--snl-muted);
  display: flex; align-items: center; gap: 10px;
}
.live-dot {
  width: 9px; height: 9px; border-radius: 50%; background: var(--snl-red);
  box-shadow: 0 0 12px var(--snl-red); animation: pulse 1.6s infinite;
}
@keyframes pulse { 50% { opacity: .35; } }
.hero h1 {
  font: 700 clamp(40px, 7vw, 76px)/0.95 Oswald, Impact, sans-serif;
  text-transform: uppercase; letter-spacing: .01em;
  margin: 18px 0 16px; color: var(--snl-ink);
  text-shadow: 0 0 28px rgba(255,59,78,0.35);
}
.lede { max-width: 640px; color: #c9cde0; font-size: 15px; line-height: 1.55; }
.lede code { background: rgba(255,255,255,0.08); padding: 1px 6px; border-radius: 5px; }

/* Stat tiles */
.tiles {
  display: grid; gap: 12px; margin: 6px 0 12px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}
.tile {
  background: var(--snl-panel); border: 1px solid var(--snl-line);
  border-radius: 14px; padding: 16px 18px; backdrop-filter: blur(6px);
}
.tile-label { font: 600 11px Inter, sans-serif; letter-spacing: .14em;
  text-transform: uppercase; color: var(--snl-muted); }
.tile-value { font: 700 34px/1.15 Oswald, sans-serif; color: var(--snl-ink);
  margin-top: 6px; }
.tile-note { font-size: 12.5px; color: var(--snl-muted); margin-top: 2px; }

/* Tabs */
.tab-nav, [role="tablist"] { border-bottom: 1px solid var(--snl-line) !important; }
[role="tab"] {
  font: 600 13px Inter, sans-serif !important; letter-spacing: .08em;
  text-transform: uppercase; color: var(--snl-muted) !important;
}
[role="tab"][aria-selected="true"] {
  color: var(--snl-ink) !important;
  border-bottom: 2px solid var(--snl-red) !important;
}

/* Panels and inputs */
.block, .form, .gr-group {
  background: var(--snl-panel) !important;
  border-color: var(--snl-line) !important; border-radius: 14px !important;
}
h2, h3 { font-family: Oswald, sans-serif !important; letter-spacing: .02em;
  text-transform: uppercase; }
input[type="range"] { accent-color: var(--snl-red); }
footer { opacity: .4; }
.prose { padding: 4px 10px !important; color: var(--snl-muted); }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill=SURFACE, body_background_fill_dark=SURFACE,
    body_text_color=INK, body_text_color_dark=INK,
    body_text_color_subdued=INK_MUTED, body_text_color_subdued_dark=INK_MUTED,
    block_background_fill="rgba(22,27,52,0.72)",
    block_background_fill_dark="rgba(22,27,52,0.72)",
    block_border_color="rgba(255,255,255,0.08)",
    block_border_color_dark="rgba(255,255,255,0.08)",
    block_label_text_color=INK_MUTED, block_label_text_color_dark=INK_MUTED,
    block_title_text_color=INK, block_title_text_color_dark=INK,
    input_background_fill="#151a33", input_background_fill_dark="#151a33",
    input_border_color="rgba(255,255,255,0.12)",
    input_border_color_dark="rgba(255,255,255,0.12)",
    background_fill_primary=SURFACE, background_fill_primary_dark=SURFACE,
    background_fill_secondary="#151a33", background_fill_secondary_dark="#151a33",
    border_color_primary="rgba(255,255,255,0.08)",
    border_color_primary_dark="rgba(255,255,255,0.08)",
    table_even_background_fill="#121731", table_even_background_fill_dark="#121731",
    table_odd_background_fill="#0f1429", table_odd_background_fill_dark="#0f1429",
    slider_color="#ff3b4e", slider_color_dark="#ff3b4e",
    button_primary_background_fill="#ff3b4e",
    button_primary_background_fill_dark="#ff3b4e",
)

# Always render dark, whatever the viewer's OS setting.
FORCE_DARK = """
() => { document.documentElement.classList.add('dark');
        document.body.classList.add('dark'); }
"""


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

with gr.Blocks(title="SNL cast renewal") as demo:
    gr.HTML(hero())
    with gr.Row():
        split = gr.Radio(["test", "val", "train", "all"], value="test",
                         label="Data split", scale=3)
        gr.Markdown(
            f"**C** = {INFO['chosen_C']} &nbsp;·&nbsp; **lr** = {INFO['chosen_lr']}"
            f" &nbsp;·&nbsp; **baseline** always predicts the training renewal "
            f"rate ({INFO['baseline_rate']:.2f}) &nbsp;·&nbsp; final models "
            f"trained on seasons 1–37, threshold set by walk-forward "
            f"validation")

    with gr.Tab("Model comparison"):
        metrics = gr.Dataframe(label="Metrics at threshold 0.5")
        pva = gr.Plot()
        with gr.Row():
            agree = gr.Plot()
            calib = gr.Plot()
        with gr.Row():
            gr.Plot(coefficient_plot())
            gr.Plot(loss_plot())
        gr.Markdown("### Feature ablation (validation): drop one group at a time")
        gr.Dataframe(ABLATION)

    with gr.Tab("Threshold & cost"):
        with gr.Row():
            model = gr.Dropdown(MODEL_NAMES, value="sklearn", label="Model")
            threshold = gr.Slider(0, 1, value=THRESHOLD, step=0.01,
                                  label="Threshold on P(renewed) "
                                        f"(chosen on validation: {THRESHOLD})")
        with gr.Row():
            c_fp = gr.Number(value=INFO["cost_missed_departure"],
                             label="Cost: predicted renewed, but the person left")
            c_fn = gr.Number(value=INFO["cost_false_alarm"],
                             label="Cost: predicted to leave, but the person stayed")
        cost_md = gr.HTML()
        with gr.Row():
            cm = gr.Plot()
            cost_curve = gr.Plot()

    with gr.Tab("Human review band"):
        gr.Markdown(
            "Below the low cut the model flags the person as leaving, above the "
            "high cut it renews them, and everyone in between goes to the "
            "producer to decide by hand.")
        with gr.Row():
            band_model = gr.Dropdown(MODEL_NAMES, value="sklearn", label="Model")
            low = gr.Slider(0, 1, value=0.6, step=0.01, label="Low cut")
            high = gr.Slider(0, 1, value=0.85, step=0.01, label="High cut")
        band_md = gr.HTML()
        band_table = gr.Dataframe()

    with gr.Tab("Data"):
        gr.Plot(target_over_time())
        feature = gr.Dropdown(FEATURES, value=FEATURES[0], label="Feature")
        dist = gr.Plot()

    comparison_outputs = [metrics, pva, agree, calib]

    def update_comparison(s):
        return (metrics_table(s), prediction_vs_actual(s),
                agreement_plot(s), calibration_plot(s))

    threshold_inputs = [model, split, threshold, c_fp, c_fn]
    threshold_outputs = [cm, cost_md, cost_curve]

    split.change(update_comparison, split, comparison_outputs)
    split.change(threshold_view, threshold_inputs, threshold_outputs)
    split.change(feature_distribution, [feature, split], dist)
    for comp in [model, threshold, c_fp, c_fn]:
        comp.change(threshold_view, threshold_inputs, threshold_outputs)
    feature.change(feature_distribution, [feature, split], dist)

    band_inputs = [band_model, split, low, high]
    for comp in [band_model, split, low, high]:
        comp.change(review_band, band_inputs, [band_table, band_md])
    demo.load(review_band, band_inputs, [band_table, band_md])

    demo.load(update_comparison, split, comparison_outputs)
    demo.load(threshold_view, threshold_inputs, threshold_outputs)
    demo.load(feature_distribution, [feature, split], dist)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=FORCE_DARK)
