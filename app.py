"""
Gradio dashboard for the Valencia Airbnb churn models.

It only LOADS what train.ipynb saved in models/ and artifacts/. Nothing is
trained here: the three saved models are read from disk and used to score
the saved test quarter (July -> September 2026).

Run with:  uv run python main.py app
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, roc_auc_score

from churn_model import (CALL_COST, SAVE_RATE, VALUE_LOST, LogisticRegressionModule,
                         business_cost, manual_predict_proba)

MODELS = Path("models")
ARTIFACTS = Path("artifacts")
MODEL_NAMES = ["sklearn", "manual torch", "nn.Module"]
MODEL_LABELS = {"sklearn": "scikit-learn", "manual torch": "Manual PyTorch loop",
                "nn.Module": "nn.Module + optim", "baseline": "Naive baseline"}

# ---------------------------------------------------------------- visual style
# Colours: three categorical slots validated for colour blind readers, plus neutral inks.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, PAGE = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f6f5f2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {"sklearn": BLUE, "manual torch": ORANGE, "nn.Module": AQUA, "baseline": MUTED}
OUTCOME_COLORS = {"stayed": BLUE, "left": ORANGE}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style(fig, title, subtitle=None, height=400, legend=True):
    """Apply one consistent look to every chart."""
    text = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{INK_2}'>{subtitle}</span>" if subtitle else "")
    top = (80 if subtitle else 56) + (34 if legend else 0)  # title, subtitle, then the legend row
    fig.update_layout(
        title=dict(text=text, x=0.01, xanchor="left", y=1, yanchor="top", pad=dict(t=16),
                   font=dict(size=15, color=INK)),
        height=height, margin=dict(l=64, r=24, t=top, b=56),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left",
                    font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS,
                     title_font=dict(size=12, color=INK_2))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, title_font=dict(size=12, color=INK_2))
    return fig


# ---------------------------------------------------------------- load artifacts (no training)
meta = json.loads((ARTIFACTS / "metadata.json").read_text())
X_test = np.load(ARTIFACTS / "X_test.npy")
test_features = pd.read_csv(ARTIFACTS / "test_features.csv")
train_features = pd.read_csv(ARTIFACTS / "train_features.csv")
y_test = test_features["left"].to_numpy()

sk_model = joblib.load(MODELS / "sklearn_logreg.joblib")
manual_params = torch.load(MODELS / "torch_manual.pt")
module_model = LogisticRegressionModule(X_test.shape[1])
module_model.load_state_dict(torch.load(MODELS / "torch_module.pt"))
module_model.eval()

proba = {
    "sklearn": sk_model.predict_proba(X_test)[:, 1],
    "manual torch": manual_predict_proba(manual_params, X_test),
    "nn.Module": module_model.predict_proba(X_test),
    "baseline": np.full(len(y_test), meta["base_rate"]),
}
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
COST_NOBODY = business_cost(y_test, proba["sklearn"], 1.01)
COST_EVERYONE = business_cost(y_test, proba["sklearn"], 0.0)
MAX_PROB_GAP = max(np.abs(proba["sklearn"] - proba[m]).max() for m in ["manual torch", "nn.Module"])
TEST_AUC = roc_auc_score(y_test, proba["sklearn"])
VAL_AUC = roc_auc_score(np.load(ARTIFACTS / "y_val.npy"),
                        sk_model.predict_proba(np.load(ARTIFACTS / "X_val.npy"))[:, 1])
BEST_T = meta["best_thresholds"]["sklearn"]
MODEL_COST = business_cost(y_test, proba["sklearn"], BEST_T)


# ---------------------------------------------------------------- small HTML pieces
def euro(x):
    return f"€{x:,.0f}"


def kpi(label, value, note="", tone=""):
    return (f"<div class='kpi {tone}'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>")


def kpi_row(items):
    return "<div class='kpi-row'>" + "".join(items) + "</div>"


header_kpis = kpi_row([
    kpi("Test listings", f"{len(y_test):,}", "short stays, July → September 2026"),
    kpi("Leave rate", f"{y_test.mean():.1%}", f"{int(y_test.sum())} listings left the platform"),
    kpi("ROC AUC on test", f"{TEST_AUC:.3f}", f"{VAL_AUC:.3f} on validation (same quarter as training)"),
    kpi("Model agreement", f"{MAX_PROB_GAP:.3f}", "largest gap in any predicted probability"),
    kpi("Cost vs calling nobody", ("+" if MODEL_COST > COST_NOBODY else "−") + euro(abs(MODEL_COST - COST_NOBODY)),
        f"scikit-learn at the validation threshold {BEST_T:.2f}",
        tone="warn" if MODEL_COST > COST_NOBODY else "good"),
])


# ---------------------------------------------------------------- tab 1: model comparison
def metrics_table():
    rows = []
    for name in ["baseline"] + MODEL_NAMES:
        p = proba[name]
        t = meta["best_thresholds"].get(name)
        rows.append({"Model": MODEL_LABELS[name],
                     "ROC AUC": round(roc_auc_score(y_test, p), 3),
                     "PR AUC": round(average_precision_score(y_test, p), 3),
                     "Log loss": round(log_loss(y_test, p, labels=[0, 1]), 4),
                     "Threshold": f"{t:.2f}" if t is not None else "—",
                     "Test cost": euro(business_cost(y_test, p, t)) if t is not None
                     else f"{euro(COST_NOBODY)} (call nobody)"})
    return pd.DataFrame(rows)


def calibration_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.45], y=[0, 0.45], mode="lines", name="Perfect calibration",
                             line=dict(color=MUTED, width=1.5, dash="dot"), hoverinfo="skip"))
    for name, dash, size in [("sklearn", "solid", 10), ("manual torch", "dash", 8), ("nn.Module", "dot", 6)]:
        p = proba[name]
        g = (pd.DataFrame({"p": p, "y": y_test, "bin": pd.qcut(p, 10, duplicates="drop")})
             .groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")))
        fig.add_trace(go.Scatter(
            x=g["p"], y=g["y"], mode="lines+markers", name=MODEL_LABELS[name],
            line=dict(color=COLORS[name], width=2, dash=dash),
            marker=dict(size=size, color=COLORS[name], line=dict(color=SURFACE, width=2)),
            customdata=g["n"],
            hovertemplate="Predicted %{x:.1%}<br>Actually left %{y:.1%}<br>%{customdata} listings<extra>"
                          + MODEL_LABELS[name] + "</extra>"))
    fig.update_xaxes(title="Average predicted probability of leaving", tickformat=".0%", rangemode="tozero")
    fig.update_yaxes(title="Share that actually left", tickformat=".0%", rangemode="tozero")
    return style(fig, "Predicted vs actual leave rate",
                 "Test listings sorted into 10 equal groups by predicted probability")


def prediction_by_outcome_plot():
    fig = go.Figure()
    for outcome, label in [(0, "stayed"), (1, "left")]:
        n = int((y_test == outcome).sum())
        for name in MODEL_NAMES:
            fig.add_trace(go.Box(
                y=proba[name][y_test == outcome], x=[MODEL_LABELS[name]] * n,
                name=f"Actually {label}", legendgroup=label, showlegend=name == "sklearn",
                marker_color=OUTCOME_COLORS[label], line=dict(width=1.5), boxpoints=False,
                fillcolor="rgba(42,120,214,0.15)" if label == "stayed" else "rgba(235,104,52,0.15)",
                hovertemplate="%{y:.1%}<extra>Actually " + label + "</extra>"))
    fig.update_layout(boxmode="group", boxgap=0.35, boxgroupgap=0.15)
    fig.update_yaxes(title="Predicted probability of leaving", tickformat=".0%")
    return style(fig, "Predictions for listings that stayed vs left",
                 "Higher boxes for listings that left mean the model separates the two groups")


def coefficient_plot():
    coef = pd.DataFrame(meta["coefficients"]).set_index("feature").drop(index="(intercept)")
    coef = coef.reindex(coef["sklearn"].abs().sort_values().index).tail(12)
    fig = go.Figure()
    for name in reversed(MODEL_NAMES):
        fig.add_trace(go.Bar(
            y=coef.index, x=coef[name], name=MODEL_LABELS[name], orientation="h",
            marker=dict(color=COLORS[name], line=dict(color=SURFACE, width=1)),
            hovertemplate="%{y}: %{x:.3f}<extra>" + MODEL_LABELS[name] + "</extra>"))
    fig.add_vline(x=0, line_color=AXIS, line_width=1)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05, legend_traceorder="reversed")
    fig.update_xaxes(title="Coefficient on standardised feature (positive = more likely to leave)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    return style(fig, "What drives the prediction: 12 largest coefficients",
                 "The three implementations should overlap almost exactly", height=520)


# ---------------------------------------------------------------- tab 2: distributions
FEATURES = [c for c in test_features.columns if c != "left"]
PERIODS = {"Training: March → July": train_features, "Test: July → September": test_features}


def target_plot():
    fig = go.Figure()
    for (label, df), color in zip(PERIODS.items(), [BLUE, ORANGE]):
        rate = df["left"].mean()
        fig.add_trace(go.Bar(
            x=["Stayed", "Left"], y=[1 - rate, rate], name=label, marker=dict(color=color),
            text=[f"{1 - rate:.1%}", f"{rate:.1%}"], textposition="outside", textfont=dict(color=INK_2),
            hovertemplate="%{x}: %{y:.1%}<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.12], title="Share of listings")
    return style(fig, "Target: did the listing leave the platform?",
                 "Only about 1 in 16 listings leaves, so accuracy is a misleading measure", height=380)


def feature_plot(feature, period):
    df = PERIODS[period]
    fig = go.Figure()
    if not pd.api.types.is_numeric_dtype(df[feature]) or df[feature].nunique() <= 2:
        rates = df.groupby(feature)["left"].agg(["mean", "size"]).reset_index()
        rates[feature] = rates[feature].astype(str)
        rates = rates.sort_values("mean")
        fig.add_trace(go.Bar(
            x=rates["mean"], y=rates[feature], orientation="h", marker=dict(color=ORANGE),
            customdata=rates["size"], text=[f"{v:.1%}" for v in rates["mean"]], textposition="outside",
            hovertemplate="%{y}: %{x:.1%} left (%{customdata} listings)<extra></extra>", showlegend=False))
        fig.update_xaxes(title="Share that left", tickformat=".0%", range=[0, rates["mean"].max() * 1.18])
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
        return style(fig, f"Leave rate by {feature}", period, height=440, legend=False)
    for outcome, label in [(0, "stayed"), (1, "left")]:
        fig.add_trace(go.Histogram(
            x=df.loc[df["left"] == outcome, feature], name=f"Listings that {label}",
            histnorm="percent", opacity=0.65, nbinsx=30, marker=dict(color=OUTCOME_COLORS[label]),
            hovertemplate="%{x}<br>%{y:.1f}% of group<extra>" + label + "</extra>"))
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title="% of group")
    return style(fig, f"Distribution of {feature}", f"{period}, each group scaled to 100%", height=440)


# ---------------------------------------------------------------- tab 3: threshold and cost
def threshold_view(model_name, threshold):
    p = proba[model_name]
    cm = confusion_matrix(y_test, p >= threshold, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cost = business_cost(y_test, p, threshold)
    diff = cost - COST_NOBODY

    kpis = kpi_row([
        kpi("Total business cost", euro(cost), f"model: {MODEL_LABELS[model_name]}"),
        kpi("Vs calling nobody", ("+" if diff > 0 else "−") + euro(abs(diff)),
            "more expensive" if diff > 0 else "saved", tone="warn" if diff > 0 else "good"),
        kpi("Hosts called", f"{tp + fp:,}", f"{euro((tp + fp) * CALL_COST)} in calls"),
        kpi("Leavers caught", f"{tp}", f"of {tp + fn} who left ({tp / max(tp + fn, 1):.0%})"),
        kpi("Precision of calls", f"{tp / max(tp + fp, 1):.0%}", "of called hosts were really leaving"),
    ])

    labels = np.array([["Correctly left alone", "Called, but would have stayed"],
                       ["Missed: left without a call", "Called and was leaving"]])
    text = [[f"<b>{cm[i, j]:,}</b><br><span style='font-size:11px'>{labels[i, j]}</span>" for j in range(2)]
            for i in range(2)]
    cm_fig = go.Figure(go.Heatmap(
        z=[[0.15, 0.55], [0.55, 1.0]], x=["Not called", "Called"], y=["Stayed", "Left"],
        text=text, texttemplate="%{text}", textfont=dict(size=15, color=INK),
        colorscale=[[0, "#eef4fc"], [1, "#86b6ef"]], showscale=False, xgap=3, ygap=3, hoverinfo="skip"))
    cm_fig.update_xaxes(title="Decision", showgrid=False)
    cm_fig.update_yaxes(title="What actually happened", autorange="reversed", showgrid=False)
    style(cm_fig, "Confusion matrix", f"Threshold {threshold:.2f} on the test quarter", height=400, legend=False)

    costs = np.array([business_cost(y_test, p, t) for t in THRESHOLDS])
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=THRESHOLDS, y=costs, mode="lines", name=MODEL_LABELS[model_name],
                               line=dict(color=COLORS[model_name], width=2),
                               hovertemplate="Threshold %{x:.2f}<br>Cost €%{y:,.0f}<extra></extra>"))
    curve.add_hline(y=COST_NOBODY, line=dict(color=MUTED, dash="dot", width=1.5),
                    annotation_text=f"Call nobody: {euro(COST_NOBODY)}", annotation_position="top right",
                    annotation_font=dict(color=INK_2, size=11))
    curve.add_vline(x=meta["best_thresholds"][model_name], line=dict(color=AXIS, dash="dash", width=1.5),
                    annotation_text="Chosen on validation", annotation_position="bottom right",
                    annotation_font=dict(color=INK_2, size=11))
    curve.add_trace(go.Scatter(x=[threshold], y=[cost], mode="markers", name="Current threshold",
                               marker=dict(size=11, color=INK, line=dict(color=SURFACE, width=2)),
                               hovertemplate="Current: €%{y:,.0f}<extra></extra>"))
    curve.update_xaxes(title="Decision threshold", range=[0, 1])
    curve.update_yaxes(title="Total cost (€)", tickprefix="€", tickformat=",.0f")
    style(curve, "Total cost by threshold", "Test quarter; the lowest point is the cheapest policy", height=400)
    return kpis, cm_fig, curve


# ---------------------------------------------------------------- page
CSS = f"""
.gradio-container {{ max-width: 1280px !important; margin: 0 auto !important; }}
#hero {{ padding: 24px 4px 4px 4px; }}
#hero h1 {{ font-size: 28px; font-weight: 700; color: {INK}; margin: 0 0 6px 0; letter-spacing: -0.01em; }}
#hero p {{ font-size: 15px; color: {INK_2}; margin: 0; max-width: 900px; line-height: 1.5; }}
.eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
           color: {BLUE}; margin-bottom: 8px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 14px 0 6px 0; }}
.kpi {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.08); border-radius: 12px; padding: 14px 16px; }}
.kpi-label {{ font-size: 12px; color: {INK_2}; font-weight: 500; }}
.kpi-value {{ font-size: 26px; font-weight: 700; color: {INK}; margin: 4px 0 2px 0; }}
.kpi-note {{ font-size: 12px; color: {MUTED}; line-height: 1.35; }}
.kpi.warn {{ border-left: 4px solid {ORANGE}; }}
.kpi.good {{ border-left: 4px solid #0ca30c; }}
.section-note {{ font-size: 14px; color: {INK_2}; line-height: 1.55; margin: 6px 0 10px 0; }}
.card {{ border-radius: 12px !important; overflow: hidden; }}
.method {{ font-size: 14px; color: {INK_2}; line-height: 1.6; max-width: 860px; }}
.method h3 {{ color: {INK}; font-size: 16px; margin: 18px 0 4px 0; }}
"""

# The charts are designed for a light background, so the page always opens in light mode.
FORCE_LIGHT = """
<script>
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u); }
</script>
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.Font("system-ui"), gr.themes.Font("-apple-system"), gr.themes.Font("Segoe UI"),
          gr.themes.Font("sans-serif")],
    radius_size=gr.themes.sizes.radius_md,
).set(body_background_fill=PAGE, block_background_fill=SURFACE, block_border_width="1px",
      block_shadow="none", block_title_text_weight="600")

with gr.Blocks(title="Valencia Airbnb host churn") as demo:
    gr.HTML(
        "<div id='hero'><div class='eyebrow'>Valencia Airbnb · host retention</div>"
        "<h1>Which listings will leave the platform next quarter?</h1>"
        "<p>Logistic regression trained three ways (scikit-learn, a manual PyTorch loop and nn.Module + optim) "
        "on March → July 2026 and tested on July → September 2026. Everything here is loaded from the files the "
        "training notebook saved; nothing is retrained.</p></div>" + header_kpis)

    with gr.Tab("Model comparison"):
        gr.HTML("<p class='section-note'>All three implementations solve the same optimisation problem, so their "
                "metrics, calibration and coefficients should match. The baseline gives every listing the same "
                "probability: the training leave rate.</p>")
        gr.Dataframe(metrics_table(), label="Test quarter results", interactive=False)
        with gr.Row(equal_height=True):
            gr.Plot(calibration_plot(), show_label=False, elem_classes="card")
            gr.Plot(prediction_by_outcome_plot(), show_label=False, elem_classes="card")
        gr.Plot(coefficient_plot(), show_label=False, elem_classes="card")

    with gr.Tab("Feature and target distributions"):
        gr.Plot(target_plot(), show_label=False, elem_classes="card")
        with gr.Row():
            feature_dd = gr.Dropdown(FEATURES, value="avail_30_share", label="Feature")
            period_radio = gr.Radio(list(PERIODS), value="Training: March → July", label="Period")
        feature_out = gr.Plot(feature_plot("avail_30_share", "Training: March → July"),
                              show_label=False, elem_classes="card")
        feature_dd.change(feature_plot, [feature_dd, period_radio], feature_out)
        period_radio.change(feature_plot, [feature_dd, period_radio], feature_out)

    with gr.Tab("Threshold and business cost"):
        gr.HTML(f"<p class='section-note'>Every host whose predicted probability is at or above the threshold gets a "
                f"call. A call costs {euro(CALL_COST)} and keeps {SAVE_RATE:.0%} of leaving hosts; a lost host costs "
                f"{euro(VALUE_LOST)} in commission. That makes a call worth it above a probability of "
                f"<b>{meta['break_even']:.2f}</b>.</p>")
        with gr.Row():
            model_dd = gr.Dropdown(MODEL_NAMES, value="sklearn", label="Model", scale=1)
            slider = gr.Slider(0.01, 0.99, value=BEST_T, step=0.01, scale=3,
                               label="Decision threshold: call hosts with a probability at or above this")
        start = threshold_view("sklearn", BEST_T)
        kpi_html = gr.HTML(start[0])
        with gr.Row(equal_height=True):
            cm_plot = gr.Plot(start[1], show_label=False, elem_classes="card")
            cost_plot = gr.Plot(start[2], show_label=False, elem_classes="card")
        model_dd.change(threshold_view, [model_dd, slider], [kpi_html, cm_plot, cost_plot])
        slider.change(threshold_view, [model_dd, slider], [kpi_html, cm_plot, cost_plot])

    with gr.Tab("Method and assumptions"):
        gr.HTML(f"""<div class='method'>
<h3>Data</h3>Inside Airbnb quarterly scrapes of Valencia (CC BY 4.0), limited to short stay listings
(minimum stay of 7 nights or less), because the March scrape only covers those.
<h3>Target</h3>A listing <b>left</b> if it is in one snapshot and missing from the next.
<h3>Split</h3>Train and validate on March → July ({len(train_features):,} listings), test once on
July → September ({len(test_features):,} listings). Preprocessing is fitted on training rows only.
<h3>Model</h3>Logistic regression with an L2 penalty (C = {meta['best_C']}). Both PyTorch versions minimise the
same objective with full batch gradient descent ({meta['epochs']:,} steps, learning rate {meta['best_lr']}),
starting from zero weights.
<h3>Business assumptions</h3>Call cost {euro(CALL_COST)}, save rate {SAVE_RATE:.0%}, value of a lost host
{euro(VALUE_LOST)}. The threshold is chosen on validation data by lowest total cost.
</div>""")


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=FORCE_LIGHT)
