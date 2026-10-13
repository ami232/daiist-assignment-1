"""
Assignment 1 — Gradio dashboard (run via `uv run python main.py app`).

Loads the preprocessor and the three trained models from artifacts/ (written
by train.py) and scores the data with them. Nothing is trained here.
"""

import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_curve

from pipeline import (ARTIFACTS, COST_PER_CALL, VALUE_PER_SUBSCRIPTION,
                      TorchLogReg, call_mask, chronological_split, engineer,
                      load_raw, predict_manual, predict_module)

# --------------------------------------------------------------------------
# Load artifacts (fail with a clear message if training hasn't run)
# --------------------------------------------------------------------------
if not (ARTIFACTS / "metrics.json").exists():
    raise SystemExit("No artifacts found: run `uv run python main.py train` first.")

metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
manual_params = torch.load(ARTIFACTS / "torch_manual_params.pt")
module_model = TorchLogReg(metrics["n_features"])
module_model.load_state_dict(torch.load(ARTIFACTS / "torch_module_state.pt"))
coefs = pd.read_csv(ARTIFACTS / "coefficients.csv")
ablation = pd.read_csv(ARTIFACTS / "ablation.csv")

raw = load_raw()
feats = engineer(raw)
train_df, val_df, test_df = chronological_split(raw)
SPLITS = {"train": train_df, "validation": val_df, "test": test_df}

MODELS = {
    "sklearn": "scikit-learn",
    "torch_manual": "PyTorch (manual loop)",
    "torch_module": "PyTorch (nn.Module + optim)",
}
X = {k: preprocessor.transform(feats.loc[v.index]).astype(np.float32)
     for k, v in SPLITS.items()}
PROBA = {
    "sklearn": {k: sk_model.predict_proba(x)[:, 1] for k, x in X.items()},
    "torch_manual": {k: predict_manual(manual_params, x) for k, x in X.items()},
    "torch_module": {k: predict_module(module_model, x) for k, x in X.items()},
}
Y = {k: v["y"].to_numpy() for k, v in SPLITS.items()}

# --------------------------------------------------------------------------
# Styling: categorical slots 1-3 for the three models (with distinct dash
# patterns, since their curves nearly coincide), gray for baselines.
# --------------------------------------------------------------------------
COLORS = {"sklearn": "#2a78d6", "torch_manual": "#eb6834", "torch_module": "#1baf7a"}
DASHES = {"sklearn": "solid", "torch_manual": "dash", "torch_module": "dot"}
SYMBOLS = {"sklearn": "circle", "torch_manual": "square", "torch_module": "diamond"}
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
NEUTRAL = "#8a8985"
SEQUENTIAL = [[0, "#eef4fc"], [1, "#1c5aa6"]]


def style(fig, title, x=None, y=None, height=420):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        template="plotly_white", height=height,
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(size=12, color="#0b0b0b"),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(title=x, gridcolor="#ececea", zeroline=False)
    fig.update_yaxes(title=y, gridcolor="#ececea", zeroline=False)
    return fig


def euros(v):
    return f"€{v:,.0f}"


# --------------------------------------------------------------------------
# Tab 1: overview tables
# --------------------------------------------------------------------------
def results_table():
    rows = []
    for key, label in MODELS.items():
        t = metrics["results"][key]["test"]
        rows.append({"Model": label, "ROC-AUC": t["roc_auc"], "PR-AUC": t["pr_auc"],
                     "Log loss": t["log_loss"]})
    cp = metrics["baselines"]["constant_prior"]
    rows.append({"Model": "Naive: constant training base rate",
                 "ROC-AUC": cp["roc_auc"], "PR-AUC": cp["pr_auc"], "Log loss": cp["log_loss"]})
    return pd.DataFrame(rows).round(4)


def capacity_table():
    rows = []
    for c in ["0.1", "0.25", "0.5"]:
        row = {"Capacity (share of list called)": f"{float(c):.0%}"}
        for key, label in MODELS.items():
            row[label] = euros(metrics["results"][key]["test_capacity"][c]["profit"])
        row["Random calls (no model)"] = euros(metrics["baselines"]["random_at_capacity"][c]["profit"])
        rows.append(row)
    row = {"Capacity (share of list called)": "100% (call everyone)"}
    for label in MODELS.values():
        row[label] = euros(metrics["baselines"]["call_everyone"]["profit"])
    row["Random calls (no model)"] = euros(metrics["baselines"]["call_everyone"]["profit"])
    rows.append(row)
    return pd.DataFrame(rows)


def agreement_table():
    a = metrics["agreement"]
    return pd.DataFrame([
        {"Compared with scikit-learn": MODELS[k],
         "Max |Δ coefficient|": v["max_abs_coef_diff"],
         "|Δ intercept|": abs(v["intercept_diff"]),
         "Max |Δ probability| (test)": v["max_abs_proba_diff_test"],
         "Mean |Δ probability| (test)": v["mean_abs_proba_diff_test"]}
        for k, v in a.items()
    ]).round(5)


def splits_table():
    return pd.DataFrame([
        {"Split": k, "Rows": s["rows"], "From": s["from"], "To": s["to"],
         "Subscription rate": f"{s['positive_rate']:.1%}"}
        for k, s in metrics["splits"].items()
    ])


def diagnostics_md():
    fa = metrics["first_attempt_row_split_diagnostic"]
    rs = metrics["random_split_diagnostic"]
    sk = metrics["results"]["sklearn"]["test"]
    h = metrics["hyperparameters"]
    return (
        f"**Final calendar split (scikit-learn):** test ROC-AUC {sk['roc_auc']:.3f}.  \n"
        f"**First attempt, 70/15/15 row split** (test {fa['test_from']} → {fa['test_to']}): "
        f"ROC-AUC {fa['roc_auc']:.3f}; mean predicted probability {fa['mean_predicted']:.1%} "
        f"vs. actual rate {fa['actual_rate']:.1%}.  \n"
        f"**Random stratified split (leaky, diagnostic only):** ROC-AUC {rs['roc_auc']:.3f}.  \n\n"
        f"Hyperparameters: C = {h['C']:g} (all three models), manual-loop lr = {h['manual_lr']} "
        f"({h['manual_epochs']} full-batch epochs), nn.Module lr = {h['module_lr']} "
        f"(Adam, {h['module_epochs']} epochs, batch {h['batch_size']}, cosine schedule)."
    )


def loss_curve_plot():
    fig = go.Figure()
    for key in ["torch_manual", "torch_module"]:
        hist = np.array(metrics["loss_history"][key])
        epochs = hist[:, 0] / hist[-1, 0]  # fraction of training, for comparability
        fig.add_trace(go.Scatter(
            x=epochs, y=hist[:, 1], name=MODELS[key], mode="lines",
            line=dict(color=COLORS[key], width=2, dash=DASHES[key]),
            hovertemplate="%{x:.0%} of training<br>objective %{y:.5f}<extra></extra>"))
    fig.update_yaxes(type="log")
    return style(fig, "Training objective (log loss + L2) during training",
                 "Fraction of training completed", "Objective (log scale)", height=360)


# --------------------------------------------------------------------------
# Tab 2: model comparison
# --------------------------------------------------------------------------
def calibration_plot(split):
    """Prediction vs. actual for a classifier: bin clients by predicted
    probability and compare each bin's mean prediction to its actual rate."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color=NEUTRAL, width=1, dash="dash"),
                             hoverinfo="skip"))
    y = Y[split]
    for key, label in MODELS.items():
        p = PROBA[key][split]
        bins = pd.qcut(p, q=10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True)
        agg = g.agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(
            x=agg["pred"], y=agg["actual"], name=label, mode="lines+markers",
            line=dict(color=COLORS[key], width=2, dash=DASHES[key]),
            marker=dict(size=9, symbol=SYMBOLS[key], line=dict(width=2, color="white")),
            customdata=agg["n"],
            hovertemplate=(f"<b>{label}</b><br>mean predicted %{{x:.1%}}<br>"
                           "actual rate %{y:.1%}<br>%{customdata} clients<extra></extra>")))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return style(fig, f"Predicted vs. actual subscription rate, by predicted-probability decile ({split})",
                 "Mean predicted probability", "Actual subscription rate", height=480)


def roc_plot(split):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random ranking",
                             line=dict(color=NEUTRAL, width=1, dash="dash"),
                             hoverinfo="skip"))
    for key, label in MODELS.items():
        fpr, tpr, _ = roc_curve(Y[split], PROBA[key][split])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=label, mode="lines",
            line=dict(color=COLORS[key], width=2, dash=DASHES[key]),
            hovertemplate=f"<b>{label}</b><br>FPR %{{x:.2f}}<br>TPR %{{y:.2f}}<extra></extra>"))
    return style(fig, f"ROC curves ({split})", "False positive rate", "True positive rate",
                 height=480)


def agreement_scatter(split):
    fig = go.Figure()
    base = PROBA["sklearn"][split]
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Identical predictions",
                             line=dict(color=NEUTRAL, width=1, dash="dash"),
                             hoverinfo="skip"))
    for key in ["torch_manual", "torch_module"]:
        fig.add_trace(go.Scattergl(
            x=base, y=PROBA[key][split], mode="markers", name=MODELS[key],
            marker=dict(size=8, symbol=SYMBOLS[key], color=COLORS[key], opacity=0.6,
                        line=dict(width=1, color="white")),
            hovertemplate=(f"scikit-learn %{{x:.3f}}<br>{MODELS[key]} %{{y:.3f}}"
                           "<extra></extra>")))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return style(fig, f"Per-client probabilities: PyTorch vs. scikit-learn ({split})",
                 "scikit-learn predicted probability", "PyTorch predicted probability",
                 height=480)


def coef_plot():
    order = coefs.reindex(coefs["sklearn"].abs().sort_values().index)
    fig = go.Figure()
    for key, label in MODELS.items():
        fig.add_trace(go.Bar(
            y=order["feature"], x=order[key], name=label, orientation="h",
            marker=dict(color=COLORS[key], line=dict(width=1, color="white")),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.4f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    return style(fig, "Coefficients (log-odds per standard deviation of each column)",
                 "Coefficient", None, height=max(500, 22 * len(order)))


def compare_views(split):
    return calibration_plot(split), roc_plot(split), agreement_scatter(split)


# --------------------------------------------------------------------------
# Tab 3: data
# --------------------------------------------------------------------------
NUMERIC_VIEW = ["age", "campaign", "pdays", "previous", "euribor3m",
                "cons.conf.idx", "cons.price.idx", "emp.var.rate", "nr.employed",
                "duration (excluded: post-call)"]
CATEGORICAL_VIEW = ["job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "day_of_week", "poutcome"]
ENGINEERED_VIEW = ["age_band", "log_prior_contacts", "days_since_contact_capped",
                   "previous_capped", "default_unknown", "recent_contact"]


def feature_plot(feature, split):
    df = SPLITS[split]
    col = feature.split(" ")[0]
    source = feats.loc[df.index] if col in ENGINEERED_VIEW else df
    values, y = source[col], df["y"]
    is_numeric = pd.api.types.is_numeric_dtype(values) and values.nunique() > 12

    fig = go.Figure()
    if is_numeric:
        # Shared bin edges so the two classes' bars line up.
        edges = np.histogram_bin_edges(values, bins=40)
        for cls, name in [(0, "Did not subscribe"), (1, "Subscribed")]:
            fig.add_trace(go.Histogram(
                x=values[y == cls], name=name, histnorm="percent",
                xbins=dict(start=edges[0], end=edges[-1], size=edges[1] - edges[0]),
                marker=dict(color=CLASS_COLORS[cls], line=dict(width=1, color="white")),
                opacity=0.65,
                hovertemplate=f"{name}<br>{col} %{{x}}<br>%{{y:.1f}}% of class<extra></extra>"))
        fig.update_layout(barmode="overlay")
        return style(fig, f"Distribution of {col} by outcome ({split}, each class sums to 100%)",
                     col, "% of clients in class")

    # Categorical: share of clients per category, plus subscription rate as text.
    g = pd.DataFrame({"v": values.astype(str), "y": y}).groupby("v")["y"]
    agg = pd.DataFrame({"n": g.size(), "rate": g.mean()}).sort_values("n")
    fig.add_trace(go.Bar(
        y=agg.index, x=agg["rate"], orientation="h", name="Subscription rate",
        marker=dict(color=CLASS_COLORS[1], line=dict(width=1, color="white")),
        text=[f"{r:.1%}  (n={n:,})" for r, n in zip(agg["rate"], agg["n"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>subscription rate %{x:.1%}<extra></extra>"))
    fig.add_vline(x=y.mean(), line=dict(color=NEUTRAL, dash="dash", width=1),
                  annotation_text=f"overall {y.mean():.1%}", annotation_position="top")
    fig.update_xaxes(tickformat=".0%", range=[0, min(1, agg["rate"].max() * 1.35 + 0.02)])
    return style(fig, f"Subscription rate by {col} ({split})", "Subscription rate", None,
                 height=max(380, 34 * len(agg) + 120))


def timeline_plot():
    g = raw.groupby("period").agg(calls=("y", "size"), rate=("y", "mean")).reset_index()
    split_of = {p: s for s, df in SPLITS.items() for p in df["period"].unique()}
    g["split"] = g["period"].map(split_of)
    fig = go.Figure()
    for s, color in [("train", "#2a78d6"), ("validation", "#eb6834"), ("test", "#1baf7a")]:
        d = g[g["split"] == s]
        fig.add_trace(go.Bar(
            x=d["period"], y=d["rate"], name=s, marker=dict(color=color),
            customdata=d["calls"],
            hovertemplate="%{x}<br>subscription rate %{y:.1%}<br>%{customdata:,} calls<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(type="category")
    return style(fig, "Monthly subscription rate, coloured by split (hover for call volume)",
                 "Month", "Subscription rate", height=380)


def volume_plot():
    g = raw.groupby("period").size().reset_index(name="calls")
    fig = go.Figure(go.Bar(x=g["period"], y=g["calls"], marker=dict(color=NEUTRAL),
                           name="Calls",
                           hovertemplate="%{x}<br>%{y:,} calls<extra></extra>"))
    fig.update_xaxes(type="category")
    return style(fig, "Monthly call volume", "Month", "Calls", height=320)


# --------------------------------------------------------------------------
# Tab 4: decision / threshold
# --------------------------------------------------------------------------
def decision_views(model_key, split, threshold, capacity_pct, cost, value):
    y, p = Y[split], PROBA[model_key][split]
    capacity = capacity_pct / 100
    call = call_mask(p, threshold, capacity)
    tp = int((call & (y == 1)).sum())
    fp = int((call & (y == 0)).sum())
    fn = int((~call & (y == 1)).sum())
    tn = int((~call & (y == 0)).sum())
    profit = tp * value - (tp + fp) * cost
    everyone = int(y.sum()) * value - len(y) * cost
    n_random = int(call.sum())
    random_profit = n_random * (y.mean() * value - cost)

    cm = np.array([[tn, fp], [fn, tp]])
    labels = [["True negative<br>(not called, wouldn't subscribe)",
               "False positive<br>(called, didn't subscribe)"],
              ["False negative<br>(not called, would have subscribed)",
               "True positive<br>(called, subscribed)"]]
    xs, ys = ["Not called", "Called"], ["Didn't subscribe", "Subscribed"]
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=xs, y=ys, colorscale=SEQUENTIAL, showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"))
    fig_cm.update_yaxes(autorange="reversed")
    style(fig_cm, f"Confusion matrix: {MODELS[model_key]}, {split}", "Decision", "Actual outcome",
          height=420)
    for i in range(2):
        for j in range(2):
            # Light ink on the dark end of the ramp, dark ink on the light end.
            dark_cell = cm[i, j] > 0.55 * max(cm.max(), 1)
            fig_cm.add_annotation(
                x=xs[j], y=ys[i], showarrow=False,
                text=f"{labels[i][j]}<br><b>{cm[i, j]:,}</b>",
                font=dict(color="#ffffff" if dark_cell else "#0b0b0b", size=12))

    # Profit vs. threshold at the current capacity and economics.
    ts = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    curve = []
    for t in ts:
        c = call_mask(p, t, capacity)
        curve.append(int((c & (y == 1)).sum()) * value - int(c.sum()) * cost)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=ts, y=curve, mode="lines", name=MODELS[model_key],
                               line=dict(color=COLORS[model_key], width=2),
                               hovertemplate="threshold %{x:.2f}<br>profit €%{y:,.0f}<extra></extra>"))
    fig_p.add_hline(y=everyone, line=dict(color=NEUTRAL, dash="dash", width=1),
                    annotation_text="call everyone (no capacity limit)",
                    annotation_position="bottom right")
    fig_p.add_trace(go.Scatter(x=[threshold], y=[profit], mode="markers", name="Current setting",
                               marker=dict(size=12, color=COLORS[model_key],
                                           line=dict(width=2, color="white")),
                               hovertemplate="current: €%{y:,.0f}<extra></extra>"))
    fig_p.add_vline(x=cost / value, line=dict(color=NEUTRAL, dash="dot", width=1),
                    annotation_text="break-even (cost / value)", annotation_position="top right")
    style(fig_p, f"Campaign profit vs. threshold (capacity {capacity_pct:.0f}%)",
          "Decision threshold (call if P(subscribe) ≥ threshold)", "Profit (€)", height=420)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    kpis = (
        f"### Profit: {euros(profit)}\n"
        f"Calls made: **{tp + fp:,}** of {len(y):,} clients "
        f"({(tp + fp) / len(y):.0%}) · subscriptions won: **{tp:,}** · "
        f"precision **{precision:.1%}** · recall **{recall:.1%}**\n\n"
        f"| Policy | Profit |\n|---|---|\n"
        f"| This model at this threshold & capacity | **{euros(profit)}** |\n"
        f"| Same number of calls, chosen at random | {euros(random_profit)} |\n"
        f"| Call everyone (ignores capacity) | {euros(everyone)} |\n"
        f"| Call no one | €0 |\n\n"
        f"Profit = {euros(value)} × subscriptions − {euros(cost)} × calls. "
        f"Break-even probability = cost / value = **{cost / value:.1%}**."
    )
    return fig_cm, fig_p, kpis


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
with gr.Blocks(title="Term-deposit call targeting") as demo:
    gr.Markdown(
        "# Term-deposit campaign: whom should the call centre phone?\n"
        "Logistic regression trained three ways (scikit-learn, a manual PyTorch loop, "
        "and `nn.Module` + `torch.optim`) on the UCI Bank Marketing data. "
        "Models are loaded from `artifacts/`; nothing is retrained here."
    )
    with gr.Tabs():
        with gr.Tab("Overview"):
            gr.Markdown("### Test-set results (Jul–Nov 2010)")
            gr.Dataframe(results_table(), interactive=False)
            gr.Markdown("### Profit with limited call capacity (test set, €5/call, €50/subscription)")
            gr.Dataframe(capacity_table(), interactive=False)
            gr.Markdown("### Do the three implementations agree?")
            gr.Dataframe(agreement_table(), interactive=False)
            gr.Plot(loss_curve_plot())
            gr.Markdown("### Splits and diagnostics")
            gr.Dataframe(splits_table(), interactive=False)
            gr.Markdown(diagnostics_md())
            gr.Markdown("### Feature-group ablation (validation set, scikit-learn, C = 1)")
            gr.Dataframe(ablation.drop(columns="val_profit_at_best_t").round(4),
                         interactive=False)

        with gr.Tab("Model comparison"):
            split_cmp = gr.Radio(["test", "validation", "train"], value="test", label="Data split")
            cal = gr.Plot()
            with gr.Row():
                roc = gr.Plot()
                agree = gr.Plot()
            gr.Plot(coef_plot())
            split_cmp.change(compare_views, split_cmp, [cal, roc, agree])
            demo.load(compare_views, split_cmp, [cal, roc, agree])

        with gr.Tab("Data & distributions"):
            gr.Plot(timeline_plot())
            gr.Plot(volume_plot())
            with gr.Row():
                feature = gr.Dropdown(
                    NUMERIC_VIEW + CATEGORICAL_VIEW + ENGINEERED_VIEW, value="poutcome",
                    label="Feature (raw columns, then engineered ones)")
                split_feat = gr.Radio(["train", "validation", "test"], value="train",
                                      label="Data split")
            feat_plot = gr.Plot()
            for ctrl in [feature, split_feat]:
                ctrl.change(feature_plot, [feature, split_feat], feat_plot)
            demo.load(feature_plot, [feature, split_feat], feat_plot)

        with gr.Tab("Threshold & business cost"):
            gr.Markdown(
                "Call a client if their predicted probability is at least the threshold **and** "
                "they rank inside the call centre's capacity (best scores first). "
                "The default threshold is each model's validation-optimal one."
            )
            with gr.Row():
                model_dd = gr.Dropdown([(v, k) for k, v in MODELS.items()], value="sklearn",
                                       label="Model")
                split_dec = gr.Radio(["test", "validation"], value="test", label="Data split")
            with gr.Row():
                threshold = gr.Slider(0, 1, step=0.01,
                                      value=metrics["results"]["sklearn"]["val"]["threshold"],
                                      label="Decision threshold")
                capacity = gr.Slider(5, 100, step=5, value=25,
                                     label="Call capacity (% of client list)")
            with gr.Row():
                cost = gr.Number(COST_PER_CALL, label="Cost per call (€)", minimum=0)
                value = gr.Number(VALUE_PER_SUBSCRIPTION, label="Value per subscription (€)",
                                  minimum=0)
            kpi = gr.Markdown()
            with gr.Row():
                cm_plot = gr.Plot()
                profit_plot = gr.Plot()
            inputs = [model_dd, split_dec, threshold, capacity, cost, value]
            outputs = [cm_plot, profit_plot, kpi]
            for ctrl in inputs:
                ctrl.change(decision_views, inputs, outputs)
            demo.load(decision_views, inputs, outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860,
                inbrowser=not os.environ.get("CI"))
