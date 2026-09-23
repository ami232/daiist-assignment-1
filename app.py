"""
Assignment 1: Gradio dashboard for Resort Hotel cancellation prediction.

Loads the artifacts written by train.py and never retrains anything.

Business cost (framing 2: flagged bookings are asked for a deposit):
- missed cancellation (FN): the full stay revenue is lost (adr x nights)
- flagged guest who would have come (FP): 30% walk away -> 0.3 x stay revenue
- flagged cancellation (TP): the deposit covers the loss -> 0
- guest not flagged who comes (TN): 0
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from train import (ARTIFACTS, CATEGORICAL_FEATURES, GUEST_LABELS, LEAD_LABELS, TARGET,
                   WEEKEND_LABELS, LogisticModel, predict_manual, predict_module)

FP_LOSS_SHARE = 0.30  # share of flagged guests who walk away when asked for a deposit
MODELS = ["sklearn", "manual_torch", "torch_module"]
LABELS = {"sklearn": "scikit-learn", "manual_torch": "Manual PyTorch",
          "torch_module": "nn.Module + optim", "baseline": "Naive baseline"}
COLORS = {"sklearn": "#2a78d6", "manual_torch": "#e8883a", "torch_module": "#3aa66a",
          "baseline": "#8a8a8a"}
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


# ---------------------------------------------------------------- load artifacts

def load_models():
    prep = joblib.load(ARTIFACTS / "preprocessing.joblib")["preprocessor"]
    sk = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
    manual = torch.load(ARTIFACTS / "manual_torch.pt")
    state = torch.load(ARTIFACTS / "torch_module.pt")
    module = LogisticModel(state["linear.weight"].shape[1])
    module.load_state_dict(state)
    module.eval()
    return prep, sk, manual, module


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / f"{name}_set.csv", parse_dates=["booking_date", "arrival_date"])
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["stay_revenue"] = df["adr"].clip(lower=0) * nights
    return df


def predict_all(df: pd.DataFrame, prep, sk, manual, module, base_rate: float) -> dict:
    X = prep.transform(df[CATEGORICAL_FEATURES])
    return {
        "sklearn": sk.predict_proba(X)[:, 1],
        "manual_torch": predict_manual(manual["w"], manual["b"], X),
        "torch_module": predict_module(module, X),
        "baseline": np.full(len(df), base_rate),
    }


metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
prep, sk, manual, module = load_models()
splits = {name: load_split(name) for name in ["train", "val", "test"]}
probs = {name: predict_all(df, prep, sk, manual, module, metrics["base_rate"])
         for name, df in splits.items() if name in ("val", "test")}


# ---------------------------------------------------------------- business cost

def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    flag = p >= threshold
    return {"tp": flag & (y == 1), "fp": flag & (y == 0),
            "fn": ~flag & (y == 1), "tn": ~flag & (y == 0)}


def business_cost(df: pd.DataFrame, p: np.ndarray, threshold: float) -> float:
    c = confusion(df[TARGET].to_numpy(), p, threshold)
    rev = df["stay_revenue"].to_numpy()
    return rev[c["fn"]].sum() + FP_LOSS_SHARE * rev[c["fp"]].sum()


def cost_curve(split: str, model: str) -> np.ndarray:
    return np.array([business_cost(splits[split], probs[split][model], t) for t in THRESHOLDS])


# The threshold is chosen on validation, never on test.
best_threshold = {m: float(THRESHOLDS[cost_curve("val", m).argmin()]) for m in MODELS}
test_curves = {m: cost_curve("test", m) for m in MODELS}


# ---------------------------------------------------------------- tab 1: threshold

def threshold_view(model: str, threshold: float):
    df, p = splits["test"], probs["test"][model]
    y = df[TARGET].to_numpy()
    c = confusion(y, p, threshold)
    n = {k: int(v.sum()) for k, v in c.items()}

    z = [[n["tn"], n["fp"]], [n["fn"], n["tp"]]]
    text = [[f"TN<br>{n['tn']:,}<br>guest comes, not asked", f"FP<br>{n['fp']:,}<br>asked, would have come"],
            [f"FN<br>{n['fn']:,}<br>missed cancellation", f"TP<br>{n['tp']:,}<br>caught cancellation"]]
    cm = go.Figure(go.Heatmap(z=z, x=["Not flagged", "Flagged (ask deposit)"],
                              y=["Came (0)", "Cancelled (1)"], text=text, texttemplate="%{text}",
                              colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix on test, threshold {threshold:.2f}",
                     yaxis_autorange="reversed", height=380, margin=dict(t=50, b=20))

    rev = df["stay_revenue"].to_numpy()
    cost = business_cost(df, p, threshold)
    do_nothing = rev[y == 1].sum()                    # nobody flagged: every cancellation lost
    ask_all = FP_LOSS_SHARE * rev[y == 0].sum()       # everyone flagged
    precision = n["tp"] / max(n["tp"] + n["fp"], 1)
    recall = n["tp"] / max(n["tp"] + n["fn"], 1)
    summary = (
        f"### Cost at threshold {threshold:.2f}: €{cost:,.0f}\n"
        f"- Missed cancellations (FN): €{rev[c['fn']].sum():,.0f} of lost stays\n"
        f"- Guests lost by asking (FP): {FP_LOSS_SHARE:.0%} × €{rev[c['fp']].sum():,.0f} "
        f"= €{FP_LOSS_SHARE * rev[c['fp']].sum():,.0f}\n\n"
        f"| Policy | Cost on test |\n|---|---|\n"
        f"| Ask nobody for a deposit | €{do_nothing:,.0f} |\n"
        f"| Ask everybody | €{ask_all:,.0f} |\n"
        f"| **This model at {threshold:.2f}** | **€{cost:,.0f}** "
        f"(saves {1 - cost / do_nothing:.0%} vs asking nobody, "
        f"{1 - cost / ask_all:.0%} vs asking everybody) |\n\n"
        f"Flagged {int(n['tp'] + n['fp']):,} of {len(y):,} bookings · "
        f"precision {precision:.2f} · recall {recall:.2f}\n\n"
        f"Threshold that minimises cost on **validation**: **{best_threshold[model]:.2f}**"
    )

    curve = go.Figure()
    for m in MODELS:
        curve.add_trace(go.Scatter(x=THRESHOLDS, y=test_curves[m], name=LABELS[m],
                                   line=dict(color=COLORS[m], width=3 if m == model else 1.5)))
    curve.add_hline(y=do_nothing, line_dash="dot", annotation_text="ask nobody")
    curve.add_vline(x=threshold, line_dash="dash", line_color="black")
    curve.add_vline(x=best_threshold[model], line_dash="dot", line_color=COLORS[model],
                    annotation_text="val optimum")
    curve.update_layout(title="Business cost on test vs threshold", xaxis_title="Threshold",
                        yaxis_title="Cost (€)", height=380, margin=dict(t=50, b=20))
    return cm, summary, curve


# ---------------------------------------------------------------- tab 2: comparison

def calibration_plot(split: str):
    df = splits[split]
    y = df[TARGET].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(color="lightgray", dash="dash")))
    for m in MODELS + ["baseline"]:
        p = probs[split][m]
        bins = pd.qcut(p, 10, duplicates="drop") if m != "baseline" else np.zeros(len(p))
        g = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True)
        agg = g.agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(x=agg["pred"], y=agg["actual"], mode="lines+markers",
                                 name=LABELS[m], line=dict(color=COLORS[m]),
                                 marker=dict(size=10 if m == "baseline" else 7),
                                 customdata=agg["n"],
                                 hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>n=%{customdata}"))
    fig.update_layout(title=f"Predicted vs actual cancellation rate ({split}, deciles of prediction)",
                      xaxis_title="Mean predicted probability", yaxis_title="Actual cancellation rate",
                      height=450, xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig


def agreement_plot(split: str):
    p = probs[split]
    fig = go.Figure()
    for m in ["manual_torch", "torch_module"]:
        fig.add_trace(go.Scattergl(x=p["sklearn"], y=p[m], mode="markers", name=LABELS[m],
                                   marker=dict(color=COLORS[m], size=4, opacity=0.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x",
                             line=dict(color="lightgray", dash="dash")))
    fig.update_layout(title="PyTorch vs scikit-learn predicted probability, per booking",
                      xaxis_title="scikit-learn", yaxis_title="PyTorch", height=450)
    return fig


def by_arrival_plot(split: str):
    df = splits[split].assign(**{m: probs[split][m] for m in MODELS})
    g = df.groupby(df["arrival_date"].dt.to_period("M").dt.to_timestamp())
    agg = g[[TARGET] + MODELS].mean()
    fig = go.Figure(go.Scatter(x=agg.index, y=agg[TARGET], name="Actual", mode="lines+markers",
                               line=dict(color="black", width=3)))
    for m in MODELS:
        fig.add_trace(go.Scatter(x=agg.index, y=agg[m], name=LABELS[m], line=dict(color=COLORS[m])))
    fig.update_layout(title=f"Actual vs mean predicted cancellation rate by arrival month ({split})",
                      yaxis_title="Cancellation rate", height=400)
    return fig


def comparison_view(split: str):
    return calibration_plot(split), agreement_plot(split), by_arrival_plot(split)


def metrics_table() -> pd.DataFrame:
    t = pd.DataFrame(metrics["test"]).T.loc[["baseline"] + MODELS]
    t.index = [LABELS[i] for i in t.index]
    return t.round(4).reset_index(names="Model (test set)")


def loss_plot():
    h = pd.read_csv(ARTIFACTS / "loss_history.csv")
    fig = go.Figure()
    for m in ["manual_torch", "torch_module"]:
        fig.add_trace(go.Scatter(x=h["epoch"], y=h[m], name=LABELS[m], line=dict(color=COLORS[m])))
    fig.update_layout(title="Training loss (log loss + L2 penalty)", xaxis_title="Epoch",
                      yaxis_title="Loss", xaxis_type="log", height=350)
    return fig


# ---------------------------------------------------------------- tab 3: distributions

DIST_FEATURES = CATEGORICAL_FEATURES + ["lead_time", "adr", "stay_revenue"]
ORDERED = {"lead_bin": LEAD_LABELS, "guests_bin": GUEST_LABELS, "weekend_bin": WEEKEND_LABELS,
           "season": ["winter", "spring", "summer", "autumn"]}


def distribution_view(feature: str, split: str):
    df = splits[split]
    fig = go.Figure()
    if feature in CATEGORICAL_FEATURES:
        g = df.groupby(feature)[TARGET].agg(["size", "mean"])
        if feature in ORDERED:  # bins keep their natural order, others sort by size
            g = g.reindex([c for c in ORDERED[feature] if c in g.index])
        else:
            g = g.sort_values("size", ascending=False)
        fig.add_trace(go.Bar(x=g.index.astype(str), y=g["size"], name="Bookings", marker_color="#9ebfe6"))
        fig.add_trace(go.Scatter(x=g.index.astype(str), y=g["mean"], name="Cancellation rate",
                                 yaxis="y2", mode="lines+markers", line=dict(color="#d6452a")))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1],
                                      title="Cancellation rate"))
    else:
        for label, value, color in [("Came", 0, "#2a78d6"), ("Cancelled", 1, "#d6452a")]:
            fig.add_trace(go.Histogram(x=df.loc[df[TARGET] == value, feature], name=label,
                                       opacity=0.6, marker_color=color, nbinsx=60))
        fig.update_layout(barmode="overlay")
    fig.update_layout(title=f"{feature} ({split})", yaxis_title="Bookings", height=450)
    return fig


def target_plot():
    rows = [(s, splits[s][TARGET].mean(), len(splits[s])) for s in ["train", "val", "test"]]
    fig = go.Figure(go.Bar(x=[f"{s}<br>n={n:,}" for s, _, n in rows], y=[r for _, r, _ in rows],
                           text=[f"{r:.1%}" for _, r, _ in rows], textposition="outside",
                           marker_color="#d6452a"))
    fig.update_layout(title="Target: cancellation rate by split", yaxis_range=[0, 0.5],
                      yaxis_title="Share cancelled", height=350)
    return fig


# ---------------------------------------------------------------- layout

with gr.Blocks(title="Resort Hotel cancellations") as demo:
    gr.Markdown(
        "# Resort Hotel: which bookings should be asked for a deposit?\n"
        "Logistic regression trained three ways on bookings made before Apr 2016, "
        "tuned on Apr–Sep 2016, evaluated on Oct 2016–Mar 2017. Nothing is retrained here."
    )
    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            model_dd = gr.Dropdown([(LABELS[m], m) for m in MODELS], value="sklearn", label="Model")
            thr = gr.Slider(0.01, 0.99, value=best_threshold["sklearn"], step=0.01,
                            label="Decision threshold (flag if P(cancel) ≥ threshold)")
        with gr.Row():
            cm_plot = gr.Plot()
            summary_md = gr.Markdown()
        curve_plot = gr.Plot()
        gr.Markdown(f"Costs: missed cancellation = full stay revenue (adr × nights); "
                    f"flagging a guest who would have come = {FP_LOSS_SHARE:.0%} of their stay revenue.")
        for ctrl in (model_dd, thr):
            ctrl.change(threshold_view, [model_dd, thr], [cm_plot, summary_md, curve_plot])
        demo.load(threshold_view, [model_dd, thr], [cm_plot, summary_md, curve_plot])

    with gr.Tab("Compare the three models"):
        gr.Dataframe(metrics_table(), label="Threshold-free metrics on test")
        split_dd = gr.Radio(["test", "val"], value="test", label="Split")
        with gr.Row():
            calib = gr.Plot()
            agree = gr.Plot()
        month = gr.Plot()
        gr.Plot(loss_plot())
        split_dd.change(comparison_view, split_dd, [calib, agree, month])
        demo.load(comparison_view, split_dd, [calib, agree, month])

    with gr.Tab("Distributions"):
        gr.Plot(target_plot())
        with gr.Row():
            feat_dd = gr.Dropdown(DIST_FEATURES, value="lead_bin", label="Feature")
            dist_split = gr.Radio(["train", "val", "test"], value="train", label="Split")
        dist_plot = gr.Plot()
        for ctrl in (feat_dd, dist_split):
            ctrl.change(distribution_view, [feat_dd, dist_split], dist_plot)
        demo.load(distribution_view, [feat_dd, dist_split], dist_plot)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
