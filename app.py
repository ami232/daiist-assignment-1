"""
Assignment 1: Gradio dashboard.

Loads ONLY what train.ipynb saved in artifacts/ (models, test data, results).
Nothing is retrained here: models are loaded and used for inference.
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

ART = Path("artifacts")
FEATURE_SETS = ["full", "deployable"]
METHODS = ["sklearn", "manual", "standard"]
COLORS = {"sklearn": "#1f77b4", "manual": "#ff7f0e", "standard": "#2ca02c"}


# Same architecture as in train.ipynb, needed to load the saved state_dict
class LogReg(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = torch.nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# ---------- load artifacts (no training) ----------
meta = json.loads((ART / "meta.json").read_text())
C_FP, C_FN = meta["C_FP"], meta["C_FN"]
results = pd.read_csv(ART / "results.csv")
y_test = np.load(ART / "y_test.npy")
train_raw = pd.read_csv(ART / "train_raw.csv")
test_raw = pd.read_csv(ART / "test_raw.csv")
val_raw = pd.read_csv(ART / "val_raw.csv")

probs = {}  # (feature_set, method) -> predicted probabilities on the test set
for fs in FEATURE_SETS:
    X = np.load(ART / f"X_test_{fs}.npy")
    Xt = torch.tensor(X)

    sk = joblib.load(ART / f"sklearn_{fs}.joblib")
    probs[(fs, "sklearn")] = sk.predict_proba(X)[:, 1]

    man = torch.load(ART / f"manual_{fs}.pt")
    with torch.no_grad():
        probs[(fs, "manual")] = torch.sigmoid(Xt @ man["w"] + man["b"]).numpy()

    std = LogReg(X.shape[1])
    std.load_state_dict(torch.load(ART / f"standard_{fs}.pt"))
    std.eval()
    with torch.no_grad():
        probs[(fs, "standard")] = torch.sigmoid(std(Xt)).numpy()


def tuned_threshold(fs, method):
    row = results[(results.features == fs) & (results.method == method)]
    return float(row.threshold.iloc[0])


def confusion(p, t):
    pred = p >= t
    tp = int(np.sum(pred & (y_test == 1)))
    fp = int(np.sum(pred & (y_test == 0)))
    fn = int(np.sum(~pred & (y_test == 1)))
    tn = int(np.sum(~pred & (y_test == 0)))
    return tp, fp, fn, tn


def cost(p, t):
    tp, fp, fn, tn = confusion(p, t)
    return C_FP * fp + C_FN * fn


# ---------- Tab 1: model comparison ----------
def calibration_plot(fs):
    """Predicted probability vs actual purchase rate, binned, for the 3 methods."""
    fig = go.Figure()
    bins = np.linspace(0, 1, 11)
    for m in METHODS:
        p = probs[(fs, m)]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        df = pd.DataFrame({"bin": idx, "p": p, "y": y_test}).groupby("bin").agg(
            pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(x=df.pred, y=df.actual, mode="lines+markers", name=m,
                                 line=dict(color=COLORS[m]), text=df.n,
                                 hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>n=%{text}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color="grey", dash="dash")))
    fig.update_layout(title=f"Predicted vs actual purchase rate (test set, {fs} features)",
                      xaxis_title="Mean predicted probability", yaxis_title="Actual purchase rate",
                      height=450)
    return fig


def agreement_plot(fs):
    """Each test session: sklearn probability vs the two PyTorch probabilities."""
    fig = go.Figure()
    base = probs[(fs, "sklearn")]
    for m in ["manual", "standard"]:
        fig.add_trace(go.Scattergl(x=base, y=probs[(fs, m)], mode="markers", name=m,
                                   marker=dict(size=4, opacity=0.5, color=COLORS[m])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical",
                             line=dict(color="grey", dash="dash")))
    diff = max(np.abs(base - probs[(fs, m)]).max() for m in ["manual", "standard"])
    fig.update_layout(title=f"Do the three implementations agree? max |Δp| = {diff:.4f}",
                      xaxis_title="sklearn probability", yaxis_title="PyTorch probability",
                      height=450)
    return fig


def loss_plot(fs):
    fig = go.Figure()
    for m in ["manual", "standard"]:
        h = meta["histories"][fs][m]
        fig.add_trace(go.Scatter(y=h, mode="lines", name=m, line=dict(color=COLORS[m])))
    fig.update_layout(title="PyTorch training loss (BCE, train set)", xaxis_title="Epoch",
                      yaxis_title="Loss", yaxis_type="log", height=350)
    return fig


def comparison_tab(fs):
    return calibration_plot(fs), agreement_plot(fs), loss_plot(fs)


# ---------- Tab 2: distributions ----------
DIST_FEATURES = ["PageValues", "ProductRelated", "ProductRelated_Duration", "BounceRates",
                 "ExitRates", "product_time_share", "sec_per_product_page", "total_pages",
                 "SpecialDay", "VisitorType"]


def distribution_plot(feature, log_x):
    d = train_raw.copy()
    d["Purchase"] = d["Revenue"].map({True: "Bought", False: "Did not buy"})
    col = feature
    if log_x and pd.api.types.is_numeric_dtype(d[feature]):
        col = f"log1p({feature})"
        d[col] = np.log1p(d[feature])
    fig = px.histogram(d, x=col, color="Purchase", barmode="overlay", histnorm="percent",
                       nbins=50, opacity=0.6,
                       title=f"{feature} by outcome (train set, normalised within each class)")
    fig.update_layout(height=450)
    return fig


def target_plot():
    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    full = pd.concat([train_raw, val_raw, test_raw])
    split_of = {m: "train" for m in month_order[:7]} | {"Oct": "val", "Nov": "val", "Dec": "test"}
    g = full.groupby("Month")["Revenue"].mean().reindex(month_order).reset_index()
    g["split"] = g.Month.map(split_of)
    fig = px.bar(g, x="Month", y="Revenue", color="split",
                 title="Purchase rate by month (target distribution across the time split)",
                 labels={"Revenue": "Purchase rate"})
    fig.update_layout(height=400)
    return fig


# ---------- Tab 3: threshold & business cost ----------
def threshold_tab(fs, method, t):
    p = probs[(fs, method)]
    tp, fp, fn, tn = confusion(p, t)
    z = [[tn, fp], [fn, tp]]
    cm = go.Figure(go.Heatmap(z=z, x=["No pop-up", "Pop-up"], y=["Did not buy", "Bought"],
                              text=z, texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix at threshold {t:.2f} (test set)",
                     xaxis_title="Predicted", yaxis_title="Actual", height=380)

    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    curve = go.Figure(go.Scatter(x=grid, y=[cost(p, g) for g in grid], mode="lines", name="model"))
    curve.add_hline(y=C_FN * y_test.sum(), line_dash="dot", annotation_text="show to nobody")
    curve.add_hline(y=C_FP * (y_test == 0).sum(), line_dash="dot", annotation_text="show to everyone")
    curve.add_vline(x=t, line_color="red", annotation_text="current")
    curve.add_vline(x=tuned_threshold(fs, method), line_dash="dash", line_color="green",
                    annotation_text="tuned on validation", annotation_position="bottom right")
    curve.update_layout(title="Total business cost vs threshold (test set)",
                        xaxis_title="Threshold", yaxis_title="Cost (€)", height=380)

    total = cost(p, t)
    best_naive = min(C_FN * y_test.sum(), C_FP * (y_test == 0).sum())
    summary = (f"### Business cost at threshold {t:.2f}: **€{total:,.0f}**\n"
               f"- False positives: {fp} × €{C_FP:g} = €{C_FP * fp:,.0f}  |  "
               f"False negatives: {fn} × €{C_FN:g} = €{C_FN * fn:,.0f}\n"
               f"- Cheapest naive policy on this test set: €{best_naive:,.0f}  →  "
               f"saving **€{best_naive - total:,.0f}**\n"
               f"- Precision {tp / max(tp + fp, 1):.2f} · Recall {tp / max(tp + fn, 1):.2f} · "
               f"Pop-ups shown: {tp + fp} of {len(y_test)} sessions")
    return cm, curve, summary


def reset_threshold(fs, method):
    return tuned_threshold(fs, method)


# ---------- layout ----------
with gr.Blocks(title="Online Shoppers: purchase prediction") as demo:
    gr.Markdown("# Online Shoppers: who gets the discount pop-up?\n"
                "Logistic regression trained three ways (scikit-learn, manual PyTorch, "
                "`nn.Module` + `optim`). Test set = December sessions. "
                "Models are loaded from `artifacts/`; nothing is retrained.")

    with gr.Tab("Model comparison"):
        res_show = results.copy()
        res_show[["test_auc", "test_logloss"]] = res_show[["test_auc", "test_logloss"]].round(4)
        gr.Dataframe(res_show, label="Test-set results (threshold tuned on Oct–Nov validation)")
        fs1 = gr.Radio(FEATURE_SETS, value="full", label="Feature set (with / without PageValues)")
        with gr.Row():
            calib = gr.Plot()
            agree = gr.Plot()
        loss = gr.Plot()
        fs1.change(comparison_tab, fs1, [calib, agree, loss])
        demo.load(comparison_tab, fs1, [calib, agree, loss])

    with gr.Tab("Distributions"):
        with gr.Row():
            feat = gr.Dropdown(DIST_FEATURES, value="PageValues", label="Feature")
            logx = gr.Checkbox(value=True, label="log1p scale (numeric features)")
        dist = gr.Plot()
        tgt = gr.Plot()
        feat.change(distribution_plot, [feat, logx], dist)
        logx.change(distribution_plot, [feat, logx], dist)
        demo.load(distribution_plot, [feat, logx], dist)
        demo.load(target_plot, None, tgt)

    with gr.Tab("Threshold & business cost"):
        gr.Markdown(f"Cost assumptions: false positive (pop-up to a non-buyer) = €{C_FP:g}, "
                    f"false negative (missed buyer) = €{C_FN:g}.")
        with gr.Row():
            fs3 = gr.Radio(FEATURE_SETS, value="deployable", label="Feature set")
            meth = gr.Radio(METHODS, value="sklearn", label="Implementation")
        thr = gr.Slider(0.01, 0.99, value=tuned_threshold("deployable", "sklearn"), step=0.01,
                        label="Decision threshold")
        summary = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        for comp in [fs3, meth]:
            comp.change(reset_threshold, [fs3, meth], thr)
        thr.change(threshold_tab, [fs3, meth, thr], [cm_plot, cost_plot, summary])
        demo.load(threshold_tab, [fs3, meth, thr], [cm_plot, cost_plot, summary])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
