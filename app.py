"""
Assignment 1 — Gradio dashboard (Bank Marketing: whom should the call center call next?).

Loads only what `main.py train` saved in `artifacts/` (preprocessor + three trained
logistic regressions) and never retrains anything. Tabs:

1. Model comparison: predicted vs actual (calibration), agreement between the three
   implementations, training curves, weights and the results table.
2. Distributions: feature / target distributions per split, and the regime change over time.
3. Threshold & business cost: threshold slider -> confusion matrix, business cost and the
   share of clients called compared with the 25% call-center capacity.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

ARTIFACTS = Path("artifacts")
MODEL_LABELS = {
    "sklearn": "scikit-learn",
    "manual_torch": "Manual PyTorch loop",
    "standard_torch": "nn.Module + torch.optim",
}
MODEL_COLORS = {"sklearn": "#1f77b4", "manual_torch": "#ff7f0e", "standard_torch": "#2ca02c"}
SPLIT_ORDER = ["train", "val", "test"]


# --------------------------------------------------------------------------- loading
class LogisticRegressionNet(torch.nn.Module):
    """Same architecture as in train.ipynb, needed to load its state_dict."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


if not (ARTIFACTS / "metadata.json").exists():
    raise SystemExit("artifacts/ not found: run `uv run python main.py train` first.")

meta = json.loads((ARTIFACTS / "metadata.json").read_text())
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_model.joblib")
manual_params = torch.load(ARTIFACTS / "manual_torch_model.pt")
torch_model = LogisticRegressionNet(meta["n_features"])
torch_model.load_state_dict(torch.load(ARTIFACTS / "standard_torch_model.pt"))
torch_model.eval()

data = pd.read_csv(ARTIFACTS / "dataset_with_split.csv")
data["period"] = data["year"].astype(str) + "-" + data["month"]
results = pd.read_csv(ARTIFACTS / "results.csv")
weights = pd.read_csv(ARTIFACTS / "weights.csv", index_col=0)

CAPACITY = meta["capacity"]

# Predictions for every row with the three saved models (inference only).
X_all = preprocessor.transform(data[meta["categorical"] + meta["numeric"]]).astype(np.float32)
Xt_all = torch.from_numpy(X_all)
with torch.no_grad():
    PROBA = {
        "sklearn": sk_model.predict_proba(X_all)[:, 1],
        "manual_torch": torch.sigmoid(Xt_all @ manual_params["w"] + manual_params["b"]).numpy(),
        "standard_torch": torch.sigmoid(torch_model(Xt_all)).numpy(),
    }


def split_mask(split: str) -> np.ndarray:
    return (data["split"] == split).to_numpy()


def top_per_period(scores: np.ndarray, periods: pd.Series, capacity: float = CAPACITY) -> np.ndarray:
    """The call-center policy from REPORT.md: call the top `capacity` share of each month."""
    d = pd.DataFrame({"score": scores, "period": periods.to_numpy()})
    rank = d.groupby("period")["score"].rank(method="first", ascending=False)
    size = d.groupby("period")["score"].transform("size")
    return (rank <= np.round(capacity * size)).to_numpy()


# --------------------------------------------------------------------------- tab 1
def calibration_figure(split: str) -> go.Figure:
    """Predicted probability vs actual subscription rate, in deciles of the prediction."""
    mask = split_mask(split)
    y = data.loc[mask, "y"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(dash="dash", color="grey")))
    for key, label in MODEL_LABELS.items():
        p = PROBA[key][mask]
        bins = pd.qcut(p, q=10, duplicates="drop")
        grouped = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(
            x=grouped["p"], y=grouped["y"], mode="lines+markers", name=label,
            line=dict(color=MODEL_COLORS[key], width=2,
                      dash="dot" if key == "standard_torch" else "solid"),
        ))
    fig.add_hline(y=y.mean(), line=dict(color="black", width=1, dash="dot"),
                  annotation_text=f"actual positive rate {y.mean():.1%}")
    fig.update_layout(
        title=f"Predicted vs actual ({split}, deciles of prediction)",
        xaxis_title="Mean predicted probability", yaxis_title="Actual subscription rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), height=480, legend=dict(x=0.01, y=0.99),
    )
    return fig


def agreement_figure(split: str) -> go.Figure:
    """Each client's probability from the two PyTorch models vs scikit-learn."""
    mask = split_mask(split)
    ref = PROBA["sklearn"][mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x",
                             line=dict(dash="dash", color="grey")))
    for key in ["manual_torch", "standard_torch"]:
        fig.add_trace(go.Scattergl(
            x=ref, y=PROBA[key][mask], mode="markers", name=MODEL_LABELS[key],
            marker=dict(size=5, opacity=0.5, color=MODEL_COLORS[key]),
        ))
    max_diff = max(np.abs(PROBA[k][mask] - ref).max() for k in ["manual_torch", "standard_torch"])
    fig.update_layout(
        title=f"PyTorch vs scikit-learn ({split}): max |diff| = {max_diff:.5f}",
        xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability", height=480,
    )
    return fig


def loss_figure() -> go.Figure:
    fig = go.Figure()
    for key in ["manual_torch", "standard_torch"]:
        hist = meta["loss_history"][key]
        fig.add_trace(go.Scatter(y=hist, mode="lines", name=MODEL_LABELS[key],
                                 line=dict(color=MODEL_COLORS[key],
                                           dash="dot" if key == "standard_torch" else "solid")))
    hp = meta["hyperparameters"]
    fig.update_layout(
        title=f"Training loss (lr = {hp['lr']}, C = {hp['C']}, {hp['epochs']:,} full-batch epochs)",
        xaxis_title="Epoch", yaxis_title="Penalised log loss (train)", xaxis_type="log", height=380,
    )
    return fig


def weights_figure(top_n: int = 15) -> go.Figure:
    w = weights.drop(index="intercept")
    w = w.reindex(w["sklearn"].abs().sort_values(ascending=True).index).tail(top_n)
    names = [n.split("__", 1)[1] for n in w.index]
    fig = go.Figure()
    for key, label in MODEL_LABELS.items():
        fig.add_trace(go.Bar(y=names, x=w[key], orientation="h", name=label,
                             marker_color=MODEL_COLORS[key]))
    fig.update_layout(
        title=f"Top {top_n} weights (standardised; + = more likely to subscribe)",
        barmode="group", height=620, xaxis_title="Weight",
    )
    return fig


def results_table() -> pd.DataFrame:
    table = results.copy()
    table["model"] = table["model"].replace(MODEL_LABELS)
    for col in ["precision@25%", "recall@25%"]:
        table[col] = (table[col] * 100).round(1).astype(str) + "%"
    table["business_cost_eur"] = table["business_cost_eur"].map("{:,.0f} EUR".format)
    return table.round(4)


# --------------------------------------------------------------------------- tab 2
RAW_FEATURES = ["age", "age_band", "job", "marital", "education", "default", "housing", "loan",
                "loans_unknown", "contact", "day_of_week", "campaign", "log_campaign", "previous",
                "poutcome", "euribor3m", "month", "emp.var.rate", "nr.employed"]
NUMERIC_RAW = {"age", "campaign", "log_campaign", "previous", "euribor3m", "emp.var.rate", "nr.employed"}


def distribution_figure(feature: str, split: str) -> go.Figure:
    d = data if split == "all" else data[data["split"] == split]
    if feature in NUMERIC_RAW:
        fig = go.Figure()
        for value, label, color in [(0, "Did not subscribe (y=0)", "#9aa5b1"),
                                    (1, "Subscribed (y=1)", "#d62728")]:
            fig.add_trace(go.Histogram(x=d.loc[d["y"] == value, feature], name=label,
                                       histnorm="probability", opacity=0.6, marker_color=color,
                                       nbinsx=40))
        fig.update_layout(barmode="overlay", yaxis_title="Share of group",
                          title=f"Distribution of {feature} by outcome ({split})")
    else:
        g = d.groupby(feature)["y"].agg(rows="size", rate="mean").sort_values("rate")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g.index.astype(str), y=g["rows"], name="Clients",
                             marker_color="#9aa5b1"))
        fig.add_trace(go.Scatter(x=g.index.astype(str), y=g["rate"], name="Subscription rate",
                                 yaxis="y2", mode="lines+markers", marker_color="#d62728"))
        fig.update_layout(
            title=f"{feature}: number of clients and subscription rate ({split})",
            yaxis_title="Clients",
            yaxis2=dict(title="Subscription rate", overlaying="y", side="right",
                        tickformat=".0%", rangemode="tozero"),
        )
    if feature in {"month", "emp.var.rate", "nr.employed"}:
        fig.add_annotation(text="Not used by the model (see REPORT.md)", xref="paper", yref="paper",
                           x=0.5, y=1.08, showarrow=False, font=dict(color="#d62728"))
    fig.update_layout(height=460, legend=dict(orientation="h", y=-0.2))
    return fig


def target_figure() -> go.Figure:
    g = data.groupby(["split"])["y"].agg(rows="size", rate="mean").reindex(SPLIT_ORDER)
    fig = go.Figure(go.Bar(x=g.index, y=g["rate"], text=[f"{r:.1%}<br>{n:,} clients"
                                                         for r, n in zip(g["rate"], g["rows"])],
                           textposition="outside", marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"]))
    fig.update_layout(title="Target distribution: subscription rate per split",
                      yaxis=dict(tickformat=".0%", range=[0, 0.7]), height=380)
    return fig


def timeline_figure() -> go.Figure:
    g = data.groupby(["year", "month"], sort=False).agg(
        calls=("y", "size"), rate=("y", "mean"), euribor=("euribor3m", "mean"),
        split=("split", "first")).reset_index()
    g["label"] = g["year"].astype(str) + "-" + g["month"]
    fig = go.Figure()
    for split, color in zip(SPLIT_ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        s = g[g["split"] == split]
        fig.add_trace(go.Bar(x=s["label"], y=s["calls"], name=f"Calls ({split})", marker_color=color,
                             opacity=0.55))
    fig.add_trace(go.Scatter(x=g["label"], y=g["rate"], name="Subscription rate", yaxis="y2",
                             mode="lines+markers", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=g["label"], y=g["euribor"] / 100, name="Euribor 3m", yaxis="y2",
                             mode="lines", line=dict(color="black", dash="dot")))
    fig.update_layout(
        title="Regime change: fewer calls, higher subscription rate",
        yaxis_title="Calls per month",
        yaxis2=dict(title="Rate", overlaying="y", side="right", tickformat=".0%", range=[0, 0.7]),
        height=460, legend=dict(orientation="h", y=-0.3),
    )
    return fig


# --------------------------------------------------------------------------- tab 3
def threshold_view(model_key: str, split: str, threshold: float, cost_fp: float, cost_fn: float):
    mask = split_mask(split)
    y = data.loc[mask, "y"].to_numpy()
    p = PROBA[model_key][mask]
    called = p >= threshold

    tp = int((called & (y == 1)).sum())
    fp = int((called & (y == 0)).sum())
    fn = int((~called & (y == 1)).sum())
    tn = int((~called & (y == 0)).sum())
    share_called = called.mean()
    precision = tp / max(called.sum(), 1)
    recall = tp / max((y == 1).sum(), 1)
    cost = fp * cost_fp + fn * cost_fn

    # Reference: the capacity policy (top 25% of each month) with the same costs.
    cap_called = top_per_period(p, data.loc[mask, "period"])
    cap_fp = int((cap_called & (y == 0)).sum())
    cap_fn = int((~cap_called & (y == 1)).sum())
    cap_cost = cap_fp * cost_fp + cap_fn * cost_fn
    cap_precision = (cap_called & (y == 1)).sum() / cap_called.sum()

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Not called (pred. 0)", "Called (pred. 1)"],
        y=["Did not subscribe (y=0)", "Subscribed (y=1)"],
        text=[[f"TN<br>{tn:,}", f"FP<br>{fp:,}<br>-{fp * cost_fp:,.0f} EUR"],
              [f"FN<br>{fn:,}<br>-{fn * cost_fn:,.0f} EUR", f"TP<br>{tp:,}"]],
        texttemplate="%{text}", textfont=dict(size=16), colorscale="Blues", showscale=False,
    ))
    cm.update_layout(title=f"Confusion matrix — {MODEL_LABELS[model_key]}, {split}, threshold {threshold:.2f}",
                     height=420, yaxis=dict(autorange="reversed"))

    over = share_called > CAPACITY + 1e-9
    status = (f"⚠️ **Over capacity**: this threshold calls {share_called:.1%} of clients, "
              f"but the call center can only call {CAPACITY:.0%}."
              if over else
              f"✅ Within capacity: {share_called:.1%} of clients called (limit {CAPACITY:.0%}).")
    summary = f"""{status}

| | This threshold | Capacity policy (top {CAPACITY:.0%} of each month) |
|---|---|---|
| Clients called | {called.sum():,} ({share_called:.1%}) | {cap_called.sum():,} ({cap_called.mean():.1%}) |
| Precision (calls that subscribe) | {precision:.1%} | {cap_precision:.1%} |
| Recall (subscribers reached) | {recall:.1%} | {(cap_called & (y == 1)).sum() / (y == 1).sum():.1%} |
| **Business cost** (FP x {cost_fp:g} + FN x {cost_fn:g} EUR) | **{cost:,.0f} EUR** | **{cap_cost:,.0f} EUR** |

Cost-only optimal threshold for these costs: C_FP / (C_FP + C_FN) = **{cost_fp / (cost_fp + cost_fn):.3f}**
(the threshold that minimises cost if there were no capacity limit).
"""

    # Curves over all thresholds, with the current one marked.
    grid = np.linspace(0.0, 1.0, 101)
    calls = np.array([(p >= t).mean() for t in grid])
    costs = np.array([((p >= t) & (y == 0)).sum() * cost_fp + ((p < t) & (y == 1)).sum() * cost_fn
                      for t in grid])
    precs = np.array([y[p >= t].mean() if (p >= t).any() else np.nan for t in grid])
    curves = go.Figure()
    curves.add_trace(go.Scatter(x=grid, y=costs, name="Business cost (EUR)", line=dict(color="#d62728")))
    curves.add_trace(go.Scatter(x=grid, y=calls, name="Share of clients called", yaxis="y2",
                                line=dict(color="#1f77b4")))
    curves.add_trace(go.Scatter(x=grid, y=precs, name="Precision", yaxis="y2",
                                line=dict(color="#2ca02c")))
    curves.add_hline(y=CAPACITY, yref="y2", line=dict(color="#1f77b4", dash="dot"),
                     annotation_text=f"capacity {CAPACITY:.0%}")
    curves.add_vline(x=threshold, line=dict(color="black", dash="dash"),
                     annotation_text=f"threshold {threshold:.2f}")
    curves.update_layout(
        title="Threshold vs cost, precision and call volume",
        xaxis_title="Decision threshold", yaxis_title="Business cost (EUR)",
        yaxis2=dict(title="Share", overlaying="y", side="right", tickformat=".0%", range=[0, 1]),
        height=420, legend=dict(orientation="h", y=-0.25),
    )
    return cm, summary, curves


# --------------------------------------------------------------------------- layout
with gr.Blocks(title="Bank Marketing — whom to call next") as demo:
    gr.Markdown(
        f"""# Bank Marketing: whom should the call center call next?
Logistic regression trained three ways (scikit-learn, manual PyTorch loop, `nn.Module` + `torch.optim`)
on a **temporal split** (train May 2008 – May 2009, validation Jun – Dec 2009, test 2010).
The call center can call **{CAPACITY:.0%} of each campaign's clients**. A wasted call costs about
**{meta['cost_fp']} EUR**, a missed subscriber about **{meta['cost_fn']} EUR**.
All models are loaded from `artifacts/`; nothing is retrained here."""
    )

    with gr.Tab("1 · Model comparison"):
        split_cmp = gr.Radio(["val", "test", "train"], value="test", label="Data split")
        with gr.Row():
            calib_plot = gr.Plot()
            agree_plot = gr.Plot()
        gr.Markdown("### Results (from training: precision/recall at 25% capacity per month)")
        gr.Dataframe(results_table(), interactive=False)
        with gr.Row():
            gr.Plot(loss_figure())
            gr.Plot(weights_figure())
        split_cmp.change(lambda s: (calibration_figure(s), agreement_figure(s)),
                         split_cmp, [calib_plot, agree_plot])

    with gr.Tab("2 · Distributions"):
        with gr.Row():
            feature = gr.Dropdown(RAW_FEATURES, value="age", label="Feature")
            split_dist = gr.Radio(["all", "train", "val", "test"], value="train", label="Data split")
        dist_plot = gr.Plot()
        with gr.Row():
            gr.Plot(target_figure())
            gr.Plot(timeline_figure())
        for control in (feature, split_dist):
            control.change(distribution_figure, [feature, split_dist], dist_plot)

    with gr.Tab("3 · Threshold & business cost"):
        with gr.Row():
            model_sel = gr.Dropdown([(v, k) for k, v in MODEL_LABELS.items()], value="sklearn",
                                    label="Model")
            split_thr = gr.Radio(["val", "test"], value="test", label="Data split")
        threshold = gr.Slider(0.0, 1.0, value=round(meta["default_threshold"], 2), step=0.01,
                              label="Decision threshold: call the client if P(subscribe) >= threshold "
                                    "(default = threshold that calls 25% of validation clients)")
        with gr.Row():
            cost_fp = gr.Number(value=meta["cost_fp"], label="Cost of a false positive (EUR per wasted call)")
            cost_fn = gr.Number(value=meta["cost_fn"], label="Cost of a false negative (EUR per missed subscriber)")
        with gr.Row():
            cm_plot = gr.Plot()
            summary_md = gr.Markdown()
        curves_plot = gr.Plot()
        inputs = [model_sel, split_thr, threshold, cost_fp, cost_fn]
        for control in inputs:
            control.change(threshold_view, inputs, [cm_plot, summary_md, curves_plot])

    demo.load(lambda: (calibration_figure("test"), agreement_figure("test"),
                       distribution_figure("age", "train"),
                       *threshold_view("sklearn", "test", round(meta["default_threshold"], 2),
                                       meta["cost_fp"], meta["cost_fn"])),
              None, [calib_plot, agree_plot, dist_plot, cm_plot, summary_md, curves_plot])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
