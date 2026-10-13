"""
Assignment 1 — Gradio dashboard for the win-back voucher model.

Run via `uv run python main.py app`. Loads what train.ipynb saved in artifacts/
(the three trained models, the scaler, the customer snapshots) and only ever
calls them for predictions: nothing is fit or retrained here.
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
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

from common import LogisticRegression, break_even_threshold, voucher_profit

torch.set_default_dtype(torch.float64)
ART = Path("artifacts")

# ---------------------------------------------------------------- load artifacts
config = json.loads((ART / "config.json").read_text())
metrics = json.loads((ART / "metrics.json").read_text())
FEATURES = config["features"]
COSTS = config["costs"]

snaps = pd.read_csv(ART / "snapshots.csv.gz", parse_dates=["cutoff"])
scaler = joblib.load(ART / "scaler.joblib")
X = torch.tensor(scaler.transform(snaps[[f"f_{f}" for f in FEATURES]].set_axis(FEATURES, axis=1)))

sk_model = joblib.load(ART / "sklearn_logreg.joblib")
manual = torch.load(ART / "manual_torch.pt")
module = LogisticRegression(len(FEATURES))
module.load_state_dict(torch.load(ART / "torch_module.pt"))
module.eval()

with torch.no_grad():
    snaps["p_sklearn"] = sk_model.predict_proba(X.numpy())[:, 1]
    snaps["p_manual"] = torch.sigmoid(X @ manual["w"] + manual["b"]).numpy()
    snaps["p_module"] = torch.sigmoid(module(X)).numpy()

MODELS = {"scikit-learn": "p_sklearn", "PyTorch manual loop": "p_manual", "PyTorch nn.Module": "p_module"}
SPLITS = {"test (cutoff 2011-09-01)": "test", "validation (cutoff 2011-06-01)": "val", "train (4 cutoffs)": "train"}
DIST_COLUMNS = {
    "days since last order": "recency_days",
    "orders in last 180 days": "n_orders",
    "revenue in last 180 days (GBP)": "revenue",
    "distinct products": "n_products",
    "share of spend on Christmas items": "xmas_share",
    "return rate": "return_rate",
} | {f"model feature: {f}": f"f_{f}" for f in FEATURES}
LOG_AXIS = {"n_orders", "revenue", "n_products"}

COEFS = pd.DataFrame(
    {
        "scikit-learn": np.append(sk_model.coef_.ravel(), sk_model.intercept_),
        "PyTorch manual loop": np.append(manual["w"].numpy(), manual["b"].numpy()),
        "PyTorch nn.Module": np.append(module.linear.weight.detach().numpy().ravel(), module.linear.bias.detach().numpy()),
    },
    index=FEATURES + ["intercept"],
)


def gbp(v):
    return f"−£{-v:,.0f}" if v < 0 else f"£{v:,.0f}"


def subset(split_label):
    return snaps[snaps["split"] == SPLITS[split_label]]


# ---------------------------------------------------------------- tab 1: comparison
def comparison(split_label):
    s = subset(split_label)
    y = s["lapsed"].to_numpy()

    # predicted vs actual: bin customers by predicted lapse probability
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="grey", dash="dot"), name="perfect"))
    for (name, col), dash in zip(MODELS.items(), ["solid", "dash", "dot"]):
        bins = pd.qcut(s[col], 10, duplicates="drop")
        g = s.groupby(bins, observed=True).agg(pred=(col, "mean"), actual=("lapsed", "mean"), n=("lapsed", "size"))
        fig.add_trace(go.Scatter(x=g["pred"], y=g["actual"], mode="lines+markers", name=name, line=dict(dash=dash),
                                 customdata=g["n"], hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>%{customdata} customers"))
    fig.update_layout(title="Predicted vs actual (deciles)",
                      xaxis_title="mean predicted lapse probability", yaxis_title="actual lapse rate",
                      height=450, legend=dict(orientation="h", y=-0.2))

    scatter = go.Figure()
    for name, col in list(MODELS.items())[1:]:
        scatter.add_trace(go.Scattergl(x=s["p_sklearn"], y=s[col], mode="markers", name=name, marker=dict(size=4, opacity=0.5)))
    scatter.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="grey", dash="dot"), name="y = x"))
    scatter.update_layout(title="Per customer: PyTorch vs sklearn",
                          xaxis_title="scikit-learn", yaxis_title="PyTorch", height=450, legend=dict(orientation="h", y=-0.2))

    base = snaps.loc[snaps["split"] == "train", "lapsed"].mean()
    rows = [{"model": "baseline: train lapse rate for everyone", "AUC": 0.5,
             "log loss": log_loss(y, np.full(len(y), base)), "max |p - p_sklearn|": np.nan}]
    for name, col in MODELS.items():
        rows.append({"model": name, "AUC": roc_auc_score(y, s[col]), "log loss": log_loss(y, s[col]),
                     "max |p - p_sklearn|": float(np.max(np.abs(s[col] - s["p_sklearn"])))})
    table = pd.DataFrame(rows)
    table[["AUC", "log loss"]] = table[["AUC", "log loss"]].round(4)
    table["max |p - p_sklearn|"] = table["max |p - p_sklearn|"].map(lambda v: "" if np.isnan(v) else f"{v:.1e}")

    coef_long = COEFS.reset_index(names="feature").melt(id_vars="feature", var_name="model", value_name="coefficient")
    coef_fig = px.bar(coef_long, y="feature", x="coefficient", color="model", barmode="group", orientation="h",
                      title="Coefficients on standardised features (the three bars should overlap)", height=650)
    coef_fig.update_layout(legend=dict(orientation="h", y=-0.1))
    return fig, scatter, table, coef_fig


# ---------------------------------------------------------------- tab 2: distributions
def distributions(column_label, split_label):
    s = subset(split_label).copy()
    col = DIST_COLUMNS[column_label]
    s["outcome"] = np.where(s["lapsed"] == 1, "lapsed", "ordered again")
    fig = px.histogram(s, x=col, color="outcome", barmode="overlay", opacity=0.6, nbins=60,
                       histnorm="probability", log_x=col in LOG_AXIS and (s[col] > 0).all(),
                       title=f"{column_label} — by outcome", height=420)
    fig.update_layout(xaxis_title=column_label, yaxis_title="share of customers")

    rates = snaps.groupby(["cutoff", "split"]).agg(lapse_rate=("lapsed", "mean"), customers=("lapsed", "size")).reset_index()
    rates["cutoff"] = rates["cutoff"].dt.strftime("%Y-%m-%d")
    target = px.bar(rates, x="cutoff", y="lapse_rate", color="split", text_auto=".2f", hover_data=["customers"],
                    title="Target: share of active customers with no order in the next 90 days", height=380)
    target.update_layout(yaxis_range=[0, 1])
    return fig, target


# ---------------------------------------------------------------- tab 3: voucher decision
def decision(model_label, split_label, threshold, discount, margin, save_rate, retained_quarters, mail_cost):
    costs = dict(discount=discount, margin=margin, save_rate=save_rate,
                 retained_quarters=retained_quarters, mail_cost=mail_cost)
    s = subset(split_label)
    p = s[MODELS[model_label]].to_numpy()
    y = s["lapsed"].to_numpy()
    spend = s["quarterly_spend"].to_numpy()
    send = p >= threshold

    cm = confusion_matrix(y, send.astype(int), labels=[1, 0])
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", height=380,
                       x=["voucher sent", "no voucher"], y=["lapsed", "ordered again"],
                       labels=dict(x="decision", y="what actually happened", color="customers"),
                       title=f"Confusion matrix at threshold {threshold:.2f}")

    profit = voucher_profit(send, y, spend, **costs)
    everyone = voucher_profit(np.ones(len(y), bool), y, spend, **costs)
    rule_days = config["recency_rule_days"]
    rule = voucher_profit(s["recency_days"].to_numpy() >= rule_days, y, spend, **costs)
    be = break_even_threshold(discount, margin, save_rate, retained_quarters)

    summary = (
        f"### Campaign profit: **{gbp(profit)}**\n"
        f"{send.sum():,} vouchers sent to {len(y):,} active customers; "
        f"{(y[send].mean() if send.any() else float('nan')):.0%} of them would have lapsed.\n\n"
        f"| policy | profit vs sending nothing |\n|---|---|\n"
        f"| this model at {threshold:.2f} | {gbp(profit)} |\n"
        f"| recency rule (no order for ≥ {rule_days} days) | {gbp(rule)} |\n"
        f"| voucher to everyone | {gbp(everyone)} |\n"
        f"| no vouchers | £0 |\n\n"
        f"Break-even lapse probability for these economics: **{be:.3f}** "
        f"(threshold picked on validation in training: {config['chosen_thresholds']['sklearn']:.2f})."
    )

    grid = np.round(np.linspace(0, 1, 101), 2)
    curve = [voucher_profit(p >= t, y, spend, **costs) for t in grid]
    curve_fig = go.Figure(go.Scatter(x=grid, y=curve, mode="lines", name="profit"))
    curve_fig.add_vline(x=threshold, line=dict(color="crimson"), annotation_text="slider")
    curve_fig.add_vline(x=be, line=dict(color="black", dash="dash"), annotation_text="break-even", annotation_position="bottom right")
    curve_fig.add_hline(y=0, line=dict(color="grey", width=1))
    curve_fig.update_layout(title="Profit vs threshold", xaxis_title="threshold on lapse probability",
                            yaxis_title="profit (GBP)", height=380)
    return cm_fig, summary, curve_fig


# ---------------------------------------------------------------- layout
def training_summary():
    tr = pd.DataFrame(metrics["test_results"]).round(4)
    tr["profit_gbp"] = tr["profit_gbp"].map(gbp)
    sci = lambda v: f"{v:.2e}"
    agree = pd.DataFrame(metrics["agreement"])
    agree["training_objective"] = agree["training_objective"].map(lambda v: f"{v:.10f}")
    agree["grad_norm_at_solution"] = agree["grad_norm_at_solution"].map(sci)
    pairs = pd.DataFrame(metrics["pair_diffs"])
    pairs[pairs.columns[1:]] = pairs[pairs.columns[1:]].map(sci)
    ablation = pd.DataFrame(metrics["ablation"]).round(4)
    return tr, agree, pairs, ablation


with gr.Blocks(title="Win-back voucher model") as demo:
    gr.Markdown(
        "# Win-back vouchers — who is about to stop ordering?\n"
        "Online Retail II (UCI), customers active in the previous 180 days, scored at quarterly cutoffs. "
        "Target: **no order in the next 90 days**. One logistic regression, trained three ways."
    )
    with gr.Tab("Model comparison"):
        split_cmp = gr.Dropdown(list(SPLITS), value=list(SPLITS)[0], label="Data")
        metrics_table = gr.Dataframe(label="Metrics on this data", interactive=False)
        with gr.Row():
            calib_plot = gr.Plot()
            scatter_plot = gr.Plot()
        coef_plot = gr.Plot()

    with gr.Tab("Distributions"):
        with gr.Row():
            column = gr.Dropdown(list(DIST_COLUMNS), value=list(DIST_COLUMNS)[0], label="Feature")
            split_dist = gr.Dropdown(list(SPLITS), value=list(SPLITS)[0], label="Data")
        dist_plot = gr.Plot()
        target_plot = gr.Plot()

    with gr.Tab("Voucher decision"):
        with gr.Row():
            model_pick = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Model")
            split_dec = gr.Dropdown(list(SPLITS)[:2], value=list(SPLITS)[0], label="Data")
        threshold = gr.Slider(0, 1, value=config["chosen_thresholds"]["sklearn"], step=0.01,
                              label="Send a voucher if predicted lapse probability ≥")
        with gr.Accordion("Voucher economics (assumptions from REPORT.md)", open=False):
            with gr.Row():
                discount = gr.Slider(0.0, 0.5, value=COSTS["discount"], step=0.01, label="discount")
                margin = gr.Slider(0.05, 0.8, value=COSTS["margin"], step=0.01, label="gross margin")
                save_rate = gr.Slider(0.0, 0.6, value=COSTS["save_rate"], step=0.01, label="save rate")
                retained = gr.Slider(1, 8, value=COSTS["retained_quarters"], step=1, label="quarters a won-back customer stays")
                mail = gr.Slider(0.0, 10.0, value=COSTS["mail_cost"], step=0.5, label="mail cost per voucher (£)")
        with gr.Row():
            cm_plot = gr.Plot()
            with gr.Column():
                profit_md = gr.Markdown()
        curve_plot = gr.Plot()

    with gr.Tab("Training summary"):
        gr.Markdown("Saved by `train.ipynb`. Test results use the threshold chosen on validation.")
        test_table = gr.Dataframe(label="Test results", interactive=False)
        with gr.Row():
            agree_table = gr.Dataframe(label="Training objective and gradient norm at each solution", interactive=False)
            pair_table = gr.Dataframe(label="Pairwise differences", interactive=False)
        ablation_table = gr.Dataframe(label="Feature-group ablation (validation)", interactive=False)
        gr.Image(str(ART / "loss_curves.png"), label="Loss curves", show_label=True)

    cmp_out = [calib_plot, scatter_plot, metrics_table, coef_plot]
    dist_in, dist_out = [column, split_dist], [dist_plot, target_plot]
    dec_in = [model_pick, split_dec, threshold, discount, margin, save_rate, retained, mail]
    dec_out = [cm_plot, profit_md, curve_plot]

    split_cmp.change(comparison, split_cmp, cmp_out)
    for c in dist_in:
        c.change(distributions, dist_in, dist_out)
    for c in dec_in:
        c.change(decision, dec_in, dec_out)

    demo.load(comparison, split_cmp, cmp_out)
    demo.load(distributions, dist_in, dist_out)
    demo.load(decision, dec_in, dec_out)
    demo.load(training_summary, None, [test_table, agree_table, pair_table, ablation_table])


if __name__ == "__main__":
    demo.launch()
