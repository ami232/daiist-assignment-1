"""
Assignment 1 - Gradio dashboard.

    uv run python main.py app

Inference only: this file loads the artifacts written by train.py (preprocessor, the
three trained models, the held-out test rows) and never fits anything.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import gradio as gr

from common import (
    ART, COSTS, LogisticRegressionNN, confusion_counts, cost_matrix, engineer_features,
    load_clean, load_meta, theoretical_threshold, total_cost,
)

if not (ART / "meta.json").exists():
    raise SystemExit("No trained artifacts found. Run `uv run python main.py train` first.")

# ------------------------------------------------------------------ load saved artifacts (no training)
meta = load_meta()
pre = joblib.load(ART / "preprocessor.joblib")
sk_model = joblib.load(ART / "sklearn_logreg.joblib")
manual = torch.load(ART / "manual_torch.pt", weights_only=True)
nn_ckpt = torch.load(ART / "nn_torch.pt", weights_only=True)
nn_model = LogisticRegressionNN(nn_ckpt["n_features"])
nn_model.load_state_dict(nn_ckpt["state_dict"])
nn_model.eval()

test = pd.read_csv(ART / "test_set.csv")
y_test = test.pop("Churn").to_numpy()
A = pre.transform(test)
At = torch.tensor(A, dtype=torch.float32)
with torch.no_grad():
    PROBS = {
        "naive baseline": np.full(len(y_test), meta["train_prior"]),
        "scikit-learn": sk_model.predict_proba(A)[:, 1],
        "manual PyTorch": torch.sigmoid(At @ manual["w"] + manual["b"]).numpy(),
        "nn.Module PyTorch": torch.sigmoid(nn_model(At)).numpy(),
    }
MODELS = list(PROBS)

# Colours: categorical slots 1-3 of a validated colour-blind-safe palette; grey for the baseline.
# Marker shapes + sizes are a second encoding so overlapping models stay distinguishable.
STYLE = {
    "naive baseline": dict(color="#898781", symbol="x", size=11),
    "scikit-learn": dict(color="#2a78d6", symbol="circle", size=15),
    "manual PyTorch": dict(color="#eb6834", symbol="square", size=10),
    "nn.Module PyTorch": dict(color="#1baf7a", symbol="diamond", size=6),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
CHURN_COLORS = {0: "#2a78d6", 1: "#eb6834"}


def base_layout(fig, title, xtitle=None, ytitle=None, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)), height=height,
        template="plotly_white", font=dict(color=MUTED),
        margin=dict(l=60, r=20, t=55, b=55),
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False)
    return fig


# ------------------------------------------------------------------ tab 1: model comparison
def reliability_plot():
    """Prediction vs actual: mean predicted churn probability vs observed churn rate, 10 equal-size bins."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(color=GRID, dash="dash", width=2), hoverinfo="skip"))
    for name, p in PROBS.items():
        s = STYLE[name]
        if name == "naive baseline":
            xs, ys = [p.mean()], [y_test.mean()]
        else:
            bins = pd.qcut(p, 10, duplicates="drop")
            g = pd.DataFrame({"p": p, "y": y_test}).groupby(bins, observed=True).mean()
            xs, ys = g["p"].to_numpy(), g["y"].to_numpy()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers" if len(xs) > 1 else "markers", name=name,
            line=dict(color=s["color"], width=2),
            marker=dict(color=s["color"], symbol=s["symbol"], size=s["size"], line=dict(color="#fcfcfb", width=1)),
            hovertemplate=name + "<br>mean predicted %{x:.3f}<br>observed churn rate %{y:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return base_layout(fig, "Predicted vs actual churn rate (test set, 10 equal-size bins)",
                       "mean predicted probability", "observed churn rate")


def convergence_plot():
    """Gap between each PyTorch run's objective and scikit-learn's optimum over training."""
    h = pd.read_csv(ART / "loss_history.csv")
    fig = go.Figure()
    for name in ["manual PyTorch", "nn.Module PyTorch"]:
        s = STYLE[name]
        gap = np.maximum(h[name].to_numpy() - meta["sklearn_objective"], 1e-9)
        fig.add_trace(go.Scatter(
            x=h["epoch"], y=gap, mode="lines", name=name, line=dict(color=s["color"], width=2 if name.startswith("manual") else 3),
            hovertemplate=name + "<br>epoch %{x}<br>gap to sklearn optimum %{y:.2e}<extra></extra>"))
    fig.update_yaxes(type="log")
    return base_layout(fig, "Training objective: gap to scikit-learn's optimum (lower = converged)",
                       "epoch (full-batch gradient descent)", "objective - sklearn objective", height=360)


def metrics_table():
    cols = ["model", "ROC-AUC", "log loss", "Brier", "accuracy@0.5", "customers flagged @t*",
            "total cost @t* (EUR)", "cost per customer (EUR)"]
    df = pd.DataFrame(meta["metrics"])[cols]
    return df.round(4)


def agreement_table():
    return pd.DataFrame({"check": list(meta["agreement"].keys()),
                         "value": [f"{v:.3g}" for v in meta["agreement"].values()]})


# ------------------------------------------------------------------ tab 2: distributions
full_X, full_y = load_clean()
full_X = engineer_features(full_X)
full_X["Churn"] = np.where(full_y == 1, "Churned", "Stayed")
FEATURES = [c for c in full_X.columns if c != "Churn"]


def target_plot():
    counts = full_X["Churn"].value_counts().reindex(["Stayed", "Churned"])
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[CHURN_COLORS[0], CHURN_COLORS[1]],
                           text=[f"{v:,} ({v / counts.sum():.1%})" for v in counts.values], textposition="outside",
                           hovertemplate="%{x}: %{y:,}<extra></extra>"))
    fig.update_yaxes(range=[0, counts.max() * 1.15])
    return base_layout(fig, "Target: customers who churned", None, "customers", height=340)


def feature_plot(feature):
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(full_X[feature]) and full_X[feature].nunique() > 8:
        for label, color in [("Stayed", CHURN_COLORS[0]), ("Churned", CHURN_COLORS[1])]:
            fig.add_trace(go.Histogram(x=full_X.loc[full_X["Churn"] == label, feature], name=label, opacity=0.65,
                                       marker_color=color, nbinsx=30))
        fig.update_layout(barmode="overlay")
        return base_layout(fig, f"Distribution of {feature} by outcome", feature, "customers")
    g = full_X.groupby(feature)["Churn"].agg(n="size", rate=lambda s: (s == "Churned").mean()).reset_index()
    g[feature] = g[feature].astype(str)
    fig.add_trace(go.Bar(x=g[feature], y=g["rate"], marker_color=CHURN_COLORS[1], name="churn rate",
                         text=[f"{r:.0%}<br>n={n:,}" for r, n in zip(g["rate"], g["n"])], textposition="outside",
                         hovertemplate=feature + " = %{x}<br>churn rate %{y:.1%}<extra></extra>"))
    fig.update_yaxes(range=[0, min(1.0, g["rate"].max() * 1.3 + 0.05)], tickformat=".0%")
    return base_layout(fig, f"Churn rate by {feature}", feature, "churn rate")


# ------------------------------------------------------------------ tab 3: threshold + cost
def threshold_view(model_name, threshold, customer_value, offer_cost, save_rate):
    costs = {"customer_value": float(customer_value), "offer_cost": float(offer_cost), "save_rate": float(save_rate)}
    p = PROBS[model_name]
    counts = confusion_counts(y_test, p, threshold)
    cost = total_cost(y_test, p, threshold, costs)
    n = len(y_test)
    do_nothing = total_cost(y_test, np.zeros(n), 0.5, costs)
    flag_all = total_cost(y_test, np.ones(n), 0.5, costs)
    t_star = theoretical_threshold(costs) if costs["save_rate"] > 0 and costs["customer_value"] > 0 else float("nan")

    z = [[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]]
    labels = [["TN: stays, no offer", "FP: stays, offer wasted"], ["FN: churns, no offer", "TP: churns, offer made"]]
    fig_cm = go.Figure(go.Heatmap(
        z=z, x=["not flagged", "flagged"], y=["actually stays", "actually churns"], colorscale=[[0, "#f1f6fd"], [1, "#184f95"]],
        showscale=False, text=[[f"{labels[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)],
        texttemplate="%{text}", textfont=dict(size=14), hoverinfo="skip", xgap=3, ygap=3))
    fig_cm.update_yaxes(autorange="reversed")
    base_layout(fig_cm, f"Confusion matrix at threshold {threshold:.2f} ({model_name})", None, None, height=360)

    ts = np.linspace(0, 1, 101)
    curve = [total_cost(y_test, p, t, costs) for t in ts]
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=ts, y=curve, mode="lines", name="total cost", line=dict(color="#2a78d6", width=2),
                                  hovertemplate="threshold %{x:.2f}<br>cost %{y:,.0f} EUR<extra></extra>"))
    fig_cost.add_hline(y=do_nothing, line=dict(color="#898781", dash="dash"), annotation_text="no offers to anyone",
                       annotation_position="top left")
    fig_cost.add_trace(go.Scatter(x=[threshold], y=[cost], mode="markers", name="selected threshold",
                                  marker=dict(color="#eb6834", size=13, symbol="diamond", line=dict(color="#fcfcfb", width=2))))
    if np.isfinite(t_star) and 0 <= t_star <= 1:
        fig_cost.add_vline(x=t_star, line=dict(color="#1baf7a", dash="dot"), annotation_text=f"formula t* = {t_star:.2f}",
                           annotation_position="bottom right")
    base_layout(fig_cost, "Total cost on the test set vs threshold", "decision threshold", "total cost (EUR)", height=380)

    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    summary = (
        f"### Total cost: **{cost:,.0f} EUR** ({cost / n:,.1f} EUR per customer)\n"
        f"- Offering to no one: {do_nothing:,.0f} EUR &nbsp;|&nbsp; offering to everyone: {flag_all:,.0f} EUR\n"
        f"- Saving vs no offers: **{do_nothing - cost:,.0f} EUR** on {n:,} test customers\n"
        f"- Flagged {tp + fp:,} customers; precision {precision:.2f}, recall {recall:.2f}\n"
        f"- Cost per outcome: " + ", ".join(f"{k} {v:,.0f}" for k, v in cost_matrix(costs).items()) + " EUR"
    )
    return fig_cm, summary, fig_cost


# ------------------------------------------------------------------ layout
with gr.Blocks(title="Telco churn: three ways, one model") as demo:
    gr.Markdown(
        "# Telco churn: logistic regression three ways\n"
        "Same split, same objective, three implementations. Everything here is loaded from saved "
        "artifacts - nothing is retrained when this app starts."
    )
    with gr.Tab("Model comparison"):
        gr.Plot(value=reliability_plot, label="Prediction vs actual")
        gr.Markdown(
            "The three trained models sit on top of one another (circle, square and diamond overlap), "
            "which is the point: the same objective gives the same predictions."
        )
        gr.Dataframe(value=metrics_table, label="Test-set metrics (cost uses the threshold t* from the cost matrix)",
                     interactive=False)
        gr.Dataframe(value=agreement_table, label="Do the three methods agree?", interactive=False)
        gr.Plot(value=convergence_plot, label="Convergence")

    with gr.Tab("Distributions"):
        gr.Plot(value=target_plot)
        feat = gr.Dropdown(FEATURES, value=FEATURES[0] if "tenure" not in FEATURES else "tenure", label="Feature")
        feat_plot = gr.Plot()
        feat.change(feature_plot, feat, feat_plot)
        demo.load(feature_plot, feat, feat_plot)

    with gr.Tab("Threshold & business cost"):
        gr.Markdown(
            "Move the threshold to see the confusion matrix and the business cost change. "
            "The three cost inputs are the placeholder assumptions from `common.py`; edit them to see how the best threshold moves."
        )
        with gr.Row():
            model_dd = gr.Dropdown(MODELS, value="scikit-learn", label="Model")
            thr = gr.Slider(0, 1, value=round(meta["theoretical_threshold"], 2), step=0.01, label="Decision threshold")
        with gr.Row():
            v_in = gr.Number(value=COSTS["customer_value"], label="Value of a churner kept / lost (EUR)")
            c_in = gr.Number(value=COSTS["offer_cost"], label="Cost of one retention offer (EUR)")
            s_in = gr.Number(value=COSTS["save_rate"], label="Offer success rate (0-1)", minimum=0, maximum=1)
        summary = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        ins = [model_dd, thr, v_in, c_in, s_in]
        outs = [cm_plot, summary, cost_plot]
        for comp in ins:
            comp.change(threshold_view, ins, outs)
        demo.load(threshold_view, ins, outs)

if __name__ == "__main__":
    demo.launch()
