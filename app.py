"""
Assignment 1 — Gradio dashboard for the Madrid Airbnb fair-price model.

Run with:  uv run python main.py app      (after `uv run python main.py train`)

This file NEVER trains anything. At startup it only LOADS what train.py saved
in artifacts/ (the fitted preprocessor + the three trained models) and uses
them to predict on the held-out test listings.
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

if not (ART / "sklearn_ridge.joblib").exists():
    raise SystemExit("No trained models found. Run `uv run python main.py train` first.")

# Colours: one fixed colour per model (colour follows the model, everywhere)
COLORS = {
    "scikit-learn Ridge": "#2a78d6",
    "Manual PyTorch loop": "#eb6834",
    "Standard PyTorch (nn + optim)": "#1baf7a",
    "Naive baseline": "#8a8985",
}
MODELS = ["scikit-learn Ridge", "Manual PyTorch loop", "Standard PyTorch (nn + optim)"]


# ---------------------------------------------------------------------------
# Load the saved artifacts (no training here)
# ---------------------------------------------------------------------------
class LinearRegressionModel(nn.Module):  # same class as in train.py, needed to load its weights
    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


metrics = json.loads((ART / "metrics.json").read_text())
preprocessor = joblib.load(ART / "preprocessor.joblib")
sk_model = joblib.load(ART / "sklearn_ridge.joblib")
manual = torch.load(ART / "manual_torch.pt")
std_state = torch.load(ART / "standard_torch.pt")
std_model = LinearRegressionModel(std_state["linear.weight"].shape[1])
std_model.load_state_dict(std_state)
std_model.eval()

train_feats = pd.read_csv(ART / "train_features.csv")
test_feats = pd.read_csv(ART / "test_features.csv")
test_info = pd.read_csv(ART / "test_predictions.csv")  # names, districts, etc. for display
coef = pd.read_csv(ART / "coefficients.csv", index_col=0)

# Predict the test set with the LOADED models
X_test = preprocessor.transform(test_feats.drop(columns="price_eur")).astype(np.float32)
X_test_t = torch.tensor(X_test)
with torch.no_grad():
    log_preds = {
        "scikit-learn Ridge": sk_model.predict(X_test),
        "Manual PyTorch loop": (X_test_t @ manual["w"] + manual["b"]).numpy(),
        "Standard PyTorch (nn + optim)": std_model(X_test_t).numpy(),
    }
preds_eur = {k: np.exp(v) for k, v in log_preds.items()}
actual = test_feats["price_eur"].to_numpy()
baseline_eur = float(test_info["pred_baseline_eur"].iloc[0])
AX_LO, AX_HI = 15, float(max(actual.max(), 3000)) * 1.2  # axis range covering every test listing


def base_layout(fig: go.Figure, title: str, x: str, y: str) -> go.Figure:
    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title=y, template="plotly_white",
        height=460, margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(size=13, color="#0b0b0b"),
    )
    fig.update_xaxes(gridcolor="#ecebe8", zeroline=False)
    fig.update_yaxes(gridcolor="#ecebe8", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Tab 1 — model comparison
# ---------------------------------------------------------------------------
def metrics_table() -> pd.DataFrame:
    rows = []
    for name, m in metrics["test_results"].items():
        rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})
    return pd.DataFrame(rows)


def pred_vs_actual(selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for name in selected:
        fig.add_trace(go.Scattergl(
            x=actual, y=preds_eur[name], mode="markers", name=name,
            marker=dict(size=6, color=COLORS[name], opacity=0.45,
                        line=dict(width=0.5, color="white")),
            customdata=test_info["name"].fillna("").str[:40],
            hovertemplate="actual €%{x:.0f}<br>predicted €%{y:.0f}<br>%{customdata}<extra>" + name + "</extra>",
        ))
    lo, hi = AX_LO, AX_HI
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction",
                             line=dict(color="#52514e", dash="dash", width=2), hoverinfo="skip"))
    fig.add_hline(y=baseline_eur, line=dict(color=COLORS["Naive baseline"], dash="dot", width=2),
                  annotation_text=f"naive baseline €{baseline_eur:.0f}", annotation_position="bottom right")
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)])
    fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)])
    return base_layout(fig, "Predicted vs actual nightly price (test set, log scale)",
                       "actual price (€)", "predicted fair price (€)")


def coef_plot() -> go.Figure:
    top = coef.reindex(coef["sklearn"].abs().sort_values(ascending=True).index).tail(15)
    fig = go.Figure()
    for col, name in zip(["sklearn", "manual_torch", "standard_torch"], MODELS):
        fig.add_trace(go.Bar(y=top.index, x=top[col], name=name, orientation="h",
                             marker_color=COLORS[name],
                             hovertemplate="%{y}: %{x:.3f}<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig = base_layout(fig, "Top 15 coefficients: do the three implementations agree?",
                      "effect on log-price (per 1 SD, or vs. reference category)", "")
    fig.update_layout(height=620)
    return fig


def loss_plot() -> go.Figure:
    fig = go.Figure()
    for key, name in [("manual_torch", "Manual PyTorch loop"), ("standard_torch", "Standard PyTorch (nn + optim)")]:
        h = metrics["loss_history"][key]
        fig.add_trace(go.Scatter(y=h, mode="lines", name=name, line=dict(color=COLORS[name], width=2)))
    fig.update_yaxes(type="log")
    return base_layout(fig, "Training loss per epoch (PyTorch versions)", "epoch", "loss (MSE + L2), log scale")


# ---------------------------------------------------------------------------
# Tab 2 — distributions
# ---------------------------------------------------------------------------
FEATURES = [c for c in train_feats.columns if c != "price_eur"]


def target_hist(scale: str) -> go.Figure:
    vals = train_feats["price_eur"]
    if scale == "log(price)":
        vals, xlab = np.log(vals), "log(price)"
    else:
        vals, xlab = vals.clip(upper=1000), "price (€) — values above €1,000 shown at €1,000"
    fig = go.Figure(go.Histogram(x=vals, nbinsx=80, marker_color="#2a78d6",
                                 marker_line=dict(color="white", width=1)))
    return base_layout(fig, f"Target distribution (train set): {scale}", xlab, "listings")


def feature_hist(feature: str) -> go.Figure:
    col = train_feats[feature]
    if col.dtype == object or col.nunique() <= 25:
        counts = col.astype(str).value_counts().sort_values()
        med = train_feats.groupby(col.astype(str))["price_eur"].median().reindex(counts.index)
        fig = go.Figure(go.Bar(y=counts.index, x=counts.values, orientation="h", marker_color="#2a78d6",
                               customdata=med.values,
                               hovertemplate="%{y}: %{x} listings<br>median price €%{customdata:.0f}<extra></extra>"))
        fig = base_layout(fig, f"{feature} (train set) — hover for median price", "listings", "")
        fig.update_layout(height=max(420, 22 * len(counts)))
        return fig
    fig = go.Figure(go.Histogram(x=col.dropna(), nbinsx=60, marker_color="#2a78d6",
                                 marker_line=dict(color="white", width=1)))
    missing = col.isna().mean() * 100
    return base_layout(fig, f"{feature} (train set) — {missing:.1f}% missing before imputation",
                       feature, "listings")


def feature_vs_price(feature: str) -> go.Figure:
    col = train_feats[feature]
    if col.dtype == object or col.nunique() <= 25:
        order = train_feats.groupby(col.astype(str))["price_eur"].median().sort_values().index
        fig = go.Figure()
        for cat in order:
            fig.add_trace(go.Box(y=train_feats.loc[col.astype(str) == cat, "price_eur"], name=cat,
                                 marker_color="#2a78d6", boxpoints=False, showlegend=False))
        fig.update_yaxes(type="log")
        return base_layout(fig, f"Price by {feature}", feature, "price (€, log scale)")
    fig = go.Figure(go.Scattergl(x=col, y=train_feats["price_eur"], mode="markers",
                                 marker=dict(size=5, color="#2a78d6", opacity=0.3)))
    fig.update_yaxes(type="log")
    return base_layout(fig, f"Price vs {feature}", feature, "price (€, log scale)")


# ---------------------------------------------------------------------------
# Tab 3 — overpricing detector (the business decision)
# ---------------------------------------------------------------------------
def detector(model_name: str, threshold_pct: float, review_cost: float):
    fair = preds_eur[model_name]
    ratio = actual / fair - 1                     # +0.5 = 50% above fair price
    flagged = ratio > threshold_pct / 100

    n_flag = int(flagged.sum())
    excess = (actual - fair)[flagged]
    total_cost = n_flag * review_cost
    summary = (
        f"### {n_flag:,} of {len(actual):,} test listings flagged ({n_flag / len(actual):.1%})\n"
        f"- Rule: flag if actual price is more than **{threshold_pct:.0f}%** above the fair price "
        f"predicted by **{model_name}**\n"
        f"- Median excess of flagged listings: **€{np.median(excess) if n_flag else 0:,.0f} per night**\n"
        f"- Review workload: {n_flag:,} × €{review_cost:,.0f} = **€{total_cost:,.0f}**"
    )

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=fair[~flagged], y=actual[~flagged], mode="markers", name="not flagged",
                               marker=dict(size=6, color="#b8b7b2", opacity=0.5),
                               hovertemplate="fair €%{x:.0f}<br>actual €%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scattergl(x=fair[flagged], y=actual[flagged], mode="markers", name="flagged as overpriced",
                               marker=dict(size=8, color="#e34948", opacity=0.8, line=dict(width=1, color="white")),
                               customdata=test_info["name"].fillna("").str[:40][flagged],
                               hovertemplate="fair €%{x:.0f}<br>actual €%{y:.0f}<br>%{customdata}<extra></extra>"))
    xs = np.array([AX_LO, AX_HI])
    fig.add_trace(go.Scatter(x=xs, y=xs * (1 + threshold_pct / 100), mode="lines", name="flagging line",
                             line=dict(color="#0b0b0b", dash="dash", width=2), hoverinfo="skip"))
    fig.update_xaxes(type="log", range=[np.log10(AX_LO), np.log10(AX_HI)])
    fig.update_yaxes(type="log", range=[np.log10(AX_LO), np.log10(AX_HI)])
    fig = base_layout(fig, "Fair (predicted) vs actual price — points above the line are flagged",
                      "fair price (€)", "actual price (€)")

    table = pd.DataFrame({
        "listing": test_info["name"], "district": test_info["neighbourhood_group_cleansed"],
        "guests": test_info["accommodates"], "actual €": actual.round(0), "fair €": fair.round(0),
        "% above fair": (ratio * 100).round(0),
    })[flagged].sort_values("% above fair", ascending=False).head(25)
    return summary, fig, table


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
hp = metrics["chosen_hyperparameters"]
with gr.Blocks(title="Madrid Airbnb overpricing detector") as demo:
    gr.Markdown(
        "# Madrid Airbnb — fair-price model & overpricing detector\n"
        "Linear regression on log(price) for entire-home listings, trained three ways. "
        "Everything below uses the models saved by `train.py`; nothing is retrained here."
    )

    with gr.Tab("1 · Compare the three models"):
        gr.Markdown(f"**Test-set results** (split by host · Ridge α={hp['ridge_alpha']} · "
                    f"PyTorch λ=α/n={hp['l2_lambda_torch']:.2e} · lr manual={hp['lr_manual']}, "
                    f"lr Adam={hp['lr_standard']} · {hp['epochs']} epochs)")
        gr.Dataframe(metrics_table(), interactive=False)
        model_pick = gr.CheckboxGroup(MODELS, value=MODELS, label="Models to show")
        pva = gr.Plot(pred_vs_actual(MODELS))
        model_pick.change(pred_vs_actual, model_pick, pva)
        gr.Plot(coef_plot())
        gr.Plot(loss_plot())

    with gr.Tab("2 · Distributions"):
        scale = gr.Radio(["price (€)", "log(price)"], value="price (€)", label="Target scale")
        tplot = gr.Plot(target_hist("price (€)"))
        scale.change(target_hist, scale, tplot)
        feat = gr.Dropdown(FEATURES, value="log_dist_to_sol_km", label="Feature")
        with gr.Row():
            fplot = gr.Plot(feature_hist("log_dist_to_sol_km"))
            rplot = gr.Plot(feature_vs_price("log_dist_to_sol_km"))
        feat.change(feature_hist, feat, fplot)
        feat.change(feature_vs_price, feat, rplot)

    with gr.Tab("3 · Overpricing detector"):
        with gr.Row():
            det_model = gr.Dropdown(MODELS, value="scikit-learn Ridge", label="Model giving the fair price")
            thr = gr.Slider(0, 300, value=50, step=5, label="Flag if actual price is more than X% above fair price")
            cost = gr.Number(value=25, label="Cost of one manual review (€)")
        s0, f0, t0 = detector("scikit-learn Ridge", 50, 25)
        det_summary = gr.Markdown(s0)
        det_plot = gr.Plot(f0)
        gr.Markdown("**Most overpriced flagged listings**")
        det_table = gr.Dataframe(t0, interactive=False)
        for ctrl in (det_model, thr, cost):
            ctrl.change(detector, [det_model, thr, cost], [det_summary, det_plot, det_table])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
