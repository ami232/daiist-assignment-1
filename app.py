"""
Assignment 1 — Gradio dashboard.

Loads ONLY what train.ipynb saved in artifacts/ (three trained models, the fitted
preprocessing parameters, the split data) and never retrains anything. Probabilities
are recomputed here by the loaded models from the saved preprocessing parameters.

Run:  uv run python main.py app
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
from sklearn.metrics import confusion_matrix, roc_auc_score

ART = Path("artifacts")
if not ART.exists():
    raise SystemExit("artifacts/ not found — run `uv run python main.py train` first.")

# --------------------------------------------------------------------------- load
metrics = json.load(open(ART / "metrics.json"))
prep = joblib.load(ART / "preprocessor.joblib")
sk_model = joblib.load(ART / "sklearn_model.joblib")
manual = torch.load(ART / "torch_manual.pt")
coef = pd.read_csv(ART / "coefficients.csv")
loss_hist = pd.read_csv(ART / "loss_history.csv")
data = pd.read_csv(ART / "predictions.csv")
live = pd.read_csv(ART / "live_predictions.csv")


class LogReg(torch.nn.Module):  # same architecture as in train.ipynb
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


std_model = LogReg(len(prep["columns"]))
std_model.load_state_dict(torch.load(ART / "torch_standard.pt"))
std_model.eval()


def transform(df: pd.DataFrame) -> np.ndarray:
    """Re-apply the preprocessing fitted on the training set (no refitting)."""
    out = df[prep["numeric"]].clip(prep["low"], prep["high"], axis=1)
    for c in prep["missing_cols"]:
        out[f"{c}_missing"] = df[c].isna().astype(float)
    out = out.fillna(prep["median"])
    for c in prep["region_cols"]:
        out[c] = (df["region"] == c.removeprefix("region_")).astype(float)
    return prep["scaler"].transform(out[prep["columns"]].to_numpy(dtype=np.float64))


def predict_all(df: pd.DataFrame) -> pd.DataFrame:
    X = transform(df)
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        df = df.copy()
        df["p_sklearn"] = sk_model.predict_proba(X)[:, 1]
        df["p_torch_manual"] = torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy()
        df["p_torch_standard"] = torch.sigmoid(std_model(Xt)).numpy()
    return df


saved = data[["p_sklearn", "p_torch_manual", "p_torch_standard"]].to_numpy()
data = predict_all(data.drop(columns=["p_sklearn", "p_torch_manual", "p_torch_standard"]))
live = predict_all(live.drop(columns=["p_sklearn", "p_torch_manual", "p_torch_standard"]))
assert np.allclose(saved, data[["p_sklearn", "p_torch_manual", "p_torch_standard"]], atol=1e-4), \
    "loaded models do not reproduce training-time predictions"

MODELS = {"scikit-learn": "p_sklearn", "PyTorch manual loop": "p_torch_manual",
          "PyTorch nn.Module + optim": "p_torch_standard"}
COLORS = {"scikit-learn": "#1f77b4", "PyTorch manual loop": "#ff7f0e",
          "PyTorch nn.Module + optim": "#2ca02c"}
FEATURES = prep["numeric"]
BEST_THR = metrics["best_threshold"]
COST_FP, COST_FN = metrics["cost_fp"], metrics["cost_fn"]


def split(name):
    return data[data["split"] == name]


# --------------------------------------------------------------------------- tab 1
def results_table():
    rows = []
    for name, r in metrics["test_results"].items():
        rows.append({"model": name, "ROC AUC": r["roc_auc"], "PR AUC": r["pr_auc"],
                     "log-loss": r["log_loss"], "precision": r["precision"],
                     "recall": r["recall"], "cost (EUR M)": r["cost_eur"] / 1e6})
    rows.append({"model": "send team to ALL countries",
                 "cost (EUR M)": metrics["cost_send_all"] / 1e6})
    rows.append({"model": "send team to NO country",
                 "cost (EUR M)": metrics["cost_send_none"] / 1e6})
    return pd.DataFrame(rows).round(3)


def coef_plot():
    c = coef[coef["feature"] != "(intercept)"].copy()
    c = c.reindex(c["sklearn"].abs().sort_values().index)
    fig = go.Figure()
    for col, label in [("sklearn", "scikit-learn"), ("torch_manual", "PyTorch manual loop"),
                       ("torch_standard", "PyTorch nn.Module + optim")]:
        fig.add_bar(y=c["feature"], x=c[col], name=label, orientation="h",
                    marker_color=COLORS[label])
    fig.update_layout(barmode="group", height=900, title="Standardised coefficients "
                      "(positive → raises probability of a >5% increase)")
    return fig


def loss_plot():
    fig = px.line(loss_hist, x="epoch", y=["torch_manual", "torch_standard"],
                  title="Training loss of the two PyTorch implementations (overlapping = identical)")
    fig.update_layout(yaxis_title="BCE + L2 loss")
    return fig


# --------------------------------------------------------------------------- tab 2
def pred_vs_actual(split_name, n_bins):
    d = split(split_name)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                    line=dict(dash="dash", color="grey"))
    for label, col in MODELS.items():
        bins = pd.qcut(d[col], int(n_bins), duplicates="drop")
        cal = d.groupby(bins, observed=True).agg(pred=(col, "mean"), actual=("target", "mean"),
                                                 n=("target", "size"))
        fig.add_scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name=label,
                        marker_color=COLORS[label], text=cal["n"],
                        hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>n=%{text}")
    fig.update_layout(title=f"Predicted probability vs actual share of increases ({split_name})",
                      xaxis_title="mean predicted probability (bin)",
                      yaxis_title="actual share with >5% real growth", height=500)

    agree = px.scatter(d, x="p_sklearn", y=["p_torch_manual", "p_torch_standard"], opacity=0.5,
                       title="Model agreement: PyTorch probabilities vs scikit-learn")
    agree.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"),
                      name="y = x")
    agree.update_layout(xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability")

    by_year = d.groupby("year").agg(actual=("target", "mean"),
                                    **{k: (v, "mean") for k, v in MODELS.items()}).reset_index()
    yr = px.line(by_year, x="year", y=["actual"] + list(MODELS), markers=True,
                 title="Average predicted probability vs actual rate, by year")
    return fig, agree, yr


# --------------------------------------------------------------------------- tab 3
def distributions(feature):
    d = data[data["split"] == "train"].copy()
    d["next year"] = d["target"].map({0: "no >5% increase", 1: ">5% increase"})
    h = px.histogram(d, x=feature, color="next year", barmode="overlay", nbins=60,
                     histnorm="probability density", opacity=0.6,
                     title=f"Distribution of {feature} by outcome (training set)")
    lo, hi = d[feature].quantile([0.01, 0.99])
    h.update_xaxes(range=[lo, hi])
    t = (data.groupby(["year", "split"])["target"].mean().reset_index())
    tr = px.bar(t, x="year", y="target", color="split",
                title="Target: share of countries with >5% real growth next year")
    tr.update_layout(yaxis_title="share positive")
    g = px.histogram(data[data["next_real_growth"].between(-0.6, 1.5)], x="next_real_growth",
                     nbins=100, title="Raw next-year real growth (dashed line = 5% label cut-off)")
    g.add_vline(x=metrics["growth_threshold"], line_dash="dash")
    return h, tr, g


# --------------------------------------------------------------------------- tab 4
def threshold_view(model_label, thr, cost_fp_k, cost_fn_k, split_name):
    d = split(split_name)
    col = MODELS[model_label]
    y, p = d["target"].to_numpy(), d[col].to_numpy()
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cfp, cfn = cost_fp_k * 1e3, cost_fn_k * 1e3
    cost = fp * cfp + fn * cfn
    cost_all = (y == 0).sum() * cfp
    cost_none = (y == 1).sum() * cfn
    best_naive = min(cost_all, cost_none)

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["predicted: no increase", "predicted: increase"],
        y=["actual: no increase", "actual: increase"], colorscale="Blues", showscale=False,
        text=[[f"TN<br>{tn}", f"FP<br>{fp}<br>wasted trips"],
              [f"FN<br>{fn}<br>missed deals", f"TP<br>{tp}"]],
        texttemplate="%{text}", textfont={"size": 18}))
    cm.update_layout(title=f"Confusion matrix — {model_label}, threshold {thr:.2f}, {split_name}",
                     yaxis_autorange="reversed", height=420)

    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [(((p >= t) & (y == 0)).sum() * cfp + ((p < t) & (y == 1)).sum() * cfn) / 1e6
             for t in grid]
    curve = go.Figure()
    curve.add_scatter(x=grid, y=costs, mode="lines", name="model")
    curve.add_hline(y=cost_all / 1e6, line_dash="dot", annotation_text="send to all")
    curve.add_hline(y=cost_none / 1e6, line_dash="dot", annotation_text="send to none")
    curve.add_vline(x=thr, line_color="red", annotation_text="your threshold")
    curve.update_layout(title="Total business cost vs threshold", xaxis_title="threshold",
                        yaxis_title="cost (EUR millions)", height=420)

    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    saving = 1 - cost / best_naive
    md = (f"### Business cost: **EUR {cost/1e6:,.1f} M**\n"
          f"- {fp} wasted campaigns × EUR {cost_fp_k:,.0f}k + {fn} missed opportunities × "
          f"EUR {cost_fn_k:,.0f}k\n"
          f"- Best naive strategy (send to all / none): EUR {best_naive/1e6:,.1f} M → "
          f"model {'saves' if saving > 0 else 'LOSES'} **{abs(saving):.1%}**\n"
          f"- Teams sent: {tp + fp} of {len(y)} country-years · precision {prec:.2f} · "
          f"recall {rec:.2f}\n"
          f"- Theoretical optimum for calibrated probabilities: "
          f"{cfp/(cfp+cfn):.2f} · validation-chosen threshold: {BEST_THR:.2f}")
    return cm, curve, md


# --------------------------------------------------------------------------- tab 5
def outlook(model_label, thr):
    col = MODELS[model_label]
    t = live[["country", "region", "milex_gdp", "real_growth", "gdp_growth", col]].copy()
    t = t.rename(columns={col: "P(>5% real increase in 2025)", "milex_gdp": "milex % GDP 2024",
                          "real_growth": "real growth 2024", "gdp_growth": "GDP growth 2024"})
    t["send team?"] = np.where(t["P(>5% real increase in 2025)"] >= thr, "YES", "")
    return t.sort_values("P(>5% real increase in 2025)", ascending=False).round(3)


# --------------------------------------------------------------------------- UI
with gr.Blocks(title="Military spending — sales targeting") as demo:
    gr.Markdown(
        "# Will a country raise real military spending by >5% next year?\n"
        "Logistic regression trained three ways (scikit-learn, manual PyTorch loop, "
        "nn.Module + optim) on SIPRI/World Bank data. Train 1990–2010 · validation "
        "2011–2016 · test 2017–2023. **All models are loaded from `artifacts/`; nothing "
        "is retrained.**")

    with gr.Tab("1 · Model comparison"):
        gr.Markdown("Test-set results at the validation-chosen threshold, vs naive baselines.")
        gr.Dataframe(results_table(), interactive=False)
        gr.Plot(coef_plot())
        gr.Plot(loss_plot())

    with gr.Tab("2 · Predicted vs actual"):
        with gr.Row():
            s2 = gr.Radio(["train", "val", "test"], value="test", label="Split")
            b2 = gr.Slider(4, 20, value=10, step=1, label="Number of probability bins")
        p2a, p2b, p2c = gr.Plot(), gr.Plot(), gr.Plot()
        for comp in (s2, b2):
            comp.change(pred_vs_actual, [s2, b2], [p2a, p2b, p2c])
        demo.load(pred_vs_actual, [s2, b2], [p2a, p2b, p2c])

    with gr.Tab("3 · Distributions"):
        f3 = gr.Dropdown(FEATURES, value="gdp_growth", label="Feature")
        p3a, p3b, p3c = gr.Plot(), gr.Plot(), gr.Plot()
        f3.change(distributions, f3, [p3a, p3b, p3c])
        demo.load(distributions, f3, [p3a, p3b, p3c])

    with gr.Tab("4 · Threshold & business cost"):
        with gr.Row():
            m4 = gr.Dropdown(list(MODELS), value="scikit-learn", label="Model")
            s4 = gr.Radio(["val", "test"], value="test", label="Split")
        t4 = gr.Slider(0.01, 0.99, value=BEST_THR, step=0.01, label="Decision threshold")
        with gr.Row():
            fp4 = gr.Number(value=COST_FP / 1e3, label="Cost of a wasted campaign (FP), EUR k")
            fn4 = gr.Number(value=COST_FN / 1e3, label="Cost of a missed opportunity (FN), EUR k")
        md4 = gr.Markdown()
        with gr.Row():
            cm4, cv4 = gr.Plot(), gr.Plot()
        ins = [m4, t4, fp4, fn4, s4]
        for comp in ins:
            comp.change(threshold_view, ins, [cm4, cv4, md4])
        demo.load(threshold_view, ins, [cm4, cv4, md4])

    with gr.Tab("5 · 2025 outlook"):
        gr.Markdown("Countries ranked by predicted probability, using their 2024 data "
                    "(no labels yet — this is what the sales team would act on).")
        with gr.Row():
            m5 = gr.Dropdown(list(MODELS), value="scikit-learn", label="Model")
            t5 = gr.Slider(0.01, 0.99, value=BEST_THR, step=0.01, label="Threshold")
        tab5 = gr.Dataframe(interactive=False)
        for comp in (m5, t5):
            comp.change(outlook, [m5, t5], tab5)
        demo.load(outlook, [m5, t5], tab5)


if __name__ == "__main__":
    demo.launch()
