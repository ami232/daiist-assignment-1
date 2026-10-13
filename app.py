"""Assignment 1 - Gradio dashboard. Loads saved artifacts; never retrains anything."""
import inspect
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
if not (ART / "meta.joblib").exists():
    raise SystemExit("artifacts/ not found. Run `uv run python main.py train` first.")

# ---------- load saved artifacts (no training here) ----------
meta = joblib.load(ART / "meta.joblib")
FN_COST, FP_COST = meta["fn_cost"], meta["fp_cost"]
THRESHOLD = meta["thresholds"]
splits = np.load(ART / "splits.npz")
X_test, y_test = splits["X_test"], splits["y_test"]
results = pd.read_csv(ART / "results.csv").round(3)
raw = pd.read_csv(ROOT / "data" / "hmeq.csv")

sk_model = joblib.load(ART / "sklearn_model.joblib")
manual = torch.load(ART / "manual_torch.pt", weights_only=True)


class LogReg(nn.Module):  # same architecture as in train
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


nn_model = LogReg(X_test.shape[1])
nn_model.load_state_dict(torch.load(ART / "nn_torch.pt", weights_only=True))
nn_model.eval()

Xs = torch.tensor(X_test)
with torch.no_grad():
    PROBS = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "manual_torch": torch.sigmoid(Xs @ manual["w"] + manual["b"]).numpy(),
        "nn_torch": torch.sigmoid(nn_model(Xs)).numpy(),
    }
MODELS = list(PROBS)

# one colour per model, used in every chart
COLORS = {"sklearn": "#4F46E5", "manual_torch": "#10B981", "nn_torch": "#F59E0B"}
REPAID, DEFAULTED = "#3B82F6", "#EF4444"


def style(fig, title):
    fig.update_layout(title=title, template="plotly_white", margin=dict(t=60, b=40))
    return fig


def confusion(p, t):
    reject = p >= t
    return (int((~reject & (y_test == 0)).sum()), int((reject & (y_test == 0)).sum()),
            int((~reject & (y_test == 1)).sum()), int((reject & (y_test == 1)).sum()))


# ---------- summary cards ----------
def kpi_html():
    p, t = PROBS["sklearn"], THRESHOLD["sklearn"]
    tn, fp, fn, tp = confusion(p, t)
    cost = (FN_COST * fn + FP_COST * fp) / len(y_test)
    base = FP_COST * (y_test == 0).sum() / len(y_test)      # naive rule: reject everyone
    gap = max(np.abs(PROBS[a] - PROBS[b]).max() for a in MODELS for b in MODELS)
    cards = [
        ("AUC (test)", f"{roc_auc_score(y_test, p):.2f}", "0.5 = coin flip, 1 = perfect"),
        ("Cost per applicant", f"{cost:,.0f} EUR", f"naive rule: {base:,.0f} EUR"),
        ("Defaults caught", f"{tp / (tp + fn):.0%}", f"{tp} of {tp + fn} defaulters"),
        ("3 models agree", f"{gap:.4f}", "max gap in predicted probability"),
    ]
    html = "".join(
        "<div style='flex:1;min-width:180px;padding:14px 16px;border-radius:12px;"
        "border:1px solid var(--border-color-primary);background:var(--background-fill-secondary)'>"
        f"<div style='font-size:13px;opacity:.7'>{a}</div>"
        f"<div style='font-size:28px;font-weight:700'>{b}</div>"
        f"<div style='font-size:12px;opacity:.7'>{c}</div></div>" for a, b, c in cards)
    return f"<div style='display:flex;gap:12px;flex-wrap:wrap'>{html}</div>"


# ---------- tab 1: predicted vs actual ----------
def calibration_plot():
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                               line=dict(dash="dash", color="grey")))
    for name, p in PROBS.items():
        frac, mean_p = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        fig.add_trace(go.Scatter(x=mean_p, y=frac, mode="lines+markers", name=name,
                                 line=dict(color=COLORS[name])))
    fig.update_layout(xaxis_title="Predicted default probability",
                      yaxis_title="Actual default rate")
    return style(fig, "Predicted vs actual (test set)")


# ---------- tab 2: distributions ----------
def distribution_plot(feature):
    d = raw.copy()
    d["Outcome"] = d["BAD"].map({0: "Repaid", 1: "Defaulted"})
    cmap = {"Repaid": REPAID, "Defaulted": DEFAULTED}
    if feature == "BAD (target)":
        c = d["Outcome"].value_counts().reset_index()
        c.columns = ["Outcome", "count"]
        fig = px.bar(c, x="Outcome", y="count", color="Outcome", color_discrete_map=cmap)
    elif raw[feature].dtype == object:
        d[feature] = d[feature].fillna("Missing")
        fig = px.histogram(d, x=feature, color="Outcome", barmode="group", color_discrete_map=cmap)
    else:
        fig = px.histogram(d, x=feature, color="Outcome", barmode="overlay", opacity=0.65,
                           nbins=40, color_discrete_map=cmap)
    return style(fig, f"{feature}: repaid vs defaulted")


# ---------- tab 3: threshold and cost ----------
def threshold_view(model_name, t):
    p = PROBS[model_name]
    tn, fp, fn, tp = confusion(p, t)
    cost = FN_COST * fn + FP_COST * fp

    cm = px.imshow([[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
                   x=["Approve", "Reject"], y=["Repaid", "Defaulted"],
                   labels=dict(x="Bank decision", y="What really happened"))
    cm.update_coloraxes(showscale=False)
    style(cm, f"Confusion matrix at threshold {t:.2f}")

    ts = np.linspace(0.01, 0.99, 99)
    ys = [(FN_COST * ((p < x) & (y_test == 1)).sum()
           + FP_COST * ((p >= x) & (y_test == 0)).sum()) / len(y_test) for x in ts]
    curve = go.Figure(go.Scatter(x=ts, y=ys, mode="lines", name="cost",
                                 line=dict(color=COLORS[model_name])))
    curve.add_trace(go.Scatter(x=[t], y=[cost / len(y_test)], mode="markers", name="selected",
                               marker=dict(size=13, color=DEFAULTED)))
    curve.add_vline(x=FP_COST / (FP_COST + FN_COST), line_dash="dash", line_color="grey",
                    annotation_text="theory: FP/(FP+FN)")
    curve.update_layout(xaxis_title="Threshold", yaxis_title="EUR per applicant")
    style(curve, "Expected cost per applicant vs threshold")

    summary = (f"### {cost:,.0f} EUR total ({cost / len(y_test):,.0f} per applicant)\n"
               f"Approved defaulters: **{fn}** x {FN_COST:,} EUR  |  "
               f"Rejected good borrowers: **{fp}** x {FP_COST:,} EUR")
    return cm, summary, curve


# ---------- UI ----------
THEME = gr.themes.Soft(primary_hue="indigo")
# Gradio 6 moved the theme from Blocks() to launch(); support both
_in_blocks = "theme" in inspect.signature(gr.Blocks.__init__).parameters
blocks_kw = {"theme": THEME} if _in_blocks else {}
launch_kw = {} if _in_blocks else {"theme": THEME}

with gr.Blocks(title="HMEQ default risk", **blocks_kw) as demo:
    gr.Markdown("# Home-equity loan default risk\n"
                "One logistic regression built three ways, used to **approve or reject** a loan.")
    gr.HTML(kpi_html())

    with gr.Tab("Models"):
        gr.Dataframe(value=results, interactive=False)
        gr.Plot(value=calibration_plot())

    with gr.Tab("Distributions"):
        choices = ["BAD (target)"] + [c for c in raw.columns if c != "BAD"]
        feat = gr.Dropdown(choices, value="BAD (target)", label="Column")
        dist = gr.Plot(value=distribution_plot("BAD (target)"))
        feat.change(distribution_plot, feat, dist)

    with gr.Tab("Threshold & cost"):
        model_dd = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
        slider = gr.Slider(0.01, 0.99, value=round(THRESHOLD[MODELS[0]], 2), step=0.01,
                           label="Threshold: reject if P(default) >= threshold")
        with gr.Row():
            cm_out, curve_out = gr.Plot(), gr.Plot()
        txt = gr.Markdown()
        ins, outs = [model_dd, slider], [cm_out, txt, curve_out]
        for c in ins:
            c.change(threshold_view, ins, outs)
        demo.load(threshold_view, ins, outs)

if __name__ == "__main__":
    demo.launch(**launch_kw)