"""
Assignment 1 - Gradio dashboard (inference only).

Loads what train.ipynb saved in artifacts/ (three trained models, the splits,
the thresholds and the FPR limit). It never trains anything.

Page layout (one page, four sections):
  1. Feature and target distributions
  2. The three models compared (calibration = predicted vs. actual, ROC)
  3. Decision threshold slider: confusion matrix + FPR vs. the COMPAS limit
  4. Audit by race and sex (race is never a model input)
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

# --------------------------------------------------------------------------
# Load artifacts (no training here)
# --------------------------------------------------------------------------
ART = Path(__file__).resolve().parent / "artifacts"
cfg = json.loads((ART / "config.json").read_text())
FEATURES = json.loads((ART / "feature_cols.json").read_text())
SPLITS = {n: pd.read_csv(ART / "splits" / f"{n}.csv", parse_dates=["compas_screening_date"])
          for n in ("train", "val", "test")}
METRICS = pd.read_csv(ART / "metrics.csv")

LIMIT = cfg["limit_fpr"]          # FPR of COMPAS (decile >= 5) on validation
COMPAS_CUT = cfg["compas_cut"]
SAVED_THR = cfg["thresholds"]     # per model, fixed on validation only

sk_model = joblib.load(ART / "models" / "sklearn_logreg.joblib")
manual = torch.load(ART / "models" / "manual_torch.pt")
std_model = nn.Linear(len(FEATURES), 1)
std_model.load_state_dict(torch.load(ART / "models" / "standard_torch.pt"))
std_model.eval()


def _to_tensor(df):
    return torch.tensor(df[FEATURES].to_numpy(dtype="float32"))


def predict(model, df):
    """Probability of reoffending for every row of df."""
    if model == "sklearn":
        return sk_model.predict_proba(df[FEATURES])[:, 1]
    with torch.no_grad():
        X = _to_tensor(df)
        if model == "manual_torch":
            return torch.sigmoid(X @ manual["w"] + manual["b"]).numpy()
        return torch.sigmoid(std_model(X).squeeze(1)).numpy()


MODELS = ["sklearn", "manual_torch", "standard_torch"]
MODEL_CHOICES = [("scikit-learn", "sklearn"),
                 ("PyTorch, manual loop (SGD)", "manual_torch"),
                 ("PyTorch, nn.Module + Adam", "standard_torch")]
STYLE = {  # colour + dash + marker, so models are distinguishable without colour
    "sklearn": dict(color="#0072B2", dash="solid", symbol="circle"),
    "manual_torch": dict(color="#D55E00", dash="dash", symbol="square"),
    "standard_torch": dict(color="#009E73", dash="dot", symbol="diamond"),
}
LABEL = dict(MODEL_CHOICES[i][::-1] for i in range(3))

Y = {n: d["y"].to_numpy() for n, d in SPLITS.items()}
PROBS = {n: {m: predict(m, d) for m in MODELS} for n, d in SPLITS.items()}


def _span(n):
    d = SPLITS[n]["compas_screening_date"]
    return f"{d.min():%b %Y} to {d.max():%b %Y}"


SPLIT_CHOICES = [(f"Validation ({_span('val')})", "val"), (f"Test ({_span('test')})", "test")]
BIG_RACES = ["African-American", "Caucasian", "Hispanic"]
SMALL_RACES = ["Other", "Asian", "Native American"]

# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def confusion(y, pred):
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
    return tn, fp, fn, tp


def rates(y, pred):
    tn, fp, fn, tp = confusion(y, pred)
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    return fpr, rec


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def pct(x):
    return "n/a" if x is None or np.isnan(x) else f"{x:.1%}"


# --------------------------------------------------------------------------
# Section 1: distributions
# --------------------------------------------------------------------------
SPLIT_FILL = {"train": ("#7f7f7f", ""), "val": ("#56B4E9", "/"), "test": ("#E69F00", "x")}


def fig_features():
    fig = go.Figure()
    for name, df in SPLITS.items():
        color, shape = SPLIT_FILL[name]
        fig.add_trace(go.Bar(x=FEATURES, y=[df[f].mean() for f in FEATURES], name=name,
                             marker=dict(color=color, pattern=dict(shape=shape))))
    fig.update_layout(template="plotly_white", barmode="group", height=430,
                      title="Share of people with each feature = 1, by split",
                      yaxis=dict(title="Share", tickformat=".0%"), xaxis=dict(tickangle=-40),
                      legend_title="Split", margin=dict(b=140))
    return fig


def fig_target():
    fig = go.Figure()
    names = list(SPLITS)
    neg = [int((Y[n] == 0).sum()) for n in names]
    pos = [int((Y[n] == 1).sum()) for n in names]
    fig.add_trace(go.Bar(x=names, y=neg, name="Did not reoffend",
                         marker=dict(color="#56B4E9", pattern=dict(shape="")),
                         text=neg, textposition="inside"))
    fig.add_trace(go.Bar(x=names, y=pos, name="Reoffended within 2 years",
                         marker=dict(color="#D55E00", pattern=dict(shape="/")),
                         text=[f"{p} ({p / (p + n):.1%})" for p, n in zip(pos, neg)],
                         textposition="inside"))
    fig.update_layout(template="plotly_white", barmode="stack", height=380,
                      title="Target (two_year_recid): people per split",
                      yaxis_title="People", legend=dict(orientation="h", y=-0.2))
    return fig


# --------------------------------------------------------------------------
# Section 2: model comparison
# --------------------------------------------------------------------------
def fig_calibration(split):
    y = Y[split]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color="#999999", dash="dashdot")))
    for m in MODELS:
        frac, mean = calibration_curve(y, PROBS[split][m], n_bins=8, strategy="quantile")
        s = STYLE[m]
        fig.add_trace(go.Scatter(x=mean, y=frac, mode="lines+markers", name=LABEL[m],
                                 line=dict(color=s["color"], dash=s["dash"]),
                                 marker=dict(symbol=s["symbol"], size=8)))
    fig.update_layout(template="plotly_white", height=430,
                      title=f"Predicted vs. actual ({split}): 8 equal-size probability bins",
                      xaxis=dict(title="Mean predicted probability", range=[0, 1]),
                      yaxis=dict(title="Actual share who reoffended", range=[0, 1]),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def fig_roc(split):
    y = Y[split]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color="#999999", dash="dashdot")))
    for m in MODELS:
        fpr, tpr, _ = roc_curve(y, PROBS[split][m])
        s = STYLE[m]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=LABEL[m],
                                 line=dict(color=s["color"], dash=s["dash"])))
        f, r = rates(y, (PROBS[split][m] >= SAVED_THR[m]).astype(int))
        fig.add_trace(go.Scatter(x=[f], y=[r], mode="markers", showlegend=False,
                                 marker=dict(symbol=s["symbol"], size=11, color=s["color"],
                                             line=dict(color="black", width=1))))
    score = SPLITS[split]["decile_score"].to_numpy()
    fpr, tpr, _ = roc_curve(y, score)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="COMPAS decile score (reference only)",
                             line=dict(color="#333333", width=2, dash="longdash")))
    f, r = rates(y, (score >= COMPAS_CUT).astype(int))
    fig.add_trace(go.Scatter(x=[f], y=[r], mode="markers", showlegend=False,
                             marker=dict(symbol="x", size=12, color="#333333")))
    fig.add_vline(x=LIMIT, line_dash="dash", line_color="#CC79A7",
                  annotation_text="FPR limit", annotation_position="top")
    fig.update_layout(template="plotly_white", height=430,
                      title=f"ROC ({split}); markers = saved thresholds, x = COMPAS decile >= {COMPAS_CUT}",
                      xaxis=dict(title="False positive rate", range=[0, 1]),
                      yaxis=dict(title="Recall (true positive rate)", range=[0, 1]),
                      legend=dict(orientation="h", y=-0.3))
    return fig


def metrics_table():
    t = METRICS.copy()
    t["model"] = t["model"].replace({"sklearn": "scikit-learn", "manual_torch": "PyTorch (manual)",
                                     "standard_torch": "PyTorch (nn.Module)"})
    for c in t.columns[1:]:
        t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return t.rename(columns={"model": "Model", "threshold": "Threshold", "AUC_val": "AUC val",
                             "AUC_test": "AUC test", "FPR_val": "FPR val", "recall_val": "Recall val",
                             "FPR_test": "FPR test", "recall_test": "Recall test"})


def compare(split):
    ps = [PROBS[split][m] for m in MODELS]
    gap = max(np.abs(a - b).max() for i, a in enumerate(ps) for b in ps[i + 1:])
    note = (f"Largest difference in predicted probability between any two models on {split}: "
            f"**{gap:.4f}**. If the curves overlap, the three methods learned the same model.")
    return fig_calibration(split), fig_roc(split), note


# --------------------------------------------------------------------------
# Sections 3 and 4: threshold, confusion matrix, audit
# --------------------------------------------------------------------------
def fig_confusion(y, pred):
    tn, fp, fn, tp = confusion(y, pred)
    z = [[tn, fp], [fn, tp]]
    text = [[f"True negatives<br><b>{tn}</b>", f"False positives<br><b>{fp}</b>"],
            [f"False negatives<br><b>{fn}</b>", f"True positives<br><b>{tp}</b>"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted low risk", "Predicted high risk"],
                               y=["Did not reoffend", "Reoffended"], text=text,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=360, title="Confusion matrix",
                      margin=dict(t=60))
    return fig


def group_table(df, y, pred, col, groups, note=""):
    rows = []
    for g in groups:
        m = (df[col] == g).to_numpy()
        n = int(m.sum())
        neg = int((y[m] == 0).sum())
        pos = n - neg
        fp = int(((pred[m] == 1) & (y[m] == 0)).sum())
        tp = int(((pred[m] == 1) & (y[m] == 1)).sum())
        lo, hi = wilson(fp, neg)
        row = {"Group": g, "People": n, "Non-reoffenders": neg, "Reoffenders": pos,
               "Flagged high risk": pct(pred[m].mean()) if n else "n/a",
               "FPR": pct(fp / neg if neg else np.nan),
               "FPR 95% interval": "n/a" if np.isnan(lo) else f"{lo:.1%} to {hi:.1%}",
               "Recall": pct(tp / pos if pos else np.nan)}
        if note:
            row["Reliability"] = note
        rows.append(row)
    return pd.DataFrame(rows)


def fig_audit(df, y, pred):
    items = [("race", g) for g in BIG_RACES] + [("sex", g) for g in ("Female", "Male")]
    labels, fprs, recs, err_lo, err_hi = [], [], [], [], []
    for col, g in items:
        m = (df[col] == g).to_numpy()
        neg = int((y[m] == 0).sum()); pos = int(m.sum()) - neg
        fp = int(((pred[m] == 1) & (y[m] == 0)).sum()); tp = int(((pred[m] == 1) & (y[m] == 1)).sum())
        f = fp / neg if neg else 0.0
        lo, hi = wilson(fp, neg)
        labels.append(f"{g}<br>(non-reoffenders: {neg})")
        fprs.append(f); recs.append(tp / pos if pos else 0.0)
        err_lo.append(0 if np.isnan(lo) else f - lo); err_hi.append(0 if np.isnan(hi) else hi - f)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=fprs, name="FPR (95% interval)",
                         marker=dict(color="#D55E00", pattern=dict(shape="/")),
                         error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo)))
    fig.add_trace(go.Bar(x=labels, y=recs, name="Recall",
                         marker=dict(color="#0072B2", pattern=dict(shape=""))))
    fig.add_hline(y=LIMIT, line_dash="dash", line_color="#CC79A7",
                  annotation_text="FPR limit (COMPAS, validation)", annotation_position="top left")
    fig.update_layout(template="plotly_white", barmode="group", height=430,
                      title="Error rates by group at the chosen threshold",
                      yaxis=dict(title="Rate", tickformat=".0%", range=[0, 1]),
                      legend=dict(orientation="h", y=-0.3))
    return fig


def update(model, split, thr):
    saved = SAVED_THR[model]
    snapped = abs(thr - saved) < 0.0005          # slider step is 0.001
    thr = saved if snapped else thr
    df, y = SPLITS[split], Y[split]
    pred = (PROBS[split][model] >= thr).astype(int)
    tn, fp, fn, tp = confusion(y, pred)
    fpr, rec = rates(y, pred)
    c_pred = (df["decile_score"].to_numpy() >= COMPAS_CUT).astype(int)
    c_fpr, c_rec = rates(y, c_pred)

    gap = fpr - LIMIT
    status = ("**within the limit**" if fpr <= LIMIT
              else f"**above the limit by {gap * 100:.1f} percentage points**")
    extra = ""
    if split == "val" and snapped:
        extra = " (on validation this equals the limit by construction, because the threshold was fixed here)"
    summary = (
        f"**Threshold {thr:.4f}** {'(saved threshold for this model)' if snapped else '(chosen with the slider)'}\n\n"
        f"- **Harm to people who would not reoffend:** {fp} of {fp + tn} wrongly flagged as high risk "
        f"= FPR **{pct(fpr)}**, against a limit of {pct(LIMIT)}: {status}{extra}.\n"
        f"- **Reoffenders detected:** {tp} of {tp + fn} = recall **{pct(rec)}**.\n"
        f"- **People flagged high risk:** {tp + fp} of {len(y)}.\n"
        f"- **COMPAS (decile >= {COMPAS_CUT}) on the same split:** FPR {pct(c_fpr)}, recall {pct(c_rec)}.\n\n"
        f"The limit is the FPR of COMPAS on validation. Recall is what this model detects while "
        f"staying inside that harm limit; no monetary price is put on a person's freedom."
    )
    race_tbl = group_table(df, y, pred, "race", BIG_RACES)
    small_tbl = group_table(df, y, pred, "race", SMALL_RACES, "Insufficient sample: do not draw conclusions")
    sex_tbl = group_table(df, y, pred, "sex", ["Female", "Male"])
    return (fig_confusion(y, pred), summary, fig_audit(df, y, pred), race_tbl, small_tbl, sex_tbl)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
INTRO = f"""
# Pretrial risk model: COMPAS data, three ways to train one logistic regression

**Scenario.** A model supports a judge's pretrial release decision by estimating the chance that a person
reoffends within two years. The threshold is set by a harm limit, not by a price: the false positive rate
(people who would not reoffend but are flagged high risk) may not exceed the FPR of COMPAS on validation
(**{LIMIT:.1%}**), and the model then tries to detect as many reoffenders as possible.

**Inputs.** The models use sex, age bins, prior-count bins, juvenile history, charge degree, charge group and two
timing indicators (17 features). **Race and the COMPAS score are never model inputs.** Race is used only in the
audit below, and the COMPAS score only as a reference.

Everything here is loaded from `artifacts/`; nothing is retrained. This is a coursework dashboard, not a tool for real decisions.
"""

with gr.Blocks(title="COMPAS pretrial risk: three-model dashboard") as demo:
    gr.Markdown(INTRO)

    gr.Markdown("## 1. Distributions")
    gr.Markdown("Splits are temporal (train, then validation, then test). Differences between the bars "
                "show how the population changes over time, for example violent charges are more common in test.")
    with gr.Row():
        gr.Plot(value=fig_features(), label="Features by split")
        gr.Plot(value=fig_target(), label="Target by split")

    gr.Markdown("## 2. The three models compared")
    split_cmp = gr.Radio(SPLIT_CHOICES, value="test", label="Data to evaluate on")
    with gr.Row():
        cal_plot = gr.Plot(label="Predicted vs. actual (calibration)")
        roc_plot = gr.Plot(label="ROC curves")
    diff_md = gr.Markdown()
    gr.Markdown("**Saved metrics.** Each model's threshold was set on validation so its FPR matches COMPAS's, "
                "then applied unchanged to test.")
    gr.Dataframe(value=metrics_table(), interactive=False, wrap=True)

    gr.Markdown("## 3. Decision threshold")
    with gr.Row():
        model_dd = gr.Dropdown(MODEL_CHOICES, value="sklearn", label="Model")
        split_rd = gr.Radio(SPLIT_CHOICES, value="test", label="Data to evaluate on")
    with gr.Row():
        thr_slider = gr.Slider(0.05, 0.95, step=0.001, value=round(SAVED_THR["sklearn"], 3),
                               label="Decision threshold (probability of reoffending)")
        reset_btn = gr.Button("Use saved threshold", scale=0)
    with gr.Row():
        conf_plot = gr.Plot(label="Confusion matrix")
        summary_md = gr.Markdown()

    gr.Markdown("## 4. Audit by group (uses the model, split and threshold chosen above)")
    gr.Markdown("Race is used here only to measure, never to predict. Rates for small groups are noisy, "
                "so every table shows the group size and a 95% interval for the FPR.")
    audit_plot = gr.Plot(label="FPR and recall by group")
    gr.Markdown("**By race: the three largest groups**")
    race_tbl = gr.Dataframe(interactive=False, wrap=True)
    gr.Markdown("**By race: groups too small for conclusions**")
    small_tbl = gr.Dataframe(interactive=False, wrap=True)
    gr.Markdown("**By sex**")
    sex_tbl = gr.Dataframe(interactive=False, wrap=True)

    controls = [model_dd, split_rd, thr_slider]
    outputs = [conf_plot, summary_md, audit_plot, race_tbl, small_tbl, sex_tbl]
    for c in controls:
        c.change(update, controls, outputs)
    reset_btn.click(lambda m: SAVED_THR[m], model_dd, thr_slider)
    split_cmp.change(compare, split_cmp, [cal_plot, roc_plot, diff_md])
    demo.load(update, controls, outputs)
    demo.load(compare, split_cmp, [cal_plot, roc_plot, diff_md])

demo.launch()
