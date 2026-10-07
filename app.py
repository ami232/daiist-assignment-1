"""
Assignment 1 — Gradio dashboard.

Run with `uv run python main.py app` (after `main.py train`).

Loads only what train.py saved in artifacts/ (models, fitted scaler, split,
thresholds, loss curves, metrics) plus the raw CSV. Nothing is retrained or
refitted here: predictions are made with the saved models at startup, and
everything else (confusion matrices, costs, curves) is computed from those
predictions as the user moves the controls.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from train import (
    BILL,
    PAY,
    PAY_AMT,
    TARGET,
    LogisticRegressionNet,
    engineer_features,
    load_raw,
    sigmoid,
)

ART = Path("artifacts")
if not (ART / "preprocessing.joblib").exists():
    raise SystemExit("artifacts/ not found: run `uv run python main.py train` first.")

# --------------------------------------------------------------------------- #
# Load artifacts (no training happens below this line)
# --------------------------------------------------------------------------- #
prep = joblib.load(ART / "preprocessing.joblib")
sk = joblib.load(ART / "sklearn_logreg.joblib")
sk_raw = joblib.load(ART / "sklearn_raw_features.joblib")
man = torch.load(ART / "manual_torch.pt")
std = LogisticRegressionNet(len(prep["features"]))
std.load_state_dict(torch.load(ART / "standard_torch.pt"))
std.eval()
summary = json.loads((ART / "metrics.json").read_text())
loss_hist = pd.read_csv(ART / "loss_history.csv", index_col="epoch")

COST_FN, COST_FP = prep["cost_fn"], prep["cost_fp"]
VAL_THRESHOLDS = prep["thresholds"]

raw = load_raw()
feats = engineer_features(raw)[prep["features"]]
X = prep["scaler"].transform(feats)
R = prep["raw_scaler"].transform(raw[prep["raw_features"]])
y = raw[TARGET].to_numpy()

w_man = man["w"].numpy()
b_man = float(man["b"])
with torch.no_grad():
    p_std_all = torch.sigmoid(std(torch.tensor(X, dtype=torch.float32))).numpy()

P_ALL = {
    "sklearn": sk.predict_proba(X)[:, 1],
    "manual_torch": sigmoid(X @ w_man + b_man),
    "standard_torch": p_std_all,
    "raw_features_sklearn": sk_raw.predict_proba(R)[:, 1],
    "baseline": np.full(len(y), prep["train_prevalence"]),
}
SPLITS = {"test": prep["idx_test"], "val": prep["idx_val"]}
SPLIT_NAME = {"test": "test set", "val": "validation set"}

# --------------------------------------------------------------------------- #
# Visual system: one palette, one Plotly template
# --------------------------------------------------------------------------- #
MODELS = ["sklearn", "manual_torch", "standard_torch"]
LABEL = {
    "sklearn": "scikit-learn",
    "manual_torch": "Manual PyTorch",
    "standard_torch": "Standard PyTorch",
    "raw_features_sklearn": "Raw features",
    "baseline": "Naive baseline",
}
# Models use the categorical slots; outcomes use muted / red so the two
# meanings never share a colour.
COLOR = {
    "sklearn": "#2a78d6",
    "manual_torch": "#eb6834",
    "standard_torch": "#1baf7a",
    "raw_features_sklearn": "#4a3aa7",
    "baseline": "#8a8984",
}
DASH = {"sklearn": "solid", "manual_torch": "dash", "standard_torch": "dot",
        "raw_features_sklearn": "solid", "baseline": "solid"}
OUTCOME_COLOR = {0: "#8a8984", 1: "#e34948"}
OUTCOME = {0: "No default", 1: "Default"}
INK, INK2, MUTED, GRID, SURFACE = "#1a1a19", "#52514e", "#8a8984", "#ebeae6", "#ffffff"

pio.templates["report"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK2),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    margin=dict(l=48, r=16, t=28, b=44),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                title=None, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, ticks="",
               title=dict(font=dict(size=11, color=MUTED)), tickfont=dict(size=11)),
    yaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, ticks="",
               title=dict(font=dict(size=11, color=MUTED)), tickfont=dict(size=11)),
    hoverlabel=dict(bgcolor=INK, font=dict(color="#ffffff", size=12), bordercolor=INK),
    bargap=0.45, bargroupgap=0.12,
    colorway=[COLOR[m] for m in MODELS],
))
pio.templates.default = "report"


def fig_(height=360, **kw):
    f = go.Figure()
    f.update_layout(height=height, **kw)
    return f


def annot(text, **kw):
    return dict(text=text, font=dict(size=10, color=MUTED), showarrow=False, **kw)


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def cost_at(y_s, p, t, c_fn=COST_FN, c_fp=COST_FP):
    pred = p >= t
    tp = int(np.sum(pred & (y_s == 1)))
    fp = int(np.sum(pred & (y_s == 0)))
    fn = int(np.sum(~pred & (y_s == 1)))
    tn = int(np.sum(~pred & (y_s == 0)))
    return c_fn * fn + c_fp * fp, (tn, fp, fn, tp)


def cost_curve(y_s, p, c_fn, c_fp):
    return np.array([cost_at(y_s, p, t, c_fn, c_fp)[0] for t in THRESHOLD_GRID])


def tiles(items):
    """items: list of (label, value, note) -> stat-tile row as HTML."""
    cells = "".join(
        f'<div class="tile"><div class="tile-label">{l}</div>'
        f'<div class="tile-value">{v}</div>'
        f'<div class="tile-note">{n}</div></div>' for l, v, n in items)
    return f'<div class="tiles">{cells}</div>'


# --------------------------------------------------------------------------- #
# Comparison tab
# --------------------------------------------------------------------------- #
def metrics_table(split="test"):
    ix = SPLITS[split]
    y_s = y[ix]
    rows = []
    for m in MODELS + ["raw_features_sklearn", "baseline"]:
        p = P_ALL[m][ix]
        t = VAL_THRESHOLDS[m]
        cost, (tn, fp, fn, tp) = cost_at(y_s, p, t)
        rows.append({
            "Model": LABEL[m],
            "ROC-AUC": roc_auc_score(y_s, p) if m != "baseline" else 0.5,
            "PR-AUC": average_precision_score(y_s, p),
            "Log-loss": log_loss(y_s, p),
            "Threshold": t,
            "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "Recall": tp / (tp + fn),
            "FP": fp, "FN": fn,
            "Cost": cost,
            "Cost / client": cost / len(y_s),
        })
    nobody = COST_FN * int(y_s.sum())
    rows.append({"Model": "Flag nobody", "ROC-AUC": 0.5, "PR-AUC": y_s.mean(),
                 "Log-loss": np.nan, "Threshold": np.nan, "Precision": 0.0,
                 "Recall": 0.0, "FP": 0, "FN": int(y_s.sum()),
                 "Cost": nobody, "Cost / client": nobody / len(y_s)})
    return pd.DataFrame(rows).round(3)


def header_tiles():
    ix = SPLITS["test"]
    y_s = y[ix]
    base = cost_at(y_s, P_ALL["baseline"][ix], VAL_THRESHOLDS["baseline"])[0] / len(y_s)
    cost = cost_at(y_s, P_ALL["sklearn"][ix], VAL_THRESHOLDS["sklearn"])[0] / len(y_s)
    _, (tn, fp, fn, tp) = cost_at(y_s, P_ALL["sklearn"][ix], VAL_THRESHOLDS["sklearn"])
    n = summary["n"]
    return tiles([
        ("Clients", f"{len(y):,}", f"{n['train']:,} train · {n['val']:,} val · {n['test']:,} test"),
        ("Default rate", f"{y.mean():.1%}", "same in every split"),
        ("ROC-AUC, test", f"{roc_auc_score(y_s, P_ALL['sklearn'][ix]):.3f}", "scikit-learn; PyTorch within 0.002"),
        ("Cost per client", f"{cost:.3f}", f"{1 - cost / base:.0%} below flagging everyone ({base:.3f})"),
        ("Defaulters caught", f"{tp / (tp + fn):.0%}", f"at threshold {VAL_THRESHOLDS['sklearn']:.2f}, "
                                                      f"{(tp + fp) / len(y_s):.0%} of clients flagged"),
    ])


def calibration_fig(split="test"):
    ix = SPLITS[split]
    f = fig_(340)
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                           line=dict(color=GRID, width=1), hoverinfo="skip"))
    for m in MODELS + ["raw_features_sklearn"]:
        df = pd.DataFrame({"p": P_ALL[m][ix], "y": y[ix]})
        df["bin"] = pd.qcut(df["p"], 10, duplicates="drop")
        g = df.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        f.add_trace(go.Scatter(
            x=g["pred"], y=g["actual"], mode="lines+markers", name=LABEL[m],
            line=dict(color=COLOR[m], dash=DASH[m], width=2),
            marker=dict(size=8, line=dict(color=SURFACE, width=2)),
            customdata=g["n"],
            hovertemplate="predicted %{x:.1%} · actual %{y:.1%} · n=%{customdata}<extra>" + LABEL[m] + "</extra>"))
    f.update_xaxes(title="Mean predicted probability (decile)", tickformat=".0%", range=[0, 0.85])
    f.update_yaxes(title="Actual default rate", tickformat=".0%", range=[0, 0.85])
    f.add_annotation(annot("perfect calibration", x=0.8, y=0.8, xanchor="right", yanchor="bottom"))
    return f


def prob_distribution_fig(split="test"):
    ix = SPLITS[split]
    f = fig_(340, boxmode="group")
    for cls in [0, 1]:
        mask = y[ix] == cls
        for m in MODELS:
            f.add_trace(go.Box(
                y=P_ALL[m][ix][mask], x=[LABEL[m]] * int(mask.sum()),
                name=OUTCOME[cls], legendgroup=OUTCOME[cls], showlegend=(m == "sklearn"),
                marker_color=OUTCOME_COLOR[cls], line=dict(width=1.5), fillcolor=OUTCOME_COLOR[cls] + "33",
                boxpoints=False, offsetgroup=str(cls), width=0.3))
    f.update_yaxes(title="Predicted probability of default", tickformat=".0%")
    return f


def roc_pr_fig(split="test"):
    ix = SPLITS[split]
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
    f.update_layout(height=340)
    for m in MODELS + ["raw_features_sklearn"]:
        p = P_ALL[m][ix]
        fpr, tpr, _ = roc_curve(y[ix], p)
        prec, rec, _ = precision_recall_curve(y[ix], p)
        kw = dict(mode="lines", line=dict(color=COLOR[m], dash=DASH[m], width=2), legendgroup=m)
        f.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{LABEL[m]} · AUC {roc_auc_score(y[ix], p):.3f}",
                               hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>", **kw), 1, 1)
        f.add_trace(go.Scatter(x=rec, y=prec, name=LABEL[m], showlegend=False,
                               hovertemplate="recall %{x:.2f} · precision %{y:.2f}<extra></extra>", **kw), 1, 2)
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                           line=dict(color=GRID, width=1), hoverinfo="skip"), 1, 1)
    f.add_hline(y=y[ix].mean(), line=dict(color=GRID, width=1), row=1, col=2)
    f.update_xaxes(title_text="False positive rate", row=1, col=1)
    f.update_yaxes(title_text="True positive rate", row=1, col=1)
    f.update_xaxes(title_text="Recall", row=1, col=2)
    f.update_yaxes(title_text="Precision", row=1, col=2)
    return f


def agreement_fig(split="test"):
    ix = SPLITS[split]
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
    f.update_layout(height=320, showlegend=False)
    for col, m in [(1, "manual_torch"), (2, "standard_torch")]:
        f.add_trace(go.Scatter(
            x=P_ALL["sklearn"][ix], y=P_ALL[m][ix], mode="markers",
            marker=dict(color=COLOR[m], size=4, opacity=0.35),
            hovertemplate="scikit-learn %{x:.3f} · " + LABEL[m] + " %{y:.3f}<extra></extra>"), 1, col)
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                               line=dict(color=MUTED, width=1), hoverinfo="skip"), 1, col)
        diff = np.abs(P_ALL[m][ix] - P_ALL["sklearn"][ix]).max()
        f.add_annotation(annot(f"{LABEL[m]} · max difference {diff:.3f}", x=0.02, y=0.98,
                               xref=f"x{col if col > 1 else ''} domain", yref=f"y{col if col > 1 else ''} domain",
                               xanchor="left", yanchor="top"))
        f.update_xaxes(title_text="scikit-learn probability", range=[0, 1], row=1, col=col)
        f.update_yaxes(title_text="PyTorch probability" if col == 1 else None, range=[0, 1], row=1, col=col)
    return f


def coef_fig():
    coefs = pd.DataFrame(summary["coefficients"]).drop(index="(intercept)")
    order = coefs["sklearn"].sort_values().index
    f = fig_(640, barmode="group", bargap=0.35, bargroupgap=0.05, margin=dict(l=160, r=16, t=28, b=44))
    for m in MODELS:
        f.add_trace(go.Bar(y=order, x=coefs.loc[order, m], orientation="h", name=LABEL[m],
                           marker=dict(color=COLOR[m], cornerradius=3),
                           hovertemplate="%{y} · %{x:.3f}<extra>" + LABEL[m] + "</extra>"))
    f.add_vline(x=0, line=dict(color=MUTED, width=1))
    f.update_xaxes(title="Coefficient per standard deviation of the feature (positive raises default risk)")
    f.update_yaxes(tickfont=dict(size=11, family="JetBrains Mono, ui-monospace, monospace"))
    return f


def agreement_table():
    a = summary["agreement"]
    rows = [{"Model": "scikit-learn", "Training objective": a["train_objective"]["sklearn"],
             "Max coefficient difference": 0.0, "Max probability difference (test)": 0.0,
             "Same decision (test)": 1.0}]
    for m in ["manual_torch", "standard_torch"]:
        rows.append({
            "Model": LABEL[m],
            "Training objective": a["train_objective"][m],
            "Max coefficient difference": a["max_abs_coef_diff_vs_sklearn"][m],
            "Max probability difference (test)": a["max_abs_test_prob_diff_vs_sklearn"][m],
            "Same decision (test)": a["test_decision_agreement_vs_sklearn"][m],
        })
    df = pd.DataFrame(rows)
    df["Training objective"] = df["Training objective"].map("{:.6f}".format)
    return df.round(4)


# --------------------------------------------------------------------------- #
# Threshold tab
# --------------------------------------------------------------------------- #
def threshold_view(model, threshold, c_fn, c_fp, split):
    ix = SPLITS[split]
    y_s, p = y[ix], P_ALL[model][ix]
    n = len(y_s)
    cost, (tn, fp, fn, tp) = cost_at(y_s, p, threshold, c_fn, c_fp)

    # Confusion matrix as four tiles in a 2×2 grid (counts and their cost).
    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted: no default", "Predicted: default"],
        y=["Actual: no default", "Actual: default"],
        text=[[f"<b>{tn:,}</b><br><span style='font-size:11px'>true negative</span>",
               f"<b>{fp:,}</b><br><span style='font-size:11px'>false positive · cost {c_fp * fp:,}</span>"],
              [f"<b>{fn:,}</b><br><span style='font-size:11px'>false negative · cost {c_fn * fn:,}</span>",
               f"<b>{tp:,}</b><br><span style='font-size:11px'>true positive</span>"]],
        texttemplate="%{text}", textfont=dict(size=15),
        colorscale=[[0, "#f3f2ef"], [1, "#d5e3f6"]], showscale=False,
        hoverinfo="skip", xgap=4, ygap=4))
    cm.update_layout(height=300, margin=dict(l=120, r=8, t=36, b=8))
    cm.update_xaxes(side="top", showgrid=False, tickfont=dict(size=11))
    cm.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11))

    # Cost per client across thresholds, against the two trivial policies.
    val_ix = SPLITS["val"]
    val_curve = cost_curve(y[val_ix], P_ALL[model][val_ix], c_fn, c_fp)
    t_opt = float(THRESHOLD_GRID[val_curve.argmin()])
    curve = cost_curve(y_s, p, c_fn, c_fp) / n
    everyone = c_fp * (y_s == 0).sum() / n
    nobody = c_fn * (y_s == 1).sum() / n
    cc = fig_(300, showlegend=False)
    cc.add_trace(go.Scatter(x=THRESHOLD_GRID, y=curve, mode="lines", line=dict(color=COLOR[model], width=2),
                            hovertemplate="threshold %{x:.2f} · cost %{y:.3f}<extra></extra>"))
    cc.add_hline(y=everyone, line=dict(color=MUTED, width=1, dash="dot"))
    cc.add_hline(y=nobody, line=dict(color=MUTED, width=1, dash="dot"))
    cc.add_annotation(annot(f"flag everyone · {everyone:.3f}", x=0.99, y=everyone, xref="x domain",
                            xanchor="right", yanchor="bottom"))
    cc.add_annotation(annot(f"flag nobody · {nobody:.3f}", x=0.99, y=nobody, xref="x domain",
                            xanchor="right", yanchor="bottom"))
    cc.add_vline(x=t_opt, line=dict(color=INK2, width=1))
    cc.add_annotation(annot(f"validation optimum {t_opt:.2f}", x=t_opt, y=0.98, yref="y domain",
                            xanchor="left", yanchor="top", xshift=4))
    cc.add_trace(go.Scatter(x=[threshold], y=[cost / n], mode="markers",
                            marker=dict(size=11, color=COLOR[model], line=dict(color=SURFACE, width=2)),
                            hovertemplate="current · cost %{y:.3f}<extra></extra>"))
    cc.update_xaxes(title="Decision threshold", range=[0, 1])
    cc.update_yaxes(title=f"Cost per client ({c_fn}·FN + {c_fp}·FP)", rangemode="tozero")

    flagged = tp + fp
    precision = tp / flagged if flagged else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    trivial = min(everyone, nobody)
    vs = (cost / n) / trivial - 1
    kpis = tiles([
        ("Total cost", f"{cost:,}", f"{cost / n:.3f} per client"),
        ("Versus best trivial policy", f"{vs:+.0%}", "flag everyone" if everyone < nobody else "flag nobody"),
        ("Clients flagged", f"{flagged / n:.0%}", f"{flagged:,} of {n:,}"),
        ("Defaulters caught", f"{recall:.0%}", "recall"),
        ("Flags correct", f"{precision:.0%}", "precision"),
        ("Cost-optimal cut-off", f"{c_fp / (c_fp + c_fn):.3f}", "FP ÷ (FP + FN), if probabilities are calibrated"),
    ])
    return cm, cc, kpis


def reset_threshold(model, c_fn, c_fp):
    val_ix = SPLITS["val"]
    curve = cost_curve(y[val_ix], P_ALL[model][val_ix], c_fn, c_fp)
    return float(THRESHOLD_GRID[curve.argmin()])


# --------------------------------------------------------------------------- #
# Features tab
# --------------------------------------------------------------------------- #
RAW_COLS = ["LIMIT_BAL", "AGE", "EDUCATION"] + PAY + BILL + PAY_AMT
EXPLORE = pd.concat([feats, raw[RAW_COLS].add_prefix("raw · ")], axis=1)
EXPLORE_COLS = list(EXPLORE.columns)


def target_fig():
    counts = pd.Series(y).value_counts().sort_index()
    f = fig_(220, bargap=0.6, margin=dict(l=8, r=8, t=8, b=36), showlegend=False)
    f.add_trace(go.Bar(
        x=[OUTCOME[i] for i in counts.index], y=counts.values,
        marker=dict(color=[OUTCOME_COLOR[i] for i in counts.index], cornerradius=4),
        text=[f"{v:,} · {v / len(y):.1%}" for v in counts.values], textposition="outside",
        textfont=dict(color=INK2), hoverinfo="skip", cliponaxis=False))
    f.update_yaxes(visible=False, range=[0, counts.max() * 1.25])
    f.update_xaxes(showgrid=False)
    return f


def split_table():
    split = pd.Series("train", index=raw.index)
    split.iloc[SPLITS["val"]] = "validation"
    split.iloc[SPLITS["test"]] = "test"
    t = pd.DataFrame({"split": split, "y": y}).groupby("split", sort=False)["y"].agg(
        Clients="size", Defaults="sum", **{"Default rate": "mean"})
    return t.reset_index().rename(columns={"split": "Split"}).round(3)


def feature_view(col):
    s = EXPLORE[col]
    discrete = s.nunique() <= 12
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, column_widths=[0.55, 0.45])
    f.update_layout(height=340)
    if discrete:
        ct = pd.crosstab(s, y, normalize="columns")
        for cls in [0, 1]:
            f.add_trace(go.Bar(x=ct.index.astype(str), y=ct[cls], name=OUTCOME[cls],
                               marker=dict(color=OUTCOME_COLOR[cls], cornerradius=3),
                               hovertemplate="%{x} · %{y:.1%}<extra>" + OUTCOME[cls] + "</extra>"), 1, 1)
        rate = pd.DataFrame({"x": s, "y": y}).groupby("x")["y"].agg(["mean", "size"])
        xs = rate.index.astype(str)
    else:
        lo, hi = s.quantile([0.005, 0.995])
        for cls in [0, 1]:
            f.add_trace(go.Histogram(x=s[(y == cls)].clip(lo, hi), name=OUTCOME[cls],
                                     marker_color=OUTCOME_COLOR[cls], opacity=0.65,
                                     histnorm="probability", nbinsx=40), 1, 1)
        f.update_layout(barmode="overlay", bargap=0.05)
        bins = pd.qcut(s, 10, duplicates="drop")
        rate = pd.DataFrame({"x": bins, "y": y}).groupby("x", observed=True)["y"].agg(["mean", "size"])
        fmt = ",.0f" if (hi - lo) > 100 else ",.2f"
        xs = [f"{iv.left:{fmt}} – {iv.right:{fmt}}" for iv in rate.index]
    f.add_trace(go.Bar(x=xs, y=rate["mean"], name="Default rate", showlegend=False,
                       marker=dict(color=OUTCOME_COLOR[1], cornerradius=3), customdata=rate["size"],
                       hovertemplate="%{x} · default rate %{y:.1%} · n=%{customdata:,}<extra></extra>"), 1, 2)
    f.add_hline(y=y.mean(), line=dict(color=MUTED, width=1, dash="dot"), row=1, col=2)
    f.add_annotation(annot(f"overall {y.mean():.1%}", x=0.99, y=y.mean(), xref="x2 domain", yref="y2",
                           xanchor="right", yanchor="bottom"))
    f.update_yaxes(title_text="Share of each outcome", tickformat=".0%", row=1, col=1)
    f.update_yaxes(title_text="Default rate", tickformat=".0%", row=1, col=2)
    f.update_xaxes(title_text="Value", row=1, col=1)
    f.update_xaxes(title_text="Value" if discrete else "Decile", row=1, col=2,
                   tickangle=0 if discrete else -35, tickfont=dict(size=10))

    stats = s.groupby(pd.Series(y).map(OUTCOME)).describe().T.round(3)
    stats.insert(0, "Statistic", stats.index)
    return f, stats


# --------------------------------------------------------------------------- #
# Training tab
# --------------------------------------------------------------------------- #
def loss_fig():
    best = summary["best_lr"]
    obj = summary["agreement"]["train_objective"]["sklearn"]
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
    f.update_layout(height=340)
    for col, m in [(1, "manual_torch"), (2, "standard_torch")]:
        cols = [c for c in loss_hist.columns if c.startswith(m + "|")]
        for c in cols:
            lr = float(c.split("=")[1])
            s = loss_hist[c].dropna()
            chosen = np.isclose(lr, best[m])
            f.add_trace(go.Scatter(
                x=s.index + 1, y=s.values, mode="lines",
                name=f"{LABEL[m]} · lr {lr}" + (" (chosen)" if chosen else ""),
                line=dict(color=COLOR[m] if chosen else "#c9c8c2", width=2.5 if chosen else 1.2),
                hovertemplate="step %{x} · %{y:.5f}<extra>lr " + str(lr) + "</extra>"), 1, col)
        f.add_hline(y=obj, line=dict(color=COLOR["sklearn"], width=1, dash="dot"), row=1, col=col)
    f.add_annotation(annot(f"scikit-learn optimum {obj:.4f}", x=0.99, y=obj, xref="x domain", yref="y",
                           xanchor="right", yanchor="bottom"))
    f.update_xaxes(type="log", title_text="Gradient step (log scale)", row=1, col=1)
    f.update_xaxes(title_text="Epoch", row=1, col=2)
    f.update_yaxes(title_text="Training objective", range=[obj - 0.004, 0.49], row=1, col=1)
    f.update_yaxes(range=[obj - 0.004, 0.49], row=1, col=2)
    return f


def tuning_figs():
    tune = pd.DataFrame(summary["tuning"])
    sk_t = tune[tune.model == "sklearn"]
    c_fig = fig_(300, showlegend=False)
    c_fig.add_trace(go.Scatter(
        x=sk_t["C"], y=sk_t["val_log_loss"], mode="lines+markers",
        line=dict(color=COLOR["sklearn"], width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
        hovertemplate="C %{x} · log-loss %{y:.5f}<extra></extra>"))
    c_fig.add_vline(x=summary["best_C"], line=dict(color=INK2, width=1))
    c_fig.add_annotation(annot(f"chosen C = {summary['best_C']}", x=np.log10(summary["best_C"]), y=0.98,
                               yref="y domain", xanchor="left", yanchor="top", xshift=4))
    c_fig.update_xaxes(type="log", title="C (stronger regularisation to the left)")
    c_fig.update_yaxes(title="Validation log-loss")

    lr_fig = fig_(300)
    for m in ["manual_torch", "standard_torch"]:
        t = tune[tune.model == m]
        lr_fig.add_trace(go.Scatter(
            x=t["lr"], y=t["val_log_loss"], mode="lines+markers", name=LABEL[m],
            line=dict(color=COLOR[m], width=2, dash=DASH[m]),
            marker=dict(size=8, line=dict(color=SURFACE, width=2)),
            hovertemplate="lr %{x} · log-loss %{y:.5f}<extra>" + LABEL[m] + "</extra>"))
    lr_fig.update_xaxes(type="log", title="Learning rate")
    lr_fig.update_yaxes(title="Validation log-loss")
    return c_fig, lr_fig


# --------------------------------------------------------------------------- #
# Client tab
# --------------------------------------------------------------------------- #
TEST_IX = np.asarray(SPLITS["test"])
STATUS = {-2: "no use", -1: "paid in full", 0: "revolving"}


def client_view(i):
    i = int(np.clip(i, 0, len(TEST_IX) - 1))
    row = TEST_IX[i]
    r = raw.iloc[row]
    hist = pd.DataFrame({
        "Month": ["Apr", "May", "Jun", "Jul", "Aug", "Sep"],
        "Repayment status": [STATUS.get(int(r[c]), f"{int(r[c])} month(s) late") for c in PAY[::-1]],
        "Statement": [f"{r[c]:,.0f}" for c in BILL[::-1]],
        "Paid": [f"{r[c]:,.0f}" for c in PAY_AMT[::-1]],
    })
    p_sk = P_ALL["sklearn"][row]
    decisions = " · ".join(
        f"{LABEL[m]} {P_ALL[m][row]:.1%} → {'lower limit' if P_ALL[m][row] >= VAL_THRESHOLDS[m] else 'keep'}"
        for m in MODELS)
    head = tiles([
        ("Client", f"#{int(r['ID'])}", f"limit {r['LIMIT_BAL']:,.0f} · age {int(r['AGE'])}"),
        ("Actual outcome", OUTCOME[int(y[row])], "next month"),
        ("P(default), scikit-learn", f"{p_sk:.1%}", f"threshold {VAL_THRESHOLDS['sklearn']:.2f}"),
        ("Decision", "Lower limit" if p_sk >= VAL_THRESHOLDS["sklearn"] else "Keep limit", decisions),
    ])

    contrib = pd.Series(sk.coef_.ravel() * X[row], index=prep["features"])
    top = contrib.reindex(contrib.abs().sort_values().index).tail(10)
    f = fig_(340, bargap=0.4, showlegend=False, margin=dict(l=160, r=16, t=16, b=44))
    f.add_trace(go.Bar(
        y=top.index, x=top.values, orientation="h",
        marker=dict(color=[OUTCOME_COLOR[1] if v > 0 else COLOR["sklearn"] for v in top.values], cornerradius=3),
        customdata=feats.iloc[row][top.index].values,
        hovertemplate="%{y} = %{customdata:.3f} · contribution %{x:+.3f}<extra></extra>"))
    f.add_vline(x=0, line=dict(color=MUTED, width=1))
    f.update_xaxes(title="Contribution to log-odds (coefficient × standardised value); positive pushes towards default")
    f.update_yaxes(tickfont=dict(size=11, family="JetBrains Mono, ui-monospace, monospace"))
    return head, hist, f, i


def random_client():
    return int(np.random.randint(len(TEST_IX)))


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
CSS = """
:root { --ink:#1a1a19; --ink2:#52514e; --muted:#8a8984; --line:#ebeae6; --paper:#f7f6f3; --card:#ffffff; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; background: var(--paper) !important; }
.gradio-container, .gradio-container * { font-family: Inter, system-ui, -apple-system, sans-serif; }
h1 { font-size: 22px !important; font-weight: 600 !important; letter-spacing: -0.01em; color: var(--ink) !important; margin: 4px 0 2px !important; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
.cap { color: var(--ink2); font-size: 13px; font-weight: 500; margin: 6px 0 -2px 2px; }
.cap em { color: var(--muted); font-style: normal; font-weight: 400; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 10px; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
.tile-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.tile-value { font-size: 26px; font-weight: 600; color: var(--ink); line-height: 1.2; margin: 4px 0 2px; }
.tile-note { font-size: 11px; color: var(--muted); }
.block:has(.prose) { border: none !important; background: transparent !important; padding: 0 !important; }
.block, .gr-box, .form { border: 1px solid var(--line) !important; box-shadow: none !important; border-radius: 8px !important; background: var(--card) !important; }
.tabs > .tab-nav { border-bottom: 1px solid var(--line) !important; }
.tab-nav button { font-size: 13px !important; font-weight: 500 !important; color: var(--muted) !important; }
.tab-nav button.selected { color: var(--ink) !important; border-bottom: 2px solid var(--ink) !important; }
label span, .label-wrap span { font-size: 12px !important; color: var(--ink2) !important; font-weight: 500 !important; }
table.table td, table.table th, .table { font-size: 12.5px !important; font-variant-numeric: tabular-nums; }
footer { display: none !important; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f7f6f3",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_shadow="none",
    button_primary_background_fill="#1a1a19",
    button_primary_text_color="#ffffff",
)


def cap(text):
    return gr.Markdown(text, elem_classes=["cap"])


with gr.Blocks(title="Credit default · monthly risk review") as demo:
    gr.Markdown("# Credit card default — monthly risk review")
    gr.Markdown(
        f"One logistic regression trained three ways, scoring each existing cardholder's probability of "
        f"defaulting next month. A missed defaulter costs {COST_FN}, a limit cut for a good client costs {COST_FP}. "
        f"Models are loaded from `artifacts/`; nothing is retrained here.", elem_classes=["sub"])
    gr.HTML(header_tiles())

    with gr.Tab("Comparison"):
        split_cmp = gr.Radio([("Test set", "test"), ("Validation set", "val")], value="test",
                             show_label=False, container=False)
        cap("Metrics at each model's validation-chosen threshold, against the naive baseline and the two trivial policies")
        tbl = gr.Dataframe(metrics_table("test"), show_label=False, interactive=False, show_search="none")
        with gr.Row():
            with gr.Column():
                cap("Predicted vs actual default rate, by risk decile")
                cal = gr.Plot(calibration_fig("test"), show_label=False)
            with gr.Column():
                cap("Predicted probability by actual outcome")
                box = gr.Plot(prob_distribution_fig("test"), show_label=False)
        cap("ROC curve *(left)* and precision–recall curve *(right)*")
        rocpr = gr.Plot(roc_pr_fig("test"), show_label=False)
        cap("Agreement between implementations: same split, same objective")
        gr.Dataframe(agreement_table(), show_label=False, interactive=False, show_search="none")
        agr = gr.Plot(agreement_fig("test"), show_label=False)
        cap("Learned coefficients, standardised features")
        gr.Plot(coef_fig(), show_label=False)
        split_cmp.change(
            lambda s: (metrics_table(s), calibration_fig(s), prob_distribution_fig(s),
                       roc_pr_fig(s), agreement_fig(s)),
            split_cmp, [tbl, cal, box, rocpr, agr])

    with gr.Tab("Threshold"):
        with gr.Row():
            model_dd = gr.Dropdown([(LABEL[m], m) for m in MODELS], value="sklearn", label="Model", scale=2)
            split_th = gr.Radio([("Test set", "test"), ("Validation set", "val")], value="test",
                                label="Evaluate on", scale=2)
            c_fn = gr.Slider(1, 20, value=COST_FN, step=1, label="Cost of a missed defaulter (FN)", scale=3)
            c_fp = gr.Slider(1, 20, value=COST_FP, step=1, label="Cost of a wrong limit cut (FP)", scale=3)
        with gr.Row():
            thr = gr.Slider(0.01, 0.99, value=VAL_THRESHOLDS["sklearn"], step=0.01,
                            label="Decision threshold: lower the limit when P(default) is at least", scale=5)
            reset = gr.Button("Validation optimum", variant="primary", scale=1)
        kpi = gr.HTML()
        with gr.Row():
            with gr.Column():
                cap("Confusion matrix at the current threshold")
                cm_plot = gr.Plot(show_label=False)
            with gr.Column():
                cap("Cost per client across thresholds")
                cost_plot = gr.Plot(show_label=False)
        inputs = [model_dd, thr, c_fn, c_fp, split_th]
        outputs = [cm_plot, cost_plot, kpi]
        for comp in inputs:
            comp.change(threshold_view, inputs, outputs)
        reset.click(reset_threshold, [model_dd, c_fn, c_fp], thr)
        model_dd.change(reset_threshold, [model_dd, c_fn, c_fp], thr)
        demo.load(threshold_view, inputs, outputs)

    with gr.Tab("Features"):
        with gr.Row():
            with gr.Column(scale=1):
                cap("Target: default next month, all clients")
                gr.Plot(target_fig(), show_label=False)
            with gr.Column(scale=1):
                cap("Stratified 70 / 15 / 15 split")
                gr.Dataframe(split_table(), show_label=False, interactive=False, show_search="none")
        feat_dd = gr.Dropdown(EXPLORE_COLS, value="DELAY_RECENT",
                              label="Feature (engineered features first, then raw columns)")
        cap("Distribution by outcome *(left)* and default rate across the feature *(right)*")
        feat_plot = gr.Plot(show_label=False)
        cap("Summary by outcome")
        feat_stats = gr.Dataframe(show_label=False, interactive=False, show_search="none")
        feat_dd.change(feature_view, feat_dd, [feat_plot, feat_stats])
        demo.load(feature_view, feat_dd, [feat_plot, feat_stats])

    with gr.Tab("Training"):
        gr.HTML(tiles([
            ("Objective", "BCE + λ‖w‖²", "identical for all three"),
            ("λ", f"{summary['lambda']:.2e}", f"= 1 / (2·C·n), C = {summary['best_C']}"),
            ("Learning rate, manual", f"{summary['best_lr']['manual_torch']}", "full batch, 2,000 steps"),
            ("Learning rate, standard", f"{summary['best_lr']['standard_torch']}", "mini-batches of 256, 60 epochs"),
        ]))
        cap("Training objective per step: manual loop *(left)* and standard workflow *(right)*, every learning rate tried")
        gr.Plot(loss_fig(), show_label=False)
        c_fig, lr_fig = tuning_figs()
        with gr.Row():
            with gr.Column():
                cap("Regularisation strength, tuned on validation log-loss")
                gr.Plot(c_fig, show_label=False)
            with gr.Column():
                cap("Learning rate, tuned on validation log-loss")
                gr.Plot(lr_fig, show_label=False)

    with gr.Tab("Client"):
        with gr.Row():
            idx = gr.Slider(0, len(TEST_IX) - 1, value=0, step=1, label="Test-set client", scale=5)
            rnd = gr.Button("Random client", variant="primary", scale=1)
        verdict = gr.HTML()
        with gr.Row():
            with gr.Column():
                cap("Six-month history")
                hist_tbl = gr.Dataframe(show_label=False, interactive=False, show_search="none")
            with gr.Column():
                cap("Largest contributions to this client's score, scikit-learn")
                contrib_plot = gr.Plot(show_label=False)
        idx.release(client_view, idx, [verdict, hist_tbl, contrib_plot, idx])
        rnd.click(random_client, None, idx).then(client_view, idx, [verdict, hist_tbl, contrib_plot, idx])
        demo.load(client_view, idx, [verdict, hist_tbl, contrib_plot, idx])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                theme=THEME, css=CSS, footer_links=[])
