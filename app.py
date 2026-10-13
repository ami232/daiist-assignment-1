"""Gradio dashboard. Loads the artifacts written by train.py and never
retrains anything: it only calls load / transform / predict."""
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn

from evaluation import business_cost
from features import build_raw_features, load_data, split_by_date

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_NAMES = ["sklearn", "pytorch_manual", "pytorch_module"]
SPLITS = ["Validation", "Test"]


class LogReg(nn.Module):
    """Same architecture as in train.py (needed to load the saved state_dict)."""
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# ---------------------------------------------------------------- artifacts
needed = ["preprocessor.joblib", "sklearn_logreg.joblib", "pytorch_manual.pt",
          "pytorch_module.pt", "results.json"]
missing = [f for f in needed if not (MODELS_DIR / f).exists()]
if missing:
    raise SystemExit(f"Missing {missing} in models/. Run `uv run python main.py train` first.")

prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
sk = joblib.load(MODELS_DIR / "sklearn_logreg.joblib")
manual = torch.load(MODELS_DIR / "pytorch_manual.pt", weights_only=True)
cfg = json.loads((MODELS_DIR / "results.json").read_text())
net = LogReg(cfg.get("n_features", len(prep.feature_names)))
net.load_state_dict(torch.load(MODELS_DIR / "pytorch_module.pt", weights_only=True))
net.eval()


def score_sklearn(X):
    return sk.predict_proba(X)[:, 1]


def score_manual(X):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X) @ manual["w"] + manual["b"]).numpy()


def score_module(X):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(X))).numpy()


SCORERS = {"sklearn": score_sklearn, "pytorch_manual": score_manual,
           "pytorch_module": score_module}

# --------------------------------------------------------------------- data
df = load_data(str(ROOT / "data" / "loans.csv"))
train, val, test = split_by_date(df)

DATA = {}
for label, part in [("Validation", val), ("Test", test)]:
    X = prep.transform(part)  # transform only: nothing is fitted here
    DATA[label] = {
        "y": part["default"].to_numpy(),
        "amt": part["loan_amnt"].to_numpy(),
        "scores": {name: fn(X) for name, fn in SCORERS.items()},
    }

SMAX = float(np.ceil(max(s.max() for d in DATA.values()
                         for s in d["scores"].values()) * 20) / 20)
_thr0 = cfg["results"]["sklearn"]["threshold"]
DEFAULT_THR = min(round(_thr0, 3), SMAX) if _thr0 is not None else 0.13
LGD0, MARGIN0 = cfg["lgd"] * 100, cfg["margin"] * 100

RAW = build_raw_features(df)
RAW["default"] = df["default"]


# -------------------------------------------------------------------- tab 1
def compare(split):
    """Prediction vs actual: mean predicted default probability per decile
    against the observed default rate, for the three models."""
    d = DATA[split]
    fig = go.Figure()
    top = 0.0
    for name, symbol in zip(MODEL_NAMES, ["circle", "x", "diamond"]):
        tmp = pd.DataFrame({"s": d["scores"][name], "y": d["y"]})
        tmp["bin"] = pd.qcut(tmp["s"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True)[["s", "y"]].mean()
        top = max(top, g.max().max())
        fig.add_trace(go.Scatter(x=g["s"], y=g["y"], mode="lines+markers", name=name,
                                 marker=dict(symbol=symbol, size=9)))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title=f"Predicted vs observed default rate by decile ({split})",
                      xaxis_title="Mean predicted default probability",
                      yaxis_title="Observed default rate", height=450)
    return fig


def metrics_table():
    rows = []
    for name, r in cfg["results"].items():
        t = r["threshold"]
        if name == "baseline_approve_all" or t is None:
            thr_txt = "approve all"
        elif name == "baseline_fico":
            thr_txt = f"reject if FICO <= {-t:.0f}"
        else:
            thr_txt = f"{t:.3f}"
        rows.append({
            "model": name,
            "threshold (chosen on validation)": thr_txt,
            "val AUC": round(r["val_auc"], 3),
            "test AUC": round(r["test_auc"], 3),
            "test PR-AUC": round(r["test_pr_auc"], 3),
            "test cost ($)": f"{r['test_cost']:,.0f}",
            "saving vs approve-all": f"{1 - r['test_cost'] / r['test_cost_approve_all']:.1%}",
        })
    return pd.DataFrame(rows)


def weights_figure():
    names = cfg["feature_names"]
    fig = go.Figure()
    fig.add_bar(y=names, x=sk.coef_[0], orientation="h", name="sklearn")
    fig.add_bar(y=names, x=manual["w"].numpy(), orientation="h", name="pytorch_manual")
    fig.add_bar(y=names, x=net.linear.weight.detach().numpy()[0], orientation="h",
                name="pytorch_module")
    fig.update_layout(barmode="group", height=950, title="Learned coefficients (standardized features)",
                      xaxis_title="Coefficient", yaxis=dict(autorange="reversed"))
    return fig


# -------------------------------------------------------------------- tab 2
def drift_figure():
    tmp = pd.concat([train.assign(split="Train"), val.assign(split="Validation"),
                     test.assign(split="Test")])
    tmp["quarter"] = tmp["issue_d"].dt.to_period("Q").astype(str)
    g = tmp.groupby(["quarter", "split"])["default"].agg(["mean", "count"]).reset_index()
    fig = go.Figure()
    for split in ["Train", "Validation", "Test"]:
        part = g[g["split"] == split]
        fig.add_bar(x=part["quarter"], y=part["mean"], name=split, customdata=part["count"],
                    hovertemplate="%{x}<br>default rate %{y:.1%}<br>n = %{customdata}<extra></extra>")
    fig.update_layout(title="Target: default rate by issue quarter (colored by split)",
                      yaxis_title="Default rate", yaxis_tickformat=".0%", height=420)
    return fig


def feature_hist(feature):
    fig = go.Figure()
    for flag, label in [(0, "Repaid"), (1, "Defaulted")]:
        x = RAW.loc[RAW["default"] == flag, feature].dropna()
        fig.add_trace(go.Histogram(x=x, name=label, histnorm="probability density",
                                   opacity=0.6, nbinsx=40))
    fig.update_layout(barmode="overlay", title=f"{feature}: repaid vs defaulted loans",
                      xaxis_title=feature, yaxis_title="Density", height=420)
    return fig


# -------------------------------------------------------------------- tab 3
def _pct(num, den):
    return f"{num / den:.1%}" if den else "n/a"


def analyze(model, split, thr, lgd_pct, margin_pct):
    d = DATA[split]
    y, amt, s = d["y"], d["amt"], d["scores"][model]
    lgd, margin = lgd_pct / 100, margin_pct / 100

    reject = s >= thr
    tp = int((reject & (y == 1)).sum())
    fp = int((reject & (y == 0)).sum())
    fn = int((~reject & (y == 1)).sum())
    tn = int((~reject & (y == 0)).sum())
    cost = business_cost(y, s, thr, amt, lgd, margin)
    approve_all = business_cost(y, s, np.inf, amt, lgd, margin)
    saving = f"{(approve_all - cost) / approve_all:.1%}" if approve_all > 0 else "n/a"
    breakeven = f"{margin / (margin + lgd):.1%}" if (margin + lgd) > 0 else "n/a"

    summary = (
        f"**{model} on the {split.lower()} set, reject if probability >= {thr:.3f}**\n\n"
        f"- Loans: {len(y):,} (actual default rate {y.mean():.1%})\n"
        f"- Rejected: {int(reject.sum()):,} ({reject.mean():.1%} of loans); "
        f"{_pct(tp, tp + fp)} of them really defaulted\n"
        f"- Defaults caught: {_pct(tp, tp + fn)} ({tp:,} of {tp + fn:,})\n"
        f"- **Business cost: ${cost:,.0f}** (approving everyone: ${approve_all:,.0f}; "
        f"saving {saving})\n"
        f"- Break-even default probability for these assumptions: {breakeven}\n"
    )

    text = [[f"Approved, repaid<br><b>{tn:,}</b>", f"Rejected, would have repaid (FP)<br><b>{fp:,}</b>"],
            [f"Approved, defaulted (FN)<br><b>{fn:,}</b>", f"Rejected, defaulted<br><b>{tp:,}</b>"]]
    cm = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Approve", "Reject"],
                              y=["Repaid", "Defaulted"], text=text, texttemplate="%{text}",
                              colorscale="Blues", showscale=False))
    cm.update_layout(title="Confusion matrix", xaxis_title="Decision",
                     yaxis_title="What actually happened", height=380)
    cm.update_yaxes(autorange="reversed")

    grid = np.linspace(0, SMAX, 121)
    costs = [business_cost(y, s, t, amt, lgd, margin) for t in grid]
    i = int(np.argmin(costs))
    cf = go.Figure()
    cf.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name="cost"))
    cf.add_hline(y=approve_all, line_dash="dash", annotation_text="approve everyone")
    cf.add_trace(go.Scatter(x=[grid[i]], y=[costs[i]], mode="markers",
                            marker=dict(size=11, symbol="star"),
                            name=f"best in hindsight on this set (t={grid[i]:.3f})"))
    cf.add_trace(go.Scatter(x=[thr], y=[cost], mode="markers",
                            marker=dict(size=12, color="red"), name="current threshold"))
    cf.update_layout(title="Business cost as a function of the threshold",
                     xaxis_title="Threshold", yaxis_title="Cost ($)", height=420)
    return summary, cm, cf


def optimal_threshold(model, lgd_pct, margin_pct):
    """Threshold minimizing cost on VALIDATION for the current assumptions."""
    d = DATA["Validation"]
    s, y, amt = d["scores"][model], d["y"], d["amt"]
    lgd, margin = lgd_pct / 100, margin_pct / 100
    cand = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    costs = [business_cost(y, s, t, amt, lgd, margin) for t in cand]
    best = float(cand[int(np.argmin(costs))])
    if business_cost(y, s, np.inf, amt, lgd, margin) <= min(costs):
        best = SMAX  # approving everyone is the cheapest rule
    return round(min(best, SMAX), 3)


# ---------------------------------------------------------------------- UI
def build_demo():
    with gr.Blocks(title="Loan default scoring") as demo:
        gr.Markdown(
            "# Loan default scoring\n"
            "Logistic regression trained three ways (scikit-learn, manual PyTorch loop, "
            "`nn.Module` + `torch.optim`) on 36-month Lending Club loans. "
            "Train: 2012-2013, validation: Jan-Jun 2014, test: Jul-Dec 2014. "
            "This app only loads the saved models; it never retrains."
        )

        with gr.Tab("1. Model comparison"):
            split_a = gr.Radio(SPLITS, value="Test", label="Evaluation set")
            calib = gr.Plot(value=compare("Test"))
            split_a.change(compare, split_a, calib)
            gr.Markdown(
                f"Max weight gap vs scikit-learn: manual loop "
                f"{cfg['weight_gap_manual_vs_sklearn']:.5f}, nn.Module "
                f"{cfg['weight_gap_module_vs_sklearn']:.5f}. The three curves overlap "
                "because the models converge to the same solution."
            )
            gr.Dataframe(value=metrics_table(), interactive=False)
            gr.Plot(value=weights_figure())

        with gr.Tab("2. Distributions"):
            gr.Plot(value=drift_figure())
            feat = gr.Dropdown(list(RAW.columns[:-1]), value="fico_mid",
                               label="Engineered feature (before scaling)")
            hist = gr.Plot(value=feature_hist("fico_mid"))
            feat.change(feature_hist, feat, hist)

        with gr.Tab("3. Threshold and business cost"):
            gr.Markdown(
                "A loan is **rejected** when its predicted default probability is at or above "
                "the threshold. Approving a loan that defaults costs LGD x amount; rejecting "
                "a loan that would have repaid costs margin x amount."
            )
            with gr.Row():
                model = gr.Radio(MODEL_NAMES, value="sklearn", label="Model")
                split_c = gr.Radio(SPLITS, value="Test", label="Evaluation set")
            thr = gr.Slider(0, SMAX, value=DEFAULT_THR, step=0.001,
                            label="Threshold: reject if predicted default probability >=")
            with gr.Row():
                lgd = gr.Slider(0, 100, value=LGD0, step=1, label="LGD (% of principal lost on a default)")
                margin = gr.Slider(0, 30, value=MARGIN0, step=1, label="Net margin (% of principal on a repaid loan)")
            btn = gr.Button("Set threshold to the validation-optimal value")
            init = analyze("sklearn", "Test", DEFAULT_THR, LGD0, MARGIN0)
            summary = gr.Markdown(value=init[0])
            with gr.Row():
                cm_plot = gr.Plot(value=init[1])
                cost_plot = gr.Plot(value=init[2])

            inputs = [model, split_c, thr, lgd, margin]
            outputs = [summary, cm_plot, cost_plot]
            for comp in inputs:
                comp.change(analyze, inputs, outputs)
            btn.click(optimal_threshold, [model, lgd, margin], thr)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()