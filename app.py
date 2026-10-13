"""Assignment 1: Gradio dashboard (Telco churn).

Loads the artifacts written by train.py. It never trains anything.
"""
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

# Definitions only: importing train.py does not train anything.
from train import ADDONS, LogReg, engineer_features

ARTIFACTS = Path("artifacts")
MODELS = ["sklearn", "manual", "module"]
LABELS = {"sklearn": "scikit-learn", "manual": "manual PyTorch", "module": "nn.Module + optim"}
COLORS = {"sklearn": "#1f77b4", "manual": "#ff7f0e", "module": "#2ca02c"}

if not (ARTIFACTS / "metrics.json").exists():
    raise SystemExit("No artifacts found. Run `uv run python main.py train` first.")

# ---------- load saved artifacts ----------
META = json.loads((ARTIFACTS / "metrics.json").read_text())
PRED = np.load(ARTIFACTS / "predictions.npz")
DATA = pd.read_csv(ARTIFACTS / "clean_data.csv")
PREP = joblib.load(ARTIFACTS / "preprocessor.joblib")
SK = joblib.load(ARTIFACTS / "sklearn_model.joblib")
_manual = torch.load(ARTIFACTS / "manual_params.pt", weights_only=True)
W_MAN = _manual["w"].numpy().ravel()
B_MAN = _manual["b"].numpy().ravel()
MOD = LogReg(len(W_MAN)).double()
MOD.load_state_dict(torch.load(ARTIFACTS / "module_state.pt", weights_only=True))
MOD.eval()

COST_FN, COST_FP = META["cost_fn"], META["cost_fp"]
THRESH = {r["model"]: r["threshold"] for r in META["results"] if r["model"] in MODELS}
RESULTS_DF = pd.DataFrame(META["results"]).round(4)


def split_key(split):
    return "val" if split == "Validation" else "test"


def confusion(y, p, thr):
    pred = p >= thr
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tp = int(np.sum((y == 1) & pred))
    return tn, fp, fn, tp


def safe_div(a, b):
    return a / b if b else 0.0


# ---------- tab 1: compare the three models ----------
def compare_models(split):
    key = split_key(split)
    y = PRED[f"y_{key}"]
    P = {m: PRED[f"p_{key}_{m}"] for m in MODELS}

    cal = go.Figure()
    cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(dash="dash", color="gray")))
    for m, width in zip(MODELS, [8, 5, 2]):  # different widths so overlapping lines stay visible
        bins = pd.qcut(P[m], 10, duplicates="drop")
        grp = pd.DataFrame({"p": P[m], "y": y}).groupby(bins, observed=True).mean()
        cal.add_trace(go.Scatter(x=grp["p"], y=grp["y"], mode="lines+markers", name=LABELS[m],
                                 line=dict(color=COLORS[m], width=width)))
    cal.update_layout(
        title=f"Predicted probability vs. actual churn rate ({split.lower()}, 10 equal-size bins)",
        xaxis_title="mean predicted churn probability", yaxis_title="actual churn rate")

    agree = go.Figure()
    for m in ["manual", "module"]:
        agree.add_trace(go.Scatter(x=P["sklearn"], y=P[m], mode="markers", name=LABELS[m],
                                   marker=dict(color=COLORS[m], size=4, opacity=0.5)))
    agree.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x",
                               line=dict(dash="dash", color="gray")))
    agree.update_layout(title="Do the three agree? PyTorch predictions vs. scikit-learn",
                        xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability")

    pairs = [("sklearn", "manual"), ("sklearn", "module"), ("manual", "module")]
    lines = [f"- {LABELS[a]} vs. {LABELS[b]}: {np.max(np.abs(P[a] - P[b])):.1e}" for a, b in pairs]
    md = "**Largest per-customer probability difference between models**\n\n" + "\n".join(lines)
    return cal, agree, md


# ---------- tab 2: distributions ----------
FEATURES = [c for c in DATA.columns if c != "Churn"]


def target_figure():
    counts = DATA["Churn"].value_counts().sort_index()
    n = len(DATA)
    vals = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    fig = go.Figure(go.Bar(x=["Stayed", "Churned"], y=vals,
                           text=[f"{v} ({v / n:.1%})" for v in vals],
                           marker_color=["#1f77b4", "#d62728"]))
    fig.update_layout(title="Target distribution (Churn)", yaxis_title="customers")
    return fig


def show_distribution(col):
    s = DATA[col]
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
        for val, name, color in [(0, "Stayed", "#1f77b4"), (1, "Churned", "#d62728")]:
            fig.add_trace(go.Histogram(x=s[DATA["Churn"] == val], name=name,
                                       opacity=0.6, marker_color=color, nbinsx=40))
        fig.update_layout(barmode="overlay", xaxis_title=col, yaxis_title="customers")
    else:
        g = DATA.groupby(col)["Churn"].agg(["mean", "count"]).reset_index()
        fig.add_trace(go.Bar(x=g[col].astype(str), y=g["mean"],
                             text=[f"n={n}" for n in g["count"]], marker_color="#d62728"))
        fig.add_hline(y=DATA["Churn"].mean(), line_dash="dash",
                      annotation_text="overall churn rate")
        fig.update_layout(xaxis_title=col, yaxis_title="churn rate", yaxis_tickformat=".0%")
    fig.update_layout(title=f"{col} vs. churn")
    return fig


# ---------- tab 3: threshold and business cost ----------
def threshold_view(model, split, thr, c_fn, c_fp):
    key = split_key(split)
    y, p = PRED[f"y_{key}"], PRED[f"p_{key}_{model}"]
    n = len(y)
    tn, fp, fn, tp = confusion(y, p, thr)
    cost = c_fn * fn + c_fp * fp
    cost_nobody = c_fn * int(y.sum())
    cost_all = c_fp * int((y == 0).sum())

    grid = np.linspace(0.01, 0.99, 99)
    curve = []
    for t in grid:
        _, f_p, f_n, _ = confusion(y, p, t)
        curve.append(c_fn * f_n + c_fp * f_p)
    best_t = float(grid[int(np.argmin(curve))])

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted: stay", "Predicted: churn"],
        y=["Actual: stay", "Actual: churn"],
        text=[[f"TN {tn}", f"FP {fp}"], [f"FN {fn}", f"TP {tp}"]],
        texttemplate="%{text}", textfont=dict(size=18),
        colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix at threshold {thr:.2f}",
                     yaxis=dict(autorange="reversed"))

    cv = go.Figure()
    cv.add_trace(go.Scatter(x=grid, y=curve, mode="lines", name="total cost"))
    cv.add_trace(go.Scatter(x=[best_t], y=[min(curve)], mode="markers", name="cheapest threshold",
                            marker=dict(symbol="star", size=14, color="#d62728")))
    cv.add_vline(x=thr, line_dash="dash")
    cv.update_layout(title="Total cost vs. threshold", xaxis_title="threshold",
                     yaxis_title="cost ($)")

    summary = (
        f"### Cost at this threshold: **${cost:,.0f}** (${cost / n:,.2f} per customer)\n"
        f"- Missed churners (FN): {fn} x ${c_fn:,.0f} = ${c_fn * fn:,.0f}\n"
        f"- Unnecessary offers (FP): {fp} x ${c_fp:,.0f} = ${c_fp * fp:,.0f}\n"
        f"- Contact nobody: ${cost_nobody:,.0f}. Contact everyone: ${cost_all:,.0f}.\n"
        f"- Recall {safe_div(tp, tp + fn):.1%}, precision {safe_div(tp, tp + fp):.1%}\n\n"
        f"Cheapest threshold on this split with these costs: **{best_t:.2f}**. "
        f"The pipeline's own threshold for this model, chosen on validation only: "
        f"**{THRESH[model]:.2f}**. (Reading the cheapest threshold off the test split is for "
        f"exploring only, because the reported result must not tune on test.)"
    )
    return cm, cv, summary


# ---------- tab 4: score one customer with the saved models ----------
CAT_INPUTS = (["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService"] + ADDONS + ["PaperlessBilling", "PaymentMethod", "Contract"])
INPUT_NAMES = ["tenure", "MonthlyCharges", "SeniorCitizen"] + CAT_INPUTS


def score_customer(*vals):
    row = dict(zip(INPUT_NAMES, vals))
    row["SeniorCitizen"] = int(row["SeniorCitizen"])
    X = PREP.transform(engineer_features(pd.DataFrame([row]))).astype(np.float64)
    with torch.no_grad():
        probs = {
            "sklearn": float(SK.predict_proba(X)[0, 1]),
            "manual": float((1 / (1 + np.exp(-(X @ W_MAN + B_MAN))))[0]),
            "module": float(torch.sigmoid(MOD(torch.tensor(X))).item()),
        }
    lines = ["| Model | Churn probability | Decision at its pipeline threshold |", "|---|---|---|"]
    for m in MODELS:
        act = "contact with offer" if probs[m] >= THRESH[m] else "no offer"
        lines.append(f"| {LABELS[m]} | {probs[m]:.1%} | {act} (threshold {THRESH[m]:.2f}) |")
    return "\n".join(lines)


# ---------- layout ----------
with gr.Blocks(title="Telco churn: three models") as demo:
    gr.Markdown("# Telco churn: scikit-learn vs. manual PyTorch vs. nn.Module\n"
                "Every number here comes from the artifacts saved by `train.py`. "
                "Nothing is retrained.")

    with gr.Tab("Compare models"):
        gr.Markdown("**Final test-set results** (each model's threshold chosen on validation)")
        gr.Dataframe(value=RESULTS_DF, interactive=False)
        split_cmp = gr.Radio(["Validation", "Test"], value="Test", label="Data split")
        cal_plot = gr.Plot()
        agree_plot = gr.Plot()
        agree_md = gr.Markdown()
        split_cmp.change(compare_models, split_cmp, [cal_plot, agree_plot, agree_md])

    with gr.Tab("Distributions"):
        gr.Plot(value=target_figure())
        col_dd = gr.Dropdown(FEATURES, value="tenure", label="Feature")
        dist_plot = gr.Plot()
        col_dd.change(show_distribution, col_dd, dist_plot)

    with gr.Tab("Threshold and cost"):
        with gr.Row():
            model_rd = gr.Radio(MODELS, value="sklearn", label="Model")
            split_rd = gr.Radio(["Validation", "Test"], value="Test", label="Data split")
        thr_sl = gr.Slider(0.01, 0.99, value=round(THRESH["sklearn"], 2), step=0.01,
                           label="Decision threshold (contact if probability >= threshold)")
        with gr.Row():
            cfn_sl = gr.Slider(50, 1000, value=COST_FN, step=10,
                               label="Cost of a missed churner ($)")
            cfp_sl = gr.Slider(5, 200, value=COST_FP, step=5,
                               label="Cost of an unnecessary offer ($)")
        use_btn = gr.Button("Use the pipeline's threshold for this model")
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        summary_md = gr.Markdown()
        th_inputs = [model_rd, split_rd, thr_sl, cfn_sl, cfp_sl]
        th_outputs = [cm_plot, curve_plot, summary_md]
        for comp in th_inputs:
            comp.change(threshold_view, th_inputs, th_outputs)
        use_btn.click(lambda m: round(THRESH[m], 2), model_rd, thr_sl)

    with gr.Tab("Score a customer"):
        gr.Markdown("Enter a customer; all three saved models score it.")
        score_inputs = [
            gr.Slider(0, 72, value=int(DATA["tenure"].median()), step=1, label="tenure (months)"),
            gr.Slider(float(np.floor(DATA["MonthlyCharges"].min())),
                      float(np.ceil(DATA["MonthlyCharges"].max())),
                      value=float(DATA["MonthlyCharges"].median()), step=0.05,
                      label="MonthlyCharges"),
            gr.Radio([("No", 0), ("Yes", 1)], value=0, label="SeniorCitizen"),
        ]
        for c in CAT_INPUTS:
            score_inputs.append(gr.Dropdown(sorted(DATA[c].unique()), value=DATA[c].mode()[0], label=c))
        score_btn = gr.Button("Score customer")
        score_md = gr.Markdown()
        score_btn.click(score_customer, score_inputs, score_md)

    demo.load(compare_models, split_cmp, [cal_plot, agree_plot, agree_md])
    demo.load(show_distribution, col_dd, dist_plot)
    demo.load(threshold_view, th_inputs, th_outputs)

if __name__ == "__main__":
    demo.launch()