"""
Gradio dashboard for the diabetes screening models.

Run with:  uv run python main.py app      (after: uv run python main.py train)

Loads only what train.py saved in artifacts/ (fitted preprocessor, the
three trained models, metrics). Nothing is trained here: the models are
only used to *predict* on the held-out test set.
"""

import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_curve

from features import ARTIFACTS, CAPACITY, business_cost, engineer, load_raw, make_target

MODEL_NAMES = ["scikit-learn", "PyTorch manual", "PyTorch nn.Module"]
COLORS = {"scikit-learn": "#2a6fdb", "PyTorch manual": "#e8833a", "PyTorch nn.Module": "#2ca58d"}

# ---------------------------------------------------------------------------
# Load artifacts (fail with a clear message if training hasn't run)
# ---------------------------------------------------------------------------
if not (ARTIFACTS / "metrics.json").exists():
    raise SystemExit("No artifacts found. Run `uv run python main.py train` first.")

metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
split = json.loads((ARTIFACTS / "split.json").read_text())
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_model.joblib")
manual = torch.load(ARTIFACTS / "torch_manual.pt")
std_state = torch.load(ARTIFACTS / "torch_standard.pt")
coefs = pd.read_csv(ARTIFACTS / "coefficients.csv")

THRESHOLD = metrics["threshold"]

# Rebuild the exact test set from the saved split indices.
df = make_target(load_raw())
features_all = engineer(df)
test_df = df.iloc[split["test_idx"]]
X_test_df = features_all.iloc[split["test_idx"]]
y_test = test_df["target"].to_numpy()
X_test = preprocessor.transform(X_test_df).astype(np.float32)

# Inference only.
Xt = torch.tensor(X_test)
with torch.no_grad():
    p_manual = torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy()
    p_std = torch.sigmoid(Xt @ std_state["linear.weight"].squeeze(0) + std_state["linear.bias"]).numpy()
PROBS = {
    "scikit-learn": sk_model.predict_proba(X_test)[:, 1],
    "PyTorch manual": p_manual,
    "PyTorch nn.Module": p_std,
}

# ---------------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------------
def results_table() -> pd.DataFrame:
    rows = []
    for name, r in metrics["test_results"].items():
        rows.append({
            "Model": name,
            "ROC-AUC": round(r["roc_auc"], 4),
            "PR-AUC": round(r["pr_auc"], 4),
            "Log loss": round(r["log_loss"], 4) if "log_loss" in r else None,
            "Invited": r["invited"],
            "Diabetics found": r["tp"],
            "Precision": round(r["precision"], 3),
            "Recall": round(r["recall"], 3),
            "Cost (EUR)": f"{r['cost_eur']:,.0f}",
        })
    return pd.DataFrame(rows)


def agreement_table() -> pd.DataFrame:
    rows = []
    for name in ["PyTorch manual", "PyTorch nn.Module"]:
        a = metrics["agreement"][name]
        rows.append({
            "Compared with scikit-learn": name,
            "Max |Δ probability|": f"{a['max_abs_prob_diff']:.1e}",
            "Mean |Δ probability|": f"{a['mean_abs_prob_diff']:.1e}",
            "Max |Δ weight|": f"{a['max_abs_coef_diff']:.1e}",
            "Same invite decision": f"{a['same_invitations']}/{a['n_test']}",
            "Final training objective": f"{a['final_objective']:.6f}",
        })
    rows.append({"Compared with scikit-learn": "scikit-learn (reference)",
                 "Final training objective": f"{metrics['agreement']['sklearn_objective']:.6f}"})
    return pd.DataFrame(rows)


def ablation_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"Feature set (5-fold CV on train)": k,
         "ROC-AUC": f"{v['roc_auc']:.4f} ± {v['roc_auc_sd']:.4f}",
         "PR-AUC": f"{v['pr_auc']:.4f} ± {v['pr_auc_sd']:.4f}"}
        for k, v in metrics["ablation"].items()
    ])


def audit_table() -> pd.DataFrame:
    a = pd.DataFrame(metrics["audit"])
    a = a.rename(columns={"attribute": "Attribute", "group": "Group", "n": "N",
                          "prevalence": "Diabetes rate", "invited_share": "Share invited",
                          "recall": "Recall", "precision": "Precision", "roc_auc": "ROC-AUC"})
    return a.round(3)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def calibration_plot() -> go.Figure:
    """Prediction vs. actual: within each band of predicted risk, what share
    of people actually have diabetes? A perfect model sits on the diagonal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(dash="dash", color="gray")))
    for name, p in PROBS.items():
        bins = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y_test, "bin": bins}).groupby("bin", observed=True).mean()
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", name=name,
                                 line=dict(color=COLORS[name])))
    fig.update_layout(title="Predicted risk vs. actual diabetes rate (test set, 10 risk deciles)",
                      xaxis_title="Mean predicted probability", yaxis_title="Observed diabetes rate",
                      height=450)
    return fig


def prob_scatter() -> go.Figure:
    """Each dot is one test person: sklearn probability vs. PyTorch probability."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                             showlegend=False))
    for name in ["PyTorch manual", "PyTorch nn.Module"]:
        fig.add_trace(go.Scattergl(x=PROBS["scikit-learn"], y=PROBS[name], mode="markers", name=name,
                                   marker=dict(size=4, opacity=0.5, color=COLORS[name])))
    fig.update_layout(title="Do the three implementations agree? (one dot per test person)",
                      xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability", height=450)
    return fig


def roc_plot() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="gray")))
    for name, p in PROBS.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name, line=dict(color=COLORS[name])))
    fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", height=450)
    return fig


def loss_plot() -> go.Figure:
    fig = go.Figure()
    for name, hist in metrics["loss_history"].items():
        fig.add_trace(go.Scatter(y=hist, mode="lines", name=name, line=dict(color=COLORS[name])))
    fig.add_hline(y=metrics["agreement"]["sklearn_objective"], line_dash="dash",
                  annotation_text="scikit-learn optimum")
    fig.update_layout(title="PyTorch training objective per epoch (log scale)", xaxis_title="Epoch",
                      yaxis_title="Loss (BCE + L2)", yaxis_type="log", height=400)
    return fig


def coef_plot(top_n: int = 20) -> go.Figure:
    top = coefs.head(int(top_n)).iloc[::-1]
    fig = go.Figure()
    for name in MODEL_NAMES:
        fig.add_trace(go.Bar(y=top["feature"], x=top[name], name=name, orientation="h",
                             marker_color=COLORS[name]))
    fig.update_layout(barmode="group", height=max(400, 28 * int(top_n)),
                      title=f"Top {int(top_n)} weights (standardised inputs; positive = higher risk)",
                      xaxis_title="Weight (log-odds per 1 SD, or vs. reference category)")
    return fig


# Distribution explorer uses the full dataset after feature engineering
# (before scaling) so values are in their natural units.
DIST_DF = features_all.assign(Diabetes=np.where(df["target"] == 1, "Diabetes", "No diabetes"))
DIST_COLUMNS = [c for c in features_all.columns]


def distribution_plot(column: str) -> go.Figure:
    data = DIST_DF[[column, "Diabetes"]].copy()
    data[column] = data[column].fillna("Missing") if data[column].dtype == object else data[column]
    if pd.api.types.is_numeric_dtype(data[column]) and data[column].nunique() > 6:
        fig = px.histogram(data, x=column, color="Diabetes", barmode="overlay", histnorm="percent",
                           nbins=40, opacity=0.6,
                           color_discrete_map={"Diabetes": "#d1495b", "No diabetes": "#2a6fdb"})
        fig.update_layout(yaxis_title="% of group")
    else:
        data[column] = data[column].astype(str).replace("nan", "Missing")
        rate = data.groupby(column)["Diabetes"].apply(lambda s: (s == "Diabetes").mean()).reset_index(name="rate")
        count = data.groupby(column).size().reset_index(name="n")
        rate = rate.merge(count, on=column)
        fig = px.bar(rate, x=column, y="rate", text="n",
                     labels={"rate": "Diabetes rate", "n": "People"})
        fig.add_hline(y=df["target"].mean(), line_dash="dash", annotation_text="overall rate")
        fig.update_traces(texttemplate="n=%{text}", marker_color="#d1495b")
    fig.update_layout(title=f"{column}: distribution by diabetes status", height=450)
    return fig


def target_plot() -> go.Figure:
    counts = pd.Series(df["target"]).map({1: "Diabetes", 0: "No diabetes"}).value_counts()
    fig = px.bar(x=counts.index, y=counts.values, text=counts.values,
                 labels={"x": "", "y": "People"}, color=counts.index,
                 color_discrete_map={"Diabetes": "#d1495b", "No diabetes": "#2a6fdb"})
    fig.update_layout(title=f"Target: {df['target'].mean():.1%} of respondents report diagnosed diabetes",
                      showlegend=False, height=350)
    return fig


# ---------------------------------------------------------------------------
# Threshold / cost explorer
# ---------------------------------------------------------------------------
def threshold_view(model: str, threshold: float, cost_test: float, cost_missed: float):
    p = PROBS[model]
    selected = p >= threshold
    r = business_cost(y_test, selected)
    cost = r["fp"] * cost_test + r["fn"] * cost_missed

    # What random invitations would achieve with the same number of tests.
    exp_tp_random = r["invited"] * y_test.mean()
    random_cost = (r["invited"] - exp_tp_random) * cost_test + (y_test.sum() - exp_tp_random) * cost_missed

    z = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
    cm = go.Figure(go.Heatmap(
        z=z, x=["Not invited", "Invited"], y=["No diabetes", "Diabetes"],
        text=[[f"TN<br>{r['tn']}", f"FP (wasted test)<br>{r['fp']}"],
              [f"FN (missed case)<br>{r['fn']}", f"TP (found)<br>{r['tp']}"]],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix — {model}, threshold {threshold:.3f}",
                     xaxis_title="Decision", yaxis_title="Actual", height=400)

    over = r["invited_share"] > CAPACITY + 1e-9
    budget = (f"⚠️ **Over budget:** invites {r['invited_share']:.1%} of people, budget covers {CAPACITY:.0%}."
              if over else f"✅ Within budget: invites {r['invited_share']:.1%} of people (budget {CAPACITY:.0%}).")
    kpis = f"""
{budget}

| | This model | Random invitations, same number of tests |
|---|---|---|
| Tests sent | {r['invited']:,} | {r['invited']:,} |
| Diabetics found | **{r['tp']:,}** of {int(y_test.sum()):,} | {exp_tp_random:,.0f} |
| Precision (share of tests that find a case) | **{r['precision']:.1%}** | {y_test.mean():.1%} |
| Recall (share of all cases found) | **{r['recall']:.1%}** | {exp_tp_random / y_test.sum():.1%} |
| Business cost | **€{cost:,.0f}** | €{random_cost:,.0f} |

Cost = wasted tests × €{cost_test:,.0f} + missed cases × €{cost_missed:,.0f}, on the {len(y_test):,}-person test set.
The default threshold ({THRESHOLD:.3f}) is the one that fills exactly the {CAPACITY:.0%} budget, chosen on training data.
"""

    grid = np.linspace(0.01, 0.9, 180)
    costs, shares = [], []
    for t in grid:
        rr = business_cost(y_test, p >= t)
        costs.append(rr["fp"] * cost_test + rr["fn"] * cost_missed)
        shares.append(rr["invited_share"])
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=grid, y=costs, name="Cost (EUR)", line=dict(color="#d1495b")))
    curve.add_trace(go.Scatter(x=grid, y=np.array(shares) * 100, name="% of people invited",
                               yaxis="y2", line=dict(color="#2a6fdb", dash="dot")))
    curve.add_vline(x=threshold, line_dash="dash", annotation_text="current")
    curve.add_vline(x=THRESHOLD, line_dash="dot", line_color="green", annotation_text="10% budget",
                    annotation_position="bottom right")
    curve.update_layout(title="Cost and share invited across thresholds", xaxis_title="Threshold",
                        yaxis=dict(title="Cost (EUR)"),
                        yaxis2=dict(title="% invited", overlaying="y", side="right"), height=400)
    return cm, kpis, curve


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
INTRO = f"""
# Diabetes screening prioritisation — New York State (CDC BRFSS 2022)

A health department can afford free glucose tests for **{CAPACITY:.0%}** of people. Which people should get the invitations?
The same L2-regularised logistic regression is trained three ways (scikit-learn, a manual PyTorch loop, and
`nn.Module` + `torch.optim`). This dashboard **only loads the saved models**; nothing is retrained here.

Test set: **{metrics['n_test']:,}** people (stratified 20% hold-out), {metrics['n_inputs']} model inputs,
C = {metrics['best_C']}, learning rate = {metrics['lr']}, {metrics['epochs']} full-batch epochs.
"""

with gr.Blocks(title="Diabetes screening models") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Model comparison"):
        gr.Markdown("### Test-set results vs. naive baselines (each selects the top 10%)")
        gr.Dataframe(results_table(), interactive=False)
        with gr.Row():
            gr.Plot(calibration_plot())
            gr.Plot(prob_scatter())
        gr.Markdown("### Agreement between the three implementations")
        gr.Dataframe(agreement_table(), interactive=False)
        with gr.Row():
            gr.Plot(roc_plot())
            gr.Plot(loss_plot())

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            model_dd = gr.Dropdown(MODEL_NAMES, value="scikit-learn", label="Model")
            thr = gr.Slider(0.01, 0.9, value=round(THRESHOLD, 3), step=0.005, label="Decision threshold (invite if risk ≥)")
        with gr.Row():
            c_test = gr.Number(value=metrics["cost_wasted_test"], label="Cost of a wasted test (EUR)")
            c_miss = gr.Number(value=metrics["cost_missed_case"], label="Cost of a missed case (EUR)")
        kpi_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        inputs = [model_dd, thr, c_test, c_miss]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, kpi_md, cost_plot])
        demo.load(threshold_view, inputs, [cm_plot, kpi_md, cost_plot])

    with gr.Tab("Data distributions"):
        gr.Plot(target_plot())
        col_dd = gr.Dropdown(DIST_COLUMNS, value="bmi", label="Feature (after engineering, before scaling)")
        dist = gr.Plot()
        col_dd.change(distribution_plot, col_dd, dist)
        demo.load(distribution_plot, col_dd, dist)

    with gr.Tab("Feature weights"):
        top_n = gr.Slider(5, len(coefs), value=20, step=1, label="How many weights to show")
        w_plot = gr.Plot()
        top_n.change(coef_plot, top_n, w_plot)
        demo.load(coef_plot, top_n, w_plot)

    with gr.Tab("Feature engineering & fairness"):
        gr.Markdown("### What feature engineering buys (5-fold CV on the training set)")
        gr.Dataframe(ablation_table(), interactive=False)
        gr.Markdown("The *leaky* row includes columns that are consequences of a diagnosis "
                    "(recent check-up, chest scan, vaccines, alcohol). It scores higher for the wrong reason, so it is not used.")
        gr.Markdown("### Audit by sex and race (test set, scikit-learn model at the budget threshold)")
        gr.Dataframe(audit_table(), interactive=False)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=not os.environ.get("CI"))
