"""
Assignment 1: Gradio dashboard for the credit-risk models.

Loads ONLY what train.ipynb saved in artifacts/ (feature transformer, three
trained models, split, metrics). Nothing is trained here: the models are only
used for inference.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_curve

from credit_pipeline import (ARTIFACTS, HOME_LEVELS, INTENT_LEVELS,
                             LOSS_GIVEN_DEFAULT, PROFIT_MARGIN, TARGET,
                             LogisticRegressionModule, business_cost,
                             load_clean_data)

MODEL_NAMES = ["sklearn", "manual_torch", "torch_module"]
COLORS = {"sklearn": "#2a78d6", "manual_torch": "#e0762b", "torch_module": "#2f9e6e", "baseline": "#8a8a8a"}

# ---------------------------------------------------------------------------
# Load artifacts (no training anywhere below)
# ---------------------------------------------------------------------------
features = joblib.load(ARTIFACTS / "features.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
manual = torch.load(ARTIFACTS / "manual_torch.pt")
std_model = LogisticRegressionModule(len(features.feature_names_))
std_model.load_state_dict(torch.load(ARTIFACTS / "torch_module.pt"))
std_model.eval()
metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
coefficients = pd.read_csv(ARTIFACTS / "coefficients.csv", index_col=0)

df = load_clean_data()
df["split"] = pd.read_csv(ARTIFACTS / "split.csv", index_col="row")["split"]
test_df = df[df["split"] == "test"].copy()
train_df = df[df["split"] == "train"].copy()


def predict_all(frame: pd.DataFrame) -> dict:
    X = features.transform(frame)
    Xt = torch.tensor(X)
    with torch.no_grad():
        return {
            "sklearn": sk_model.predict_proba(X)[:, 1],
            "manual_torch": torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy(),
            "torch_module": torch.sigmoid(std_model(Xt)).numpy(),
        }


test_probs = predict_all(test_df)
y_test = test_df[TARGET].to_numpy()
for name, p in test_probs.items():
    test_df[f"p_{name}"] = p

# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------
def results_table() -> pd.DataFrame:
    t = pd.DataFrame(metrics["test_results"])
    t["total_cost"] = t["total_cost"].map(lambda v: f"{v:,.0f}")
    return t.round(4)


def calibration_plot() -> go.Figure:
    """Predicted vs actual: bin test applicants by predicted probability, compare to observed default rate."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(dash="dash", color="#aaa")))
    for name in MODEL_NAMES:
        bins = pd.qcut(test_df[f"p_{name}"], 10, duplicates="drop")
        g = test_df.groupby(bins, observed=True).agg(pred=(f"p_{name}", "mean"), actual=(TARGET, "mean"))
        fig.add_trace(go.Scatter(x=g["pred"], y=g["actual"], mode="lines+markers", name=name,
                                 line=dict(color=COLORS[name])))
    fig.add_hline(y=metrics["base_rate"], line_dash="dot", line_color=COLORS["baseline"],
                  annotation_text="naive baseline (train default rate)")
    fig.update_layout(title="Predicted vs actual default rate (test set, deciles)",
                      xaxis_title="mean predicted P(default)", yaxis_title="actual default rate",
                      height=450)
    return fig


def roc_plot() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="baseline (AUC 0.5)",
                             line=dict(dash="dash", color=COLORS["baseline"])))
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y_test, test_df[f"p_{name}"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name, line=dict(color=COLORS[name])))
    fig.update_layout(title="ROC curves (test)", xaxis_title="false positive rate",
                      yaxis_title="true positive rate (defaults caught)", height=450)
    return fig


def agreement_plot(other: str) -> go.Figure:
    fig = px.scatter(test_df, x="p_sklearn", y=f"p_{other}", color=test_df[TARGET].map({0: "repaid", 1: "defaulted"}),
                     opacity=0.4, labels={"p_sklearn": "sklearn P(default)", f"p_{other}": f"{other} P(default)",
                                          "color": "actual"},
                     title=f"Per-applicant agreement: sklearn vs {other}")
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#555"),
                             showlegend=False))
    fig.update_layout(height=450)
    return fig


def coef_plot() -> go.Figure:
    c = coefficients.drop(index="(intercept)")
    long = c.reset_index(names="feature").melt(id_vars="feature", var_name="model", value_name="coef")
    fig = px.bar(long, y="feature", x="coef", color="model", barmode="group", orientation="h",
                 color_discrete_map=COLORS, title="Coefficients (standardised features)")
    fig.update_layout(height=650)
    return fig


def loss_plot() -> go.Figure:
    h = metrics["loss_history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(h["manual_torch_every_50_steps"])) * 50,
                             y=h["manual_torch_every_50_steps"], name="manual_torch (per GD step)",
                             line=dict(color=COLORS["manual_torch"])))
    fig.add_trace(go.Scatter(x=np.arange(1, len(h["torch_module_per_epoch"]) + 1),
                             y=h["torch_module_per_epoch"], name="torch_module (per epoch)",
                             line=dict(color=COLORS["torch_module"]), xaxis="x2"))
    fig.update_layout(title="Training loss", yaxis_title="train loss (log-loss + L2)", height=400,
                      xaxis=dict(title="full-batch gradient steps"),
                      xaxis2=dict(title="mini-batch epochs", overlaying="x", side="top"))
    return fig


# ---------------------------------------------------------------------------
# Tab 2: data distributions
# ---------------------------------------------------------------------------
engineered_train = features.unscaled(train_df)
DIST_OPTIONS = sorted(set(["person_age", "person_income", "person_emp_length", "loan_amnt",
                           "loan_percent_income", "cb_person_cred_hist_length", "person_home_ownership",
                           "loan_intent", "cb_person_default_on_file", "loan_grade", "loan_int_rate"])
                      | set(features.feature_names_))


def distribution_plot(column: str) -> go.Figure:
    data = train_df.join(engineered_train[[c for c in engineered_train.columns if c not in train_df]])
    outcome = data[TARGET].map({0: "repaid", 1: "defaulted"})
    if data[column].dtype == object or data[column].nunique() <= 8:
        rate = data.groupby(column)[TARGET].agg(["mean", "size"]).reset_index()
        fig = px.bar(rate, x=column, y="mean", text="size",
                     labels={"mean": "default rate", "size": "n"},
                     title=f"Default rate by {column} (bar label = number of applicants, train split)")
        fig.update_yaxes(tickformat=".0%")
    else:
        clipped = data[column].clip(upper=data[column].quantile(0.99))
        fig = px.histogram(x=clipped, color=outcome, barmode="overlay", histnorm="probability density",
                           nbins=50, opacity=0.6, labels={"x": column, "color": "outcome"},
                           title=f"{column} by outcome (train split, top 1% clipped for readability)")
    fig.update_layout(height=450)
    return fig


def target_plot() -> go.Figure:
    counts = df[TARGET].map({0: "repaid", 1: "defaulted"}).value_counts()
    fig = px.bar(x=counts.index, y=counts.values, labels={"x": "loan_status", "y": "applicants"},
                 title=f"Target distribution (all data): {df[TARGET].mean():.1%} default")
    fig.update_layout(height=350)
    return fig


# ---------------------------------------------------------------------------
# Tab 3: threshold + business cost
# ---------------------------------------------------------------------------
def threshold_view(model_name, threshold, lgd, margin):
    p = test_df[f"p_{model_name}"]
    c = business_cost(y_test, p, test_df["loan_amnt"], threshold, lgd, margin)
    base = business_cost(y_test, np.zeros(len(y_test)), test_df["loan_amnt"], 1.0, lgd, margin)

    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       x=["approve (pred. repay)", "reject (pred. default)"],
                       y=["actually repaid", "actually defaulted"],
                       title=f"Confusion matrix, {model_name}, threshold {threshold:.2f} (test)")
    cm_fig.update_layout(height=420, coloraxis_showscale=False)

    ts = np.round(np.arange(0.02, 0.99, 0.01), 2)
    costs = [business_cost(y_test, p, test_df["loan_amnt"], t, lgd, margin)["total_cost"] for t in ts]
    curve = go.Figure(go.Scatter(x=ts, y=costs, mode="lines", name="model"))
    curve.add_hline(y=base["total_cost"], line_dash="dot", line_color=COLORS["baseline"],
                    annotation_text="approve everyone (baseline)")
    curve.add_vline(x=threshold, line_dash="dash", line_color="#d33")
    best_t = ts[int(np.argmin(costs))]
    curve.update_layout(title=f"Total cost vs threshold (test minimum at {best_t:.2f}; "
                              f"theory: margin/(margin+LGD) = {margin / (margin + lgd):.2f})",
                        xaxis_title="threshold on P(default)", yaxis_title="cost", height=420,
                        title_font_size=13)

    n = len(y_test)
    approved = c["tn"] + c["fn"]
    summary = (
        f"### Business cost on {n:,} test applicants\n"
        f"- **Total cost: {c['total_cost']:,.0f}** (baseline 'approve everyone': {base['total_cost']:,.0f} → "
        f"saving **{base['total_cost'] - c['total_cost']:,.0f}**)\n"
        f"- Losses from approved defaulters (FN = {c['fn']}): {c['default_losses']:,.0f}\n"
        f"- Profit lost on rejected good customers (FP = {c['fp']}): {c['lost_profit']:,.0f}\n"
        f"- Approval rate: {approved / n:.1%}; defaults caught: {c['tp'] / max(c['tp'] + c['fn'], 1):.1%}\n"
        f"- Threshold chosen on *validation* during training: {metrics['thresholds'][model_name]:.2f}"
    )
    return cm_fig, curve, summary


# ---------------------------------------------------------------------------
# Tab 4: score one applicant
# ---------------------------------------------------------------------------
def score_applicant(age, income, emp_length, home, intent, amount, prior_default, hist, threshold):
    row = pd.DataFrame([{
        "person_age": age, "person_income": income, "person_emp_length": emp_length,
        "person_home_ownership": home, "loan_intent": intent, "loan_amnt": amount,
        "loan_percent_income": amount / income, "cb_person_default_on_file": prior_default,
        "cb_person_cred_hist_length": hist,
    }])
    probs = predict_all(row)
    lines = [f"| model | P(default) | decision at {threshold:.2f} |", "|---|---|---|"]
    for name, p in probs.items():
        lines.append(f"| {name} | {p[0]:.1%} | {'**REJECT**' if p[0] >= threshold else 'approve'} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Credit risk: loan approval") as demo:
    gr.Markdown("# Loan approval: probability of default, trained three ways\n"
                "Same L2 logistic regression via scikit-learn, a manual PyTorch loop and nn.Module + torch.optim. "
                "All numbers are on the held-out **test** set unless stated otherwise.")

    with gr.Tab("Model comparison"):
        gr.Dataframe(results_table(), label="Test-set comparison (thresholds chosen on validation)")
        with gr.Row():
            gr.Plot(calibration_plot())
            gr.Plot(roc_plot())
        with gr.Row():
            other = gr.Radio(["manual_torch", "torch_module"], value="torch_module", label="Compare sklearn with")
        agree = gr.Plot(agreement_plot("torch_module"))
        other.change(agreement_plot, other, agree)
        with gr.Row():
            gr.Plot(coef_plot())
            gr.Plot(loss_plot())

    with gr.Tab("Data"):
        gr.Plot(target_plot())
        col = gr.Dropdown(DIST_OPTIONS, value="loan_to_income", label="Feature")
        dist = gr.Plot(distribution_plot("loan_to_income"))
        col.change(distribution_plot, col, dist)

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            m = gr.Dropdown(MODEL_NAMES, value="sklearn", label="Model")
            t = gr.Slider(0.02, 0.98, value=metrics["thresholds"]["sklearn"], step=0.01,
                          label="Reject if P(default) ≥")
        with gr.Row():
            lgd = gr.Slider(0.1, 1.0, value=LOSS_GIVEN_DEFAULT, step=0.05, label="Loss given default (share of loan lost)")
            mg = gr.Slider(0.01, 0.30, value=PROFIT_MARGIN, step=0.01, label="Profit margin on a repaid loan")
        summary = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        inputs = [m, t, lgd, mg]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, cost_plot, summary])
        demo.load(threshold_view, inputs, [cm_plot, cost_plot, summary])

    with gr.Tab("Score an applicant"):
        with gr.Row():
            a_age = gr.Number(30, label="Age")
            a_inc = gr.Number(40000, label="Yearly income")
            a_emp = gr.Number(3, label="Years employed")
            a_hist = gr.Number(5, label="Credit history (years)")
        with gr.Row():
            a_home = gr.Dropdown(HOME_LEVELS, value="RENT", label="Home ownership")
            a_int = gr.Dropdown(INTENT_LEVELS, value="PERSONAL", label="Loan purpose")
            a_amt = gr.Number(12000, label="Loan amount")
            a_prior = gr.Radio(["N", "Y"], value="N", label="Previous default on file")
        a_t = gr.Slider(0.02, 0.98, value=metrics["thresholds"]["sklearn"], step=0.01, label="Threshold")
        btn = gr.Button("Score")
        out = gr.Markdown()
        btn.click(score_applicant, [a_age, a_inc, a_emp, a_home, a_int, a_amt, a_prior, a_hist, a_t], out)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
