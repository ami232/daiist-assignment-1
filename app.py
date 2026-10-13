"""
Gradio dashboard (run with: uv run python main.py app).

Loads the artifacts written by train.py and only runs inference with them;
nothing is trained or fitted here.
"""
import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

from pipeline import (
    ARTIFACTS, CATEGORICAL, NUMERIC, LogisticRegressionTorch, business_cost,
    engineer, load_data, time_split,
)

MODEL_LABELS = {
    "sklearn": "scikit-learn",
    "torch_manual": "PyTorch (manual loop)",
    "torch_module": "PyTorch (nn.Module + optim)",
}
MODEL_COLORS = {"sklearn": "#1f3a5f", "torch_manual": "#2a9d8f", "torch_module": "#e9a03b"}
OUTCOME_COLORS = {"Repaid": "#4c78a8", "Defaulted": "#c0392b"}

# --- Load data and saved artifacts (inference only) -------------------------
metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_model.joblib")
manual = torch.load(ARTIFACTS / "torch_manual.pt")
loss_history = pd.read_csv(ARTIFACTS / "loss_history.csv")

df = load_data()
train_df, test_df = time_split(df)
X_test = preprocessor.transform(engineer(test_df))
y_test = test_df["Default"].to_numpy()
loan_test = test_df["loan_amnt"].to_numpy()

torch_model = LogisticRegressionTorch(X_test.shape[1])
torch_model.load_state_dict(torch.load(ARTIFACTS / "torch_module.pt"))
torch_model.eval()

X_test_t = torch.tensor(X_test, dtype=torch.float64)
with torch.no_grad():
    PROBS = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "torch_manual": torch.sigmoid(X_test_t @ manual["w"] + manual["b"]).numpy(),
        "torch_module": torch.sigmoid(torch_model(X_test_t)).numpy(),
    }

BEST_THRESHOLD = metrics["best_threshold"]
APPROVE_ALL_COST = business_cost(y_test, np.zeros(len(y_test)), loan_test, threshold=1.0)
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

features_all = engineer(df)
features_all["Outcome"] = np.where(df["Default"] == 1, "Defaulted", "Repaid")


# --- Tab 1: model comparison --------------------------------------------------
def metrics_table():
    cost_col = f"Cost at threshold {BEST_THRESHOLD:.2f} (USD)"
    rows = [{
        "Model": "Baseline (approve everyone)",
        "ROC AUC": 0.5,
        "Log loss": round(log_loss(y_test, np.full(len(y_test), train_df["Default"].mean())), 4),
        cost_col: f"{APPROVE_ALL_COST:,.0f}",
    }]
    for key, p in PROBS.items():
        rows.append({
            "Model": MODEL_LABELS[key],
            "ROC AUC": round(roc_auc_score(y_test, p), 4),
            "Log loss": round(log_loss(y_test, p), 4),
            cost_col: f"{business_cost(y_test, p, loan_test, BEST_THRESHOLD):,.0f}",
        })
    return pd.DataFrame(rows)


def calibration_plot():
    """Predicted vs actual: group test loans into 10 bins of predicted risk."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], mode="lines", name="Perfect calibration",
                             line=dict(color="grey", dash="dash")))
    for key, p in PROBS.items():
        bins = pd.qcut(p, 10, duplicates="drop")
        grouped = pd.DataFrame({"p": p, "y": y_test}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(x=grouped["p"], y=grouped["y"], mode="lines+markers",
                                 name=MODEL_LABELS[key], line=dict(color=MODEL_COLORS[key])))
    fig.update_layout(title="Predicted vs actual default rate (2015 test loans, 10 risk bins)",
                      xaxis_title="Average predicted probability of default",
                      yaxis_title="Actual default rate", height=450)
    return fig


def agreement_plot():
    fig = go.Figure()
    for key in ["torch_manual", "torch_module"]:
        fig.add_trace(go.Scattergl(x=PROBS["sklearn"], y=PROBS[key], mode="markers",
                                   name=MODEL_LABELS[key], opacity=0.5,
                                   marker=dict(size=4, color=MODEL_COLORS[key])))
    fig.update_layout(title="Do the three implementations agree? (each dot is one test loan)",
                      xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability",
                      height=450)
    return fig


def loss_plot():
    fig = go.Figure()
    for key in ["torch_manual", "torch_module"]:
        fig.add_trace(go.Scatter(x=loss_history["epoch"] + 1, y=loss_history[key], mode="lines",
                                 name=MODEL_LABELS[key], line=dict(color=MODEL_COLORS[key])))
    fig.update_layout(title="Training loss of the two PyTorch versions", xaxis_title="Epoch",
                      yaxis_title="Regularized log loss", xaxis_type="log", height=400)
    return fig


# --- Tab 2: distributions -------------------------------------------------------
def feature_plot(feature):
    fig = go.Figure()
    if feature in CATEGORICAL:
        count = features_all[feature].value_counts()
        rate = (features_all["Outcome"] == "Defaulted").groupby(features_all[feature]).mean()
        order = count.index
        fig.add_trace(go.Bar(x=order, y=rate[order], marker_color=OUTCOME_COLORS["Defaulted"],
                             text=[f"n={count[c]}" for c in order], textposition="outside"))
        fig.update_layout(title=f"Default rate by {feature}", yaxis_title="Default rate")
    else:
        for outcome, color in OUTCOME_COLORS.items():
            values = features_all.loc[features_all["Outcome"] == outcome, feature]
            fig.add_trace(go.Histogram(x=values, name=outcome, marker_color=color,
                                       opacity=0.6, histnorm="probability density", nbinsx=40))
        fig.update_layout(barmode="overlay", title=f"Distribution of {feature} by outcome",
                          xaxis_title=feature, yaxis_title="Density")
    fig.update_layout(height=450)
    return fig


def target_plot():
    by_year = df.groupby(df["issue_d"].dt.year)["Default"].agg(["mean", "size"])
    colors = ["#1f3a5f" if year < 2015 else "#e9a03b" for year in by_year.index]
    fig = go.Figure(go.Bar(x=by_year.index.astype(str), y=by_year["mean"], marker_color=colors,
                           text=[f"n={n}" for n in by_year["size"]], textposition="outside"))
    fig.update_layout(title="Target: default rate by issue year (blue = training, orange = test)",
                      yaxis_title="Default rate", height=400)
    return fig


# --- Tab 3: threshold and business cost ---------------------------------------
def threshold_view(model_label, threshold):
    key = {v: k for k, v in MODEL_LABELS.items()}[model_label]
    p = PROBS[key]
    reject = (p >= threshold).astype(int)
    cm = confusion_matrix(y_test, reject, labels=[0, 1])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Approved", "Rejected"], y=["Repaid", "Defaulted"],
        text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
    ))
    fig_cm.update_layout(title=f"Confusion matrix at threshold {threshold:.2f}",
                         xaxis_title="Model decision", yaxis_title="What actually happened",
                         yaxis_autorange="reversed", height=400)

    costs = [business_cost(y_test, p, loan_test, t) for t in THRESHOLDS]
    cost = business_cost(y_test, p, loan_test, threshold)
    fig_cost = go.Figure(go.Scatter(x=THRESHOLDS, y=costs, mode="lines",
                                    line=dict(color=MODEL_COLORS[key]), name="Cost"))
    fig_cost.add_hline(y=APPROVE_ALL_COST, line_dash="dash", line_color="grey",
                       annotation_text="Approve everyone")
    fig_cost.add_vline(x=threshold, line_color="#c0392b")
    fig_cost.update_layout(title="Business cost on 2015 loans for every threshold",
                           xaxis_title="Threshold (reject if probability of default ≥ threshold)",
                           yaxis_title="Cost (USD)", height=400)

    summary = (
        f"**Cost at this threshold:** ${cost:,.0f}  \n"
        f"**Approving everyone would cost:** ${APPROVE_ALL_COST:,.0f}  \n"
        f"**Saving vs approving everyone:** ${APPROVE_ALL_COST - cost:,.0f} "
        f"({(APPROVE_ALL_COST - cost) / APPROVE_ALL_COST:.1%})  \n"
        f"**Loans rejected:** {reject.mean():.1%}  \n"
        f"Threshold chosen on the 2014 validation slice: {BEST_THRESHOLD:.2f}. "
        f"Cost assumptions: a default loses {metrics['loss_if_default']:.0%} of the amount lent; "
        f"a wrongly rejected loan misses {metrics['profit_if_repaid']:.0%} in interest."
    )
    return fig_cm, fig_cost, summary


# --- Layout ---------------------------------------------------------------------
with gr.Blocks(title="Loan default model") as demo:
    gr.Markdown(
        "# Loan default model\n"
        "Lending Club loans, trained on 2012–2014 and tested on 2015. "
        "The same logistic regression was trained three ways; this app only loads the saved models."
    )
    with gr.Tab("Model comparison"):
        gr.Dataframe(metrics_table(), interactive=False)
        gr.Plot(calibration_plot())
        gr.Plot(agreement_plot())
        gr.Plot(loss_plot())
    with gr.Tab("Distributions"):
        feature_choice = gr.Dropdown(NUMERIC + CATEGORICAL, value="fico", label="Feature")
        feature_out = gr.Plot(feature_plot("fico"))
        feature_choice.change(feature_plot, feature_choice, feature_out)
        gr.Plot(target_plot())
    with gr.Tab("Threshold and cost"):
        with gr.Row():
            model_choice = gr.Radio(list(MODEL_LABELS.values()), value="scikit-learn", label="Model")
            threshold = gr.Slider(0.01, 0.99, value=BEST_THRESHOLD, step=0.01,
                                  label="Reject a loan if its probability of default is at least")
        summary_out = gr.Markdown()
        with gr.Row():
            cm_out = gr.Plot()
            cost_out = gr.Plot()
        outputs = [cm_out, cost_out, summary_out]
        for control in (model_choice, threshold):
            control.change(threshold_view, [model_choice, threshold], outputs)
        demo.load(threshold_view, [model_choice, threshold], outputs)

if __name__ == "__main__":
    demo.launch()
