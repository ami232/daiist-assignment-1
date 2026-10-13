import json
import pickle
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

with open(ARTIFACTS / "test_data.json") as f:
    td = json.load(f)

with open(ARTIFACTS / "training_curves.json") as f:
    curves = json.load(f)

with open(ARTIFACTS / "distributions.json") as f:
    dist = json.load(f)

X_test = np.array(td["X_test"])
y_test = np.array(td["y_test"])
dates_test = td["dates_test"]
feature_names = td["feature_names"]
n_features = td["n_features"]
majority_class = td["majority_class"]

with open(ARTIFACTS / "sklearn_model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

manual_weights = torch.load(ARTIFACTS / "manual_pytorch.pt", weights_only=True)
W_manual = manual_weights["W"]
b_manual = manual_weights["b"]


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


std_model = LogisticRegressionModel(n_features)
std_model.load_state_dict(torch.load(ARTIFACTS / "std_pytorch.pt", weights_only=True))
std_model.eval()

X_te_t = torch.tensor(X_test, dtype=torch.float32)

y_prob_sklearn = sklearn_model.predict_proba(X_test)[:, 1]

with torch.no_grad():
    y_prob_manual = torch.sigmoid(X_te_t @ W_manual + b_manual).squeeze().numpy()

with torch.no_grad():
    y_prob_std = torch.sigmoid(std_model(X_te_t)).squeeze().numpy()

MODEL_PROBS = {
    "Scikit-learn LR": y_prob_sklearn,
    "Manual PyTorch LR": y_prob_manual,
    "Standard PyTorch LR": y_prob_std,
}

y_baseline = np.full_like(y_test, majority_class)
results = {}
for name, y_pred, y_prob in [
    ("Naive baseline", y_baseline, np.full(len(y_test), float(majority_class))),
    ("Scikit-learn LR", (y_prob_sklearn >= 0.5).astype(int), y_prob_sklearn),
    ("Manual PyTorch LR", (y_prob_manual >= 0.5).astype(int), y_prob_manual),
    ("Standard PyTorch LR", (y_prob_std >= 0.5).astype(int), y_prob_std),
]:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

COST_FP = 10_000
COST_FN = 5_000


def build_comparison_table():
    header = "| Model | Accuracy | Precision | Recall | F1 | AUC |\n|---|---|---|---|---|---|\n"
    rows = ""
    for name, m in results.items():
        rows += f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} |\n"
    return header + rows


def plot_pred_vs_actual():
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Scikit-learn LR", "Manual PyTorch LR", "Standard PyTorch LR"],
        horizontal_spacing=0.06,
    )
    for i, (name, probs) in enumerate(MODEL_PROBS.items(), 1):
        colors = ["#2ecc71" if y == 1 else "#e74c3c" for y in y_test]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_test))),
                y=probs,
                mode="markers",
                marker=dict(color=colors, size=4, opacity=0.6),
                name=name,
                showlegend=False,
                hovertemplate="Date: %{customdata}<br>Prob: %{y:.3f}<br>Actual: %{text}",
                customdata=dates_test,
                text=["Up" if y == 1 else "Down" for y in y_test],
            ),
            row=1, col=i,
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=i)

    fig.update_layout(
        title="Predicted Probability vs Actual (Green=Up, Red=Down)",
        height=400, template="plotly_white",
    )
    fig.update_yaxes(title_text="P(Up)", range=[0, 1])
    fig.update_xaxes(title_text="Test sample index")
    return fig


def plot_roc_curves():
    fig = go.Figure()
    for name, probs in MODEL_PROBS.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_val:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400, template="plotly_white",
    )
    return fig


def plot_prob_distributions():
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(MODEL_PROBS.keys()),
        horizontal_spacing=0.06,
    )
    for i, (name, probs) in enumerate(MODEL_PROBS.items(), 1):
        for label, color, label_name in [(0, "#e74c3c", "Down"), (1, "#2ecc71", "Up")]:
            mask = y_test == label
            fig.add_trace(
                go.Histogram(
                    x=probs[mask], nbinsx=30, opacity=0.6,
                    marker_color=color, name=label_name,
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )
    fig.update_layout(
        title="Distribution of Predicted Probabilities by Actual Class",
        barmode="overlay", height=350, template="plotly_white",
    )
    fig.update_xaxes(title_text="P(Up)")
    return fig


def plot_feature_distribution(feature_name):
    train_vals = dist["features"][feature_name]["train"]
    test_vals = dist["features"][feature_name]["test"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train_vals, nbinsx=50, opacity=0.6, name="Train", marker_color="#3498db"))
    fig.add_trace(go.Histogram(x=test_vals, nbinsx=50, opacity=0.6, name="Test", marker_color="#e67e22"))
    fig.update_layout(
        title=f"Distribution of '{feature_name}' — Train vs Test",
        barmode="overlay", height=400, template="plotly_white",
        xaxis_title=feature_name, yaxis_title="Count",
    )
    return fig


def plot_target_distribution():
    train_counts = [dist["target_train"].count(0), dist["target_train"].count(1)]
    test_counts = [dist["target_test"].count(0), dist["target_test"].count(1)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Down (0)", "Up (1)"], y=train_counts, name="Train", marker_color="#3498db"))
    fig.add_trace(go.Bar(x=["Down (0)", "Up (1)"], y=test_counts, name="Test", marker_color="#e67e22"))
    fig.update_layout(
        title="Target Distribution — Train vs Test",
        barmode="group", height=350, template="plotly_white",
        yaxis_title="Count",
    )
    return fig


def threshold_analysis(threshold, model_name):
    probs = MODEL_PROBS[model_name]
    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    total_cost = fp * COST_FP + fn * COST_FN
    baseline_cost = sum(1 for y in y_test if y == 0) * COST_FP

    cm_fig = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Pred Down", "Pred Up"],
        y=["Actual Down", "Actual Up"],
        text=[[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ))
    cm_fig.update_layout(
        title=f"Confusion Matrix — {model_name} (threshold={threshold:.2f})",
        height=350, template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )

    metrics_text = (
        f"### Metrics at threshold = {threshold:.2f}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Accuracy | {acc:.4f} |\n"
        f"| Precision | {prec:.4f} |\n"
        f"| Recall | {rec:.4f} |\n"
        f"| F1 Score | {f1:.4f} |\n\n"
        f"### Business Cost Analysis\n\n"
        f"- **False Positives** (bad longs): {fp} × ${COST_FP:,} = **${fp * COST_FP:,}**\n"
        f"- **False Negatives** (missed rallies): {fn} × ${COST_FN:,} = **${fn * COST_FN:,}**\n"
        f"- **Total trading cost**: **${total_cost:,}**\n"
        f"- **Naive baseline cost** (always go long): **${baseline_cost:,}**\n"
        f"- **Savings vs baseline**: **${baseline_cost - total_cost:,}**"
    )

    return cm_fig, metrics_text


def plot_training_curves():
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Manual PyTorch", "Standard PyTorch (nn.Module)"],
        horizontal_spacing=0.08,
    )
    epochs = list(range(1, len(curves["manual_train"]) + 1))

    fig.add_trace(go.Scatter(x=epochs, y=curves["manual_train"], name="Train", line=dict(color="#3498db")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=curves["manual_val"], name="Val", line=dict(color="#e74c3c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=curves["std_train"], name="Train", line=dict(color="#3498db"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=curves["std_val"], name="Val", line=dict(color="#e74c3c"), showlegend=False), row=1, col=2)

    fig.update_layout(title="Training & Validation Loss", height=400, template="plotly_white")
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="BCE Loss")
    return fig


with gr.Blocks(title="S&P 500 Direction Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# S&P 500 Next-Day Direction Predictor\n"
        "*Logistic regression trained 3 ways on daily technical indicators (2014-2024). "
        "Business framing: systematic trading desk deciding whether to go long overnight.*"
    )

    with gr.Tab("Model Comparison"):
        gr.Markdown("## Three-Model Comparison (Test Set)")
        gr.Markdown(build_comparison_table())
        gr.Plot(value=plot_pred_vs_actual(), label="Predicted Prob vs Actual")
        gr.Plot(value=plot_roc_curves(), label="ROC Curves")
        gr.Plot(value=plot_prob_distributions(), label="Probability Distributions")

    with gr.Tab("Feature & Target Distributions"):
        gr.Markdown("## Explore Feature Distributions (Train vs Test)")
        gr.Plot(value=plot_target_distribution(), label="Target Distribution")
        feature_dropdown = gr.Dropdown(
            choices=feature_names, value=feature_names[0], label="Select feature"
        )
        feature_plot = gr.Plot(value=plot_feature_distribution(feature_names[0]))
        feature_dropdown.change(fn=plot_feature_distribution, inputs=feature_dropdown, outputs=feature_plot)

    with gr.Tab("Threshold Tuner"):
        gr.Markdown(
            "## Decision Threshold Explorer\n"
            "Adjust the classification threshold and see how the confusion matrix and "
            "business cost change. The trading desk can tune this to its risk appetite.\n\n"
            f"**Cost assumptions:** FP (bad long) = ${COST_FP:,} | FN (missed rally) = ${COST_FN:,}"
        )
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(MODEL_PROBS.keys()),
                value="Scikit-learn LR",
                label="Model",
            )
            threshold_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Decision Threshold"
            )
        cm_plot = gr.Plot()
        metrics_md = gr.Markdown()

        def update_threshold(threshold, model_name):
            return threshold_analysis(threshold, model_name)

        threshold_slider.change(fn=update_threshold, inputs=[threshold_slider, model_selector], outputs=[cm_plot, metrics_md])
        model_selector.change(fn=update_threshold, inputs=[threshold_slider, model_selector], outputs=[cm_plot, metrics_md])
        demo.load(fn=update_threshold, inputs=[threshold_slider, model_selector], outputs=[cm_plot, metrics_md])

    with gr.Tab("Training Curves"):
        gr.Markdown("## Loss Curves (Manual vs Standard PyTorch)")
        gr.Plot(value=plot_training_curves(), label="Training Curves")

demo.launch()
