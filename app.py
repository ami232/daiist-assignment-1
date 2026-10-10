import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Load everything saved during training
sk_model = joblib.load("models/sklearn_logreg.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")
numeric_cols = joblib.load("models/numeric_cols.joblib")

with open("models/metrics.json") as f:
    saved_metrics = json.load(f)

X_test = pd.read_csv("models/X_test.csv")
X_test_raw = pd.read_csv("models/X_test_raw.csv")  # unscaled, for human-readable distribution plots
y_test = pd.read_csv("models/y_test.csv").squeeze()  # squeeze turns the 1-column df into a Series

n_features = X_test.shape[1]

# Manual PyTorch model: just the saved weight/bias tensors
manual_state = torch.load("models/manual_pytorch.pt")
w_manual, b_manual = manual_state["w"], manual_state["b"]

#Standard PyTorch model: needs the same class definition used in train.ipynb
class LogisticRegressionModule(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

standard_model = LogisticRegressionModule(n_features)
standard_model.load_state_dict(torch.load("models/standard_pytorch.pt"))
standard_model.eval()

#Precompute predicted probabilities for all three models on the test set
X_test_t = torch.tensor(X_test.astype(float).values, dtype=torch.float32)

sk_proba = sk_model.predict_proba(X_test)[:, 1]

with torch.no_grad():
    manual_proba = torch.sigmoid(X_test_t @ w_manual + b_manual).numpy().flatten()
    standard_proba = standard_model(X_test_t).numpy().flatten()

model_probas = {
    "scikit-learn": sk_proba,
    "Manual PyTorch": manual_proba,
    "Standard PyTorch": standard_proba,
}

#Core logic functions

def get_confusion_and_cost(model_name, threshold):
    """Given a model and a threshold, compute predictions, confusion matrix,
    and a business-cost number based on a calculated 333:1 cost ratio
    (lost tuition from a missed dropout vs. an advisor check-in)."""
    proba = model_probas[model_name]
    preds = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    cost_per_false_negative = 10000  # lost tuition from a missed at-risk student (~€5,000/yr x ~2 yrs)
    cost_per_false_positive = 30     # advisor check-in (~1 hr x ~€30/hr)
    total_cost = (fn * cost_per_false_negative) + (fp * cost_per_false_positive)

    return cm, total_cost, tn, fp, fn, tp


def plot_confusion_matrix(model_name, threshold):
    cm, total_cost, tn, fp, fn, tp = get_confusion_and_cost(model_name, threshold)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Dropout", "Dropout"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} @ threshold {threshold:.2f}")
    plt.tight_layout()

    cost_text = (
        f"**Estimated business cost: €{total_cost:,.0f}**\n\n"
        f"False negatives (missed dropouts): {fn} × €10,000 = €{fn*10000:,.0f}\n\n"
        f"False positives (false alarms): {fp} × €30 = €{fp*30:,.0f}"
    )
    return fig, cost_text


def plot_prediction_vs_actual():
    """Boxplots of predicted probability, grouped by true outcome, for each model.
    A good model should show the 'Not Dropout' box low and the 'Dropout' box
    high, with minimal overlap."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, proba) in zip(axes, model_probas.items()):
        data_to_plot = [proba[y_test == 0], proba[y_test == 1]]
        ax.boxplot(data_to_plot, tick_labels=["Actual: Not Dropout", "Actual: Dropout"])
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Default threshold (0.5)")
        ax.set_title(name)
        ax.set_ylabel("Predicted probability of dropout")
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_metrics_table():
    df = pd.DataFrame(saved_metrics).T
    df = df.round(3)
    df.insert(0, "model", df.index)
    df = df.reset_index(drop=True)
    return df


#Features found meaningful during EDA
eda_features = [
    "Age at enrollment",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (approved)",
    "sem1_pass_rate",
    "financial_risk_score",
    "has_zero_grade",
]


def plot_feature_distribution(feature_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    all_values = X_test_raw[feature_name]
    n_unique = all_values.nunique()
    is_integer_like = (all_values.dropna() % 1 == 0).all()

    if n_unique <= 15 and is_integer_like:
        min_val, max_val = int(all_values.min()), int(all_values.max())
        bins = np.arange(min_val, max_val + 2) - 0.5
        for label, name in [(0, "Not Dropout"), (1, "Dropout")]:
            subset = X_test_raw[y_test == label][feature_name]
            ax.hist(subset, bins=bins, alpha=0.5, label=name, density=True)
        ax.set_xticks(range(min_val, max_val + 1))
    else:
        bins = np.linspace(all_values.min(), all_values.max(), 15)
        for label, name in [(0, "Not Dropout"), (1, "Dropout")]:
            subset = X_test_raw[y_test == label][feature_name]
            ax.hist(subset, bins=bins, alpha=0.5, label=name, density=True)

    ax.set_title(f"{feature_name} distribution by outcome")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    return fig


#Gradio layout

with gr.Blocks(title="Student Dropout Risk Dashboard") as demo:
    gr.Markdown("# Student Dropout Risk Dashboard")
    gr.Markdown(
        "Comparing scikit-learn, manual PyTorch, and standard PyTorch logistic "
        "regression, trained to predict dropout risk after 1st-semester results "
        "are available."
    )

    with gr.Tab("Model Comparison"):
        gr.Markdown("### Three-method comparison vs. naive baseline")
        gr.DataFrame(value=plot_metrics_table())
        gr.Markdown("### Prediction vs. Actual (test set)")
        gr.Markdown(
            "Each box shows the spread of predicted dropout probabilities for "
            "students in that true outcome group. A good model pushes the "
            "'Not Dropout' box low and the 'Dropout' box high, with little overlap."
        )
        gr.Plot(value=plot_prediction_vs_actual())

    with gr.Tab("Feature Distributions"):
        gr.Markdown("### Feature Distributions")
        gr.Markdown(
            "The y-axis shows density, not the actual number of student count. Bars are scaled so each "
            "group's (Not Dropout / Dropout) total area sums to 1. This makes the "
            "two groups' shapes comparable even though there are roughly twice as "
            "many non-dropouts as dropouts in the data. A taller bar means that "
            "range of values is relatively more common within that group, not "
            "that more students overall fall into it."
        )
        feature_dropdown = gr.Dropdown(
            choices=eda_features, value=eda_features[0], label="Select a feature (from EDA findings)"
        )
        dist_plot = gr.Plot()
        feature_dropdown.change(fn=plot_feature_distribution, inputs=feature_dropdown, outputs=dist_plot)
        demo.load(fn=plot_feature_distribution, inputs=feature_dropdown, outputs=dist_plot)

    with gr.Tab("Threshold & Business Cost"):
        gr.Markdown("### Decision threshold and business cost")
        gr.Markdown(
            "The costs assume that a missed dropout (false negative) costs ~€10,000 in lost "
            "tuition, versus ~€30 for an unnecessary advisor check-in (false "
            "positive) so I used a ~333:1 ratio reflecting that missing an at-risk student "
            "is far more costly than a false alarm."
        )
        model_dropdown = gr.Dropdown(
            choices=list(model_probas.keys()), value="scikit-learn", label="Select a model"
        )
        threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Decision threshold")
        cm_plot = gr.Plot()
        cost_display = gr.Markdown()

        def update_threshold_view(model_name, threshold):
            fig, cost_text = plot_confusion_matrix(model_name, threshold)
            return fig, cost_text

        model_dropdown.change(fn=update_threshold_view, inputs=[model_dropdown, threshold_slider], outputs=[cm_plot, cost_display])
        threshold_slider.change(fn=update_threshold_view, inputs=[model_dropdown, threshold_slider], outputs=[cm_plot, cost_display])
        demo.load(fn=update_threshold_view, inputs=[model_dropdown, threshold_slider], outputs=[cm_plot, cost_display])

if __name__ == "__main__":
    demo.launch()
