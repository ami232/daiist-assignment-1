"""
Assignment 1 — Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file — main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py app`, this file must build and launch a Gradio
app (a `demo` that calls `.launch()`) using only what your training stage
already produced — it must never retrain anything itself. At minimum, the
dashboard must let you:

- Compare your three trained models via a prediction-vs-actual plot.
- See feature and/or target distributions.
- For classification: move a decision-threshold slider and watch the
  confusion matrix, and a business-cost number tied to your REPORT.md's
  framing, change with it.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score
)
# ---------------------------------------
# 1. LOAD SAVED ARTIFACTS
# ---------------------------------------

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# Models and preprocessing
preprocessor = joblib.load(
    ARTIFACTS_DIR / "preprocessor.joblib"
)

sklearn_model = joblib.load(
    ARTIFACTS_DIR / "sklearn_model.joblib"
)

# Previously calculated results
model_results = pd.read_csv(
    ARTIFACTS_DIR / "model_comparison.csv",
    index_col=0
)

distribution_data = pd.read_csv(
    ARTIFACTS_DIR / "data_distributions.csv"
)

validation_data = pd.read_csv(
    ARTIFACTS_DIR / "validation_predictions.csv"
)

# ---------------------------------------
# 2. MODEL COMPARISON CHART
# ---------------------------------------

def plot_model_comparison(metric):
    fig, ax = plt.subplots(figsize=(8, 4))

    model_results[metric].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(f"Model Comparison - {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    return fig


# ---------------------------------------
# 3. PREDICTIONS VS ACTUAL RESULTS
# ---------------------------------------

def plot_predictions(model_name):

    column_map = {
        "Scikit-learn": "sklearn_probability",
        "Manual PyTorch": "manual_probability",
        "Standard PyTorch": "standard_probability"
    }

    probabilities = validation_data[
        column_map[model_name]
    ]

    actual = validation_data["actual"]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(
        range(len(actual)),
        actual,
        label="Actual subscription",
        alpha=0.45,
        s=15
    )

    ax.scatter(
        range(len(probabilities)),
        probabilities,
        label="Predicted probability",
        alpha=0.45,
        s=15
    )

    ax.set_title(f"{model_name}: Predictions vs Actual")
    ax.set_xlabel("Validation customer index")
    ax.set_ylabel("Subscription / Probability")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig


# ---------------------------------------
# 4. CREATE GRADIO DASHBOARD
# ---------------------------------------

# ---------------------------------------
# DATA EXPLORATION
# ---------------------------------------

def plot_distribution(feature):

    fig, ax = plt.subplots(figsize=(8, 4))

    if feature == "Subscription Outcome":
        counts = distribution_data["y"].value_counts()
        counts = counts.reindex(["no", "yes"], fill_value=0)

        ax.bar(["No", "Yes"], counts.values)
        ax.set_ylabel("Number of Customers")

    elif feature == "Age":
        ax.hist(
            distribution_data["age"],
            bins=20,
            edgecolor="white"
        )
        ax.set_xlabel("Customer Age")
        ax.set_ylabel("Number of Customers")

    elif feature == "Job":
        counts = distribution_data["job"].value_counts()

        ax.barh(counts.index, counts.values)
        ax.invert_yaxis()
        ax.set_xlabel("Number of Customers")

    ax.set_title(f"Distribution: {feature}")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    return fig

# ---------------------------------------
# THRESHOLD SIMULATOR
# ---------------------------------------

def simulate_threshold(threshold):

    actual = validation_data["actual"].to_numpy()

    probabilities = validation_data[
        "sklearn_probability"
    ].to_numpy()

    # Convert probabilities into binary predictions
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        actual, predictions, labels=[0, 1]
    ).ravel()

    # Business assumptions
    CALL_COST = 5
    SUBSCRIPTION_VALUE = 50

    calls = int(tp + fp)

    net_value = (
        tp * SUBSCRIPTION_VALUE
        - calls * CALL_COST
    )

    # Evaluation metrics
    precision = precision_score(
        actual, predictions, zero_division=0
    )

    recall = recall_score(
        actual, predictions, zero_division=0
    )

    # Create confusion matrix plot
    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig, ax = plt.subplots(figsize=(6, 4))

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

    ax.set_xlabel("Predicted Subscription")
    ax.set_ylabel("Actual Subscription")
    ax.set_title(
        f"Confusion Matrix — Threshold {threshold:.2f}"
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()

    # Create business summary
    summary = f"""
### Results at Threshold {threshold:.2f}

| Metric | Value |
|---|---:|
| Customers selected for calls | {calls} |
| Subscribers identified | {tp} |
| Missed subscribers | {fn} |
| Precision | {precision:.1%} |
| Recall | {recall:.1%} |
| Estimated net value | €{net_value:,.0f} |

**Business assumptions:** €5 per call and €50 per successful subscription.

*Financial values are hypothetical and do not represent proven incremental campaign profit.*
"""

    return fig, summary

with gr.Blocks(title="Bank Marketing Dashboard") as demo:

    gr.Markdown("""
    # Bank Marketing Prediction Dashboard

    Predicting customer subscriptions to term deposits
    using three implementations of logistic regression.

    Dataset: UCI Bank Marketing (4,119 observations).
    """)

    with gr.Tab("Model Comparison"):

        gr.Markdown("""
        ## Comparing Logistic Regression Implementations

        All three models were trained using the same
        training data and preprocessing transformations.

        The evaluation results below come from the
        validation dataset.
        """)

        gr.Dataframe(
            value=model_results.reset_index(),
            label="Validation Performance",
            interactive=False
        )

        metric_selector = gr.Dropdown(
            choices=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC"
            ],
            value="F1",
            label="Select Evaluation Metric"
        )

        comparison_plot = gr.Plot(
            value=plot_model_comparison("F1")
        )

        metric_selector.change(
            fn=plot_model_comparison,
            inputs=metric_selector,
            outputs=comparison_plot
        )

    with gr.Tab("Predictions vs Actual"):

        gr.Markdown("""
        ## Predictions vs Actual Subscriptions

        Compare the predicted probabilities from the
        three models with the actual customer outcomes.
        """)

        model_selector = gr.Dropdown(
            choices=[
                "Scikit-learn",
                "Manual PyTorch",
                "Standard PyTorch"
            ],
            value="Scikit-learn",
            label="Select Model"
        )

        prediction_plot = gr.Plot(
            value=plot_predictions("Scikit-learn")
        )

        model_selector.change(
            fn=plot_predictions,
            inputs=model_selector,
            outputs=prediction_plot
        )

    with gr.Tab("Data Exploration"):

        gr.Markdown("""
        ## Exploring the Bank Marketing Dataset

        Examine the distribution of customer characteristics
        and subscription outcomes across 4,119 observations.
        """)

        feature_selector = gr.Dropdown(
            choices=[
                "Subscription Outcome",
                "Age",
                "Job"
            ],
            value="Subscription Outcome",
            label="Select Feature"
        )

        distribution_plot = gr.Plot(
            value=plot_distribution("Subscription Outcome")
        )

        feature_selector.change(
            fn=plot_distribution,
            inputs=feature_selector,
            outputs=distribution_plot
        )
    
    with gr.Tab("Threshold Simulator"):

        gr.Markdown("""
        ## Interactive Classification Threshold

        Adjust the probability threshold to explore
        how marketing decisions change.

        Lower thresholds generally identify more potential
        subscribers but require more calls.

        Results use the validation dataset.
        """)

        threshold_slider = gr.Slider(
            minimum=0.05,
            maximum=0.90,
            step=0.05,
            value=0.10,
            label="Classification Threshold"
        )

        with gr.Row():

            confusion_plot = gr.Plot(
                label="Confusion Matrix"
            )

            business_summary = gr.Markdown()

        # Update when the slider changes
        threshold_slider.change(
            fn=simulate_threshold,
            inputs=threshold_slider,
            outputs=[
                confusion_plot,
                business_summary
            ]
        )

        # Display initial results
        demo.load(
            fn=simulate_threshold,
            inputs=threshold_slider,
            outputs=[
                confusion_plot,
                business_summary
            ]
        )


# ---------------------------------------
# 5. LAUNCH APPLICATION
# ---------------------------------------

demo.launch()