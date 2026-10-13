"""Gradio dashboard for the saved Bank Marketing classification models."""

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


MODELS_DIR = Path(__file__).resolve().parent / "models"
ARTIFACT_PATH = MODELS_DIR / "bank_marketing_artifacts.joblib"
MANUAL_MODEL_PATH = MODELS_DIR / "manual_pytorch.pt"
STANDARD_MODEL_PATH = MODELS_DIR / "standard_pytorch.pt"

for path in (ARTIFACT_PATH, MANUAL_MODEL_PATH, STANDARD_MODEL_PATH):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing trained artifact: {path}. Run `uv run python main.py train` first."
        )

artifacts = joblib.load(ARTIFACT_PATH)
preprocessor = artifacts["preprocessor"]
sklearn_model = artifacts["sklearn_model"]
model_names = artifacts["model_names"]
test_features_ready = artifacts["test_features_ready"]
validation_features_ready = artifacts["validation_features_ready"]
test_actual = np.asarray(artifacts["test_actual"], dtype=np.int64)
validation_actual = np.asarray(artifacts["validation_actual"], dtype=np.int64)
feature_data = artifacts["test_features"].copy()

manual_state = torch.load(
    MANUAL_MODEL_PATH, map_location="cpu", weights_only=True
)
manual_weights = manual_state["weights"].to(dtype=torch.float64)
manual_bias = manual_state["bias"].to(dtype=torch.float64)


class TorchLogisticRegression(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, 1, dtype=torch.float64)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


standard_model = TorchLogisticRegression(artifacts["input_size"])
standard_model.load_state_dict(
    torch.load(STANDARD_MODEL_PATH, map_location="cpu", weights_only=True)
)
standard_model.eval()


def torch_probabilities(features: np.ndarray) -> dict[str, np.ndarray]:
    feature_tensor = torch.tensor(features, dtype=torch.float64)
    with torch.no_grad():
        manual = torch.sigmoid(feature_tensor @ manual_weights + manual_bias)
        standard = torch.sigmoid(standard_model(feature_tensor))
    return {
        "manual PyTorch": manual.numpy().ravel(),
        "standard PyTorch": standard.numpy().ravel(),
    }


def probabilities_for(features: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = {
        "scikit-learn": sklearn_model.predict_proba(features)[:, 1]
    }
    probabilities.update(torch_probabilities(features))
    return probabilities


test_probabilities = probabilities_for(test_features_ready)
validation_probabilities = probabilities_for(validation_features_ready)


def metrics_for(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision_yes": precision_score(actual, predicted, zero_division=0),
        "recall_yes": recall_score(actual, predicted, zero_division=0),
        "f1_yes": f1_score(actual, predicted, zero_division=0),
        "roc_auc": roc_auc_score(actual, probabilities),
        "average_precision": average_precision_score(actual, probabilities),
    }


test_metrics = pd.DataFrame(
    {
        name: metrics_for(test_actual, test_probabilities[name])
        for name in model_names
    }
).T
test_metrics.loc["naive majority baseline"] = metrics_for(
    test_actual,
    np.full(len(test_actual), test_actual.mean()),
)

prediction_rows = []
for model_name in model_names:
    prediction_rows.extend(
        {
            "Model": model_name,
            "Actual outcome": "yes" if actual == 1 else "no",
            "Predicted probability of yes": probability,
        }
        for actual, probability in zip(
            test_actual, test_probabilities[model_name], strict=True
        )
    )
prediction_plot = px.box(
    pd.DataFrame(prediction_rows),
    x="Actual outcome",
    y="Predicted probability of yes",
    color="Model",
    facet_col="Model",
    points=False,
    category_orders={"Actual outcome": ["no", "yes"]},
    title="Predicted probability by actual outcome (held-out test set)",
)
prediction_plot.update_layout(showlegend=False)
prediction_plot.update_yaxes(range=[0, 1])


def make_confusion_matrix(
    model_name: str,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> tuple[go.Figure, str]:
    predicted = (
        validation_probabilities[model_name] >= threshold
    ).astype(np.int64)
    matrix = confusion_matrix(validation_actual, predicted, labels=[0, 1])
    figure = px.imshow(
        matrix,
        x=["Predicted no", "Predicted yes"],
        y=["Actual no", "Actual yes"],
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Validation confusion matrix at threshold {threshold:.2f}",
    )
    figure.update_layout(coloraxis_showscale=False)

    false_positives = int(matrix[0, 1])
    false_negatives = int(matrix[1, 0])
    total_cost = (
        false_positives * false_positive_cost
        + false_negatives * false_negative_cost
    )
    average_cost = total_cost / len(validation_actual)
    summary = (
        f"False positives: {false_positives} × {false_positive_cost:g} units; "
        f"false negatives: {false_negatives} × {false_negative_cost:g} units. "
        f"Total validation cost: {total_cost:.1f} units "
        f"({average_cost:.3f} per customer)."
    )
    return figure, summary


def make_feature_distribution(feature_name: str) -> go.Figure:
    data = feature_data[[feature_name]].copy()
    data["Actual outcome"] = np.where(test_actual == 1, "yes", "no")
    if pd.api.types.is_numeric_dtype(data[feature_name]):
        figure = px.histogram(
            data,
            x=feature_name,
            color="Actual outcome",
            barmode="overlay",
            opacity=0.65,
            title=f"Distribution of {feature_name} by actual outcome (test set)",
        )
    else:
        figure = px.histogram(
            data,
            x=feature_name,
            color="Actual outcome",
            barmode="group",
            title=f"Distribution of {feature_name} by actual outcome (test set)",
        )
    return figure


target_distribution = px.bar(
    x=list(artifacts["target_distribution"].keys()),
    y=list(artifacts["target_distribution"].values()),
    labels={"x": "Subscription outcome", "y": "Number of customers"},
    title="Target distribution in the full dataset",
)

initial_confusion, initial_cost = make_confusion_matrix(
    model_names[0], 0.5, 1.0, 3.0
)
feature_names = feature_data.columns.tolist()

with gr.Blocks(title="Bank Marketing model comparison") as demo:
    gr.Markdown(
        """
        # Bank Marketing: model comparison

        The models and preprocessor below are loaded from the training artifacts;
        this app does not train anything. Test-set metrics and predicted-vs-actual
        plots are kept separate from the validation-set threshold controls.

        Business costs are relative, user-adjustable units, not measured euros.
        The default relative costs are 1 unit per false positive and 3 per false
        negative. They are assumptions because the dataset contains no call-cost
        or subscription-value information.
        """
    )

    with gr.Tab("Compare models"):
        gr.Markdown("Metrics use the held-out test set at the default threshold of 0.50.")
        gr.Dataframe(
            value=test_metrics.reset_index(names="model"),
            label="Test-set classification metrics (threshold = 0.50)",
            interactive=False,
        )
        gr.Plot(value=prediction_plot, label="Predicted probability vs actual outcome")

    with gr.Tab("Threshold and business cost"):
        gr.Markdown(
            "Use validation data to explore the threshold. Cost is "
            "`false positives × cost per unnecessary contact + "
            "false negatives × cost per missed subscription`."
        )
        with gr.Row():
            selected_model = gr.Dropdown(
                choices=model_names,
                value=model_names[0],
                label="Model",
            )
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.01,
                label="Decision threshold",
            )
        with gr.Row():
            false_positive_cost = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.5,
                label="Relative cost of an unnecessary contact",
            )
            false_negative_cost = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=3.0,
                step=0.5,
                label="Relative cost of a missed subscription (false negative)",
            )
        confusion_plot = gr.Plot(value=initial_confusion, label="Confusion matrix")
        cost_summary = gr.Markdown(value=initial_cost)
        threshold_inputs = [
            selected_model,
            threshold,
            false_positive_cost,
            false_negative_cost,
        ]
        threshold_outputs = [confusion_plot, cost_summary]
        for control in threshold_inputs:
            control.change(
                fn=lambda model, cutoff, fp_cost, fn_cost: make_confusion_matrix(
                    model, cutoff, fp_cost, fn_cost
                ),
                inputs=threshold_inputs,
                outputs=threshold_outputs,
            )

    with gr.Tab("Distributions"):
        gr.Plot(value=target_distribution, label="Target distribution")
        feature_selector = gr.Dropdown(
            choices=feature_names,
            value=feature_names[0],
            label="Feature to inspect",
        )
        feature_plot = gr.Plot(
            value=make_feature_distribution(feature_names[0]),
            label="Feature distribution",
        )
        feature_selector.change(
            fn=make_feature_distribution,
            inputs=feature_selector,
            outputs=feature_plot,
        )


demo.launch()
