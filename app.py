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

"""
Assignment 1 — Gradio dashboard.

This app loads the artifacts produced during training and never retrains
the models. It lets the user:

- Compare the three logistic regression implementations.
- Inspect the target distribution.
- Explore how the classification threshold affects the confusion matrix.
- See how the threshold affects expected incremental business profit.
- Review final test-set performance using the threshold selected on validation.
"""

import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 1. Load dataset and saved training artifacts
# ============================================================

df = pd.read_csv("data/hotel_bookings.csv")

preprocessor = joblib.load("artifacts/preprocessor.joblib")
sk_model = joblib.load("artifacts/sklearn_logistic.joblib")

manual_params = torch.load(
    "artifacts/manual_pytorch.pt",
    weights_only=True
)

with open("artifacts/business_config.json", "r") as f:
    business_config = json.load(f)


# ============================================================
# 2. Rebuild the standard PyTorch model architecture
# ============================================================

class LogisticRegressionTorch(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


standard_model = LogisticRegressionTorch(
    business_config["n_features"]
)

standard_model.load_state_dict(
    torch.load(
        "artifacts/standard_pytorch.pt",
        weights_only=True
    )
)

standard_model.eval()


# ============================================================
# 3. Recreate the same dataset used during training
# ============================================================

df = df[
    (df["hotel"] == "Resort Hotel") &
    (df["arrival_date_year"] == 2016)
].copy()

df["country"] = df["country"].fillna("Unknown")


# ============================================================
# 4. Recreate the same engineered features
# ============================================================

df["total_nights"] = (
    df["stays_in_weekend_nights"]
    + df["stays_in_week_nights"]
)

df["party_size"] = (
    df["adults"]
    + df["children"].fillna(0)
    + df["babies"]
)

df["has_children"] = (
    (df["children"].fillna(0) + df["babies"]) > 0
).astype(int)

df["has_previous_cancellation"] = (
    df["previous_cancellations"] > 0
).astype(int)

df["has_special_requests"] = (
    df["total_of_special_requests"] > 0
).astype(int)


# ============================================================
# 5. Recreate booking and arrival dates
# ============================================================

df["arrival_date"] = pd.to_datetime(
    df["arrival_date_year"].astype(str)
    + "-"
    + df["arrival_date_month"]
    + "-"
    + df["arrival_date_day_of_month"].astype(str)
)

df["booking_date"] = (
    df["arrival_date"]
    - pd.to_timedelta(df["lead_time"], unit="D")
)


# ============================================================
# 6. Drop the same columns removed during training
# ============================================================

columns_to_drop = [
    "hotel",
    "arrival_date_year",
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
    "company",
    "agent",
]

df = df.drop(columns=columns_to_drop)


# ============================================================
# 7. Recreate the chronological train / validation / test split
# ============================================================

df = df.sort_values("booking_date").reset_index(drop=True)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

initial_train = df.iloc[:train_end].copy()
initial_val = df.iloc[train_end:val_end].copy()

train_cutoff = initial_train["booking_date"].max()
val_cutoff = initial_val["booking_date"].max()

val_df = df[
    (df["booking_date"] >= train_cutoff)
    & (df["booking_date"] < val_cutoff)
].copy()

test_df = df[
    df["booking_date"] >= val_cutoff
].copy()


# ============================================================
# 8. Define exactly the same model features
# ============================================================

numeric_features = [
    "lead_time",
    "is_repeated_guest",
    "previous_bookings_not_canceled",
    "adr",
    "required_car_parking_spaces",
    "total_nights",
    "party_size",
    "has_children",
    "has_previous_cancellation",
    "has_special_requests",
]

categorical_features = [
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

features = numeric_features + categorical_features


# ============================================================
# 9. Prepare validation and test inputs
# ============================================================

X_val = val_df[features]
y_val = val_df["is_canceled"]

X_test = test_df[features]
y_test = test_df["is_canceled"]

# Important:
# The saved preprocessor is only used with transform().
# Nothing is fitted again inside the app.

X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

X_val_t = torch.tensor(
    X_val_processed,
    dtype=torch.float32
)

X_test_t = torch.tensor(
    X_test_processed,
    dtype=torch.float32
)


# ============================================================
# 10. Generate probabilities from the three trained models
# ============================================================

# ---------- Scikit-learn ----------

sk_val_prob = sk_model.predict_proba(
    X_val_processed
)[:, 1]

sk_test_prob = sk_model.predict_proba(
    X_test_processed
)[:, 1]


# ---------- Manual PyTorch ----------

manual_w = manual_params["weights"]
manual_b = manual_params["bias"]

with torch.no_grad():

    manual_val_logits = (
        X_val_t @ manual_w + manual_b
    )

    manual_val_prob = torch.sigmoid(
        manual_val_logits
    ).numpy()

    manual_test_logits = (
        X_test_t @ manual_w + manual_b
    )

    manual_test_prob = torch.sigmoid(
        manual_test_logits
    ).numpy()


# ---------- Standard PyTorch ----------

with torch.no_grad():

    standard_val_logits = standard_model(
        X_val_t
    ).squeeze()

    standard_val_prob = torch.sigmoid(
        standard_val_logits
    ).numpy()

    standard_test_logits = standard_model(
        X_test_t
    ).squeeze()

    standard_test_prob = torch.sigmoid(
        standard_test_logits
    ).numpy()


# ============================================================
# 11. Business-value helper
# ============================================================

def calculate_expected_profit(
    predictions,
    actual,
    booking_df
):

    booking_value = (
        booking_df["adr"].values
        * booking_df["total_nights"].values
    )

    actual = np.asarray(actual)

    tp_mask = (
        (predictions == 1)
        & (actual == 1)
    )

    fp_mask = (
        (predictions == 1)
        & (actual == 0)
    )

    tp_profit = (
        business_config["save_probability"]
        * business_config["contribution_margin"]
        * booking_value[tp_mask]
        - business_config["intervention_cost"]
    ).sum()

    fp_cost = (
        business_config["intervention_cost"]
        * fp_mask.sum()
    )

    return tp_profit - fp_cost


# ============================================================
# 12. Target distribution plot
# ============================================================

def plot_target_distribution():

    fig, ax = plt.subplots()

    counts = y_test.value_counts().sort_index()

    ax.bar(
        ["Not canceled", "Canceled"],
        [
            counts.get(0, 0),
            counts.get(1, 0)
        ]
    )

    ax.set_title(
        "Final test-set target distribution"
    )

    ax.set_ylabel(
        "Number of bookings"
    )

    return fig


# ============================================================
# 13. Prediction-vs-actual comparison plot
# ============================================================

def calibration_points(
    probabilities,
    actual,
    n_bins=10
):

    temp = pd.DataFrame({
        "probability": probabilities,
        "actual": np.asarray(actual),
    })

    temp["bin"] = pd.cut(
        temp["probability"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True
    )

    grouped = (
        temp.groupby(
            "bin",
            observed=True
        )
        .agg(
            mean_prediction=("probability", "mean"),
            actual_rate=("actual", "mean")
        )
        .dropna()
    )

    return grouped


def plot_model_comparison():

    fig, ax = plt.subplots()

    sk_points = calibration_points(
        sk_test_prob,
        y_test
    )

    manual_points = calibration_points(
        manual_test_prob,
        y_test
    )

    standard_points = calibration_points(
        standard_test_prob,
        y_test
    )

    ax.plot(
        sk_points["mean_prediction"],
        sk_points["actual_rate"],
        marker="o",
        label="Scikit-learn"
    )

    ax.plot(
        manual_points["mean_prediction"],
        manual_points["actual_rate"],
        marker="o",
        label="Manual PyTorch"
    )

    ax.plot(
        standard_points["mean_prediction"],
        standard_points["actual_rate"],
        marker="o",
        label="Standard PyTorch"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect prediction"
    )

    ax.set_xlabel(
        "Mean predicted cancellation probability"
    )

    ax.set_ylabel(
        "Actual cancellation rate"
    )

    ax.set_title(
        "Predicted probability vs actual cancellation rate"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()

    return fig


# ============================================================
# 14. Interactive threshold evaluation
#     Uses VALIDATION, not test
# ============================================================

def evaluate_threshold(threshold):

    predictions = (
        sk_val_prob >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_val,
        predictions
    )

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Predicted 0",
        "Predicted 1"
    ])

    ax.set_yticklabels([
        "Actual 0",
        "Actual 1"
    ])

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    ax.set_title(
        f"Validation confusion matrix "
        f"(threshold = {threshold:.2f})"
    )

    expected_profit = calculate_expected_profit(
        predictions,
        y_val.values,
        val_df
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    metrics_text = (
        f"Expected incremental profit: "
        f"€{expected_profit:,.2f}\n\n"
        f"Precision: {precision:.3f}\n"
        f"Recall: {recall:.3f}\n"
        f"F1: {f1:.3f}"
    )

    return fig, metrics_text


# ============================================================
# 15. Final frozen test-set evaluation
# ============================================================

optimal_threshold = float(
    business_config["optimal_threshold"]
)

final_test_pred = (
    sk_test_prob >= optimal_threshold
).astype(int)

final_test_accuracy = accuracy_score(
    y_test,
    final_test_pred
)

final_test_precision = precision_score(
    y_test,
    final_test_pred,
    zero_division=0
)

final_test_recall = recall_score(
    y_test,
    final_test_pred,
    zero_division=0
)

final_test_f1 = f1_score(
    y_test,
    final_test_pred,
    zero_division=0
)

final_test_profit = calculate_expected_profit(
    final_test_pred,
    y_test.values,
    test_df
)

final_test_text = (
    f"Frozen threshold: {optimal_threshold:.2f}\n\n"
    f"Accuracy: {final_test_accuracy:.3f}\n"
    f"Precision: {final_test_precision:.3f}\n"
    f"Recall: {final_test_recall:.3f}\n"
    f"F1: {final_test_f1:.3f}\n\n"
    f"Expected incremental profit: "
    f"€{final_test_profit:,.2f}"
)


# ============================================================
# 16. Initial slider outputs
# ============================================================

initial_confusion_plot, initial_threshold_text = (
    evaluate_threshold(optimal_threshold)
)


# ============================================================
# 17. Build Gradio dashboard
# ============================================================

with gr.Blocks(
    title="Hotel Cancellation Dashboard"
) as demo:

    gr.Markdown(
        """
        # Hotel Booking Cancellation Dashboard

        This dashboard compares three implementations of the
        same logistic regression model for predicting booking
        cancellations at a resort hotel.

        The interactive threshold analysis uses the validation
        period, while the final test results use the threshold
        previously selected on validation.
        """
    )

    gr.Markdown(
        "## Model comparison"
    )

    with gr.Row():

        gr.Plot(
            value=plot_model_comparison(),
            label="Prediction vs actual"
        )

        gr.Plot(
            value=plot_target_distribution(),
            label="Target distribution"
        )


    gr.Markdown(
        "## Interactive validation threshold"
    )

    threshold_slider = gr.Slider(
        minimum=0.01,
        maximum=0.95,
        value=optimal_threshold,
        step=0.01,
        label="Cancellation probability threshold"
    )

    with gr.Row():

        confusion_plot = gr.Plot(
            value=initial_confusion_plot,
            label="Validation confusion matrix"
        )

        threshold_output = gr.Textbox(
            value=initial_threshold_text,
            label="Validation metrics and business value",
            interactive=False,
            lines=7
        )


    threshold_slider.change(
        fn=evaluate_threshold,
        inputs=threshold_slider,
        outputs=[
            confusion_plot,
            threshold_output
        ]
    )


    gr.Markdown(
        "## Final test-set evaluation"
    )

    gr.Textbox(
        value=final_test_text,
        label="Final out-of-sample performance",
        interactive=False,
        lines=8
    )


# ============================================================
# 18. Launch app
# ============================================================

if __name__ == "__main__":
    demo.launch()