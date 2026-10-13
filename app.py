"""
Hotel cancellation dashboard.

Loads what train.ipynb saved in artifacts/ and never trains or refits
anything. The raw CSV is read only for the distribution charts, prepared with
the same clean and add_features functions used in training.

Run with: uv run python main.py app
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from pipeline import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FALSE_ALARM_COST,
    MISSED_CANCELLATION_COST,
    NUMERIC_FEATURES,
    CancellationModel,
    add_features,
    business_cost,
    clean,
    stay_value,
)

ARTIFACTS = Path("artifacts")
DATA_PATH = Path("data/hotel_reservations.csv")
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

MODELS = ["scikit-learn", "PyTorch (manual)", "PyTorch (standard)"]
ALL_MODELS = ["Baseline"] + MODELS
SPLITS = {"Test (Oct to Dec 2018)": "test", "Validation (Jul to Sep 2018)": "validation"}

# ==================== Load the saved artifacts (no training) ====================

results = json.loads((ARTIFACTS / "results.json").read_text())
predictions = pd.read_csv(ARTIFACTS / "predictions.csv", parse_dates=["arrival"])
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_model.joblib")
manual_params = torch.load(ARTIFACTS / "manual_model.pt")
standard_model = CancellationModel(len(results["features"])).double()
standard_model.load_state_dict(torch.load(ARTIFACTS / "standard_model.pt"))
standard_model.eval()

data = add_features(clean(pd.read_csv(DATA_PATH)))
data["total_nights"] = data["no_of_weekend_nights"] + data["no_of_week_nights"]

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
CHOSEN_THRESHOLD = results["thresholds"]["scikit-learn"]

# ==================== Chart style (reference palette, light surface) ====================

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

STAYED_COLOR = "#2a78d6"  # blue
CANCELED_COLOR = "#eb6834"  # orange
MODEL_COLORS = {"scikit-learn": "#2a78d6", "PyTorch (manual)": "#eb6834", "PyTorch (standard)": "#1baf7a"}
RAISES_RISK = "#e34948"  # red
LOWERS_RISK = "#2a78d6"  # blue
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]


def style(fig, title, x_title, y_title, height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK)),
        height=height,
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        margin=dict(l=60, r=30, t=80, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def euros(value):
    return f"€{value:,.0f}"


# ==================== Tab 1: model comparison ====================


def results_table():
    table = pd.DataFrame(results["test_results"]).T
    table.index.name = "Model"
    table = table.reset_index()
    for column in ["ROC AUC", "Log loss"]:
        table[column] = table[column].map("{:.3f}".format)
    for column in ["Accuracy", "Precision", "Recall"]:
        table[column] = table[column].map("{:.1%}".format)
    for column in ["Cost (€)", "Saved vs baseline (€)"]:
        table[column] = table[column].map(euros)
    table["Threshold"] = table["Threshold"].map("{:.2f}".format)
    return table


def calibration_figure(split_label):
    """Predicted vs actual: bookings grouped by predicted probability."""
    part = predictions[predictions["split"] == SPLITS[split_label]]
    bins = np.linspace(0, 1, 11)
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
    )
    # Same data, three models: descending marker sizes keep all three visible.
    for model, size in zip(MODELS, [18, 13, 8]):
        column = f"proba_{model}"
        groups = part.groupby(pd.cut(part[column], bins, include_lowest=True), observed=True)
        summary = groups.agg(predicted=(column, "mean"), actual=("y", "mean"), bookings=("y", "size"))
        fig.add_scatter(
            x=summary["predicted"], y=summary["actual"], mode="lines+markers", name=model,
            line=dict(color=MODEL_COLORS[model], width=2),
            marker=dict(size=size, color=MODEL_COLORS[model], line=dict(color=SURFACE, width=2)),
            customdata=summary["bookings"],
            hovertemplate=(
                f"<b>{model}</b><br>Average prediction: %{{x:.1%}}<br>"
                "Actually canceled: %{y:.1%}<br>Bookings: %{customdata:,}<extra></extra>"
            ),
        )
    style(fig, "Predicted vs actual cancellation rate", "Average predicted probability", "Share actually canceled", 460)
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


# ==================== Tab 2: threshold and business cost ====================


def split_data(split_label):
    part = predictions[predictions["split"] == SPLITS[split_label]]
    return part["y"].to_numpy(), part["stay_value"].to_numpy(), part


# Cost at every threshold, computed once at startup for each model and split.
COST_CURVES = {
    (model, split): np.array([
        business_cost(y, (part[f"proba_{model}"].to_numpy() >= t).astype(int), value)
        for t in THRESHOLDS
    ])
    for model in ALL_MODELS
    for split in SPLITS
    for y, value, part in [split_data(split)]
}


def threshold_summary(model, split_label, threshold):
    y, value, part = split_data(split_label)
    flagged = (part[f"proba_{model}"].to_numpy() >= threshold).astype(int)
    cost = business_cost(y, flagged, value)
    baseline_flagged = (part["proba_Baseline"].to_numpy() >= results["thresholds"]["Baseline"]).astype(int)
    baseline_cost = business_cost(y, baseline_flagged, value)
    caught = ((flagged == 1) & (y == 1)).sum()
    precision = caught / flagged.sum() if flagged.sum() else 0.0
    recall = caught / y.sum()
    return (
        f"### Cost of mistakes: {euros(cost)}\n"
        f"Saved vs baseline ({euros(baseline_cost)}): **{euros(baseline_cost - cost)}**  \n"
        f"Flagged as likely to cancel: **{flagged.sum():,}** of {len(y):,} bookings  \n"
        f"Precision: **{precision:.1%}** of flagged bookings really canceled  \n"
        f"Recall: **{recall:.1%}** of cancellations were caught"
    )


def confusion_figure(model, split_label, threshold):
    y, value, part = split_data(split_label)
    flagged = part[f"proba_{model}"].to_numpy() >= threshold
    canceled = y == 1
    cells = [  # (row, column, mask, label, cost multiplier)
        (0, 0, ~canceled & ~flagged, "Correct: no action", 0),
        (0, 1, ~canceled & flagged, "False alarm: guest relocated", FALSE_ALARM_COST),
        (1, 0, canceled & ~flagged, "Missed: room stays empty", MISSED_CANCELLATION_COST),
        (1, 1, canceled & flagged, "Correct: room resold", 0),
    ]
    counts = np.zeros((2, 2), dtype=int)
    for row, column, mask, _, _ in cells:
        counts[row, column] = mask.sum()

    x_labels = ["Model: won't cancel", "Model: will cancel"]
    y_labels = ["Actually stayed", "Actually canceled"]
    fig = go.Figure(go.Heatmap(
        z=counts, x=x_labels, y=y_labels, xgap=2, ygap=2, showscale=False,
        colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
        hovertemplate="%{y}, %{x}<br>Bookings: %{z:,}<extra></extra>",
    ))
    for row, column, mask, label, multiplier in cells:
        cost = multiplier * value[mask].sum()
        text = f"<b>{counts[row, column]:,}</b><br>{label}"
        if multiplier:
            text += f"<br>Cost: {euros(cost)}"
        dark_cell = counts[row, column] > counts.max() * 0.5
        fig.add_annotation(
            x=x_labels[column], y=y_labels[row], text=text, showarrow=False,
            font=dict(color="#ffffff" if dark_cell else INK, size=13),
        )
    style(fig, f"Confusion matrix at threshold {threshold:.2f}", "", "", 400)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(margin=dict(t=110))
    return fig


def cost_curve_figure(model, split_label, threshold):
    costs = COST_CURVES[(model, split_label)]
    best = THRESHOLDS[costs.argmin()]
    current = costs[np.argmin(np.abs(THRESHOLDS - threshold))]
    fig = go.Figure()
    fig.add_scatter(
        x=THRESHOLDS, y=costs, mode="lines", name="Total cost",
        line=dict(color=STAYED_COLOR, width=2),
        hovertemplate="Threshold %{x:.2f}<br>Cost €%{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=[best], y=[costs.min()], mode="markers+text", name="Lowest cost",
        marker=dict(size=10, color=INK, line=dict(color=SURFACE, width=2)),
        text=[f"lowest: {best:.2f}"], textposition="bottom center",
        hovertemplate="Lowest cost at %{x:.2f}<br>€%{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=[threshold], y=[current], mode="markers", name="Slider",
        marker=dict(size=12, color=CANCELED_COLOR, line=dict(color=SURFACE, width=2)),
        hovertemplate="Slider at %{x:.2f}<br>€%{y:,.0f}<extra></extra>",
    )
    style(fig, "Business cost at every threshold", "Threshold", "Cost of mistakes (€)", 400)
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", rangemode="tozero")
    return fig


def probability_histogram(model, split_label, threshold):
    y, _, part = split_data(split_label)
    proba = part[f"proba_{model}"]
    fig = go.Figure()
    for outcome, label, color in [(0, "Stayed", STAYED_COLOR), (1, "Canceled", CANCELED_COLOR)]:
        fig.add_histogram(
            x=proba[y == outcome], name=label, marker_color=color,
            xbins=dict(start=0, end=1, size=0.02),
            hovertemplate=f"{label}<br>Probability %{{x}}<br>Bookings: %{{y:,}}<extra></extra>",
        )
    fig.add_vline(x=threshold, line=dict(color=INK, width=2))
    fig.add_annotation(
        x=threshold, y=1, yref="paper", text=f"threshold {threshold:.2f}: flagged to the right",
        showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=INK),
    )
    style(fig, "Predicted probabilities by actual outcome", "Predicted probability of canceling", "Bookings", 400)
    fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return fig


def update_threshold_tab(model, split_label, threshold):
    return (
        threshold_summary(model, split_label, threshold),
        confusion_figure(model, split_label, threshold),
        cost_curve_figure(model, split_label, threshold),
        probability_histogram(model, split_label, threshold),
    )


# ==================== Tab 3: data distributions ====================

NUMERIC_VIEWS = {
    "Lead time (days)": "lead_time",
    "Room price (€ per night)": "avg_price_per_room",
    "Total nights": "total_nights",
    "Special requests": "no_of_special_requests",
    "Adults": "no_of_adults",
    "Children": "no_of_children",
}
CATEGORY_VIEWS = {
    "Market segment": "market_segment_type",
    "Room type": "room_type_reserved",
    "Meal plan": "type_of_meal_plan",
    "Arrival weekday": "arrival_weekday",
    "Parking space requested": "required_car_parking_space",
    "Repeated guest": "repeated_guest",
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def distribution_figure(feature_label):
    if feature_label in NUMERIC_VIEWS:
        column = NUMERIC_VIEWS[feature_label]
        fig = go.Figure()
        for outcome, label, color in [(0, "Stayed", STAYED_COLOR), (1, "Canceled", CANCELED_COLOR)]:
            fig.add_histogram(
                x=data.loc[data["y"] == outcome, column], name=label, marker_color=color, nbinsx=40,
                hovertemplate=f"{label}<br>{feature_label}: %{{x}}<br>Bookings: %{{y:,}}<extra></extra>",
            )
        style(fig, f"{feature_label}: stayed vs canceled (all 36,275 bookings)", feature_label, "Bookings", 420)
        fig.update_layout(barmode="stack", bargap=0.08)
        return fig

    column = CATEGORY_VIEWS[feature_label]
    summary = data.groupby(column)["y"].agg(rate="mean", bookings="size").reset_index()
    if column == "arrival_weekday":
        summary = summary.set_index(column).loc[WEEKDAYS].reset_index()
    else:
        summary = summary.sort_values("rate")
    if column in ("required_car_parking_space", "repeated_guest"):
        summary[column] = summary[column].map({0: "No", 1: "Yes"})
    fig = go.Figure(go.Bar(
        x=summary[column].astype(str), y=summary["rate"], marker_color=STAYED_COLOR,
        customdata=summary["bookings"],
        hovertemplate="%{x}<br>Cancellation rate: %{y:.1%}<br>Bookings: %{customdata:,}<extra></extra>",
    ))
    style(fig, f"Cancellation rate by {feature_label.lower()} (all bookings)", feature_label, "Cancellation rate", 420)
    fig.update_layout(barcornerradius=4, bargap=0.4)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def drift_figure():
    monthly = data.groupby(data["arrival"].dt.to_period("M"))["y"].agg(rate="mean", bookings="size")
    months = monthly.index.to_timestamp()
    fig = go.Figure()
    periods = [
        ("Train", "2017-07-01", "2018-07-01", 0.10),
        ("Validation", "2018-07-01", "2018-10-01", 0.20),
        ("Test", "2018-10-01", "2019-01-01", 0.30),
    ]
    for label, start, end, opacity in periods:
        fig.add_vrect(
            x0=start, x1=end, fillcolor=MUTED, opacity=opacity, line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font=dict(color=INK_SECONDARY),
        )
    fig.add_scatter(
        x=months, y=monthly["rate"], mode="lines+markers", name="Cancellation rate",
        line=dict(color=CANCELED_COLOR, width=2),
        marker=dict(size=8, color=CANCELED_COLOR, line=dict(color=SURFACE, width=2)),
        customdata=monthly["bookings"],
        hovertemplate="%{x|%b %Y}<br>Cancellation rate: %{y:.1%}<br>Bookings: %{customdata:,}<extra></extra>",
    )
    style(fig, "Cancellation rate by arrival month", "Arrival month", "Cancellation rate", 420)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return fig


# ==================== Tab 4: coefficients ====================

FEATURE_LABELS = {
    "no_of_adults": "Adults",
    "no_of_children": "Children",
    "avg_price_per_room": "Room price",
    "no_of_special_requests": "Special requests",
    "log_lead_time": "Lead time (log)",
    "log_total_nights": "Total nights (log)",
    "weekend_share": "Share of nights on weekend",
    "required_car_parking_space": "Parking space requested",
    "repeated_guest": "Repeated guest",
    "invalid_date": "Arrival date was 29 Feb 2018",
    "zero_nights": "Zero-night booking",
    "is_free": "Free room",
    "has_prev_cancellation": "Has canceled before",
    "has_prev_stay": "Has stayed before",
    "type_of_meal_plan": "Meal plan",
    "room_type_reserved": "Room type",
    "market_segment_type": "Market segment",
    "arrival_weekday": "Arrival weekday",
}


def readable_feature(name):
    """'categorical__market_segment_type_Offline' -> 'Market segment: Offline (vs Online)'."""
    kind, feature = name.split("__", 1)
    if kind != "categorical":
        label = FEATURE_LABELS[feature]
        return f"{label} (per std. dev.)" if kind == "numeric" else label
    column = next(c for c in CATEGORICAL_FEATURES if feature.startswith(c + "_"))
    category = feature[len(column) + 1:].replace("infrequent_sklearn", "rare categories")
    reference = results["reference_categories"][column]
    return f"{FEATURE_LABELS[column]}: {category} (vs {reference})"


def coefficients_figure():
    coefficients = pd.Series(results["coefficients"]).sort_values()
    labels = [readable_feature(name) for name in coefficients.index]
    colors = [RAISES_RISK if value > 0 else LOWERS_RISK for value in coefficients]
    fig = go.Figure(go.Bar(
        x=coefficients, y=labels, orientation="h", marker_color=colors,
        customdata=np.exp(coefficients),
        hovertemplate="%{y}<br>Coefficient: %{x:.3f}<br>Odds multiplied by %{customdata:.2f}<extra></extra>",
    ))
    style(fig, "What raises (red) or lowers (blue) the risk of cancellation", "Coefficient (change in log-odds)", "", 900)
    fig.update_layout(barcornerradius=4, bargap=0.3, margin=dict(l=330))
    fig.update_yaxes(gridcolor=SURFACE)
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    return fig


# ==================== Tab 5: predict one booking ====================


def predict_probabilities(rows):
    """Run prepared booking rows through the saved preprocessor and all three models."""
    X = preprocessor.transform(rows[FEATURES])
    X_t = torch.tensor(X, dtype=torch.float64)
    with torch.no_grad():
        manual = torch.sigmoid(X_t @ manual_params["weight"] + manual_params["bias"]).numpy()
        standard = torch.sigmoid(standard_model(X_t)).numpy()
    return {
        "scikit-learn": sk_model.predict_proba(X)[:, 1],
        "PyTorch (manual)": manual,
        "PyTorch (standard)": standard,
    }


def predict_booking(adults, children, weekend_nights, week_nights, meal_plan, parking, room_type,
                    lead_time, arrival, segment, repeated, prev_cancellations, prev_stays, price, requests):
    try:
        arrival_date = pd.Timestamp(arrival)
    except (ValueError, TypeError):
        raise gr.Error("Please enter the arrival date as YYYY-MM-DD.")

    booking = pd.DataFrame([{
        "no_of_adults": int(adults),
        "no_of_children": int(children),
        "no_of_weekend_nights": int(weekend_nights),
        "no_of_week_nights": int(week_nights),
        "type_of_meal_plan": meal_plan,
        "required_car_parking_space": int(parking),
        "room_type_reserved": room_type,
        "lead_time": int(lead_time),
        "arrival_year": arrival_date.year,
        "arrival_month": arrival_date.month,
        "arrival_date": arrival_date.day,
        "market_segment_type": segment,
        "repeated_guest": int(repeated),
        "no_of_previous_cancellations": int(prev_cancellations),
        "no_of_previous_bookings_not_canceled": int(prev_stays),
        "avg_price_per_room": float(price),
        "no_of_special_requests": int(requests),
    }])
    prepared = add_features(clean(booking))
    probabilities = {model: float(p[0]) for model, p in predict_probabilities(prepared).items()}

    p = probabilities["scikit-learn"]
    value = float(stay_value(prepared).iloc[0])
    cost_if_overbook = (1 - p) * FALSE_ALARM_COST * value
    cost_if_keep = p * MISSED_CANCELLATION_COST * value
    if p >= CHOSEN_THRESHOLD:
        decision = "**Likely to cancel: count this room as available and overbook it.**"
    else:
        decision = "**Likely to stay: keep the room reserved for this guest.**"

    rows = "\n".join(f"| {model} | {prob:.1%} |" for model, prob in probabilities.items())
    return (
        f"| Model | Probability of canceling |\n|---|---|\n{rows}\n\n"
        f"{decision}  \n"
        f"Decision threshold: {CHOSEN_THRESHOLD:.2f} (chosen on validation to minimise cost)\n\n"
        f"Stay value: **{euros(value)}** ({euros(float(price))} × {int(weekend_nights) + int(week_nights)} nights)  \n"
        f"Expected cost if you overbook: (1 − {p:.2f}) × {FALSE_ALARM_COST} × {euros(value)} = **{euros(cost_if_overbook)}**  \n"
        f"Expected cost if you keep the room: {p:.2f} × {MISSED_CANCELLATION_COST} × {euros(value)} = **{euros(cost_if_keep)}**\n\n"
        f"Taken at face value, these costs favour overbooking only above "
        f"{FALSE_ALARM_COST / (FALSE_ALARM_COST + MISSED_CANCELLATION_COST):.2f}. The threshold is lower "
        f"({CHOSEN_THRESHOLD:.2f}) because the model's probabilities run low on recent bookings, so the "
        f"threshold that minimised real cost on validation sits below the theoretical one."
    )


# ==================== Layout ====================

with gr.Blocks(title="Hotel cancellation dashboard") as demo:
    gr.Markdown(
        "# Hotel cancellation dashboard\n"
        "Predicts which bookings will be canceled, so a revenue manager can decide how many rooms "
        "to overbook. A missed cancellation leaves a room empty (cost: 1× the stay); a false alarm "
        "means relocating a guest (cost: 1.5× the stay). Logistic regression is trained three ways "
        "(scikit-learn, manual PyTorch, standard PyTorch); this app only loads the saved models."
    )

    with gr.Tab("Model comparison"):
        gr.Markdown("### Test set results (Oct to Dec 2018), each model at its validation-chosen threshold")
        gr.Dataframe(results_table(), interactive=False)
        calibration_split = gr.Radio(list(SPLITS), value=list(SPLITS)[0], label="Period")
        calibration_plot = gr.Plot()
        gr.Markdown(
            "The three models give identical probabilities, so their lines lie on top of each other "
            "(the marker sizes differ so all three stay visible). Points above the dotted line mean "
            "the models underestimate cancellations: they learned from a period with fewer of them."
        )

    with gr.Tab("Threshold and cost"):
        with gr.Row():
            threshold_model = gr.Dropdown(ALL_MODELS, value="scikit-learn", label="Model")
            threshold_split = gr.Radio(list(SPLITS), value=list(SPLITS)[0], label="Period")
        threshold_slider = gr.Slider(0.01, 0.99, value=CHOSEN_THRESHOLD, step=0.01, label="Decision threshold")
        threshold_text = gr.Markdown()
        with gr.Row():
            confusion_plot = gr.Plot()
            cost_plot = gr.Plot()
        histogram_plot = gr.Plot()

    with gr.Tab("Data"):
        feature_picker = gr.Dropdown(
            list(NUMERIC_VIEWS) + list(CATEGORY_VIEWS), value="Lead time (days)", label="Feature"
        )
        distribution_plot = gr.Plot()
        gr.Plot(drift_figure())
        gr.Markdown(
            "Cancellation rates change a lot over time, which is why the data is split by arrival "
            "date and why month features were left out: with one year of data, the model cannot "
            "tell a seasonal pattern from a change between years."
        )

    with gr.Tab("What drives cancellations"):
        gr.Plot(coefficients_figure())
        gr.Markdown(
            "Coefficients from the scikit-learn model (the PyTorch ones are identical). Numeric "
            "features were standardised, so their bar is the effect of one standard deviation. "
            "Categories are compared with the most common one. Hover to see how much each feature "
            "multiplies the odds of canceling."
        )

    with gr.Tab("Predict one booking"):
        gr.Markdown("Enter a booking. It goes through the same cleaning, features, preprocessor and saved models as in training.")
        with gr.Row():
            adults = gr.Number(2, label="Adults", precision=0, minimum=0)
            children = gr.Number(0, label="Children", precision=0, minimum=0)
            weekend_nights = gr.Number(1, label="Weekend nights", precision=0, minimum=0)
            week_nights = gr.Number(2, label="Week nights", precision=0, minimum=0)
        with gr.Row():
            arrival = gr.Textbox("2018-10-15", label="Arrival date (YYYY-MM-DD)")
            lead_time = gr.Number(57, label="Lead time (days booked ahead)", precision=0, minimum=0)
            price = gr.Number(100, label="Room price (€ per night)", minimum=0)
            requests = gr.Number(0, label="Special requests", precision=0, minimum=0)
        with gr.Row():
            segment = gr.Dropdown(sorted(data["market_segment_type"].unique()), value="Online", label="Market segment")
            room_type = gr.Dropdown(sorted(data["room_type_reserved"].unique()), value="Room_Type 1", label="Room type")
            meal_plan = gr.Dropdown(sorted(data["type_of_meal_plan"].unique()), value="Meal Plan 1", label="Meal plan")
        with gr.Row():
            parking = gr.Checkbox(False, label="Parking space requested")
            repeated = gr.Checkbox(False, label="Repeated guest")
            prev_cancellations = gr.Number(0, label="Previous cancellations", precision=0, minimum=0)
            prev_stays = gr.Number(0, label="Previous stays (not canceled)", precision=0, minimum=0)
        predict_button = gr.Button("Predict", variant="primary")
        prediction_text = gr.Markdown()

    # Wiring: every control redraws its charts as soon as it changes.
    calibration_split.change(calibration_figure, calibration_split, calibration_plot)
    threshold_inputs = [threshold_model, threshold_split, threshold_slider]
    threshold_outputs = [threshold_text, confusion_plot, cost_plot, histogram_plot]
    for control in threshold_inputs:
        control.change(update_threshold_tab, threshold_inputs, threshold_outputs)
    feature_picker.change(distribution_figure, feature_picker, distribution_plot)
    predict_button.click(
        predict_booking,
        [adults, children, weekend_nights, week_nights, meal_plan, parking, room_type,
         lead_time, arrival, segment, repeated, prev_cancellations, prev_stays, price, requests],
        prediction_text,
    )

    # Draw the initial charts when the page opens.
    demo.load(calibration_figure, calibration_split, calibration_plot)
    demo.load(update_threshold_tab, threshold_inputs, threshold_outputs)
    demo.load(distribution_figure, feature_picker, distribution_plot)

if __name__ == "__main__":
    demo.launch()
