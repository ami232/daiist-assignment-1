"""Interactive dashboard. Loads saved models; never fits or updates a model."""
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import gradio as gr
from features import METHODS, ROOT, date_splits, load_data, rental_units, scale_features
from models import DTYPE, LinearDemand

ARTIFACTS = ROOT / "artifacts"
REQUIRED = ["preprocessing.joblib", "sklearn.joblib", "manual.pt", "standard.pt", "baseline.joblib", "metrics.json", "test_predictions.csv", "learning_curves.json"]
missing = [name for name in REQUIRED if not (ARTIFACTS / name).exists()]
if missing:
    raise SystemExit("Saved models are missing. Run 'uv run python main.py train' first. Missing: " + ", ".join(missing))
torch.set_num_threads(1)
bundle = joblib.load(ARTIFACTS / "preprocessing.joblib")
sklearn_model = joblib.load(ARTIFACTS / "sklearn.joblib")
baseline_model = joblib.load(ARTIFACTS / "baseline.joblib")
manual_model = torch.load(ARTIFACTS / "manual.pt", map_location="cpu", weights_only=True)
standard_model = LinearDemand(len(bundle["feature_names"]))
standard_model.load_state_dict(torch.load(ARTIFACTS / "standard.pt", map_location="cpu", weights_only=True))
standard_model.eval()
summary = json.loads((ARTIFACTS / "metrics.json").read_text())
histories = json.loads((ARTIFACTS / "learning_curves.json").read_text())
test = pd.read_csv(ARTIFACTS / "test_predictions.csv", parse_dates=["timestamp"])
data = load_data()
splits = date_splits(data)
ALL_METHODS = METHODS + ["Training-mean baseline"]
COLORS = ["#087f8c", "#6653a4", "#df7b24", "#717b88"]


def style(figure, title):
    figure.update_layout(template="plotly_white", title=title, font=dict(family="Arial", size=13),
                         margin=dict(l=40, r=25, t=60, b=40), height=380,
                         legend=dict(orientation="h", y=-0.22))
    return figure


def comparison_plots(method):
    figure = px.scatter(test, x="cnt", y=method, opacity=0.4, color_discrete_sequence=["#087f8c"],
                        labels={"cnt": "Actual rentals in an hour", method: "Predicted rentals in an hour"})
    low, high = min(0, float(test[method].min())), float(max(test["cnt"].max(), test[method].max()))
    figure.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect prediction", line=dict(color="#df7b24", dash="dash")))
    scatter = style(figure, f"{method}: predicted vs actual")
    daily = test.groupby("dteday", as_index=False)[["cnt", method]].sum()
    timeline = go.Figure()
    timeline.add_trace(go.Scatter(x=daily["dteday"], y=daily["cnt"], name="Actual", line=dict(color="#252f3d")))
    timeline.add_trace(go.Scatter(x=daily["dteday"], y=daily[method], name="Predicted", line=dict(color="#087f8c")))
    timeline.update_yaxes(title="Rentals per day (hourly predictions summed)")
    return scatter, style(timeline, "Demand through the held-out test period")


DISTRIBUTIONS = {"Hourly rentals (target)": "cnt", "Hour of day": "hr", "Season code": "season", "Month": "mnth", "Day of week": "weekday"}


def explore(feature, subset):
    frame = data if subset == "All data" else data.loc[splits[subset.lower()]]
    column = DISTRIBUTIONS[feature]
    histogram = px.histogram(frame, x=column, nbins=40, color_discrete_sequence=["#087f8c"], labels={column: feature})
    histogram.update_yaxes(title="Recorded hours")
    hourly = frame.groupby(["hr", "season"], as_index=False)["cnt"].mean()
    hourly["season"] = hourly["season"].astype(str)
    seasonal = px.line(hourly, x="hr", y="cnt", color="season", markers=True,
                       color_discrete_sequence=COLORS,
                       labels={"hr": "Hour of day", "cnt": "Mean observed hourly rentals", "season": "Season code"})
    seasonal.update_xaxes(dtick=3)
    return style(histogram, f"{feature} · {subset}"), style(seasonal, f"Observed hourly patterns by season · {subset}")


def predict_scenario(day, hour):
    rows = data.loc[data["dteday"] == day]
    if rows.empty:
        raise gr.Error("Choose one of the recorded historical dates.")
    scenario = rows.iloc[[0]].copy()
    scenario["hr"] = int(hour)
    x = scale_features(scenario, bundle)
    tx = torch.tensor(x, dtype=DTYPE)
    with torch.no_grad():
        predictions = [
            rental_units(sklearn_model.predict(x), bundle)[0],
            rental_units((tx @ manual_model["weights"] + manual_model["bias"]).numpy(), bundle)[0],
            rental_units(standard_model(tx).numpy(), bundle)[0],
            float(baseline_model.predict(x)[0]),
        ]
    actual_row = rows.loc[rows["hr"] == int(hour), "cnt"]
    actual = "No recorded observation for this hour" if actual_row.empty else f"{int(actual_row.iloc[0])} rentals"
    note = (f"**{day}, {int(hour):02d}:00** · Season code {int(scenario['season'].iloc[0])} · "
            f"{'Working day' if int(scenario['workingday'].iloc[0]) else 'Day off'}\n\n"
            f"**Recorded actual:** {actual}. These are system-wide rentals, not a shop's required bike stock.")
    table = pd.DataFrame({"Method": ALL_METHODS, "Predicted hourly rentals": np.round(predictions, 2)})
    return table, note

metrics_rows = []
for method in ALL_METHODS:
    measured = summary["evaluation"]["test"][method]
    metrics_rows.append({"Method": method, "RMSE (rentals/h)": round(measured["RMSE"], 3),
                         "MAE (rentals/h)": round(measured["MAE"], 3), "R²": round(measured["R2"], 4)})
metrics_table = pd.DataFrame(metrics_rows)
experiment_table = pd.DataFrame([
    {"Feature set": name, "Columns": values["features"], "Validation RMSE": round(values["RMSE"], 3)}
    for name, values in summary["feature_experiment_validation"].items()
])
curve_data = pd.DataFrame([{**point, "Method": method} for method, points in histories.items() for point in points])
curve = style(px.line(curve_data, x="epoch", y="objective", color="Method", color_discrete_sequence=COLORS),
              "Training objective as the two loops learn")

with gr.Blocks(title="Bike demand planner", analytics_enabled=False) as demo:
    gr.HTML('''<div class="hero"><div class="eyebrow">BIKE DEMAND / CALENDAR-ONLY PLANNING</div>
    <h1>See demand before the busy hours.</h1>
    <p>Explore hourly rental forecasts, compare three ways of learning the same model,
    and inspect the seasonal patterns behind the predictions.</p></div>''')
    with gr.Row():
        gr.Markdown("### 17,379 hours\nWashington DC · 2011–2012")
        gr.Markdown("### Three implementations\nOne linear model · saved predictions")
        gr.Markdown(f"### {summary['evaluation']['test'][METHODS[0]]['RMSE']:.1f} test RMSE\nRentals per hour · large misses count more")
    gr.Markdown("**Planning scope:** this project illustrates a city-center shop scenario using system-wide historical rentals. It cannot determine the exact bike stock needed by a shop.")
    with gr.Tab("Compare models"):
        gr.Markdown("### How well do the predictions match?\nThe final test is October–December 2012. Earlier dates were used for training and validation. Lower RMSE and MAE are better; R² compares against the test-period mean.")
        gr.Dataframe(metrics_table, interactive=False, label="Same test hours for every method", wrap=True)
        method = gr.Dropdown(ALL_METHODS, value=METHODS[0], label="Choose a saved model")
        scatter_value, timeline_value = comparison_plots(METHODS[0])
        with gr.Row():
            scatter = gr.Plot(scatter_value)
            timeline = gr.Plot(timeline_value)
        method.change(comparison_plots, method, [scatter, timeline])
        with gr.Accordion("Feature experiment and model agreement", open=False):
            gr.Markdown("The feature experiment uses validation hours only. Separate hour and season effects are compared with a distinct hourly pattern for each season.")
            gr.Dataframe(experiment_table, interactive=False, label="Validation feature comparison")
            gr.Dataframe(pd.DataFrame([{"Pair": pair, "Maximum difference (rentals/h)": value} for pair, value in summary["test_prediction_max_difference"].items()]),
                         interactive=False, label="Prediction agreement on the test set")
            gr.Plot(curve)
            gr.Markdown("Both PyTorch loops minimize mean squared error plus an L2 penalty on weights. scikit-learn uses the equivalent Ridge objective. All comparisons retain the unrounded predictions.")
    with gr.Tab("Explore the data"):
        gr.Markdown("### When are rentals recorded?\nSeason codes are the four categories supplied with the dataset. Missing hours are absent observations and have not been filled with zero rentals.")
        with gr.Row():
            distribution_feature = gr.Dropdown(list(DISTRIBUTIONS), value="Hourly rentals (target)", label="Distribution")
            subset = gr.Dropdown(["Train", "Validation", "Test", "All data"], value="Train", label="Date subset")
        histogram_value, seasonal_value = explore("Hourly rentals (target)", "Train")
        with gr.Row():
            histogram = gr.Plot(histogram_value)
            seasonal = gr.Plot(seasonal_value)
        distribution_feature.change(explore, [distribution_feature, subset], [histogram, seasonal])
        subset.change(explore, [distribution_feature, subset], [histogram, seasonal])
    with gr.Tab("Try a calendar date"):
        gr.Markdown("### Compare the saved models on a historical scenario\nChoose a recorded date and hour. Predictions use year, day type, weekday, and season–hour indicators. Weather is excluded. The available dates keep this demonstration within the dataset's historical scope.")
        with gr.Row():
            day = gr.Dropdown(sorted(data["dteday"].unique().tolist()), value="2012-10-01", label="Calendar date")
            hour = gr.Slider(0, 23, value=12, step=1, label="Hour of day")
        predict_button = gr.Button("Compare rental estimates", variant="primary")
        initial_table, initial_note = predict_scenario("2012-10-01", 12)
        scenario_note = gr.Markdown(initial_note)
        scenario_table = gr.Dataframe(initial_table, interactive=False, label="Hourly rental estimates")
        predict_button.click(predict_scenario, [day, hour], [scenario_table, scenario_note], api_name="predict_calendar")
        gr.Markdown("These estimates are rental counts, not required inventory. Daily totals above are sums over recorded hours. A calendar-only model cannot account for unexpected weather, events, or changes in the service.")
    gr.Markdown("Source: [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset) · Fanaee-T (2013) · CC BY 4.0. The dashboard loads models already saved by the training stage.")

CSS = """
.gradio-container {max-width: 1260px !important; margin: auto;}
.hero {background: #103c41; color: #f7fbfa; padding: 34px 38px; border-radius: 18px; margin-bottom: 14px;}
.hero h1 {font-size: 34px; line-height: 1.15; font-weight: 650; color: #f7fbfa; margin: 12px 0;}
.hero p {max-width: 800px; color: #d5e7e5; font-size: 16px; line-height: 1.6;}
.eyebrow {font-size: 11px; letter-spacing: 2px; font-weight: 700; color: #a5d6ca;}
"""

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False,
                share=False, show_error=True, theme=gr.themes.Soft(primary_hue="teal"), css=CSS,
                footer_links=[])
