"""
Assignment 1: Gradio dashboard for the bike-demand forecast.

Run with:  uv run python main.py app   (after main.py train)

Everything here is loaded from artifacts/ written by train.py. Nothing is
fitted at startup: the scaler and all three models are only used to predict.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from features import build_features, prepare_dataset
from train import LinearRegressionModel, predict_torch_manual, predict_torch_nn, to_counts

ARTIFACTS = Path("artifacts")
MODELS = ["scikit-learn Ridge", "PyTorch manual loop", "PyTorch nn.Module + optim"]
BASELINES = ["Baseline: same hour last week", "Baseline: hour x day-type average"]
COLORS = {MODELS[0]: "#2a6fdb", MODELS[1]: "#e8871e", MODELS[2]: "#2ca58d",
          BASELINES[0]: "#9a9a9a", BASELINES[1]: "#c4c4c4", "Actual": "#222222"}

if not (ARTIFACTS / "meta.json").exists():
    raise SystemExit("artifacts/ not found. Run `uv run python main.py train` first.")

# ---- Load the trained artifacts ---------------------------------------------
meta = json.loads((ARTIFACTS / "meta.json").read_text())
feature_names = meta["feature_names"]
scaler = joblib.load(ARTIFACTS / "scaler.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_ridge.joblib")
manual_params = torch.load(ARTIFACTS / "torch_manual.pt")
nn_model = LinearRegressionModel(len(feature_names))
nn_model.load_state_dict(torch.load(ARTIFACTS / "torch_nn.pt"))
nn_model.eval()

test_df = pd.read_csv(ARTIFACTS / "test_predictions.csv", parse_dates=["timestamp"])
coefs = pd.read_csv(ARTIFACTS / "coefficients.csv", index_col="feature")

# Full dataset for the distribution views (no fitting, only plotting).
data = prepare_dataset()


# ---- Model comparison --------------------------------------------------------
def metrics_table():
    rows = [{"Model": name, **{k: round(v, 3) for k, v in m.items()}}
            for name, m in meta["test_metrics"].items()]
    return pd.DataFrame(rows)


def pred_vs_actual(selected):
    fig = go.Figure()
    for name in selected:
        fig.add_trace(go.Scattergl(
            x=test_df["cnt"], y=test_df[name], mode="markers", name=name,
            marker=dict(size=4, opacity=0.45, color=COLORS[name])))
    top = float(max(test_df["cnt"].max(), test_df[selected].to_numpy().max() if selected else 0))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect forecast",
                             line=dict(color="black", dash="dash", width=1)))
    fig.update_layout(title="Predicted vs actual rentals per hour (test: Oct-Dec 2012)",
                      xaxis_title="Actual", yaxis_title="Predicted", height=520)
    return fig


week_starts = pd.date_range(test_df["timestamp"].min().normalize(), test_df["timestamp"].max(), freq="7D")
WEEK_CHOICES = [d.strftime("%Y-%m-%d") for d in week_starts]


def week_timeseries(week_start, selected):
    start = pd.Timestamp(week_start)
    part = test_df[(test_df["timestamp"] >= start) & (test_df["timestamp"] < start + pd.Timedelta(days=7))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=part["timestamp"], y=part["cnt"], name="Actual",
                             line=dict(color=COLORS["Actual"], width=2)))
    for name in selected:
        fig.add_trace(go.Scatter(x=part["timestamp"], y=part[name], name=name,
                                 line=dict(color=COLORS[name], width=1.5)))
    fig.update_layout(title=f"Hourly rentals, week starting {week_start}",
                      yaxis_title="Rentals per hour", height=420)
    return fig


def error_by_hour(selected):
    fig = go.Figure()
    for name in selected:
        mae = (test_df[name] - test_df["cnt"]).abs().groupby(test_df["hr"]).mean()
        fig.add_trace(go.Scatter(x=mae.index, y=mae.values, name=name, mode="lines+markers",
                                 line=dict(color=COLORS[name])))
    fig.update_layout(title="Mean absolute error by hour of day (test)",
                      xaxis_title="Hour", yaxis_title="MAE (bikes/hour)", height=380)
    return fig


def loss_curves():
    fig = go.Figure()
    for name, losses in meta["loss_curves"].items():
        dash = "dot" if "nn.Module" in name else "solid"
        fig.add_trace(go.Scatter(x=list(range(1, len(losses) + 1)), y=losses, name=name, line=dict(color=COLORS[name], dash=dash)))
    fig.update_layout(title=f"PyTorch training loss (lr={meta['learning_rate']}, alpha={meta['alpha']})",
                      xaxis_title="Epoch (full-batch step, log scale)", yaxis_title="Loss (log scale)",
                      xaxis_type="log", yaxis_type="log", height=380)
    return fig


def ablation_table():
    return pd.DataFrame(meta["feature_ablation"]).round(2)


def agreement_table():
    return pd.DataFrame([{"Check": k, "Value": f"{v:.2e}"} for k, v in meta["agreement"].items()])


# ---- Distributions -----------------------------------------------------------
RAW_FEATURES = {
    "Rentals per hour (target, cnt)": "cnt",
    "log(1 + rentals) (model target)": "log_cnt",
    "Temperature (normalised, temp)": "temp",
    "Humidity (normalised, hum)": "hum",
    "Wind speed (normalised)": "windspeed",
    "Hour of day": "hr",
    "Weather situation (1 clear - 4 heavy rain)": "weathersit",
    "Month": "mnth",
    "Rentals same hour last week": "cnt_lag_week",
}
data["log_cnt"] = np.log1p(data["cnt"])


def distribution(label, split_by):
    col = RAW_FEATURES[label]
    color = None if split_by == "none" else split_by
    plot_df = data.assign(workingday=data["workingday"].map({1: "working day", 0: "weekend/holiday"}))
    fig = px.histogram(plot_df, x=col, color=color, nbins=50, barmode="overlay", opacity=0.65,
                       category_orders={"split": ["train", "val", "test"]})
    fig.update_layout(title=f"Distribution of {label}", height=420)
    return fig


def hourly_profile():
    prof = (data.assign(day=data["workingday"].map({1: "working day", 0: "weekend/holiday"}))
            .groupby(["hr", "day"])["cnt"].mean().reset_index())
    fig = px.line(prof, x="hr", y="cnt", color="day", markers=True)
    fig.update_layout(title="Average rentals by hour: why hour x day-type features are needed",
                      xaxis_title="Hour", yaxis_title="Mean rentals", height=380)
    return fig


def temp_relationship():
    binned = data.assign(temp_c=(data["temp"] * 41).round()).groupby("temp_c")["cnt"].mean().reset_index()
    fig = px.line(binned, x="temp_c", y="cnt", markers=True)
    fig.update_layout(title="Average rentals by temperature: a curve that flattens above ~30°C, which temp^2 lets a linear model follow",
                      xaxis_title="Temperature (°C)", yaxis_title="Mean rentals", height=380)
    return fig


# ---- Coefficients ------------------------------------------------------------
def coef_plot(top_n):
    top = coefs["scikit-learn Ridge"].abs().sort_values(ascending=False).head(int(top_n)).index
    long = coefs.loc[top].reset_index().melt(id_vars="feature", var_name="model", value_name="coef")
    fig = px.bar(long, y="feature", x="coef", color="model", barmode="group", orientation="h",
                 color_discrete_map=COLORS)
    fig.update_layout(title="Largest standardised coefficients (effect on log demand per 1 std. dev.)",
                      height=max(400, 22 * int(top_n) * 3 // 2), yaxis=dict(autorange="reversed"))
    return fig


# ---- What-if forecast ----------------------------------------------------------
def what_if(hour, day_type, month, year, weather, temp_c, humidity, wind_kmh, last_week):
    row = pd.DataFrame([{
        "hr": int(hour),
        "workingday": int(day_type == "Working day"),
        "holiday": int(day_type == "Public holiday"),
        "mnth": int(month),
        "yr": int(year == 2012),
        "weathersit": {"Clear": 1, "Mist / cloudy": 2, "Rain / snow": 3}[weather],
        # Same normalisation as the UCI file: temp/41, hum/100, wind/67.
        "temp": temp_c / 41.0,
        "hum": humidity / 100.0,
        "windspeed": wind_kmh / 67.0,
        "cnt_lag_week": float(last_week),
    }])
    X = scaler.transform(build_features(row)[feature_names])
    preds = {
        MODELS[0]: to_counts(sk_model.predict(X))[0],
        MODELS[1]: to_counts(predict_torch_manual(manual_params, X))[0],
        MODELS[2]: to_counts(predict_torch_nn(nn_model, X))[0],
    }
    return pd.DataFrame([{"Model": k, "Forecast rentals this hour": round(float(v), 1)} for k, v in preds.items()])


# ---- Layout ------------------------------------------------------------------
with gr.Blocks(title="Bike demand forecast") as demo:
    gr.Markdown(
        "# Hourly bike-rental forecast: one ridge regression, three implementations\n"
        "Capital Bikeshare (Washington D.C.), 2011-2012. Models trained on Jan 2011 to Sep 2012 "
        "and evaluated on the unseen test period Oct-Dec 2012. All views load saved artifacts; "
        "nothing is retrained here."
    )
    with gr.Tab("Model comparison"):
        gr.Markdown("### Test-set metrics (bikes per hour)")
        gr.Dataframe(metrics_table(), interactive=False)
        gr.Markdown("The three implementations produce the same predictions to within 0.001 bikes, "
                    "so their points and lines sit exactly on top of each other and only the last one drawn "
                    "is visible. Tick a baseline to see how a naive forecast compares.")
        series = gr.CheckboxGroup(MODELS + BASELINES, value=MODELS, label="Series to show")
        scatter = gr.Plot(pred_vs_actual(MODELS))
        week = gr.Dropdown(WEEK_CHOICES, value=WEEK_CHOICES[0], label="Week to inspect")
        ts = gr.Plot(week_timeseries(WEEK_CHOICES[0], MODELS))
        err = gr.Plot(error_by_hour(MODELS))
        series.change(pred_vs_actual, series, scatter)
        series.change(week_timeseries, [week, series], ts)
        series.change(error_by_hour, series, err)
        week.change(week_timeseries, [week, series], ts)

    with gr.Tab("Do the three agree?"):
        gr.Markdown(
            "All three minimise the same objective, `mean((y - Xw - b)^2) + (alpha/n)·||w||^2` "
            "on the same standardised features. scikit-learn solves it exactly; both PyTorch "
            "versions reach it by gradient descent from the same zero start."
        )
        gr.Dataframe(agreement_table(), interactive=False)
        gr.Plot(loss_curves())

    with gr.Tab("Features & distributions"):
        gr.Markdown("### Feature ablation: each engineering step added in turn "
                    "(least squares, trained on train, scored on validation, bikes/hour)")
        gr.Dataframe(ablation_table(), interactive=False)
        feat = gr.Dropdown(list(RAW_FEATURES), value=list(RAW_FEATURES)[0], label="Variable")
        split_by = gr.Radio(["none", "split", "workingday"], value="none", label="Colour by")
        dist = gr.Plot(distribution(list(RAW_FEATURES)[0], "none"))
        feat.change(distribution, [feat, split_by], dist)
        split_by.change(distribution, [feat, split_by], dist)
        gr.Plot(hourly_profile())
        gr.Plot(temp_relationship())

    with gr.Tab("Coefficients"):
        top_n = gr.Slider(5, len(feature_names), value=20, step=1, label="Number of features")
        coef_fig = gr.Plot(coef_plot(20))
        top_n.change(coef_plot, top_n, coef_fig)

    with gr.Tab("What-if forecast"):
        gr.Markdown("Set the conditions for one hour and see what each saved model forecasts.")
        with gr.Row():
            hour = gr.Slider(0, 23, value=8, step=1, label="Hour")
            day_type = gr.Radio(["Working day", "Weekend", "Public holiday"], value="Working day", label="Day type")
            month = gr.Slider(1, 12, value=10, step=1, label="Month")
            year = gr.Radio([2011, 2012], value=2012, label="Year")
        with gr.Row():
            weather = gr.Radio(["Clear", "Mist / cloudy", "Rain / snow"], value="Clear", label="Weather")
            temp_c = gr.Slider(-5, 40, value=18, step=1, label="Temperature (°C)")
            humidity = gr.Slider(0, 100, value=60, step=1, label="Humidity (%)")
            wind = gr.Slider(0, 60, value=12, step=1, label="Wind speed (km/h)")
            last_week = gr.Number(value=500, label="Rentals in this hour last week")
        out = gr.Dataframe(interactive=False)
        inputs = [hour, day_type, month, year, weather, temp_c, humidity, wind, last_week]
        gr.Button("Forecast").click(what_if, inputs, out)
        demo.load(what_if, inputs, out)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
