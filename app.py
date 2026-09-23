"""Gradio dashboard for the saved bikeshare models.

This file does not train. It loads the Ridge model, the manual PyTorch
weights, and the nn.Module state dict written by train.py, and scores the
same test matrix those fits were evaluated on.
"""

from __future__ import annotations

import os
import sys

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

MODEL_COLORS = {
    "sklearn": "#1f4b99",
    "manual": "#c46b1a",
    "standard": "#1f7a4d",
}
MODEL_LABELS = {
    "sklearn": "scikit-learn Ridge",
    "manual": "manual PyTorch loop",
    "standard": "nn.Module + SGD",
}

DIST_COLUMNS = {
    "cnt": "Hourly rentals (target)",
    "hr": "Hour of day",
    "temp_c": "Temperature (°C)",
    "hum_pct": "Humidity (%)",
    "wind": "Wind speed",
    "workingday": "Working day (1 = yes)",
    "holiday": "Holiday (1 = yes)",
    "weathersit": "Weather situation code",
    "lag_1": "Rentals, previous hour",
    "lag_24": "Rentals, same hour yesterday",
    "lag_168": "Rentals, same hour last week",
    "precip": "Precipitation flag",
    "morning_rush": "Weekday morning rush flag",
    "evening_rush": "Weekday evening rush flag",
}


class LinearRegressor(torch.nn.Module):
    """Matches train.LinearRegressor so the saved state dict loads."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, dtype=torch.float64)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `uv run python main.py train` first. "
            "The app does not train models itself."
        )
    return path


def load_artifacts() -> dict:
    prep = joblib.load(_require(ARTIFACTS / "preprocessor.joblib"))
    ridge = joblib.load(_require(ARTIFACTS / "sklearn_ridge.joblib"))
    manual = torch.load(_require(ARTIFACTS / "torch_manual.pt"), weights_only=True)
    dashboard = joblib.load(_require(ARTIFACTS / "dashboard.joblib"))
    n_features = len(prep["feature_names"])
    standard = LinearRegressor(n_features)
    state = torch.load(_require(ARTIFACTS / "torch_standard.pt"), weights_only=True)
    standard.load_state_dict(state)
    standard.eval()
    frame = pd.read_csv(_require(ARTIFACTS / "distributions.csv"), parse_dates=["ts"])
    return {
        "prep": prep,
        "ridge": ridge,
        "manual_w": manual["w"].numpy().reshape(-1),
        "manual_b": float(manual["b"].reshape(-1)[0]),
        "standard": standard,
        "dashboard": dashboard,
        "frame": frame,
    }


LOADED = load_artifacts()


def to_bikes(pred_scaled: np.ndarray) -> np.ndarray:
    prep = LOADED["prep"]
    return np.clip(pred_scaled * prep["y_std"] + prep["y_mean"], 0, None)


def predict_test() -> dict[str, np.ndarray]:
    """Run each saved model on the stored test matrix. No fitting."""
    x_test = LOADED["dashboard"]["x_test"]
    ridge_pred = to_bikes(LOADED["ridge"].predict(x_test))
    manual_pred = to_bikes(x_test @ LOADED["manual_w"] + LOADED["manual_b"])
    with torch.no_grad():
        standard_scaled = LOADED["standard"](torch.tensor(x_test, dtype=torch.float64)).numpy()
    standard_pred = to_bikes(standard_scaled)
    return {"sklearn": ridge_pred, "manual": manual_pred, "standard": standard_pred}


PREDICTIONS = predict_test()
TEST_TS = pd.to_datetime(LOADED["dashboard"]["ts_test"])
Y_TEST = LOADED["dashboard"]["y_test"]
LAG168 = LOADED["dashboard"]["lag168_test"]
TRAIN_MEAN = float(LOADED["dashboard"]["train_mean"])
METRICS = LOADED["dashboard"]["metrics"]


def _metric_rows() -> list[list[str]]:
    rows = []
    labels = [
        ("sklearn", "scikit-learn Ridge"),
        ("manual", "manual PyTorch loop"),
        ("standard", "nn.Module + SGD"),
    ]
    for key, label in labels:
        scores = METRICS["models"][key]["test"]
        rows.append([label, f"{scores['mae']:.2f}", f"{scores['rmse']:.2f}", f"{scores['r2']:.3f}"])
    mean_scores = METRICS["baselines"]["train_mean"]
    week_scores = METRICS["baselines"]["same_hour_last_week"]
    rows.append(["baseline: training mean", f"{mean_scores['mae']:.2f}", f"{mean_scores['rmse']:.2f}", f"{mean_scores['r2']:.3f}"])
    rows.append(["baseline: same hour last week", f"{week_scores['mae']:.2f}", f"{week_scores['rmse']:.2f}", f"{week_scores['r2']:.3f}"])
    return rows


def comparison_plots(model_name: str, show_baseline: bool, day_offset: int):
    pred = PREDICTIONS[model_name]
    actual = Y_TEST
    color = MODEL_COLORS[model_name]

    scatter = go.Figure()
    scatter.add_trace(go.Scatter(
        x=actual,
        y=pred,
        mode="markers",
        marker={"size": 5, "opacity": 0.35, "color": color},
        name=MODEL_LABELS[model_name],
        hovertemplate="actual %{x:.0f}<br>predicted %{y:.0f}<extra></extra>",
    ))
    upper = float(max(actual.max(), pred.max()))
    scatter.add_trace(go.Scatter(
        x=[0, upper],
        y=[0, upper],
        mode="lines",
        line={"color": "#333", "dash": "dash"},
        name="perfect prediction",
    ))
    scatter.update_layout(
        title="Test quarter: predicted rentals vs actual",
        xaxis_title="Actual rentals",
        yaxis_title="Predicted rentals",
        template="plotly_white",
        height=460,
        legend={"orientation": "h"},
    )

    start = TEST_TS.min() + pd.Timedelta(days=int(day_offset))
    end = start + pd.Timedelta(days=14)
    mask = (TEST_TS >= start) & (TEST_TS < end)
    window_ts = TEST_TS[mask]
    series = go.Figure()
    series.add_trace(go.Scatter(
        x=window_ts, y=actual[mask], mode="lines", name="actual",
        line={"color": "#222", "width": 2},
    ))
    for key, label in MODEL_LABELS.items():
        series.add_trace(go.Scatter(
            x=window_ts, y=PREDICTIONS[key][mask], mode="lines", name=label,
            line={"color": MODEL_COLORS[key], "width": 1.4},
        ))
    if show_baseline:
        series.add_trace(go.Scatter(
            x=window_ts, y=LAG168[mask], mode="lines", name="same hour last week",
            line={"color": "#888", "dash": "dot", "width": 1.5},
        ))
    series.update_layout(
        title=f"Fourteen days from {start:%d %b %Y}",
        xaxis_title="Hour",
        yaxis_title="Rentals",
        template="plotly_white",
        height=420,
        legend={"orientation": "h"},
    )

    gap = PREDICTIONS["manual"] - PREDICTIONS["sklearn"]
    gap_fig = go.Figure()
    gap_fig.add_trace(go.Scatter(
        x=TEST_TS, y=gap, mode="lines", name="manual − sklearn",
        line={"color": MODEL_COLORS["manual"], "width": 1},
    ))
    gap_fig.add_hline(y=0, line_dash="dash", line_color="#333")
    gap_fig.update_layout(
        title="Manual forecast minus Ridge, in bikes. A flat line at 0 means they match.",
        yaxis_title="Bikes",
        yaxis={"range": [-0.01, 0.01]},
        template="plotly_white",
        height=320,
    )

    stride = LOADED["dashboard"]["loss_stride"]
    epochs = np.arange(len(LOADED["dashboard"]["manual_loss"])) * stride
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(
        x=epochs, y=LOADED["dashboard"]["manual_loss"], name="manual loop",
        line={"color": MODEL_COLORS["manual"]},
    ))
    loss_fig.add_trace(go.Scatter(
        x=epochs, y=LOADED["dashboard"]["standard_loss"], name="nn.Module + SGD",
        line={"color": MODEL_COLORS["standard"], "dash": "dash"},
    ))
    loss_fig.update_layout(
        title="Training loss on standardized rentals (every 10th epoch)",
        xaxis_title="Epoch",
        yaxis_title="Mean squared error + L2",
        template="plotly_white",
        height=360,
        legend={"orientation": "h"},
    )
    return scatter, series, gap_fig, loss_fig, _metric_rows()


def distribution_plots(column: str, split_name: str):
    frame = LOADED["frame"]
    if split_name != "all":
        frame = frame.loc[frame["split"] == split_name]
    values = frame[column].dropna()
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=values, nbinsx=40, marker_color="#1f4b99"))
    hist.update_layout(
        title=DIST_COLUMNS.get(column, column),
        xaxis_title=DIST_COLUMNS.get(column, column),
        yaxis_title="Hours",
        template="plotly_white",
        height=420,
    )

    profile_source = LOADED["frame"] if split_name == "all" else frame
    grouped = profile_source.groupby(["workingday", "hr"], as_index=False)["cnt"].mean()
    profile = go.Figure()
    for flag, label, color in (
        (1, "Working day", "#1f4b99"),
        (0, "Weekend or holiday", "#c46b1a"),
    ):
        part = grouped.loc[grouped["workingday"] == flag].sort_values("hr")
        profile.add_trace(go.Scatter(
            x=part["hr"], y=part["cnt"], mode="lines+markers", name=label,
            line={"color": color},
        ))
    profile.update_layout(
        title="Mean rentals by hour — the shape a plain hour number cannot fit",
        xaxis_title="Hour of day",
        yaxis_title="Mean rentals",
        template="plotly_white",
        height=420,
        legend={"orientation": "h"},
        xaxis={"dtick": 1},
    )
    return hist, profile


def _cost_curve(pred: np.ndarray, actual: np.ndarray, under: float, over: float, grid: np.ndarray) -> np.ndarray:
    costs = []
    for buffer in grid:
        deployed = pred + buffer
        short = np.maximum(actual - deployed, 0.0)
        extra = np.maximum(deployed - actual, 0.0)
        costs.append(under * short.sum() + over * extra.sum())
    return np.asarray(costs)


def staffing(model_name: str, buffer: float):
    pred = PREDICTIONS[model_name]
    actual = Y_TEST
    under_cost = float(METRICS["cost_under"])
    over_cost = float(METRICS["cost_over"])
    deployed = pred + buffer
    short = np.maximum(actual - deployed, 0.0)
    extra = np.maximum(deployed - actual, 0.0)
    total = under_cost * short.sum() + over_cost * extra.sum()
    hours_short = int((short > 0).sum())
    hours_extra = int((extra > 0).sum())

    grid = np.arange(0, 121)
    curve = _cost_curve(pred, actual, under_cost, over_cost, grid)
    best = int(grid[int(np.argmin(curve))])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=curve, mode="lines", name="test-quarter cost",
        line={"color": "#1f4b99"},
    ))
    fig.add_vline(x=buffer, line_dash="dash", line_color="#c46b1a")
    fig.add_trace(go.Scatter(
        x=[best], y=[curve[best]], mode="markers", name=f"lowest cost at +{best} bikes",
        marker={"size": 10, "color": "#1f7a4d"},
    ))
    fig.update_layout(
        title="Cost of the Q4 2012 test quarter if every forecast is raised by a buffer",
        xaxis_title="Extra bikes added to the forecast",
        yaxis_title="Total cost ($)",
        template="plotly_white",
        height=420,
        legend={"orientation": "h"},
    )
    summary = (
        f"**{MODEL_LABELS[model_name]}**, buffer **+{buffer:.0f}** bikes\n\n"
        f"- Test-quarter cost: **${total:,.0f}**\n"
        f"- Bikes short, summed over hours: **{short.sum():,.0f}** across **{hours_short}** hours\n"
        f"- Extra bikes staged, summed over hours: **{extra.sum():,.0f}** across **{hours_extra}** hours\n"
        f"- Lowest point on this curve: **+{best}** bikes "
        f"(${curve[best]:,.0f}). The training script's newsvendor guess was "
        f"+{METRICS['suggested_buffer_bikes']:.0f}, from the "
        f"{METRICS['critical_fractile']:.0%} quantile of test residuals."
    )
    table = [
        ["Hours short (actual > deployed)", str(hours_short)],
        ["Hours with bikes left over", str(hours_extra)],
        ["Hours exactly matching", str(int(np.isclose(actual, deployed).sum()))],
        ["Cost of stockouts ($5 × bikes short)", f"{under_cost * short.sum():,.0f}"],
        ["Cost of extra bikes ($1 × bikes extra)", f"{over_cost * extra.sum():,.0f}"],
    ]
    return fig, summary, table


def build_demo() -> gr.Blocks:
    agreement = METRICS["agreement"]
    sizes = METRICS["split_sizes"]
    intro = f"""
# One hour ahead: how many bikes should be available?

Capital Bikeshare, system-wide, 2011–2012. The model predicts hourly rentals
so a rebalancing desk can stage bikes before the hour starts. Training used
{sizes['train']:,} hours through 30 June 2012, validation used Jul–Sep 2012,
and every number on this page is the untouched test quarter, Oct–Dec 2012.

The three lines are the same linear regression. Largest test-set gap between
the manual PyTorch forecast and Ridge is
**{agreement['max_abs_test_pred_sklearn_vs_manual']:.5f} bikes**.
"""
    max_offset = max(int((TEST_TS.max() - TEST_TS.min()).days) - 13, 0)

    with gr.Blocks(title="Bikeshare demand") as demo:
        gr.Markdown(intro)
        with gr.Tabs():
            with gr.Tab("Compare models"):
                with gr.Row():
                    model_choice = gr.Radio(
                        choices=[
                            ("scikit-learn Ridge", "sklearn"),
                            ("manual PyTorch loop", "manual"),
                            ("nn.Module + SGD", "standard"),
                        ],
                        value="sklearn",
                        label="Prediction-vs-actual model",
                    )
                    baseline_choice = gr.Checkbox(
                        value=True,
                        label="Show same-hour-last-week baseline on the time series",
                    )
                scatter = gr.Plot()
                metrics = gr.Dataframe(
                    headers=["Model", "Test MAE (bikes)", "Test RMSE", "Test R²"],
                    datatype=["str", "str", "str", "str"],
                    label="Same test quarter, same features",
                )
                day_offset = gr.Slider(
                    minimum=0, maximum=max_offset, step=1, value=0,
                    label="Start day of the 14-day window (0 = 1 Oct 2012)",
                )
                series = gr.Plot()
                gap_plot = gr.Plot()
                loss_plot = gr.Plot()

            with gr.Tab("Distributions"):
                gr.Markdown(
                    "Histograms use the hours that survived the lag warmup. "
                    "The hour profile is why rush-hour flags exist: weekdays "
                    "spike at 8:00 and 17:00, weekends spike in the middle of the day."
                )
                with gr.Row():
                    column = gr.Dropdown(
                        choices=[(label, key) for key, label in DIST_COLUMNS.items()],
                        value="cnt",
                        label="Column",
                    )
                    split_name = gr.Radio(
                        choices=["all", "train", "val", "test"],
                        value="all",
                        label="Split",
                    )
                hist = gr.Plot()
                profile = gr.Plot()

            with gr.Tab("Staffing cost"):
                gr.Markdown(
                    "The fit predicts a mean count. The desk does not stage the mean. "
                    "Missing a rental is priced at **$5**, and an extra bike on a van "
                    "at **$1**. Both numbers are planning assumptions. "
                    "Raising the forecast by the slider is the regression version of "
                    "moving a decision threshold: stockouts fall, idle bikes rise, "
                    "and the total cost moves with them."
                )
                with gr.Row():
                    cost_model = gr.Radio(
                        choices=[
                            ("scikit-learn Ridge", "sklearn"),
                            ("manual PyTorch loop", "manual"),
                            ("nn.Module + SGD", "standard"),
                        ],
                        value="sklearn",
                        label="Forecast",
                    )
                    buffer = gr.Slider(
                        minimum=0, maximum=120, step=1, value=0,
                        label="Extra bikes added on top of the forecast",
                    )
                cost_summary = gr.Markdown()
                cost_plot = gr.Plot()
                cost_table = gr.Dataframe(
                    headers=["Quantity on the test quarter", "Value"],
                    datatype=["str", "str"],
                )

        def _refresh_comparison(model_name, show_baseline, offset):
            return comparison_plots(model_name, show_baseline, offset)

        def _refresh_dist(column_name, which_split):
            return distribution_plots(column_name, which_split)

        def _refresh_cost(model_name, buffer_value):
            return staffing(model_name, buffer_value)

        for control in (model_choice, baseline_choice, day_offset):
            control.change(
                _refresh_comparison,
                inputs=[model_choice, baseline_choice, day_offset],
                outputs=[scatter, series, gap_plot, loss_plot, metrics],
            )
        demo.load(
            _refresh_comparison,
            inputs=[model_choice, baseline_choice, day_offset],
            outputs=[scatter, series, gap_plot, loss_plot, metrics],
        )
        for control in (column, split_name):
            control.change(_refresh_dist, inputs=[column, split_name], outputs=[hist, profile])
        demo.load(_refresh_dist, inputs=[column, split_name], outputs=[hist, profile])
        for control in (cost_model, buffer):
            control.change(_refresh_cost, inputs=[cost_model, buffer], outputs=[cost_plot, cost_summary, cost_table])
        demo.load(_refresh_cost, inputs=[cost_model, buffer], outputs=[cost_plot, cost_summary, cost_table])

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860)
