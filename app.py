"""
Assignment 1 - Gradio dashboard.  Run with:  uv run python main.py app

Everything shown comes from artifacts/ written by train.py. The app only LOADS
the saved models and runs inference with them; it never trains anything.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from pipeline import (
    ARTIFACTS_DIR, TARGET, LinearRegressionTorch, build_features, load_data,
    split_masks, to_bikes,
)

# ---------------------------------------------------------------------------
# 1. Load artifacts (fail with a clear message if training has not run)
# ---------------------------------------------------------------------------
REQUIRED = ["preprocessing.joblib", "sklearn_model.joblib", "manual_torch.pt",
            "standard_torch.pt", "test_predictions.csv", "test_metrics.csv",
            "val_tuning.csv", "coefficients.csv", "loss_curves.csv", "summary.json"]
missing = [name for name in REQUIRED if not (ARTIFACTS_DIR / name).exists()]
if missing:
    raise SystemExit(f"Missing artifacts {missing}. Run `uv run python main.py train` first.")

pre = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
sk_model = joblib.load(ARTIFACTS_DIR / "sklearn_model.joblib")
manual_params = torch.load(ARTIFACTS_DIR / "manual_torch.pt")          # {"w", "b"}
std_model = LinearRegressionTorch(len(pre["feature_names"]))
std_model.load_state_dict(torch.load(ARTIFACTS_DIR / "standard_torch.pt"))
std_model.eval()

summary = json.loads((ARTIFACTS_DIR / "summary.json").read_text())
metrics = pd.read_csv(ARTIFACTS_DIR / "test_metrics.csv")
tuning = pd.read_csv(ARTIFACTS_DIR / "val_tuning.csv")
coefs = pd.read_csv(ARTIFACTS_DIR / "coefficients.csv")
losses = pd.read_csv(ARTIFACTS_DIR / "loss_curves.csv")
saved = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv", parse_dates=["timestamp"])

MODELS = {"sklearn": "scikit-learn", "manual_torch": "Manual PyTorch",
          "standard_torch": "Standard PyTorch"}
BASELINES = {"baseline_mean": "Baseline: training mean",
             "baseline_hour_profile": "Baseline: hour x day-type profile"}
# Okabe-Ito colour-blind-safe palette, plus distinct dash styles so the
# lines stay distinguishable without relying on colour alone.
STYLE = {
    "actual": dict(color="#222222", dash="solid", width=2),
    "sklearn": dict(color="#0072B2", dash="solid", width=2),
    "manual_torch": dict(color="#E69F00", dash="dash", width=2),
    "standard_torch": dict(color="#009E73", dash="dot", width=3),
    "baseline_mean": dict(color="#999999", dash="dashdot", width=1.5),
    "baseline_hour_profile": dict(color="#CC79A7", dash="dashdot", width=1.5),
}
LAYOUT = dict(template="plotly_white", height=430, margin=dict(l=60, r=20, t=50, b=50),
              legend=dict(orientation="h", y=-0.18))


def predict_all(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    """Run the three LOADED models on raw rows (inference only). Output in bikes/hour."""
    X = pre["scaler"].transform(build_features(raw, pre["prep"]))
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return {
            "sklearn": to_bikes(sk_model.predict(X)),
            "manual_torch": to_bikes((X_t @ manual_params["w"] + manual_params["b"]).numpy()),
            "standard_torch": to_bikes(std_model(X_t).numpy()),
        }


# ---------------------------------------------------------------------------
# 2. Data for the views
# ---------------------------------------------------------------------------
data = load_data()
masks = split_masks(data)
data["split"] = np.select([masks["train"], masks["val"], masks["test"]],
                          ["train", "val", "test"], default="other")

test = data[masks["test"]].reset_index(drop=True)
for name, pred in predict_all(test).items():
    test[name] = pred
assert (test["timestamp"].values == saved["timestamp"].values).all()
for name in BASELINES:                       # baselines are look-ups, not models
    test[name] = saved[name].to_numpy()
# Sanity check: live inference must reproduce what train.py saved.
replay_gap = max(float(np.abs(test[n] - saved[n]).max()) for n in MODELS)
test_days = sorted(test["timestamp"].dt.normalize().unique())


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
def fig_scatter(choice: str) -> go.Figure:
    keys = list(MODELS) if choice == "All three" else [k for k, v in MODELS.items() if v == choice]
    fig = go.Figure()
    for key in keys:
        fig.add_trace(go.Scattergl(
            x=test[TARGET], y=test[key], mode="markers", name=MODELS[key],
            marker=dict(color=STYLE[key]["color"], size=5, opacity=0.45),
            customdata=test["timestamp"].dt.strftime("%a %d %b %Y, %H:00"),
            hovertemplate="%{customdata}<br>actual %{x}<br>predicted %{y:.0f}<extra></extra>"))
    top = float(max(test[TARGET].max(), test[keys].max().max()))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect prediction",
                             line=dict(color="#222222", dash="dash", width=1)))
    fig.update_layout(title="Predicted vs actual, test set (Oct-Dec 2012)",
                      xaxis_title="Actual rentals per hour",
                      yaxis_title="Predicted rentals per hour", **LAYOUT)
    return fig


def fig_timeline(start_day: int, n_days: int, show_baselines: bool) -> go.Figure:
    start = test_days[int(start_day)]
    window = test[(test["timestamp"] >= start)
                  & (test["timestamp"] < start + pd.Timedelta(days=int(n_days)))]
    series = {"actual": ("Actual", window[TARGET])}
    series.update({k: (v, window[k]) for k, v in MODELS.items()})
    if show_baselines:
        series.update({k: (v, window[k]) for k, v in BASELINES.items()})
    fig = go.Figure()
    for key, (label, y) in series.items():
        fig.add_trace(go.Scatter(x=window["timestamp"], y=y, mode="lines", name=label,
                                 line=STYLE[key]))
    fig.update_layout(title=f"Hour by hour from {start:%a %d %b %Y}",
                      yaxis_title="Rentals per hour", **LAYOUT)
    return fig


DISTRIBUTIONS = {   # label -> (function of the raw rows, is_discrete)
    "Rentals per hour (target)": (lambda d: d[TARGET], False),
    "log(1 + rentals), the scale the model is trained on": (lambda d: np.log1p(d[TARGET]), False),
    "Temperature (C)": (lambda d: d["temp"] * 41, False),
    "Humidity (%), raw: note the sensor-fault zeros": (lambda d: d["hum"] * 100, False),
    "Wind speed (km/h)": (lambda d: d["windspeed"] * 67, False),
    "Hour of day": (lambda d: d["hr"], True),
    "Weather situation (1 clear .. 4 extreme)": (lambda d: d["weathersit"], True),
}
SPLIT_STYLE = {"train": STYLE["sklearn"], "val": STYLE["manual_torch"], "test": STYLE["standard_torch"]}


def fig_distribution(variable: str) -> go.Figure:
    """Step histograms on SHARED bin edges, normalised per split so sizes don't matter."""
    func, discrete = DISTRIBUTIONS[variable]
    values = func(data)
    if discrete:
        edges = np.arange(values.min() - 0.5, values.max() + 1.5)
    else:
        edges = np.histogram_bin_edges(values, bins=40)
    fig = go.Figure()
    for split, style in SPLIT_STYLE.items():
        counts, _ = np.histogram(func(data[data["split"] == split]), bins=edges)
        share = np.append(counts / counts.sum(), 0)
        fig.add_trace(go.Scatter(x=edges, y=share, mode="lines", line_shape="hv",
                                 name=f"{split} ({counts.sum():,} rows)", line=style))
    fig.update_layout(title=f"{variable}: share of rows per split",
                      yaxis_title="Share of rows in the split", **LAYOUT)
    return fig


def fig_hour_profile() -> go.Figure:
    fit = data[data["split"] != "test"]
    profile = fit.groupby(["hr", "workingday"])[TARGET].mean().unstack()
    fig = go.Figure()
    for flag, label, style in [(1, "Working day", "sklearn"), (0, "Weekend or holiday", "manual_torch")]:
        fig.add_trace(go.Scatter(x=profile.index, y=profile[flag], mode="lines+markers",
                                 name=label, line=STYLE[style]))
    fig.update_layout(title="Why hour x day-type indicators: mean rentals by hour (train + val)",
                      xaxis_title="Hour of day", yaxis_title="Mean rentals per hour", **LAYOUT)
    return fig


def fig_coefficients() -> go.Figure:
    fig = go.Figure()
    for key in ["manual_torch", "standard_torch"]:
        fig.add_trace(go.Scatter(
            x=coefs["sklearn"], y=coefs[key], mode="markers", name=MODELS[key],
            marker=dict(color=STYLE[key]["color"], size=9 if key == "manual_torch" else 5),
            text=coefs["feature"], hovertemplate="%{text}<br>sklearn %{x:.5f}<br>torch %{y:.5f}<extra></extra>"))
    lo, hi = coefs[["sklearn"]].min().iloc[0], coefs[["sklearn"]].max().iloc[0]
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Identical",
                             line=dict(color="#222222", dash="dash", width=1)))
    fig.update_layout(title="Each of the 56 coefficients: PyTorch vs scikit-learn",
                      xaxis_title="scikit-learn coefficient (standardised features)",
                      yaxis_title="PyTorch coefficient", **LAYOUT)
    return fig


def fig_convergence() -> go.Figure:
    fig = go.Figure()
    for key in ["manual_torch", "standard_torch"]:
        gap = (losses[key] - losses["sklearn_optimum"]).abs().clip(lower=1e-10)
        fig.add_trace(go.Scatter(x=losses["epoch"], y=gap, mode="lines", name=MODELS[key],
                                 line=STYLE[key]))
    fig.update_layout(title="Distance to scikit-learn's exact optimum during training",
                      xaxis_title="Epoch (full-batch gradient step)",
                      yaxis_title="|training MSE - optimal MSE|, log scale",
                      yaxis_type="log", yaxis_exponentformat="power", **LAYOUT)
    fig.add_annotation(x=losses["epoch"].iloc[-1], y=-6.6, xanchor="right", yanchor="bottom", showarrow=False,
                       text="float32 precision floor: no further progress possible",
                       font=dict(color="#555555", size=12))
    return fig


DAY_TYPES = {"Working day": (1, 0), "Weekend": (0, 0), "Public holiday (weekday)": (0, 1)}
WEATHER = {"Clear or partly cloudy": 1, "Mist or cloudy": 2, "Light rain or snow": 3}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def what_if(day_type, month, temp_c, hum_pct, wind_kmh, weather, hour):
    workingday, holiday = DAY_TYPES[day_type]
    rows = pd.DataFrame({
        "yr": 1, "mnth": MONTHS.index(month) + 1, "hr": range(24),
        "workingday": workingday, "holiday": holiday, "weathersit": WEATHER[weather],
        "temp": temp_c / 41, "hum": hum_pct / 100, "windspeed": wind_kmh / 67,
    })
    preds = predict_all(rows)
    fig = go.Figure()
    for key, label in MODELS.items():
        fig.add_trace(go.Scatter(x=rows["hr"], y=preds[key], mode="lines", name=label,
                                 line=STYLE[key]))
    fig.add_vline(x=hour, line=dict(color="#999999", width=1))
    fig.update_layout(title="Predicted rentals across the day for these conditions",
                      xaxis_title="Hour of day", yaxis_title="Predicted rentals per hour", **LAYOUT)
    at_hour = " | ".join(f"{label}: **{preds[key][int(hour)]:.0f}**" for key, label in MODELS.items())
    return fig, f"Predicted rentals at {int(hour):02d}:00: {at_hour}"


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model"].map({**MODELS, **BASELINES}).fillna(out["model"])
    return out.round(3)


agreement = pd.DataFrame(summary["agreement"]).T.reset_index(names="implementation")
agreement["implementation"] = agreement["implementation"].map(MODELS)
for col in agreement.columns[1:]:
    agreement[col] = agreement[col].map(lambda v: f"{v:.1e}")

# ---------------------------------------------------------------------------
# 4. Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Bike demand forecast dashboard") as demo:
    gr.Markdown(
        "# Hourly bike demand forecast\n"
        "Capital Bikeshare, Washington D.C. One linear regression on log(1 + rentals), "
        "trained three ways. Trained on Jan 2011 to Sep 2012; tested on Oct to Dec 2012. "
        "Every number below comes from the saved artifacts; nothing is retrained here.")

    with gr.Tab("Results"):
        chosen = summary["chosen"]
        gr.Markdown(
            f"Hyper-parameters picked by {chosen['selected_by']} (Jul to Sep 2012): "
            f"regularisation alpha = {chosen['alpha']}, learning rate = {chosen['learning_rate']}, "
            f"{chosen['epochs']} epochs. Live inference with the loaded models reproduces the "
            f"saved test predictions to within {replay_gap:.1e} bikes.")
        gr.Markdown("### Test set, evaluated once (MAE and RMSE in rentals per hour)")
        gr.Dataframe(fmt(metrics), interactive=False)
        gr.Markdown("### Validation tuning")
        gr.Dataframe(tuning.round(3), interactive=False)

    with gr.Tab("Predicted vs actual"):
        model_pick = gr.Radio(["All three", *MODELS.values()], value="All three",
                              label="Model")
        gr.Markdown("The three models overlap almost perfectly, so with *All three* only the "
                    "last one drawn is visible. Pick one at a time to compare.")
        scatter = gr.Plot(fig_scatter("All three"))
        model_pick.change(fig_scatter, model_pick, scatter)
        with gr.Row():
            start_day = gr.Slider(0, len(test_days) - 1, value=0, step=1,
                                  label="Start day (days after 1 Oct 2012)")
            n_days = gr.Slider(1, 21, value=7, step=1, label="Days to show")
            show_base = gr.Checkbox(value=False, label="Show baselines")
        timeline = gr.Plot(fig_timeline(0, 7, False))
        for control in (start_day, n_days, show_base):
            control.change(fig_timeline, [start_day, n_days, show_base], timeline)

    with gr.Tab("Distributions"):
        variable = gr.Dropdown(list(DISTRIBUTIONS), value=list(DISTRIBUTIONS)[0],
                               label="Variable")
        dist_plot = gr.Plot(fig_distribution(list(DISTRIBUTIONS)[0]))
        variable.change(fig_distribution, variable, dist_plot)
        gr.Plot(fig_hour_profile())

    with gr.Tab("Three implementations"):
        gr.Markdown("Same features, same split, same loss. Do they land on the same weights?")
        gr.Dataframe(agreement, interactive=False)
        gr.Plot(fig_coefficients())
        gr.Plot(fig_convergence())

    with gr.Tab("What-if"):
        gr.Markdown("Set the conditions for a day in 2012 and see what each saved model predicts.")
        with gr.Row():
            with gr.Column(scale=1):
                day_type = gr.Radio(list(DAY_TYPES), value="Working day", label="Day type")
                month = gr.Dropdown(MONTHS, value="Jun", label="Month")
                weather = gr.Radio(list(WEATHER), value="Clear or partly cloudy", label="Weather")
                temp_c = gr.Slider(0, 41, value=22, step=1, label="Temperature (C)")
                hum = gr.Slider(10, 100, value=60, step=1, label="Humidity (%)")
                wind = gr.Slider(0, 57, value=12, step=1, label="Wind speed (km/h)")
                hour = gr.Slider(0, 23, value=8, step=1, label="Hour")
            with gr.Column(scale=2):
                inputs = [day_type, month, temp_c, hum, wind, weather, hour]
                first_fig, first_text = what_if(*[c.value for c in inputs])
                wi_plot = gr.Plot(first_fig)
                wi_text = gr.Markdown(first_text)
        for control in inputs:
            control.change(what_if, inputs, [wi_plot, wi_text])

if __name__ == "__main__":
    demo.launch()   # http://127.0.0.1:7860, the address the CI checks
