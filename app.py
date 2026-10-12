"""Assignment 1 - Gradio dashboard.

Loads the artifacts saved by train.py (models, preprocessing, predictions).
Nothing is trained here.
"""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import gradio as gr
from pathlib import Path

from train import TRAIN_END, VAL_END, apply_scaler, build_features, evaluate, to_tensor

ART = Path("artifacts")
MODEL_KEYS = ["sklearn", "manual", "standard"]
LABELS = {
    "baseline": "Baseline (average)",
    "sklearn": "scikit-learn",
    "manual": "Manual PyTorch",
    "standard": "nn.Module + optim",
}
LABEL_TO_KEY = {v: k for k, v in LABELS.items()}

# ---- load saved artifacts (no training) ----
preds = pd.read_csv(ART / "predictions.csv", parse_dates=["dteday"])
prep = joblib.load(ART / "preprocessing.joblib")
sk_model = joblib.load(ART / "sklearn_model.joblib")
manual = torch.load(ART / "manual_model.pt")
std_model = nn.Linear(len(prep["columns"]), 1)
std_model.load_state_dict(torch.load(ART / "standard_model.pt"))
std_model.eval()

# ---- raw data, only for the distribution plots ----
raw = pd.read_csv("data/hour.csv", parse_dates=["dteday"])
raw["split"] = np.where(raw["dteday"] <= TRAIN_END, "train",
                        np.where(raw["dteday"] <= VAL_END, "val", "test"))
DIST_COLUMNS = ["cnt", "temp", "hum", "windspeed", "hr", "season", "weathersit", "workingday"]


# ---------------- tab 1: model comparison ----------------
def compare(split, model_labels):
    part = preds[preds["split"] == split]
    rows = []
    for key in ["baseline"] + MODEL_KEYS:
        rows.append({"model": LABELS[key], **evaluate(part["cnt"], part[f"pred_{key}"])})
    table = pd.DataFrame(rows)

    fig = go.Figure()
    for label in model_labels:
        key = LABEL_TO_KEY[label]
        fig.add_trace(go.Scatter(x=part["cnt"], y=part[f"pred_{key}"], mode="markers",
                                 name=label, opacity=0.4, marker=dict(size=5)))
    top = float(part["cnt"].max())
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect prediction",
                             line=dict(color="black", dash="dash")))
    fig.update_layout(xaxis_title="Actual rentals per hour", yaxis_title="Predicted rentals per hour",
                      title=f"Predicted vs actual ({split} set)", height=480)
    return fig, table


# ---------------- tab 2: distributions ----------------
def distribution(column, split):
    data = raw if split == "all" else raw[raw["split"] == split]
    fig = px.histogram(data, x=column, color="split" if split == "all" else None,
                       nbins=40, barmode="overlay", opacity=0.7,
                       title=f"Distribution of {column} ({split})")
    fig.update_layout(height=420)
    return fig


# ---------------- tab 3: under-prediction and safety buffer ----------------
def buffer_view(model_label, split, buffer):
    key = LABEL_TO_KEY[model_label]
    part = preds[preds["split"] == split].copy()
    part["err"] = part["cnt"] - (part[f"pred_{key}"] + buffer)   # positive = too low
    by_hour = part.groupby(["hr", "workingday"])["err"].mean().reset_index()
    by_hour["day type"] = by_hour["workingday"].map({0: "non-working day", 1: "working day"})
    fig = px.bar(by_hour, x="hr", y="err", color="day type", barmode="group",
                 labels={"hr": "Hour of day", "err": "Average actual minus predicted"},
                 title="Average error by hour (above 0 = predicting too low)")
    fig.update_layout(height=420)

    too_low = float((part["err"] > 0).mean() * 100)
    idle = float(np.maximum(-part["err"], 0).mean())
    mae = float(part["err"].abs().mean())
    text = (f"**With a buffer of {buffer} bikes added to every prediction:**\n\n"
            f"- Hours where the prediction is too low: **{too_low:.1f}%**\n"
            f"- Average spare bikes per hour (idle if the prediction is too high): **{idle:.1f}**\n"
            f"- MAE: **{mae:.1f}** bikes")
    return fig, text


# ---------------- tab 4: try a prediction with the saved models ----------------
SEASONS = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
WEATHER = {"Clear / few clouds": 1, "Mist / cloudy": 2, "Light rain or snow": 3, "Heavy rain or snow": 4}


def try_prediction(year, hour, day_type, season, weather, temp_c, humidity):
    row = pd.DataFrame([{
        "yr": 1 if year == 2012 else 0,
        "hr": int(hour),
        "workingday": 1 if day_type == "Working day" else 0,
        "season": SEASONS[season],
        "weathersit": WEATHER[weather],
        "temp": temp_c / 41,          # the dataset stores temperature divided by 41
        "hum": humidity / 100,        # and humidity divided by 100
    }])
    X = build_features(row).reindex(columns=prep["columns"], fill_value=0.0)
    X = apply_scaler(X, prep["mean"], prep["std"])
    t = to_tensor(X)
    out = {
        "scikit-learn": float(sk_model.predict(X)[0]),
        "Manual PyTorch": float((t @ manual["w"] + manual["b"]).item()),
        "nn.Module + optim": float(std_model(t).item()),
    }
    lines = [f"- {name}: **{max(v, 0):.0f}** bikes" for name, v in out.items()]
    return "**Predicted rentals in that hour (negative predictions shown as 0):**\n\n" + "\n".join(lines)


# ---------------- layout ----------------
with gr.Blocks(title="Bike rental demand") as demo:
    gr.Markdown("# Hourly bike rental demand\nThree linear regressions (scikit-learn, manual PyTorch, "
                "nn.Module + optim) trained on the same chronological split. All models are loaded "
                "from `artifacts/`; nothing is trained here.")

    with gr.Tab("Model comparison"):
        split_in = gr.Radio(["test", "val"], value="test", label="Data split")
        models_in = gr.CheckboxGroup([LABELS[k] for k in MODEL_KEYS], value=[LABELS[k] for k in MODEL_KEYS],
                                     label="Models in the plot")
        plot_out = gr.Plot()
        table_out = gr.Dataframe(label="Metrics (signed_error: positive = predicting too low)")
        for comp in (split_in, models_in):
            comp.change(compare, [split_in, models_in], [plot_out, table_out])
        demo.load(compare, [split_in, models_in], [plot_out, table_out])

    with gr.Tab("Distributions"):
        col_in = gr.Dropdown(DIST_COLUMNS, value="cnt", label="Column")
        dsplit_in = gr.Radio(["all", "train", "val", "test"], value="all", label="Data split")
        dist_out = gr.Plot()
        for comp in (col_in, dsplit_in):
            comp.change(distribution, [col_in, dsplit_in], dist_out)
        demo.load(distribution, [col_in, dsplit_in], dist_out)

    with gr.Tab("Under-prediction and safety buffer"):
        gr.Markdown("Predicting too low costs more than predicting too high. Add a buffer of extra "
                    "bikes to every prediction and see what it does to each kind of error.")
        bmodel_in = gr.Dropdown([LABELS[k] for k in MODEL_KEYS], value=LABELS["sklearn"], label="Model")
        bsplit_in = gr.Radio(["test", "val"], value="test", label="Data split")
        buffer_in = gr.Slider(0, 100, value=0, step=5, label="Safety buffer (bikes added to each prediction)")
        bplot_out = gr.Plot()
        btext_out = gr.Markdown()
        for comp in (bmodel_in, bsplit_in, buffer_in):
            comp.change(buffer_view, [bmodel_in, bsplit_in, buffer_in], [bplot_out, btext_out])
        demo.load(buffer_view, [bmodel_in, bsplit_in, buffer_in], [bplot_out, btext_out])

    with gr.Tab("Try a prediction"):
        year_in = gr.Radio([2011, 2012], value=2012, label="Year")
        hour_in = gr.Slider(0, 23, value=17, step=1, label="Hour of day")
        day_in = gr.Radio(["Working day", "Non-working day"], value="Working day", label="Day type")
        season_in = gr.Dropdown(list(SEASONS), value="Fall", label="Season")
        weather_in = gr.Dropdown(list(WEATHER), value="Clear / few clouds", label="Weather")
        temp_in = gr.Slider(-8, 41, value=20, step=1, label="Temperature (°C)")
        hum_in = gr.Slider(0, 100, value=60, step=1, label="Humidity (%)")
        pred_btn = gr.Button("Predict")
        pred_out = gr.Markdown()
        pred_btn.click(try_prediction,
                       [year_in, hour_in, day_in, season_in, weather_in, temp_in, hum_in], pred_out)

if __name__ == "__main__":
    demo.launch()