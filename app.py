"""
Assignment 1 - Gradio dashboard (run with: uv run python main.py app).

Loads the three models saved by train.py and the reports in artifacts/.
Nothing is trained here: the models are only used to predict.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from features import (ARTIFACTS, FEATURE_SETS, apply_scaler, build_features, load_data,
                      time_split, to_bikes)

if not (ARTIFACTS / "meta.json").exists():
    raise SystemExit("No trained models found. Run `uv run python main.py train` first.")

# ------------------------------------------------------------------ load artifacts
meta = json.loads((ARTIFACTS / "meta.json").read_text())
curves = json.loads((ARTIFACTS / "curves.json").read_text())
results = pd.read_csv(ARTIFACTS / "results.csv")
ablation = pd.read_csv(ARTIFACTS / "ablation.csv")
coefs = pd.read_csv(ARTIFACTS / "coefficients.csv")

sk_model = joblib.load(ARTIFACTS / "model_sklearn.joblib")
manual = torch.load(ARTIFACTS / "model_manual.pt")
std_model = torch.nn.Linear(len(meta["feature_names"]), 1)
std_model.load_state_dict(torch.load(ARTIFACTS / "model_standard.pt"))
std_model.eval()

MODELS = ["scikit-learn", "PyTorch manual loop", "PyTorch nn.Module + optim"]
COLORS = {"scikit-learn": "#2a78d6", "PyTorch manual loop": "#eb6834",
          "PyTorch nn.Module + optim": "#1baf7a", "Actual": "#52514e"}
DASH = {"scikit-learn": "solid", "PyTorch manual loop": "dash", "PyTorch nn.Module + optim": "dot"}
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}


def predict_all(raw_rows: pd.DataFrame) -> dict:
    """Rebuild the training features for raw rows and predict bikes with each saved model."""
    X = apply_scaler(build_features(raw_rows, **FEATURE_SETS[meta["feature_set"]]), meta["scaler"])
    X = X[meta["feature_names"]].to_numpy(dtype=float)
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return {"scikit-learn": to_bikes(sk_model.predict(X)),
                "PyTorch manual loop": to_bikes((Xt @ manual["w"] + manual["b"]).numpy()),
                "PyTorch nn.Module + optim": to_bikes(std_model(Xt).squeeze(1).numpy())}


df = load_data()
df["split"] = time_split(df)
df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
for name, p in predict_all(df).items():
    df[name] = p
test = df[df["split"] == "test"]
TEST_WEEKS = sorted(test["dteday"].dt.to_period("W").astype(str).unique())


def style(fig, title, x, y, height=420):
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=height, template="plotly_white",
                      margin=dict(l=60, r=20, t=60, b=50), legend=dict(orientation="h", y=-0.2))
    return fig


# ------------------------------------------------------------------ tab 1: models
def scatter_plot(model, split):
    d = df[df["split"] == split]
    top = float(max(d["cnt"].max(), d[model].max()))
    fig = go.Figure()
    fig.add_scatter(x=d["cnt"], y=d[model], mode="markers", name=model,
                    marker=dict(size=5, color=COLORS[model], opacity=0.35),
                    customdata=d["timestamp"].dt.strftime("%a %d %b %Y %Hh"),
                    hovertemplate="%{customdata}<br>actual %{x:.0f}<br>predicted %{y:.0f}<extra></extra>")
    fig.add_scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect prediction",
                    line=dict(color=COLORS["Actual"], width=2, dash="dash"), hoverinfo="skip")
    mae = np.abs(d[model] - d["cnt"]).mean()
    return style(fig, f"{model} on the {split} set: predicted vs actual (MAE {mae:.1f} bikes)",
                 "Actual rentals in the hour", "Predicted rentals", 480)


def week_plot(week):
    d = test[test["dteday"].dt.to_period("W").astype(str) == week]
    fig = go.Figure()
    fig.add_scatter(x=d["timestamp"], y=d["cnt"], mode="lines", name="Actual",
                    line=dict(color=COLORS["Actual"], width=2))
    for m in MODELS:
        fig.add_scatter(x=d["timestamp"], y=d[m], mode="lines", name=m,
                        line=dict(color=COLORS[m], width=2, dash=DASH[m]))
    fig.update_layout(hovermode="x unified")
    return style(fig, "One test week, hour by hour (the three model lines overlap when they agree)",
                 "", "Rentals per hour")


def coef_plot():
    fig = go.Figure()
    lim = [float(coefs["scikit-learn"].min()), float(coefs["scikit-learn"].max())]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="Identical", hoverinfo="skip",
                    line=dict(color=COLORS["Actual"], width=1, dash="dash"))
    for m in MODELS[1:]:
        fig.add_scatter(x=coefs["scikit-learn"], y=coefs[m], mode="markers", name=m, text=coefs["feature"],
                        marker=dict(size=9, color=COLORS[m], symbol="circle" if m == MODELS[1] else "x"),
                        hovertemplate="%{text}<br>sklearn %{x:.4f}<br>PyTorch %{y:.4f}<extra></extra>")
    return style(fig, "Do the three implementations learn the same coefficients?",
                 "scikit-learn coefficient", "PyTorch coefficient")


def loss_plot():
    fig = go.Figure()
    palette = ["#86b6ef", "#2a78d6", "#0d366b"]
    for (lr, hist), c in zip(curves["lr_search_manual"].items(), palette):
        fig.add_scatter(x=curves["epoch"], y=hist, mode="lines", name=f"manual loop, lr={lr}",
                        line=dict(color=c, width=2))
    fig.add_scatter(x=curves["epoch"], y=curves["final_standard"], mode="lines",
                    name=f"nn.Module + optim, lr={meta['learning_rate']}",
                    line=dict(color=COLORS["PyTorch nn.Module + optim"], width=2, dash="dot"))
    fig.update_yaxes(type="log")
    return style(fig, "Training loss per epoch (log scale), by learning rate",
                 "Epoch (full-batch gradient step)", "MSE on log(1+rentals) + L2 penalty")


# ------------------------------------------------------------------ tab 2: data
DIST_COLS = {"Target: rentals per hour (cnt)": "cnt", "Target as modelled: log(1 + rentals)": "log_cnt",
             "Temperature (normalised)": "temp", "Humidity": "hum", "Wind speed": "windspeed",
             "Weather situation (1 clear - 4 storm)": "weathersit", "Hour of day": "hr"}
df["log_cnt"] = np.log1p(df["cnt"])


def dist_plot(label):
    col = DIST_COLS[label]
    fig = go.Figure()
    for s in ["train", "val", "test"]:
        fig.add_histogram(x=df.loc[df["split"] == s, col], name=s, histnorm="percent", nbinsx=40,
                          marker=dict(color=SPLIT_COLORS[s], line=dict(color="white", width=1)), opacity=0.7)
    fig.update_layout(barmode="overlay")
    return style(fig, f"{label}: distribution in each split", label, "% of hours in the split")


def profile_plot():
    prof = df.groupby(["workingday", "hr"])["cnt"].mean().reset_index()
    fig = go.Figure()
    for wd, name, c in [(1, "Working day", "#2a78d6"), (0, "Weekend / holiday", "#eb6834")]:
        d = prof[prof["workingday"] == wd]
        fig.add_scatter(x=d["hr"], y=d["cnt"], mode="lines+markers", name=name, line=dict(color=c, width=2))
    return style(fig, "Why hour x working-day: two different daily shapes", "Hour of day", "Average rentals")


def trend_plot():
    daily = df.groupby(["dteday", "split"])["cnt"].sum().reset_index()
    fig = go.Figure()
    for s in ["train", "val", "test"]:
        d = daily[daily["split"] == s]
        fig.add_scatter(x=d["dteday"], y=d["cnt"], mode="lines", name=s, line=dict(color=SPLIT_COLORS[s], width=2))
    return style(fig, "Daily rentals over time, coloured by split: the service is growing",
                 "", "Rentals per day")


def ablation_plot():
    fig = go.Figure()
    for col, c in [("val MAE", "#eb6834"), ("test MAE", "#1baf7a")]:
        fig.add_bar(y=ablation["Feature set"], x=ablation[col], name=col, orientation="h",
                    marker=dict(color=c, line=dict(color="white", width=2)),
                    text=ablation[col].round(0), textposition="outside")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, float(ablation[["val MAE", "test MAE"]].max().max()) * 1.15])
    return style(fig, "What each feature step bought (scikit-learn, lower is better)",
                 "Mean absolute error (bikes per hour)", "")


# ------------------------------------------------------------------ tab 3: what-if
WEATHER = {"Clear": 1, "Mist / cloudy": 2, "Light rain or snow": 3}


def what_if(date, hour, workingday, temp_c, hum, wind, weather):
    row = pd.DataFrame({"dteday": [pd.Timestamp(date)], "hr": [int(hour)],
                        "workingday": [1 if workingday == "Working day" else 0],
                        "temp": [(temp_c + 8) / 47],        # dataset's own normalisation: (t+8)/(39+8)
                        "hum": [hum / 100], "windspeed": [wind / 67], "weathersit": [WEATHER[weather]]})
    return pd.DataFrame({"Model": MODELS,
                         "Predicted rentals in that hour": [round(float(p[0]), 1) for p in predict_all(row).values()]})


# ------------------------------------------------------------------ layout
with gr.Blocks(title="Bike-share demand: one model, three implementations") as demo:
    gr.Markdown("# Bike-share hourly demand forecast\n"
                "Linear regression on log(1 + rentals), trained three ways on the same chronological split. "
                "Models are loaded from `artifacts/`; nothing is retrained here.")
    with gr.Tab("Model comparison"):
        gr.Markdown(f"Ridge strength alpha = {meta['alpha']}, learning rate = {meta['learning_rate']}, "
                    f"{meta['epochs']} full-batch epochs. Bias < 0 means under-forecasting (empty docks).")
        gr.Dataframe(results.round(2), label="Validation and test metrics, in bikes per hour")
        with gr.Row():
            model_dd = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
            split_dd = gr.Dropdown(["test", "val", "train"], value="test", label="Split")
        scatter = gr.Plot(scatter_plot(MODELS[0], "test"))
        model_dd.change(scatter_plot, [model_dd, split_dd], scatter)
        split_dd.change(scatter_plot, [model_dd, split_dd], scatter)
        week_dd = gr.Dropdown(TEST_WEEKS, value=TEST_WEEKS[1], label="Test week")
        week = gr.Plot(week_plot(TEST_WEEKS[1]))
        week_dd.change(week_plot, week_dd, week)
        with gr.Row():
            gr.Plot(coef_plot())
            gr.Plot(loss_plot())
        gr.Dataframe(pd.DataFrame({"Check": list(meta["agreement"]), "Value": [round(v, 5) for v in meta["agreement"].values()]}),
                     label="Agreement between implementations")
    with gr.Tab("Data & features"):
        dist_dd = gr.Dropdown(list(DIST_COLS), value=list(DIST_COLS)[0], label="Variable")
        dist = gr.Plot(dist_plot(list(DIST_COLS)[0]))
        dist_dd.change(dist_plot, dist_dd, dist)
        gr.Plot(trend_plot())
        with gr.Row():
            gr.Plot(profile_plot())
            gr.Plot(ablation_plot())
        gr.Dataframe(ablation.round(1), label="Feature ablation table")
    with gr.Tab("What-if forecast"):
        gr.Markdown("Describe tomorrow's hour and see what each saved model predicts.")
        with gr.Row():
            date_in = gr.Textbox("2012-12-15", label="Date (YYYY-MM-DD, sets the growth trend)")
            hour_in = gr.Slider(0, 23, 8, step=1, label="Hour")
            wd_in = gr.Radio(["Working day", "Weekend / holiday"], value="Working day", label="Day type")
        with gr.Row():
            temp_in = gr.Slider(-8, 39, 15, step=1, label="Temperature (C)")
            hum_in = gr.Slider(0, 100, 60, step=1, label="Humidity (%)")
            wind_in = gr.Slider(0, 57, 12, step=1, label="Wind speed (km/h)")
            weather_in = gr.Radio(list(WEATHER), value="Clear", label="Weather")
        inputs = [date_in, hour_in, wd_in, temp_in, hum_in, wind_in, weather_in]
        out = gr.Dataframe(what_if("2012-12-15", 8, "Working day", 15, 60, 12, "Clear"))
        for comp in inputs:
            comp.change(what_if, inputs, out)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
