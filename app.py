"""
Gradio dashboard: Seoul bike rentals.

Loads the models and data saved by train.py. Nothing is trained here.
Run with:  uv run python main.py app
"""

from pathlib import Path

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")  # draw plots as images, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ARTIFACTS = Path("artifacts")

# ------------------------------------------------------------------ Load saved artifacts
saved = joblib.load(ARTIFACTS / "data.joblib")
info = saved["info"]
X = saved["X"]                      # already scaled in train.py
y = info["rented_bike_count"].to_numpy(dtype=float)

sk_model = joblib.load(ARTIFACTS / "sklearn_model.joblib")
manual = torch.load(ARTIFACTS / "torch_manual.pt")             # {"w": ..., "b": ...}
standard = nn.Linear(X.shape[1], 1)
standard.load_state_dict(torch.load(ARTIFACTS / "torch_standard.pt"))

# Predictions from the loaded models (predicting only, no training).
X_t = torch.tensor(X)
with torch.no_grad():
    log_preds = {
        "scikit-learn": sk_model.predict(X),
        "PyTorch manual": (X_t @ manual["w"] + manual["b"]).numpy(),
        "PyTorch standard": standard(X_t).squeeze(-1).numpy(),
    }
preds = {name: np.expm1(p) for name, p in log_preds.items()}  # log scale -> bikes
preds["Naive baseline (train mean)"] = np.full(len(y), saved["baseline_mean"])

rush = ((~info["day_off"]) & info["hour"].isin([7, 8, 9, 17, 18, 19])).to_numpy()
SPLITS = {"Validation (Sep 2018)": "val", "Test (Oct–Nov 2018)": "test"}
MODELS = ["scikit-learn", "PyTorch manual", "PyTorch standard"]


# ------------------------------------------------------------------ Tab 1: model comparison
def compare_models(split_label):
    mask = (info["split"] == SPLITS[split_label]).to_numpy()

    rows = []
    for name, p in preds.items():
        rows.append({
            "Model": name,
            "RMSE": np.sqrt(np.mean((y[mask] - p[mask]) ** 2)),
            "Rush-hour RMSE": np.sqrt(np.mean((y[mask & rush] - p[mask & rush]) ** 2)),
            "MAE": np.mean(np.abs(y[mask] - p[mask])),
        })
    table = pd.DataFrame(rows).round(1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    top = max(y[mask].max(), max(preds[m][mask].max() for m in MODELS))
    for ax, name in zip(axes, MODELS):
        ax.scatter(y[mask], preds[name][mask], s=6, alpha=0.4)
        ax.plot([0, top], [0, top], color="red", linestyle="--", label="perfect prediction")
        ax.set_title(name)
        ax.set_xlabel("actual rentals per hour")
    axes[0].set_ylabel("predicted rentals per hour")
    axes[0].legend()
    fig.suptitle(f"Predicted vs actual: {split_label}")
    fig.tight_layout()

    diff = max(np.abs(preds[m][mask] - preds["scikit-learn"][mask]).max() for m in MODELS[1:])
    note = f"Largest prediction difference between the PyTorch versions and scikit-learn: **{diff:.2f} bikes**."
    return table, fig, note


# ------------------------------------------------------------------ Tab 2: distributions
DIST_COLUMNS = ["rented_bike_count", "hour", "temperature", "humidity", "wind_speed",
                "visibility", "solar_radiation", "rainfall", "snowfall"]


def plot_distribution(column):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for split, label in [("train", "Train (Dec–Aug)"), ("val", "Validation (Sep)"), ("test", "Test (Oct–Nov)")]:
        values = info.loc[info["split"] == split, column]
        ax.hist(values, bins=40, alpha=0.5, density=True, label=label)
    ax.set_xlabel(column)
    ax.set_ylabel("share of hours")
    ax.set_title(f"Distribution of {column} by split")
    ax.legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Tab 3: safety buffer
def buffer_stats(split_mask, buffer_pct):
    planned = preds["scikit-learn"][split_mask] * (1 + buffer_pct / 100)
    actual = y[split_mask]
    short = actual > planned
    return {
        "hours_short": short.mean() * 100,
        "rush_hours_short": short[rush[split_mask]].mean() * 100,
        "bikes_short_per_hour": np.clip(actual - planned, 0, None).mean(),
        "idle_bikes_per_hour": np.clip(planned - actual, 0, None).mean(),
    }


def safety_buffer(split_label, buffer_pct):
    mask = (info["split"] == SPLITS[split_label]).to_numpy()
    s = buffer_stats(mask, buffer_pct)
    summary = (
        f"Planning for **forecast + {buffer_pct:.0f}%**:\n\n"
        f"- Hours with too few bikes: **{s['hours_short']:.1f}%** "
        f"(weekday rush hours: **{s['rush_hours_short']:.1f}%**)\n"
        f"- Average bikes missing per hour: **{s['bikes_short_per_hour']:.0f}**\n"
        f"- Average idle bikes per hour: **{s['idle_bikes_per_hour']:.0f}**"
    )

    buffers = np.arange(0, 101, 5)
    curve = [buffer_stats(mask, b) for b in buffers]
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(buffers, [c["hours_short"] for c in curve], marker="o", color="tab:red",
             label="% hours with too few bikes")
    ax1.set_xlabel("safety buffer (%)")
    ax1.set_ylabel("% hours with too few bikes", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(buffers, [c["idle_bikes_per_hour"] for c in curve], marker="o", color="tab:blue",
             label="idle bikes per hour")
    ax2.set_ylabel("average idle bikes per hour", color="tab:blue")
    ax1.axvline(buffer_pct, color="grey", linestyle="--")
    ax1.set_title("Trade-off: fewer shortages vs more idle bikes")
    fig.tight_layout()
    return summary, fig


# ------------------------------------------------------------------ Layout
with gr.Blocks(title="Seoul bike rentals") as demo:
    gr.Markdown(
        "# Seoul bike rentals: next-day hourly forecast\n"
        "Linear regression trained three ways (scikit-learn, manual PyTorch, standard PyTorch). "
        "All models are loaded from `artifacts/`, nothing is retrained here."
    )

    with gr.Tab("Model comparison"):
        split_cmp = gr.Dropdown(list(SPLITS), value="Test (Oct–Nov 2018)", label="Period")
        metrics_table = gr.Dataframe(label="Metrics (in bikes)")
        agreement_note = gr.Markdown()
        scatter_plot = gr.Plot()
        split_cmp.change(compare_models, split_cmp, [metrics_table, scatter_plot, agreement_note])

    with gr.Tab("Distributions"):
        column = gr.Dropdown(DIST_COLUMNS, value="rented_bike_count", label="Column")
        dist_plot = gr.Plot()
        column.change(plot_distribution, column, dist_plot)

    with gr.Tab("Safety buffer"):
        gr.Markdown(
            "Under-predicting is the costly mistake (customers find no bike), so the operator "
            "can plan for the forecast plus a safety buffer. Uses the scikit-learn predictions; "
            "the three models are practically identical."
        )
        split_buf = gr.Dropdown(list(SPLITS), value="Test (Oct–Nov 2018)", label="Period")
        buffer = gr.Slider(0, 100, value=0, step=5, label="Safety buffer (%)")
        buffer_text = gr.Markdown()
        buffer_plot = gr.Plot()
        for control in [split_buf, buffer]:
            control.change(safety_buffer, [split_buf, buffer], [buffer_text, buffer_plot])

    # Fill every tab when the page first opens.
    demo.load(compare_models, split_cmp, [metrics_table, scatter_plot, agreement_note])
    demo.load(plot_distribution, column, dist_plot)
    demo.load(safety_buffer, [split_buf, buffer], [buffer_text, buffer_plot])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
