"""
Gradio dashboard for the NBA opening line model.

It only loads what train.py saved in artifacts/. Nothing is trained here.
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

ART = Path(__file__).resolve().parent / "artifacts"

if not (ART / "config.json").exists():
    raise SystemExit("No saved models found. Run `uv run python main.py train` first.")

# ---------------------------------------------------------------
# Load saved artifacts
# ---------------------------------------------------------------
config = json.loads((ART / "config.json").read_text())
FEATURES = config["features"]
TARGET = config["target"]

scaler = joblib.load(ART / "scaler.joblib")
sk_model = joblib.load(ART / "sklearn_model.joblib")

manual = torch.load(ART / "manual_torch.pt")
w_man, b_man = manual["w"], manual["b"]

std_model = torch.nn.Linear(len(FEATURES), 1)
std_model.load_state_dict(torch.load(ART / "standard_torch.pt"))
std_model.eval()

test = pd.read_csv(ART / "test_games.csv")
train = pd.read_csv(ART / "train_games.csv")
results = pd.read_csv(ART / "test_results.csv", index_col=0)
coefs = pd.read_csv(ART / "coefficients.csv", index_col=0)
home_by_season = pd.read_csv(ART / "home_edge_by_season.csv")

# ---------------------------------------------------------------
# Predictions from the saved models (inference only)
# ---------------------------------------------------------------
X_test = scaler.transform(test[FEATURES])
with torch.no_grad():
    X_t = torch.tensor(X_test, dtype=torch.float32)
    pred_manual = (X_t @ w_man + b_man).numpy()
    pred_standard = std_model(X_t).squeeze(-1).numpy()

PREDS = {
    "Naive baseline": np.full(len(test), config["baseline_value"]),
    "scikit-learn": sk_model.predict(X_test),
    "Manual PyTorch": pred_manual,
    "Standard PyTorch": pred_standard,
}
MODELS = ["scikit-learn", "Manual PyTorch", "Standard PyTorch"]
y_test = test[TARGET].values
test["label"] = test["GAME_DATE_EST"] + "  " + test["away"] + " @ " + test["home"]


# ---------------------------------------------------------------
# Tab 1. Predicted vs actual
# ---------------------------------------------------------------
def pred_vs_actual(model_names):
    fig = go.Figure()
    for name in model_names:
        fig.add_trace(go.Scatter(
            x=PREDS[name], y=y_test, mode="markers", name=name, opacity=0.45,
            text=test["label"], hovertemplate="%{text}<br>predicted %{x:.1f}<br>actual %{y}<extra></extra>",
        ))
    lim = [-50, 80]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect line", line=dict(dash="dash", color="grey")))
    fig.update_layout(
        xaxis_title="predicted home margin (points)", yaxis_title="actual home margin (points)",
        title=f"Test season {config['test_season']}-{config['test_season'] - 1999}, {len(test)} games",
        height=520,
    )
    return fig


def loss_curves():
    fig = go.Figure()
    for name, hist in config["loss_history"].items():
        fig.add_trace(go.Scatter(y=hist, mode="lines", name=name))
    fig.update_layout(xaxis_title="epoch", yaxis_title="training loss (MSE + penalty)",
                      yaxis_type="log", height=350, title="PyTorch training loss")
    return fig


# ---------------------------------------------------------------
# Tab 2. Distributions
# ---------------------------------------------------------------
def distribution(column):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train[column], name=f"train ({config['train_label']})", opacity=0.6, histnorm="percent"))
    fig.add_trace(go.Histogram(x=test[column], name=f"test ({config['test_season']}-{config['test_season'] - 1999})", opacity=0.6, histnorm="percent"))
    fig.update_layout(barmode="overlay", xaxis_title=column, yaxis_title="% of games", height=450)
    return fig


# ---------------------------------------------------------------
# Tab 3. How often the line is badly off
# ---------------------------------------------------------------
def line_misses(points):
    rows = []
    for name, p in PREDS.items():
        miss = np.abs(y_test - p) > points
        rows.append({"model": name, f"games missed by more than {points} pts": int(miss.sum()),
                     "% of games": round(100 * miss.mean(), 1)})
    table = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(x=table["model"], y=table["% of games"]))
    fig.update_layout(yaxis_title="% of test games", height=380,
                      title=f"Share of games where the opening line was off by more than {points} points")
    return table, fig


# ---------------------------------------------------------------
# Tab 4. Look up one game
# ---------------------------------------------------------------
def one_game(label):
    row = test[test["label"] == label].iloc[0]
    i = row.name
    line = pd.DataFrame({
        "model": list(PREDS),
        "predicted home margin": [round(float(PREDS[m][i]), 1) for m in PREDS],
    })
    result = f"Final: {row['away']} {int(row['PTS_away'])}, {row['home']} {int(row['PTS_home'])} (home margin {int(row[TARGET]):+d})"
    feats = row[FEATURES].to_frame("value").reset_index().rename(columns={"index": "feature"})
    feats["value"] = feats["value"].astype(float).round(2)
    return line, result, feats


# ---------------------------------------------------------------
# Tab 5. Home court by season
# ---------------------------------------------------------------
def home_court_chart():
    s = home_by_season
    colors = ["#d62728" if season in (2019, 2020) else "#1f77b4" for season in s["SEASON"]]
    labels = [f"{y}-{(y + 1) % 100:02d}" for y in s["SEASON"]]
    fig = go.Figure(go.Bar(x=labels, y=s["avg_home_margin"], marker_color=colors,
                           text=s["home_win_pct"].astype(str) + "% home wins", textposition="outside"))
    fig.update_layout(yaxis_title="average home margin (points)", height=450,
                      title="Home court advantage by season (red = COVID seasons, not used)")
    return fig


# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
with gr.Blocks(title="NBA opening line") as demo:
    gr.Markdown(
        "# NBA opening line model\n"
        "Linear regression that predicts the home team's final margin before tip off, "
        f"trained three ways. Trained on {config['train_label']}, tested on "
        f"{config['test_season']}-{config['test_season'] - 1999}."
    )

    with gr.Tab("Compare models"):
        picked = gr.CheckboxGroup(MODELS + ["Naive baseline"], value=MODELS, label="Models to plot")
        scatter = gr.Plot(value=pred_vs_actual(MODELS))
        picked.change(pred_vs_actual, picked, scatter)
        gr.Markdown("### Test results")
        gr.Dataframe(results.reset_index().rename(columns={"index": "model"}))
        gap = config["max_prediction_gap"]
        gr.Markdown("Largest difference between two models on any single test game: "
                    + ", ".join(f"{k} {v:.3f} pts" for k, v in gap.items()))
        gr.Markdown("### Coefficients (points per 1 standard deviation of the feature)")
        gr.Dataframe(coefs.reset_index().rename(columns={"index": "feature"}))
        gr.Plot(value=loss_curves())

    with gr.Tab("Distributions"):
        col = gr.Dropdown([TARGET] + FEATURES, value=TARGET, label="Column")
        hist = gr.Plot(value=distribution(TARGET))
        col.change(distribution, col, hist)

    with gr.Tab("Line misses"):
        gr.Markdown("A line that is far off invites sharp bettors. Move the slider to choose what counts as badly off.")
        pts = gr.Slider(1, 20, value=10, step=1, label="Off by more than (points)")
        first_table, first_fig = line_misses(10)
        miss_table = gr.Dataframe(value=first_table)
        miss_plot = gr.Plot(value=first_fig)
        pts.change(line_misses, pts, [miss_table, miss_plot])

    with gr.Tab("Look up a game"):
        game = gr.Dropdown(test["label"].tolist(), value=test["label"].iloc[0], label="Test game")
        first = one_game(test["label"].iloc[0])
        game_line = gr.Dataframe(value=first[0], label="Opening line by model")
        game_result = gr.Markdown(first[1])
        game_feats = gr.Dataframe(value=first[2], label="Features the model saw (before the game)")
        game.change(one_game, game, [game_line, game_result, game_feats])

    with gr.Tab("Home court by season"):
        gr.Plot(value=home_court_chart())

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
