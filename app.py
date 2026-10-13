"""
Assignment 1 — Gradio dashboard.

Loads the three models saved by train.py and shows them. Nothing is trained
here: this file only reads files from models/ and makes predictions.

Run with:  uv run python main.py app
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
import torch.nn as nn

from features import ALL_FEATURES, SKILLS, add_features, clean_data

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

if not (MODELS_DIR / "results.json").exists():
    raise SystemExit("No saved models found. Run `uv run python main.py train` first.")


# ---------------------------------------------------------------------------
# 1. Load what train.py saved
# ---------------------------------------------------------------------------
with open(MODELS_DIR / "results.json") as f:
    results = json.load(f)

scaler = joblib.load(MODELS_DIR / "scaler.joblib")
sklearn_model = joblib.load(MODELS_DIR / "sklearn_model.joblib")

manual_params = torch.load(MODELS_DIR / "manual_model.pt")
manual_w = manual_params["w"]
manual_b = manual_params["b"]

torch_model = nn.Linear(len(ALL_FEATURES), 1)
torch_model.load_state_dict(torch.load(MODELS_DIR / "torch_model.pt"))
torch_model.eval()

test_df = pd.read_csv(MODELS_DIR / "test_data.csv")

# Full cleaned dataset, only used to draw the distributions.
full_df = add_features(clean_data(pd.read_csv(ROOT / "data" / "fifa20_players.csv")))

MODEL_NAMES = ["scikit-learn", "PyTorch manual loop", "PyTorch nn.Module + optim"]


# ---------------------------------------------------------------------------
# 2. Prediction helper (used by every tab)
# ---------------------------------------------------------------------------
def predict_euros(df):
    """Return the predicted value in euros from each of the three models."""
    X = scaler.transform(df[ALL_FEATURES])
    X_t = torch.tensor(X, dtype=torch.float32)

    sklearn_log = sklearn_model.predict(X)
    with torch.no_grad():
        manual_log = (X_t @ manual_w + manual_b).numpy()
        torch_log = torch_model(X_t).squeeze(-1).numpy()

    # The models predict log(value), so exp() takes us back to euros.
    return {
        "scikit-learn": np.exp(sklearn_log),
        "PyTorch manual loop": np.exp(manual_log),
        "PyTorch nn.Module + optim": np.exp(torch_log),
    }


def format_euros(value):
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.2f}M"
    return f"€{value / 1_000:.0f}K"


# Predictions for the test players, computed once when the app starts.
test_predictions = predict_euros(test_df)


# ---------------------------------------------------------------------------
# 3. Tab "Compare models"
# ---------------------------------------------------------------------------
def metrics_table():
    table = pd.DataFrame(results["metrics"]).T
    table = table.rename(columns={
        "rmse_log": "RMSE (log)",
        "r2_log": "R² (log)",
        "mae_euros": "Mean error (€)",
        "median_percent_error": "Typical error (%)",
    })
    table["RMSE (log)"] = table["RMSE (log)"].round(4)
    table["R² (log)"] = table["R² (log)"].round(4)
    table["Mean error (€)"] = table["Mean error (€)"].round(0)
    table["Typical error (%)"] = table["Typical error (%)"].round(1)
    return table.reset_index(names="Model")


def prediction_vs_actual_plot(model_choice):
    if model_choice == "All three":
        models_to_show = MODEL_NAMES
    else:
        models_to_show = [model_choice]

    fig = go.Figure()
    for name in models_to_show:
        fig.add_trace(go.Scatter(
            x=test_df["value_eur"], y=test_predictions[name],
            mode="markers", name=name, opacity=0.5,
            text=test_df["short_name"],
        ))

    # Diagonal line: a perfect prediction would sit exactly on it.
    low = test_df["value_eur"].min()
    high = test_df["value_eur"].max()
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines",
                             name="perfect prediction",
                             line=dict(color="black", dash="dash")))
    fig.update_xaxes(type="log", title="Actual value (€)")
    fig.update_yaxes(type="log", title="Predicted value (€)")
    fig.update_layout(title="Prediction vs. actual on the test set", height=520)
    return fig


def loss_curves_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=results["manual_loss_history"], name="PyTorch manual loop"))
    fig.add_trace(go.Scatter(y=results["torch_loss_history"], name="PyTorch nn.Module + optim"))
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(type="log", title="Training loss")
    fig.update_layout(title="Loss curves saved during training", height=400)
    return fig


def weights_table():
    return pd.DataFrame(results["weights"]).round(4)


# ---------------------------------------------------------------------------
# 4. Tab "Distributions"
# ---------------------------------------------------------------------------
DISTRIBUTION_COLUMNS = ["value_eur", "log_value"] + ALL_FEATURES


def distribution_plot(column):
    fig = px.histogram(full_df, x=column, nbins=50, color="position_group",
                       title=f"Distribution of {column}")
    fig.update_layout(height=450)
    return fig


def age_curve_plot():
    by_age = full_df.groupby("age")["value_eur"].median().reset_index()
    fig = px.line(by_age, x="age", y="value_eur", markers=True,
                  title="Median value by age (why we need a squared age term)")
    fig.update_yaxes(title="Median value (€)")
    fig.update_layout(height=400)
    return fig


# ---------------------------------------------------------------------------
# 5. Tab "Bargain finder" (the business view)
# ---------------------------------------------------------------------------
def find_bargains(budget_millions, position, how_many):
    table = test_df[["short_name", "player_positions", "position_group", "age", "value_eur"]].copy()
    table["predicted_eur"] = test_predictions["scikit-learn"]

    table = table[table["value_eur"] <= budget_millions * 1_000_000]
    if position != "all":
        table = table[table["position_group"] == position]

    # A "bargain" is a player whose real price is below what the model says
    # a player like him is usually worth.
    table["discount_percent"] = (1 - table["value_eur"] / table["predicted_eur"]) * 100
    table = table.sort_values("discount_percent", ascending=False).head(int(how_many))

    table["value_eur"] = table["value_eur"].apply(format_euros)
    table["predicted_eur"] = table["predicted_eur"].apply(format_euros)
    table["discount_percent"] = table["discount_percent"].round(1)
    table = table.drop(columns=["position_group"])
    table.columns = ["Player", "Positions", "Age", "Actual value", "Model value", "Discount (%)"]
    return table


# ---------------------------------------------------------------------------
# 6. Tab "Try a player"
# ---------------------------------------------------------------------------
def predict_one_player(age, potential, height, position, foot, skill_moves, weak_foot,
                       reputation, pace, shooting, passing, dribbling, defending, physic):
    # Build a one-row table that looks like the original dataset...
    first_position = {"defender": "CB", "midfielder": "CM", "attacker": "ST"}[position]
    player = pd.DataFrame([{
        "age": age, "potential": potential, "height_cm": height,
        "player_positions": first_position, "preferred_foot": foot,
        "skill_moves": skill_moves, "weak_foot": weak_foot,
        "international_reputation": reputation,
        "pace": pace, "shooting": shooting, "passing": passing,
        "dribbling": dribbling, "defending": defending, "physic": physic,
    }])
    # ...and send it through the same feature function used in training.
    player = add_features(player)
    predictions = predict_euros(player)

    rows = [[name, format_euros(predictions[name][0])] for name in MODEL_NAMES]
    return pd.DataFrame(rows, columns=["Model", "Predicted value"])


# ---------------------------------------------------------------------------
# 7. Build the page
# ---------------------------------------------------------------------------
with gr.Blocks(title="FIFA 20 player value") as demo:
    gr.Markdown(
        "# FIFA 20 player value: one linear regression, trained three ways\n"
        "A low-budget club wants to find players whose price is lower than what "
        "their profile is usually worth. All models were trained by `train.py` "
        "and are only **loaded** here."
    )

    with gr.Tab("Compare models"):
        gr.Markdown("### Test set results (players the models never saw)")
        gr.Dataframe(value=metrics_table(), interactive=False)
        model_choice = gr.Dropdown(["All three"] + MODEL_NAMES, value="All three", label="Model to show")
        scatter = gr.Plot(value=prediction_vs_actual_plot("All three"))
        model_choice.change(prediction_vs_actual_plot, inputs=model_choice, outputs=scatter)
        gr.Plot(value=loss_curves_plot())
        gr.Markdown(f"### Weights learned by each method (alpha = {results['best_alpha']})")
        gr.Dataframe(value=weights_table(), interactive=False)

    with gr.Tab("Distributions"):
        column_choice = gr.Dropdown(DISTRIBUTION_COLUMNS, value="value_eur", label="Column")
        histogram = gr.Plot(value=distribution_plot("value_eur"))
        column_choice.change(distribution_plot, inputs=column_choice, outputs=histogram)
        gr.Markdown("Compare `value_eur` with `log_value`: the log is what makes the target fit a line.")
        gr.Plot(value=age_curve_plot())

    with gr.Tab("Bargain finder"):
        gr.Markdown(
            "Test-set players whose actual value is furthest **below** the model's value. "
            "This is a shortlist for scouts to look at, not a buy order: a big discount "
            "can also mean the model is missing something about the player."
        )
        with gr.Row():
            budget = gr.Slider(0.1, 110, value=5, step=0.1, label="Maximum price (million €)")
            position_filter = gr.Dropdown(["all", "defender", "midfielder", "attacker"], value="all", label="Position")
            how_many = gr.Slider(5, 50, value=15, step=5, label="Players to show")
        bargains = gr.Dataframe(value=find_bargains(5, "all", 15), interactive=False)
        for control in [budget, position_filter, how_many]:
            control.change(find_bargains, inputs=[budget, position_filter, how_many], outputs=bargains)

    with gr.Tab("Try a player"):
        with gr.Row():
            age = gr.Slider(16, 40, value=24, step=1, label="Age")
            potential = gr.Slider(48, 95, value=75, step=1, label="Potential")
            height = gr.Slider(155, 205, value=180, step=1, label="Height (cm)")
        with gr.Row():
            position = gr.Dropdown(["defender", "midfielder", "attacker"], value="midfielder", label="Position")
            foot = gr.Dropdown(["Right", "Left"], value="Right", label="Preferred foot")
            skill_moves = gr.Slider(1, 5, value=3, step=1, label="Skill moves")
            weak_foot = gr.Slider(1, 5, value=3, step=1, label="Weak foot")
            reputation = gr.Slider(1, 5, value=1, step=1, label="International reputation")
        with gr.Row():
            skill_sliders = [gr.Slider(15, 96, value=65, step=1, label=skill.capitalize()) for skill in SKILLS]
        player_inputs = [age, potential, height, position, foot, skill_moves, weak_foot, reputation] + skill_sliders
        predict_button = gr.Button("Predict value")
        prediction_output = gr.Dataframe(interactive=False)
        predict_button.click(predict_one_player, inputs=player_inputs, outputs=prediction_output)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
