from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

REQUIRED = [
    ARTIFACT_DIR / "sklearn_model.joblib",
    ARTIFACT_DIR / "preprocessor.joblib",
    ARTIFACT_DIR / "manual_torch.pt",
    ARTIFACT_DIR / "standard_torch.pt",
    ARTIFACT_DIR / "test_predictions.csv",
    ARTIFACT_DIR / "model_comparison.csv",
    ARTIFACT_DIR / "target_distribution.csv",
    ARTIFACT_DIR / "hero_pick_counts.csv",
    ARTIFACT_DIR / "metadata.json",
]
missing = [path.name for path in REQUIRED if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Missing training artifacts: " + ", ".join(missing) + ". Run `uv run python main.py train` first."
    )

sklearn_model = joblib.load(ARTIFACT_DIR / "sklearn_model.joblib")
preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
manual_state = torch.load(ARTIFACT_DIR / "manual_torch.pt", map_location="cpu")
standard_state = torch.load(ARTIFACT_DIR / "standard_torch.pt", map_location="cpu")

predictions_df = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
comparison_df = pd.read_csv(ARTIFACT_DIR / "model_comparison.csv")
target_df = pd.read_csv(ARTIFACT_DIR / "target_distribution.csv")
hero_pick_df = pd.read_csv(ARTIFACT_DIR / "hero_pick_counts.csv")
metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))

scaler = preprocessor["scaler"]
hero_ids = [int(x) for x in preprocessor["hero_ids"]]
hero_names = {int(k): v for k, v in preprocessor["hero_names"].items()}
hero_meta = {int(k): v for k, v in preprocessor["hero_meta"].items()}
roles = list(preprocessor["roles"])
attributes = list(preprocessor["attributes"])
hero_index = {hero_id: i for i, hero_id in enumerate(hero_ids)}
name_to_id = {name: hero_id for hero_id, name in hero_names.items()}
hero_choices = sorted(name_to_id)


standard_model = torch.nn.Sequential(
    torch.nn.Linear(len(preprocessor["feature_names"]), 1),
    torch.nn.Sigmoid(),
)
standard_model.load_state_dict(standard_state)
standard_model.eval()


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -30, 30))
    return 1.0 / (1.0 + np.exp(-value))


def featurize_draft(radiant_ids: list[int], dire_ids: list[int]) -> np.ndarray:
    vector = np.zeros(len(hero_ids) + len(roles) + len(attributes), dtype=np.float32)

    for hero_id in radiant_ids:
        vector[hero_index[hero_id]] = 1.0
    for hero_id in dire_ids:
        vector[hero_index[hero_id]] = -1.0

    offset = len(hero_ids)
    for i, role in enumerate(roles):
        radiant_count = sum(role in hero_meta[h]["roles"] for h in radiant_ids)
        dire_count = sum(role in hero_meta[h]["roles"] for h in dire_ids)
        vector[offset + i] = float(radiant_count - dire_count)

    offset += len(roles)
    for i, attribute in enumerate(attributes):
        radiant_count = sum(hero_meta[h]["primary_attr"] == attribute for h in radiant_ids)
        dire_count = sum(hero_meta[h]["primary_attr"] == attribute for h in dire_ids)
        vector[offset + i] = float(radiant_count - dire_count)

    return vector


def prediction_vs_actual_figure(model_column: str = "sklearn_prob"):
    labels = {
        "sklearn_prob": "scikit-learn",
        "manual_torch_prob": "Manual PyTorch",
        "standard_torch_prob": "PyTorch nn.Module",
    }
    sample = predictions_df.iloc[:: max(1, len(predictions_df) // 1200)].copy()
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.035, len(sample))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample["actual"] + jitter,
            y=sample[model_column],
            mode="markers",
            marker={"size": 6, "opacity": 0.45},
            hovertemplate="Actual=%{customdata}<br>Predicted Radiant win probability=%{y:.3f}<extra></extra>",
            customdata=sample["actual"],
            name=labels[model_column],
        )
    )
    fig.update_layout(
        title=f"Prediction vs. actual outcome — {labels[model_column]}",
        xaxis={"title": "Actual outcome (0 = Dire win, 1 = Radiant win)", "tickmode": "array", "tickvals": [0, 1]},
        yaxis={"title": "Predicted probability of Radiant win", "range": [0, 1]},
        margin={"l": 50, "r": 20, "t": 55, "b": 50},
        height=420,
    )
    return fig


def target_distribution_figure():
    labels = ["Dire win" if int(x) == 0 else "Radiant win" for x in target_df["radiant_win"]]
    fig = go.Figure(go.Bar(x=labels, y=target_df["count"], text=target_df["count"], textposition="auto"))
    fig.update_layout(
        title="Target distribution",
        xaxis_title="Match outcome",
        yaxis_title="Number of matches",
        height=380,
        margin={"l": 50, "r": 20, "t": 55, "b": 50},
    )
    return fig


def hero_pick_figure(top_n: int = 20):
    top_n = int(top_n)
    data = hero_pick_df.head(top_n).sort_values("pick_count", ascending=True)
    fig = go.Figure(go.Bar(x=data["pick_count"], y=data["hero"], orientation="h"))
    fig.update_layout(
        title=f"Top {top_n} most-picked heroes in the modeling sample",
        xaxis_title="Picks",
        yaxis_title="Hero",
        height=max(450, 25 * top_n),
        margin={"l": 120, "r": 20, "t": 55, "b": 50},
    )
    return fig


def threshold_analysis(threshold: float, model_column: str):
    threshold = float(threshold)
    y_true = predictions_df["actual"].to_numpy(dtype=int)
    probabilities = predictions_df[model_column].to_numpy(dtype=float)
    y_pred = (probabilities >= threshold).astype(int)

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    cost = fp + fn
    accuracy = (tp + tn) / len(y_true)

    matrix = np.array([[tn, fp], [fn, tp]])
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Dire", "Predicted Radiant"],
            y=["Actual Dire", "Actual Radiant"],
            text=matrix,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        title=f"Confusion matrix at threshold {threshold:.2f}",
        height=390,
        margin={"l": 90, "r": 20, "t": 55, "b": 60},
    )

    summary = (
        f"### Threshold result\n"
        f"**Accuracy:** {accuracy:.3f}  \n"
        f"**Business cost:** {cost} wrong draft calls  \n"
        f"Cost definition: **FP + FN**, where each incorrect draft recommendation costs 1 unit.  \n"
        f"FP={fp}, FN={fn}, TP={tp}, TN={tn}."
    )
    return fig, summary


def predict_draft(*hero_names_selected: str):
    if any(name is None for name in hero_names_selected):
        return pd.DataFrame(), "Select all 10 heroes first."

    if len(set(hero_names_selected)) != 10:
        return pd.DataFrame(), "Each hero can only appear once in a Dota draft."

    radiant_names = list(hero_names_selected[:5])
    dire_names = list(hero_names_selected[5:])
    radiant_ids = [name_to_id[name] for name in radiant_names]
    dire_ids = [name_to_id[name] for name in dire_names]

    raw = featurize_draft(radiant_ids, dire_ids).reshape(1, -1)
    scaled = scaler.transform(raw).astype(np.float32)

    sklearn_prob = float(sklearn_model.predict_proba(scaled)[0, 1])

    manual_weights = np.asarray(manual_state["weights"], dtype=np.float32)
    manual_bias = float(manual_state["bias"])
    manual_prob = sigmoid(float(scaled[0] @ manual_weights + manual_bias))

    with torch.no_grad():
        standard_prob = float(
            standard_model(torch.tensor(scaled, dtype=torch.float32))
            .squeeze(-1)
            .item()
        )

    result = pd.DataFrame(
        {
            "Model": ["scikit-learn", "Manual PyTorch", "PyTorch nn.Module"],
            "Radiant win probability": [sklearn_prob, manual_prob, standard_prob],
            "Dire win probability": [1 - sklearn_prob, 1 - manual_prob, 1 - standard_prob],
        }
    )
    result["Radiant win probability"] = result["Radiant win probability"].map(lambda x: round(x, 4))
    result["Dire win probability"] = result["Dire win probability"].map(lambda x: round(x, 4))

    average_probability = float(np.mean([sklearn_prob, manual_prob, standard_prob]))
    if average_probability >= 0.55:
        verdict = f"### Draft read: Radiant favored ({average_probability:.1%} average predicted win probability)"
    elif average_probability <= 0.45:
        verdict = f"### Draft read: Dire favored ({1-average_probability:.1%} average predicted win probability)"
    else:
        verdict = f"### Draft read: Close draft ({average_probability:.1%} average Radiant win probability)"

    verdict += "\nThis is a draft-only estimate. It does not know player skill, lanes, execution, items, or in-game events."
    return result, verdict


model_column_choices = {
    "scikit-learn": "sklearn_prob",
    "Manual PyTorch": "manual_torch_prob",
    "PyTorch nn.Module": "standard_torch_prob",
}


def update_prediction_plot(model_name: str):
    return prediction_vs_actual_figure(model_column_choices[model_name])


def update_threshold(threshold: float, model_name: str):
    return threshold_analysis(threshold, model_column_choices[model_name])


initial_heroes = hero_pick_df["hero"].head(10).tolist()
if len(initial_heroes) < 10:
    initial_heroes = hero_choices[:10]

with gr.Blocks(title="Dota 2 Draft Win Probability") as demo:
    gr.Markdown(
        "# Dota 2 Draft Win Probability\n"
        "Logistic regression trained three ways on the same time-ordered match split. "
        "The dashboard only loads artifacts created by `train.py`; it never retrains models."
    )

    with gr.Tab("Model comparison"):
        gr.Dataframe(
            value=comparison_df.round(4),
            interactive=False,
            label="Test-set results (threshold = 0.50 for accuracy)",
        )
        plot_model = gr.Radio(
            choices=list(model_column_choices), value="scikit-learn", label="Model"
        )
        prediction_plot = gr.Plot(value=prediction_vs_actual_figure())
        plot_model.change(update_prediction_plot, inputs=plot_model, outputs=prediction_plot)

    with gr.Tab("Data distributions"):
        gr.Markdown(
            f"Modeling sample: **{metadata['model_rows']:,}** recent valid matches | "
            f"Train **{metadata['train_rows']:,}** | Validation **{metadata['validation_rows']:,}** | "
            f"Test **{metadata['test_rows']:,}** | Features **{metadata['n_features']}**"
        )
        gr.Plot(value=target_distribution_figure())
        top_n = gr.Slider(10, 40, value=20, step=5, label="Number of heroes to show")
        pick_plot = gr.Plot(value=hero_pick_figure(20))
        top_n.change(hero_pick_figure, inputs=top_n, outputs=pick_plot)

    with gr.Tab("Threshold analysis"):
        threshold_model = gr.Radio(
            choices=list(model_column_choices), value="scikit-learn", label="Model"
        )
        threshold_slider = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=0.50,
            step=0.01,
            label="Radiant decision threshold",
        )
        confusion_plot = gr.Plot()
        threshold_summary = gr.Markdown()

        threshold_slider.change(
            update_threshold,
            inputs=[threshold_slider, threshold_model],
            outputs=[confusion_plot, threshold_summary],
        )
        threshold_model.change(
            update_threshold,
            inputs=[threshold_slider, threshold_model],
            outputs=[confusion_plot, threshold_summary],
        )
        demo.load(
            update_threshold,
            inputs=[threshold_slider, threshold_model],
            outputs=[confusion_plot, threshold_summary],
        )

    with gr.Tab("Draft Battle"):
        gr.Markdown(
            "Pick five unique heroes for each side. The same engineered features used during training "
            "are built from the selected draft and passed through all three saved models."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Radiant")
                radiant_inputs = [
                    gr.Dropdown(hero_choices, value=initial_heroes[i], label=f"Radiant {i+1}")
                    for i in range(5)
                ]
            with gr.Column():
                gr.Markdown("### Dire")
                dire_inputs = [
                    gr.Dropdown(hero_choices, value=initial_heroes[i+5], label=f"Dire {i+1}")
                    for i in range(5)
                ]

        predict_button = gr.Button("Evaluate draft", variant="primary")
        draft_table = gr.Dataframe(interactive=False, label="Saved-model predictions")
        draft_verdict = gr.Markdown()
        predict_button.click(
            predict_draft,
            inputs=radiant_inputs + dire_inputs,
            outputs=[draft_table, draft_verdict],
        )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
