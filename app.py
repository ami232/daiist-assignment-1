"""
Assignment 1 — Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file — main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py app`, this file must build and launch a Gradio
app (a `demo` that calls `.launch()`) using only what your training stage
already produced — it must never retrain anything itself. At minimum, the
dashboard must let you:

- Compare your three trained models via a prediction-vs-actual plot.
- See feature and/or target distributions.
- For classification: move a decision-threshold slider and watch the
  confusion matrix, and a business-cost number tied to your REPORT.md's
  framing, change with it.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""


import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
import matplotlib.pyplot as plt

# ============================================================
# 1. Cargar todo lo que train.py guardo, sin reentrenar nada
# ============================================================
scaler = joblib.load("models/scaler.pkl")
sklearn_model = joblib.load("models/sklearn_model.pkl")

with open("models/feature_cols.json") as f:
    feature_cols = json.load(f)

test_df = pd.read_csv("models/test_data.csv")
test_df["dteday"] = pd.to_datetime(test_df["dteday"])

n_features = len(feature_cols)

manual_weights = torch.load("models/manual_model.pt")
w = manual_weights["w"]
b = manual_weights["b"]

standard_model = nn.Linear(n_features, 1)
standard_model.load_state_dict(torch.load("models/standard_model.pt"))
standard_model.eval()

# ============================================================
# 2. Predicciones de los 3 modelos sobre TODO el test set
#    (se calculan una sola vez al arrancar, nunca se reentrena nada)
# ============================================================
X_test = test_df[feature_cols].values
y_test = test_df["cnt"].values

X_test_scaled = scaler.transform(X_test)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

y_pred_sklearn = sklearn_model.predict(X_test_scaled)

with torch.no_grad():
    y_pred_manual = (X_test_t @ w + b).numpy().flatten()
    y_pred_standard = standard_model(X_test_t).numpy().flatten()

test_df["pred_sklearn"] = y_pred_sklearn
test_df["pred_manual"] = y_pred_manual
test_df["pred_standard"] = y_pred_standard

MODEL_PRED_COLS = {
    "Scikit-learn": "pred_sklearn",
    "PyTorch manual": "pred_manual",
    "PyTorch estandar": "pred_standard",
}


# ============================================================
# 3. Funciones que generan cada grafico
# ============================================================
def plot_pred_vs_actual(model_name, hora_min, hora_max):
    pred_col = MODEL_PRED_COLS[model_name]

    subset = test_df[(test_df["hr"] >= hora_min) & (test_df["hr"] <= hora_max)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(subset["cnt"], subset[pred_col], alpha=0.3, s=10)

    max_val = max(subset["cnt"].max(), subset[pred_col].max())
    ax.plot([0, max_val], [0, max_val], "r--", label="Prediccion perfecta")

    ax.set_xlabel("Alquileres reales (cnt)")
    ax.set_ylabel("Alquileres predichos")
    ax.set_title(model_name + ": prediccion vs. real (horas " + str(hora_min) + "-" + str(hora_max) + ")")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_distribution(feature_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(test_df[feature_name], bins=30, color="steelblue", edgecolor="black")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion de " + feature_name + " (test set)")
    fig.tight_layout()
    return fig


def plot_target_distribution():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(test_df["cnt"], bins=30, color="seagreen", edgecolor="black")
    ax.set_xlabel("cnt (alquileres por hora)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion del target (test set)")
    fig.tight_layout()
    return fig


def compute_metrics_by_hour_range(hora_min, hora_max):
    subset = test_df[(test_df["hr"] >= hora_min) & (test_df["hr"] <= hora_max)]

    rows = []
    for model_name, pred_col in MODEL_PRED_COLS.items():
        errors = np.abs(subset["cnt"] - subset[pred_col])
        mae = errors.mean()
        rows.append([model_name, round(mae, 2), len(subset)])

    return pd.DataFrame(rows, columns=["Modelo", "MAE", "N filas en este rango"])


# ============================================================
# 4. Construccion del dashboard
# ============================================================
with gr.Blocks(title="Bike Sharing - Comparacion de modelos") as demo:
    gr.Markdown("# Dashboard: prediccion de demanda de bicicletas compartidas")
    gr.Markdown(
        "Los 3 modelos fueron entrenados previamente en train.py y aqui solo "
        "se cargan sus pesos guardados, esta app nunca reentrena nada."
    )

    with gr.Tab("Prediccion vs. Real"):
        with gr.Row():
            model_selector = gr.Radio(
                choices=list(MODEL_PRED_COLS.keys()),
                value="Scikit-learn",
                label="Modelo",
            )
        with gr.Row():
            hora_min_slider = gr.Slider(0, 23, value=0, step=1, label="Hora minima")
            hora_max_slider = gr.Slider(0, 23, value=23, step=1, label="Hora maxima")

        with gr.Row():
            pred_plot = gr.Plot()
        with gr.Row():
            metrics_table = gr.Dataframe(label="MAE por modelo en el rango seleccionado")

        def update_pred_tab(model_name, hora_min, hora_max):
            fig = plot_pred_vs_actual(model_name, hora_min, hora_max)
            table = compute_metrics_by_hour_range(hora_min, hora_max)
            return fig, table

        model_selector.change(
            update_pred_tab,
            inputs=[model_selector, hora_min_slider, hora_max_slider],
            outputs=[pred_plot, metrics_table],
        )
        hora_min_slider.change(
            update_pred_tab,
            inputs=[model_selector, hora_min_slider, hora_max_slider],
            outputs=[pred_plot, metrics_table],
        )
        hora_max_slider.change(
            update_pred_tab,
            inputs=[model_selector, hora_min_slider, hora_max_slider],
            outputs=[pred_plot, metrics_table],
        )

        demo.load(
            update_pred_tab,
            inputs=[model_selector, hora_min_slider, hora_max_slider],
            outputs=[pred_plot, metrics_table],
        )

    with gr.Tab("Distribuciones"):
        gr.Markdown("### Distribucion del target")
        target_plot = gr.Plot(value=plot_target_distribution())

        gr.Markdown("### Distribucion de una feature")
        feature_selector = gr.Dropdown(
            choices=feature_cols, value="temp", label="Elige una feature"
        )
        feature_plot = gr.Plot()

        feature_selector.change(
            plot_feature_distribution, inputs=feature_selector, outputs=feature_plot
        )
        demo.load(plot_feature_distribution, inputs=feature_selector, outputs=feature_plot)


if __name__ == "__main__":
    demo.launch()
    
