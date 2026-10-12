# Assignment 1 - Gradio dashboard

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import gradio as gr

from common import transform

MODELS_DIR = "models"

prep = joblib.load(f"{MODELS_DIR}/preprocessing.joblib")
bedroom_imputer, ohe, scaler = prep["bedroom_imputer"], prep["ohe"], prep["scaler"]
feature_names, y_mean, y_std = prep["feature_names"], prep["y_mean"], prep["y_std"]

sk_model = joblib.load(f"{MODELS_DIR}/sklearn_ridge.joblib")

manual_state = torch.load(f"{MODELS_DIR}/manual_pytorch.pt", weights_only=True)
w_manual, b_manual = manual_state["w"], manual_state["b"]

linear = torch.nn.Linear(len(feature_names), 1)
linear.load_state_dict(torch.load(f"{MODELS_DIR}/standard_pytorch.pt", weights_only=True))
linear.eval()

with open(f"{MODELS_DIR}/metrics.json") as f:
    metrics = json.load(f)
with open(f"{MODELS_DIR}/test_predictions.json") as f:
    test_preds = json.load(f)
with open(f"{MODELS_DIR}/distributions.json") as f:
    dist = json.load(f)

METHOD_LABELS = {
    "baseline": "Baseline (predict mean)",
    "sklearn": "scikit-learn Ridge",
    "manual_pytorch": "Manual PyTorch loop",
    "standard_pytorch": "Standard PyTorch (nn.Module)",
}
COLORS = {"baseline": "#999999", "sklearn": "#1f77b4",
          "manual_pytorch": "#ff7f0e", "standard_pytorch": "#2ca02c"}


# Tab 1: model comparison

def metrics_bar_chart():
    methods = list(metrics.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[METHOD_LABELS[m] for m in methods],
        y=[metrics[m]["test"]["MAE"] for m in methods],
        marker_color=[COLORS[m] for m in methods],
        text=[f"${metrics[m]['test']['MAE']:,.0f}" for m in methods],
        textposition="outside",
    ))
    fig.update_layout(title="Test-set MAE by method (lower is better)",
                       yaxis_title="Mean Absolute Error ($)", showlegend=False,
                       margin=dict(t=50, b=20))
    return fig


def pred_vs_actual_plot(selected_methods):
    y_true = np.array(test_preds["y_true"])
    fig = go.Figure()
    lo, hi = y_true.min(), y_true.max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(dash="dash", color="black"),
                              name="Perfect prediction", showlegend=True))
    for m in (selected_methods or []):
        fig.add_trace(go.Scatter(
            x=y_true, y=np.array(test_preds[m]), mode="markers",
            marker=dict(size=4, opacity=0.4, color=COLORS[m]),
            name=METHOD_LABELS[m],
        ))
    fig.update_layout(title="Predicted vs. actual district value (test set)",
                       xaxis_title="Actual value ($)", yaxis_title="Predicted value ($)",
                       margin=dict(t=50, b=20))
    return fig


def comparison_table():
    rows = []
    for m in metrics:
        t = metrics[m]["test"]
        rows.append([METHOD_LABELS[m], f"${t['MAE']:,.0f}", f"${t['RMSE']:,.0f}", f"{t['R2']:.3f}"])
    return pd.DataFrame(rows, columns=["Method", "MAE", "RMSE", "R2"])


# Tab 2: distributions

def target_distribution():
    fig = px.histogram(x=dist["median_house_value"], nbins=50,
                        labels={"x": "Median house value ($)"},
                        title="Target distribution: district median house value")
    fig.update_layout(margin=dict(t=50, b=20))
    return fig


def feature_distribution(feature):
    fig = px.histogram(x=dist[feature], nbins=50, labels={"x": feature},
                        title=f"Feature distribution: {feature}")
    fig.update_layout(margin=dict(t=50, b=20))
    return fig


def ocean_proximity_counts():
    s = pd.Series(dist["ocean_proximity"]).value_counts()
    fig = px.bar(x=s.index, y=s.values, labels={"x": "ocean_proximity", "y": "count"},
                 title="Districts by ocean_proximity (training set)")
    fig.update_layout(margin=dict(t=50, b=20))
    return fig


def price_map():
    fig = px.scatter(
        x=dist["longitude"], y=dist["latitude"],
        color=dist["median_house_value"],
        color_continuous_scale="viridis",
        labels={"x": "longitude", "y": "latitude", "color": "median house value ($)"},
        title="Geographic price map (training set)",
        opacity=0.5,
    )
    fig.update_layout(margin=dict(t=50, b=20), yaxis=dict(scaleanchor="x"))
    return fig


# Tab 3: live prediction

def predict(longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
            population, households, median_income, ocean_proximity):
    row = pd.DataFrame([{
        "longitude": longitude, "latitude": latitude,
        "housing_median_age": housing_median_age, "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms, "population": population,
        "households": households, "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])
    X, _, _ = transform(row, bedroom_imputer, ohe, scaler=scaler)
    X = X[feature_names]

    sk_pred = float(sk_model.predict(X)[0])

    X_t = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        manual_pred = float((X_t @ w_manual + b_manual).item() * y_std + y_mean)
        standard_pred = float(linear(X_t).item() * y_std + y_mean)

    return f"${sk_pred:,.0f}", f"${manual_pred:,.0f}", f"${standard_pred:,.0f}"


with gr.Blocks(title="California District Value Estimator") as demo:
    gr.Markdown(
        "# California District Value Estimator\n"
        "Three implementations of the same linear regression model, trained once"
    )

    with gr.Tab("Model comparison"):
        gr.Markdown(
            "All three methods converge to essentially the same solution"
        )
        gr.Plot(metrics_bar_chart)
        gr.DataFrame(comparison_table, label="Test-set metrics")
        method_select = gr.CheckboxGroup(
            choices=[("scikit-learn Ridge", "sklearn"),
                     ("Manual PyTorch loop", "manual_pytorch"),
                     ("Standard PyTorch", "standard_pytorch")],
            value=["sklearn"], label="Show on scatter plot",
        )
        scatter_plot = gr.Plot(lambda m: pred_vs_actual_plot(m), inputs=method_select)
        method_select.change(pred_vs_actual_plot, inputs=method_select, outputs=scatter_plot)

    with gr.Tab("Data distributions"):
        gr.Plot(target_distribution)
        with gr.Row():
            feat_select = gr.Dropdown(
                choices=["median_income", "housing_median_age"],
                value="median_income", label="Feature",
            )
            feat_plot = gr.Plot(lambda f: feature_distribution(f), inputs=feat_select)
            feat_select.change(feature_distribution, inputs=feat_select, outputs=feat_plot)
        gr.Plot(ocean_proximity_counts)
        gr.Plot(price_map)

    with gr.Tab("Try a prediction"):
        gr.Markdown("Describe a hypothetical district and compare what each saved model predicts.")
        with gr.Row():
            with gr.Column():
                longitude = gr.Slider(-124.35, -114.31, value=-118.5, label="longitude")
                latitude = gr.Slider(32.54, 41.95, value=34.27, label="latitude")
                housing_median_age = gr.Slider(1, 52, value=28, step=1, label="housing_median_age")
                median_income = gr.Slider(0.5, 15.0, value=3.5, label="median_income (tens of $1000s)")
            with gr.Column():
                total_rooms = gr.Slider(2, 15000, value=2100, label="total_rooms (district total)")
                total_bedrooms = gr.Slider(2, 3000, value=440, label="total_bedrooms (district total)")
                population = gr.Slider(3, 15000, value=1180, label="population (district total)")
                households = gr.Slider(2, 4000, value=410, label="households (district total)")
                ocean_proximity = gr.Dropdown(
                    list(ohe.categories_[0]), value="INLAND", label="ocean_proximity"
                )
        predict_btn = gr.Button("Predict", variant="primary")
        with gr.Row():
            sk_out = gr.Textbox(label="scikit-learn Ridge")
            manual_out = gr.Textbox(label="Manual PyTorch loop")
            standard_out = gr.Textbox(label="Standard PyTorch")
        predict_btn.click(
            predict,
            inputs=[longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                    population, households, median_income, ocean_proximity],
            outputs=[sk_out, manual_out, standard_out],
        )

if __name__ == "__main__":
    demo.launch()