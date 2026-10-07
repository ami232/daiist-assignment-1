"""Gradio dashboard for the Walmart weekly sales models.

Everything here comes from the artifacts saved by train.py. This file loads them and makes
predictions with them, but never fits or retrains anything.
"""
import json
import os
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from features import FLAGS, NUMERIC, add_features, load_data, split_by_time
from models import LinearRegressionModel

ARTIFACTS = Path(__file__).parent / "artifacts"
MODELS = ["scikit-learn", "Manual PyTorch", "Standard PyTorch"]
BASELINE = "Baseline (store average)"
UNITS = {"Temperature": "°F", "Fuel_Price": "$/gal", "Unemployment": "pts"}
WEEK_TYPES = {  # week type -> (flag column, the month it falls in)
    "Normal week": (None, None),
    "Super Bowl": ("is_superbowl", 2),
    "Labor Day": ("is_labor_day", 9),
    "Thanksgiving": ("is_thanksgiving", 11),
    "Pre-Christmas peak": ("is_pre_christmas", 12),
    "Post-Christmas": ("is_post_christmas", 12),
}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------- load what train.py saved
pre = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_ridge.joblib")
manual = torch.load(ARTIFACTS / "manual_torch.pt")
std_model = LinearRegressionModel(len(pre.get_feature_names_out()))
std_model.load_state_dict(torch.load(ARTIFACTS / "standard_torch.pt"))
std_model.eval()
store_avg = pd.read_csv(ARTIFACTS / "baseline_store_avg.csv", index_col="Store")["Weekly_Sales"]
coefs = pd.read_csv(ARTIFACTS / "coefficients.csv", index_col=0)
metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
scaler = pre.named_transformers_["num"]
std_of = dict(zip(scaler.feature_names_in_, scaler.scale_))
mean_of = dict(zip(scaler.feature_names_in_, scaler.mean_))


def predict_all(frame):
    """Dollar predictions from the three saved models (each one predicts log sales)."""
    X = pre.transform(frame)
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        log_preds = {
            "scikit-learn": sk_model.predict(X),
            "Manual PyTorch": (X_t @ manual["w"] + manual["b"]).numpy().ravel(),
            "Standard PyTorch": std_model(X_t).numpy().ravel(),
        }
    return {name: np.exp(p) for name, p in log_preds.items()}


# ---------------------------------------------------------------- data + test-year predictions
df = add_features(load_data())
train, val, test = split_by_time(df)
df["Split"] = np.select([df.index.isin(train.index), df.index.isin(val.index)], ["train", "validation"], "test")
test = test.copy()
for name, pred in predict_all(test).items():
    test[name] = pred
test[BASELINE] = test["Store"].map(store_avg).to_numpy()
STORE_CHOICES = ["All stores"] + [str(s) for s in sorted(df["Store"].unique())]


def select_store(store):
    return test if store == "All stores" else test[test["Store"] == int(store)]


# ---------------------------------------------------------------- tab 1: model comparison
def results_table():
    t = pd.DataFrame(metrics["test_results"]).T
    return pd.DataFrame({
        "Model": t.index,
        "WMAE ($, holidays x5)": t["WMAE"].map("{:,.0f}".format),
        "MAE ($)": t["MAE"].map("{:,.0f}".format),
        "MAPE (%)": t["MAPE_%"].map("{:.2f}".format),
        "R²": t["R2"].map("{:.4f}".format),
    }).reset_index(drop=True)


def agreement_table():
    rows = [{"Comparison": k, "Max |coefficient diff|": f"{v['max_abs_coef_diff']:.5f}",
             "Max test prediction diff (%)": f"{v['max_test_pred_diff_%']:.3f}"}
            for k, v in metrics["agreement"].items()]
    return pd.DataFrame(rows)


def time_series_plot(store):
    weekly = select_store(store).groupby("Date")[["Weekly_Sales", BASELINE] + MODELS].sum()
    fig = go.Figure()
    fig.add_scatter(x=weekly.index, y=weekly["Weekly_Sales"], name="Actual", line=dict(color="black", width=3))
    fig.add_scatter(x=weekly.index, y=weekly[BASELINE], name=BASELINE, line=dict(color="grey", dash="dot"))
    for model, dash in zip(MODELS, ["solid", "dash", "dot"]):
        fig.add_scatter(x=weekly.index, y=weekly[model], name=model, line=dict(dash=dash))
    fig.update_layout(title=f"Test year: actual vs. predicted weekly sales ({store})",
                      yaxis_title="Weekly sales ($)", hovermode="x unified", height=450)
    return fig


def scatter_plot(store):
    data = select_store(store)
    special = (data[FLAGS].sum(axis=1) > 0).to_numpy()
    colors = np.where(special, "crimson", "steelblue").tolist()
    lo, hi = data["Weekly_Sales"].min() * 0.9, data["Weekly_Sales"].max() * 1.1
    fig = make_subplots(rows=1, cols=3, subplot_titles=MODELS, shared_yaxes=True)
    for i, model in enumerate(MODELS, start=1):
        fig.add_scatter(x=data["Weekly_Sales"], y=data[model], mode="markers", showlegend=False,
                        marker=dict(size=5, color=colors, opacity=0.5), row=1, col=i)
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                        line=dict(color="grey", dash="dash"), row=1, col=i)
        fig.update_xaxes(title_text="Actual ($)", row=1, col=i)
    fig.update_yaxes(title_text="Predicted ($)", row=1, col=1)
    fig.update_layout(title="Predicted vs. actual (red = holiday weeks; dashed line = perfect prediction)", height=420)
    return fig


# ---------------------------------------------------------------- tab 2: drivers
def pct(coef):
    return 100 * (np.exp(coef) - 1)  # log-target coefficient -> % change in sales


def drivers_plot(method):
    c = coefs[method]
    labels = [f"{f} (+1 std = {std_of[f]:.2f} {UNITS.get(f, '')})" for f in NUMERIC] + FLAGS
    values = pct(c[NUMERIC + FLAGS].to_numpy())
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker_color=np.where(values >= 0, "seagreen", "indianred").tolist(),
                           text=[f"{v:+.1f}%" for v in values], textposition="outside"))
    fig.update_layout(title=f"Drivers of weekly sales ({method})", xaxis_title="% change in weekly sales",
                      height=420, yaxis=dict(autorange="reversed"))
    return fig


def months_plot(method):
    c = coefs[method]
    values = [0.0] + [pct(c[f"Month_{m}"]) for m in range(2, 13)]
    fig = go.Figure(go.Bar(x=MONTH_NAMES, y=values, text=[f"{v:+.1f}%" for v in values], textposition="outside"))
    fig.update_layout(title=f"Seasonality: each month vs. January ({method})",
                      yaxis_title="% change in weekly sales", height=380)
    return fig


# ---------------------------------------------------------------- tab 3: distributions
DIST_COLUMNS = ["Weekly_Sales", "log_sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]


def distribution_plot(column):
    fig = px.histogram(df, x=column, color="Split", barmode="overlay", opacity=0.6, nbins=60,
                       category_orders={"Split": ["train", "validation", "test"]})
    fig.update_layout(title=f"Distribution of {column} by split", height=420)
    return fig


def sales_by_week_type_plot():
    week_type = pd.Series("Normal week", index=df.index)
    for label, (flag, _) in WEEK_TYPES.items():
        if flag:
            week_type[df[flag] == 1] = label
    fig = px.box(df.assign(**{"Week type": week_type}), x="Week type", y="Weekly_Sales",
                 category_orders={"Week type": list(WEEK_TYPES)})
    fig.update_layout(title="Weekly sales by week type (all stores, all years)", height=420)
    return fig


# ---------------------------------------------------------------- tab 4: what-if
def what_if(store, month_name, week_type, temperature, fuel, unemployment):
    flag, flag_month = WEEK_TYPES[week_type]
    month = flag_month or MONTH_NAMES.index(month_name) + 1
    row = {**mean_of,  # any driver without a slider is held at its training average
           "Temperature": temperature, "Fuel_Price": fuel, "Unemployment": unemployment,
           "Store": int(store), "Month": month, **{f: int(f == flag) for f in FLAGS}}
    preds = predict_all(pd.DataFrame([row]))
    out = pd.DataFrame({"Model": MODELS + [BASELINE],
                        "Predicted weekly sales ($)": [f"{preds[m][0]:,.0f}" for m in MODELS]
                        + [f"{store_avg[int(store)]:,.0f}"]})
    note = f"Month used: **{MONTH_NAMES[month - 1]}**" + (" (set by the week type)." if flag_month else ".")
    return out, note


# ---------------------------------------------------------------- layout
with gr.Blocks(title="Walmart weekly sales: three ways, one model") as demo:
    gr.Markdown("# Walmart weekly sales: one linear model, trained three ways\n"
                "Ridge-regularised linear regression on log weekly sales, trained with scikit-learn, "
                "a manual PyTorch loop and a standard PyTorch workflow. All results below come from the "
                "saved models; nothing is retrained here. Test year: Nov 2011 to Oct 2012.")

    with gr.Tab("Model comparison"):
        gr.Markdown("### Test-year results")
        gr.Dataframe(value=results_table(), interactive=False)
        gr.Markdown("### Do the three implementations agree?")
        gr.Dataframe(value=agreement_table(), interactive=False)
        store_dd = gr.Dropdown(STORE_CHOICES, value="All stores", label="Store")
        ts = gr.Plot()
        sc = gr.Plot()
        store_dd.change(time_series_plot, store_dd, ts)
        store_dd.change(scatter_plot, store_dd, sc)

    with gr.Tab("Drivers"):
        gr.Markdown("Each bar is a model coefficient turned into a % effect on weekly sales. Holiday flags are "
                    "relative to a normal week **in the same month**; continuous drivers are per +1 standard "
                    "deviation, holding store, month and holidays fixed. These are associations, not causes.")
        method = gr.Radio(MODELS, value="scikit-learn", label="Implementation")
        drv = gr.Plot()
        mon = gr.Plot()
        method.change(drivers_plot, method, drv)
        method.change(months_plot, method, mon)

    with gr.Tab("Distributions"):
        col_dd = gr.Dropdown(DIST_COLUMNS, value="Weekly_Sales", label="Column")
        dist = gr.Plot()
        col_dd.change(distribution_plot, col_dd, dist)
        gr.Plot(value=sales_by_week_type_plot())

    with gr.Tab("What-if"):
        gr.Markdown("Set up a hypothetical week and see what each saved model predicts.")
        with gr.Row():
            wi_store = gr.Dropdown(STORE_CHOICES[1:], value="1", label="Store")
            wi_month = gr.Dropdown(MONTH_NAMES, value="Jun", label="Month")
            wi_type = gr.Radio(list(WEEK_TYPES), value="Normal week", label="Week type")
        with gr.Row():
            wi_temp = gr.Slider(float(df["Temperature"].min()), float(df["Temperature"].max()),
                                value=round(float(mean_of["Temperature"]), 1), label="Temperature (°F)")
            wi_fuel = gr.Slider(float(df["Fuel_Price"].min()), float(df["Fuel_Price"].max()),
                                value=round(float(mean_of["Fuel_Price"]), 2), label="Fuel price ($/gal)")
            wi_unemp = gr.Slider(float(df["Unemployment"].min()), float(df["Unemployment"].max()),
                                 value=round(float(mean_of["Unemployment"]), 2), label="Unemployment (%)")
        wi_out = gr.Dataframe(interactive=False)
        wi_note = gr.Markdown()
        wi_inputs = [wi_store, wi_month, wi_type, wi_temp, wi_fuel, wi_unemp]
        for comp in wi_inputs:
            comp.change(what_if, wi_inputs, [wi_out, wi_note])

    demo.load(time_series_plot, store_dd, ts)
    demo.load(scatter_plot, store_dd, sc)
    demo.load(drivers_plot, method, drv)
    demo.load(months_plot, method, mon)
    demo.load(distribution_plot, col_dd, dist)
    demo.load(what_if, wi_inputs, [wi_out, wi_note])


if __name__ == "__main__":
    # Open a browser tab locally; skip that on CI (GitHub Actions sets CI=true), which only pings the server.
    demo.launch(inbrowser=not os.environ.get("CI"))
