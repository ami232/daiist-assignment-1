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
"""
app.py - Step 10: Gradio dashboard.

Run with: uv run python main.py app

"""
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

from business import COST_FALSE_ALARM, COST_MISSED_CANCELLATION, business_cost
from evaluate import MODEL_NAMES, load_models, load_splits, predict_all
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, get_X_y

LABELS = {"sklearn": "sklearn", "manual_torch": "Manual PyTorch",
          "standard_torch": "Standard PyTorch"}
# Same colours everywhere: red = cancelled, blue = stayed
OUTCOME_COLORS = {"Cancelled": "#d62728", "Stayed": "#1f77b4"}

# ---------------------------------------------------------------
# Load everything ONCE at startup (inference only, no training)
# ---------------------------------------------------------------
train_df, val_df, test_df = load_splits()
X_test, y_test = get_X_y(test_df)
y_test = y_test.to_numpy()
pre, models = load_models()
probs = predict_all(pre, models, X_test)          # P(cancel) on the test set
never_cost = business_cost(y_test, np.zeros(len(y_test)), 0.5)
always_cost = business_cost(y_test, np.ones(len(y_test)), 0.5)


# ---------------------------------------------------------------
# Tab 1: compare the three models
# ---------------------------------------------------------------
def comparison_table():
    rows = []
    for name in MODEL_NAMES:
        p = probs[name]
        rows.append({
            "Model": LABELS[name],
            "Accuracy @0.5": round(float(((p >= 0.5) == y_test).mean()), 3),
            "ROC-AUC": round(float(roc_auc_score(y_test, p)), 3),
            "Log loss": round(float(log_loss(y_test, np.clip(p, 1e-7, 1 - 1e-7))), 3),
            "Cost @0.5 (EUR)": int(business_cost(y_test, p, 0.5)),
        })
    return pd.DataFrame(rows)


def prediction_vs_actual_plot():
    """Calibration plot: predicted probability vs what actually happened."""
    fig = go.Figure()
    bins = np.linspace(0, 1, 11)
    for name in MODEL_NAMES:
        p = probs[name]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        mean_pred, actual_rate = [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() >= 20:                      # skip near-empty bins
                mean_pred.append(p[mask].mean())
                actual_rate.append(y_test[mask].mean())
        fig.add_trace(go.Scatter(x=mean_pred, y=actual_rate, mode="lines+markers",
                                 name=LABELS[name]))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title="Predicted vs actual cancellation rate (test set, 10 bins)",
                      xaxis_title="Average predicted P(cancel)",
                      yaxis_title="Actual share cancelled",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], height=450)
    return fig


def probability_histogram(model_label):
    name = {v: k for k, v in LABELS.items()}[model_label]
    df = pd.DataFrame({"P(cancel)": probs[name],
                       "Actual": np.where(y_test == 1, "Cancelled", "Stayed")})
    fig = px.histogram(df, x="P(cancel)", color="Actual", nbins=40,
                       barmode="overlay", opacity=0.6,
                       color_discrete_map=OUTCOME_COLORS,
                       title=f"{model_label}: predicted probability by actual outcome")
    fig.update_layout(height=400)
    return fig


# ---------------------------------------------------------------
# Tab 2: feature and target distributions
# ---------------------------------------------------------------
def target_plot():
    rates = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Cancellation rate": [train_df[TARGET].mean(), val_df[TARGET].mean(),
                              test_df[TARGET].mean()],
    })
    fig = px.bar(rates, x="Split", y="Cancellation rate", text_auto=".1%",
                 title="Target: share of bookings cancelled in each split")
    fig.update_layout(yaxis_tickformat=".0%", height=350)
    return fig


def feature_plot(feature):
    df = train_df.copy()
    df["Outcome"] = np.where(df[TARGET] == 1, "Cancelled", "Stayed")
    if feature in NUMERIC_FEATURES:
        fig = px.histogram(df, x=feature, color="Outcome", nbins=40,
                           barmode="overlay", opacity=0.6,
                           color_discrete_map=OUTCOME_COLORS,
                           title=f"{feature}: distribution by outcome (train set)")
    else:
        rate = (df.groupby(feature)[TARGET].agg(["mean", "size"])
                  .reset_index().sort_values("size", ascending=False).head(15))
        fig = px.bar(rate, x=feature, y="mean", hover_data=["size"],
                     title=f"{feature}: cancellation rate per category "
                           f"(train set, 15 most common)")
        fig.update_layout(yaxis_title="Cancellation rate", yaxis_tickformat=".0%")
    fig.update_layout(height=420)
    return fig


# ---------------------------------------------------------------
# Tab 3: threshold slider -> confusion matrix + business cost
# ---------------------------------------------------------------
def threshold_view(model_label, threshold):
    name = {v: k for k, v in LABELS.items()}[model_label]
    p = probs[name]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    cost = fn * COST_MISSED_CANCELLATION + fp * COST_FALSE_ALARM

    # Confusion matrix as a heatmap
    z = [[tn, fp], [fn, tp]]
    text = [[f"Stayed, no deposit<br>{tn}", f"False alarm<br>{fp}"],
            [f"Missed cancellation<br>{fn}", f"Caught cancellation<br>{tp}"]]
    cm = go.Figure(go.Heatmap(z=z, x=["No deposit", "Ask deposit"],
                              y=["Stayed", "Cancelled"], text=text,
                              texttemplate="%{text}", colorscale="Blues",
                              showscale=False))
    cm.update_layout(title=f"Confusion matrix at threshold {threshold:.2f}",
                     xaxis_title="Model decision", yaxis_title="What happened",
                     yaxis_autorange="reversed", height=400)

    # Cost for every threshold, with the current one marked
    ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [business_cost(y_test, p, t) for t in ts]
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=ts, y=costs, mode="lines", name=model_label))
    curve.add_hline(y=never_cost, line_dash="dot", annotation_text="Never ask")
    curve.add_hline(y=always_cost, line_dash="dot", annotation_text="Always ask")
    curve.add_vline(x=threshold, line_dash="dash", line_color="red")
    curve.update_layout(title="Business cost on the test set vs threshold",
                        xaxis_title="Threshold", yaxis_title="Cost (EUR)", height=400)

    summary = (
        f"### Business cost: **EUR {cost:,}**\n"
        f"- Missed cancellations: {fn} x EUR {COST_MISSED_CANCELLATION} = EUR {fn * COST_MISSED_CANCELLATION:,}\n"
        f"- False alarms (good guests asked for a deposit): {fp} x EUR {COST_FALSE_ALARM} = EUR {fp * COST_FALSE_ALARM:,}\n"
        f"- Deposits requested: {tp + fp} of {len(y_test):,} bookings ({(tp + fp) / len(y_test):.0%})\n\n"
        f"For comparison: never asking costs **EUR {never_cost:,}**, "
        f"always asking costs **EUR {always_cost:,}**."
    )
    return cm, curve, summary


# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
with gr.Blocks(title="City Hotel cancellation risk") as demo:
    gr.Markdown(
        "# City Hotel - booking cancellation risk\n"
        "Predicts at booking time whether a reservation will be cancelled, so "
        "high-risk bookings can be asked for a deposit (refundable up to 14 days "
        f"before arrival). Test set: {len(y_test):,} bookings arriving "
        f"{test_df['arrival_date'].min().date()} to {test_df['arrival_date'].max().date()}."
    )

    with gr.Tab("1. Compare the three models"):
        gr.Dataframe(value=comparison_table(), interactive=False)
        gr.Plot(value=prediction_vs_actual_plot())
        hist_model = gr.Dropdown(list(LABELS.values()), value="sklearn", label="Model")
        hist_plot = gr.Plot()
        hist_model.change(probability_histogram, hist_model, hist_plot)

    with gr.Tab("2. Data distributions"):
        gr.Plot(value=target_plot())
        feat = gr.Dropdown(NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                           value="log_lead_time", label="Feature")
        feat_plot = gr.Plot()
        feat.change(feature_plot, feat, feat_plot)

    with gr.Tab("3. Threshold & business cost"):
        with gr.Row():
            thr_model = gr.Dropdown(list(LABELS.values()), value="sklearn", label="Model")
            thr = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Threshold: ask for a deposit if P(cancel) >=")
        cost_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        for control in (thr_model, thr):
            control.change(threshold_view, [thr_model, thr], [cm_plot, curve_plot, cost_md])

    # Fill the interactive plots when the page first opens
    demo.load(probability_histogram, hist_model, hist_plot)
    demo.load(feature_plot, feat, feat_plot)
    demo.load(threshold_view, [thr_model, thr], [cm_plot, curve_plot, cost_md])


if __name__ == "__main__":
    demo.launch()