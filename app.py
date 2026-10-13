"""Assignment 1 — Gradio dashboard.

Student dropout early-warning models."""

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score

ARTIFACTS = "artifacts"
COLORS = {"Graduate": "#636EFA", "Dropout": "#EF553B"}   # same colours in every chart

# 1. Load everything the training stage saved
sk_model = joblib.load(f"{ARTIFACTS}/sklearn_model.joblib")
preprocessor = joblib.load(f"{ARTIFACTS}/preprocessor.joblib")
data = joblib.load(f"{ARTIFACTS}/dashboard_data.joblib")

manual = torch.load(f"{ARTIFACTS}/manual_pytorch.pt")          # {"w": ..., "b": ...}
std_model = nn.Sequential(nn.Linear(data["n_features"], 1), nn.Sigmoid())
std_model.load_state_dict(torch.load(f"{ARTIFACTS}/standard_pytorch.pt"))
std_model.eval()

X_test = data["X_test"]
y_test = data["y_test"].values
clean_data = data["clean_data"]
COST_CONTACT = data["cost_contact"]
COST_MISSED = data["cost_missed"]
DEFAULT_THRESHOLD = data["threshold"]

# 2. Predict the test set with the saved models
X_test_prep = preprocessor.transform(X_test)
X_test_t = torch.tensor(X_test_prep, dtype=torch.float32)
with torch.no_grad():
    probs = {
        "scikit-learn": sk_model.predict_proba(X_test_prep)[:, 1],
        "Manual PyTorch": torch.sigmoid(X_test_t @ manual["w"] + manual["b"]).numpy(),
        "Standard PyTorch": std_model(X_test_t).squeeze(-1).numpy(),
    }

# 3. Tab 1: compare the three models
def scores_table(threshold, cost_contact, cost_missed):
    rows = []
    for name, p in probs.items():
        pred = (p >= threshold).astype(int)
        contacted = int(pred.sum())
        missed = int(((pred == 0) & (y_test == 1)).sum())
        rows.append({
            "model": name,
            "ROC-AUC": roc_auc_score(y_test, p),
            "log loss": log_loss(y_test, p),
            "recall": recall_score(y_test, pred, zero_division=0),
            "precision": precision_score(y_test, pred, zero_division=0),
            "contacted": contacted,
            "missed": missed,
            "cost (€)": cost_contact * contacted + cost_missed * missed,
        })
    return pd.DataFrame(rows).round(3)

def prediction_vs_actual_plot():
    long = pd.concat([
        pd.DataFrame({
            "model": name,
            "predicted dropout probability": p,
            "actual outcome": np.where(y_test == 1, "Dropout", "Graduate"),
        })
        for name, p in probs.items()
    ])
    fig = px.box(long, x="model", y="predicted dropout probability",
                 color="actual outcome", points="all", color_discrete_map=COLORS,
                 title="Predicted probability vs actual outcome (test set)")
    fig.add_hline(y=DEFAULT_THRESHOLD, line_dash="dash", annotation_text="20% threshold")
    return fig

# 4. Tab 2: distributions
plot_data = clean_data.copy()
plot_data["outcome"] = np.where(plot_data["dropout"] == 1, "Dropout", "Graduate")
FEATURES = [col for col in clean_data.columns if col != "dropout"]

def target_plot():
    counts = plot_data["outcome"].value_counts().reset_index()
    counts.columns = ["outcome", "students"]
    return px.bar(counts, x="outcome", y="students", color="outcome", text="students", color_discrete_map=COLORS,
                  title="Target: dropouts vs graduates (all 3,630 students)")

def feature_plot(feature):
    return px.histogram(plot_data, x=feature, color="outcome", barmode="overlay", color_discrete_map=COLORS,
                        histnorm="percent", opacity=0.6,
                        title=f"{feature}: dropouts vs graduates (% of each group)")

# 5. Tab 3: threshold and cost
def confusion_plot(pred):
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fig = px.imshow([[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
                    x=["Not contacted", "Contacted"], y=["Graduated", "Dropped out"],
                    labels=dict(x="Model decision", y="Actual outcome", color="students"),
                    title="Confusion matrix")
    return fig, tn, fp, fn, tp

def cost_curve(p, threshold, cost_contact, cost_missed):
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [cost_contact * (p >= t).sum() + cost_missed * ((p < t) & (y_test == 1)).sum()
             for t in thresholds]
    best = thresholds[int(np.argmin(costs))]
    fig = px.line(x=thresholds, y=costs, labels={"x": "threshold", "y": "total cost (€)"},
                  title="Total cost at every threshold")
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="your threshold")
    fig.add_vline(x=best, line_dash="dot", line_color="green",
                  annotation_text="lowest cost", annotation_position="bottom right")
    return fig, best, min(costs)

def update_threshold_tab(model_name, threshold, cost_contact, cost_missed):
    cost_contact = cost_contact or 0
    cost_missed = cost_missed or 0
    p = probs[model_name]
    pred = (p >= threshold).astype(int)
    cm_fig, tn, fp, fn, tp = confusion_plot(pred)
    cost = cost_contact * (tp + fp) + cost_missed * fn
    everyone = cost_contact * len(y_test)
    nobody = cost_missed * int(y_test.sum())
    curve_fig, best, best_cost = cost_curve(p, threshold, cost_contact, cost_missed)
    rule = cost_contact / cost_missed if cost_missed > 0 else float("nan")
    summary = (
        f"### Total cost: €{cost:,.0f}\n"
        f"- **Contacted:** {tp + fp} students ({tp} real dropouts, {fp} who would have graduated)\n"
        f"- **Missed dropouts:** {fn}\n"
        f"- **For comparison:** contacting everyone costs €{everyone:,.0f}; "
        f"contacting nobody costs €{nobody:,.0f}\n"
        f"- **Rule from your costs:** contact a student if their risk is above "
        f"{cost_contact:,.0f} ÷ {cost_missed:,.0f} = **{rule:.0%}**\n"
        f"- **Lowest cost on the chart below:** €{best_cost:,.0f} at threshold {best:.2f} "
        f"(test set, for exploring only)"
    )
    return cm_fig, summary, curve_fig

# 6. The dashboard
with gr.Blocks(title="Student dropout early warning") as demo:
    gr.Markdown(
        "# Student dropout early warning\n"
        "Logistic regression trained three ways (scikit-learn, manual PyTorch, standard PyTorch) "
        "to flag students at risk of dropping out at the end of their first semester. "
        "All results are on the test set (726 students)."
    )

    with gr.Tab("Compare models"):
        gr.Markdown("Each dot is a student. A good model gives dropouts high probabilities "
                    "and graduates low ones. The three models should look almost identical.")
        gr.Plot(value=prediction_vs_actual_plot())
        gr.Dataframe(value=scores_table(DEFAULT_THRESHOLD, COST_CONTACT, COST_MISSED))

    with gr.Tab("Distributions"):
        gr.Plot(value=target_plot())
        feature_in = gr.Dropdown(FEATURES, value="pass_rate_1st_sem", label="Feature")
        feature_out = gr.Plot(value=feature_plot("pass_rate_1st_sem"))
        feature_in.change(feature_plot, feature_in, feature_out)

    with gr.Tab("Threshold & cost"):
        gr.Markdown("Move the threshold or change the costs to see who gets contacted "
                    "and what it costs. Default costs: €300 per student contacted, "
                    "€1,500 per missed dropout.")
        with gr.Row():
            model_in = gr.Dropdown(list(probs.keys()), value="scikit-learn", label="Model")
            threshold_in = gr.Slider(0.01, 0.99, value=DEFAULT_THRESHOLD, step=0.01,
                                     label="Decision threshold")
        with gr.Row():
            contact_in = gr.Number(value=COST_CONTACT, label="Cost of contacting one student (€)")
            missed_in = gr.Number(value=COST_MISSED, label="Cost of missing one dropout (€)")
        with gr.Row():
            cm_out = gr.Plot()
            summary_out = gr.Markdown()
        curve_out = gr.Plot()

        inputs = [model_in, threshold_in, contact_in, missed_in]
        outputs = [cm_out, summary_out, curve_out]
        for component in inputs:
            component.change(update_threshold_tab, inputs, outputs)
        demo.load(update_threshold_tab, inputs, outputs)

if __name__ == "__main__":
    demo.launch()