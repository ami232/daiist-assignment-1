"""
app.py — Assignment 1 Gradio dashboard. Run: uv run python main.py app
Loads only what train.py saved; nothing is retrained
"""

from __future__ import annotations
 
import json
 
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
 
from train import ARTIFACT_DIR, ChurnModel, campaign_value, evaluate
 
METRICS = ARTIFACT_DIR / "metrics"
LABELS = {0: "Stayed", 1: "Churned"}
 
# Load artifacts and predict once
 
def load_predictions():
    data = np.load(ARTIFACT_DIR / "data" / "test_processed.npz")
    X_test, y_test = data["X_test"], data["y_test"]
 
    sklearn_model = joblib.load(ARTIFACT_DIR / "models" / "sklearn_logreg.joblib")
    baseline = joblib.load(ARTIFACT_DIR / "models" / "naive_baseline.joblib")
    manual = torch.load(ARTIFACT_DIR / "models" / "manual_pytorch_logreg.pt", weights_only=True)
    standard = ChurnModel(X_test.shape[1]).double()
    standard.load_state_dict(torch.load(ARTIFACT_DIR / "models" / "standard_pytorch_logreg.pt",
                                        weights_only=True))
    standard.eval()
 
    X_t = torch.tensor(X_test, dtype=torch.float64)
    with torch.no_grad():
        manual_probs = torch.sigmoid(X_t @ manual["w"] + manual["b"]).numpy()
        standard_probs = torch.sigmoid(standard(X_t)).numpy()
    probs = {"scikit-learn": sklearn_model.predict_proba(X_test)[:, 1],
             "Manual PyTorch": manual_probs,
             "Standard PyTorch": standard_probs}
    return y_test, probs, baseline.predict_proba(X_test)[:, 1]
 
 
META = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
Y_TEST, PROBS, BASELINE_PROBS = load_predictions()
MODEL_NAMES = list(PROBS)
THRESHOLD = META["threshold"]
 
TEST_DF = pd.read_csv(ARTIFACT_DIR / "data" / "test_customers.csv")
TRAIN_DF = pd.read_csv(ARTIFACT_DIR / "data" / "train_customers.csv")
BILLS = TEST_DF["MonthlyCharges"].to_numpy()
MONTH_TO_MONTH = (TEST_DF["Contract"] == "Month-to-month").to_numpy().astype(float)
 
AGREEMENT = pd.read_csv(METRICS / "agreement.csv")
COEFS = pd.read_csv(METRICS / "coefficients.csv", index_col=0)
ABLATION = pd.read_csv(METRICS / "ablation.csv")
SENSITIVITY = pd.read_csv(METRICS / "sensitivity.csv")
LOSS_HISTORY = json.loads((METRICS / "loss_history.json").read_text())
 
DISTRIBUTION_COLUMNS = ["Churn (target)", "tenure", "log_tenure", "MonthlyCharges", "TotalCharges",
                        "Contract", "contract_months", "PaymentMethod", "electronic_check",
                        "InternetService", "SeniorCitizen", "Partner", "Dependents",
                        "PaperlessBilling", "PhoneService", "MultipleLines", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                        "StreamingMovies", "gender"]

 
# TAB 1 — MODEL COMPARISON
def metrics_table():
    rows = [evaluate(n, Y_TEST, p, BILLS, THRESHOLD) for n, p in PROBS.items()]
    rows.append(evaluate("Naive baseline", Y_TEST, BASELINE_PROBS, BILLS, THRESHOLD))
    return pd.DataFrame(rows).round(4)
 
 
def calibration_plot():
    """Prediction vs actual: customers grouped by predicted probability (10 bins);
    a well-calibrated model sits on the diagonal."""
    fig = go.Figure()
    for name, probs in PROBS.items():
        bins = pd.cut(probs, np.linspace(0, 1, 11), include_lowest=True)
        grouped = pd.DataFrame({"p": probs, "y": Y_TEST, "bin": bins}).groupby("bin", observed=True)
        fig.add_trace(go.Scatter(x=grouped["p"].mean(), y=grouped["y"].mean(),
                                 mode="lines+markers", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title="Predicted vs. actual churn rate (test set, 10 groups; "
                            "the three model lines overlap because the models agree)",
                      xaxis_title="Average predicted probability", yaxis_title="Actual churn rate")
    return fig
 
 
def strip_plot():
    long_df = pd.concat([
        pd.DataFrame({"Model": n, "Actual": pd.Series(Y_TEST).map(LABELS), "Predicted probability": p})
        for n, p in PROBS.items()])
    fig = px.strip(long_df, x="Actual", y="Predicted probability", color="Actual", facet_col="Model",
                   title="Predicted probability for each test customer, by actual outcome")
    fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4))
    fig.add_hline(y=THRESHOLD, line_dash="dash", annotation_text=f"threshold {THRESHOLD}")
    fig.update_layout(showlegend=False)
    return fig
 
 
def loss_curve_plot():
    optimum = LOSS_HISTORY["scikit-learn optimum"]
    fig = go.Figure()
    for name in ["Manual PyTorch", "Standard PyTorch"]:
        gap = np.clip(np.array(LOSS_HISTORY[name]) - optimum, 1e-15, None)
        fig.add_trace(go.Scatter(y=gap, mode="lines", name=name))
    fig.update_layout(title="Training objective minus scikit-learn's minimum (log scale; "
                            "the flat line at 1e-15 is the limit of computer precision)",
                      xaxis_title="Epoch", yaxis_title="Gap to minimum", yaxis_type="log")
    return fig

 
# TAB 2 — DISTRIBUTIONS
def distribution_plot(column: str):
    df = TRAIN_DF.assign(Outcome=TRAIN_DF["Churn"].map(LABELS))
    if column == "Churn (target)":
        counts = df["Outcome"].value_counts().reset_index()
        return px.bar(counts, x="Outcome", y="count", title="Target: churned vs. stayed (training set)")
    if pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() > 10:
        return px.histogram(df, x=column, color="Outcome", barmode="overlay", nbins=40,
                            title=f"{column} by outcome (training set)")
    rate = df.groupby(column)["Churn"].agg(["mean", "size"]).reset_index()
    rate["Churn rate %"] = (rate["mean"] * 100).round(1)
    fig = px.bar(rate, x=column, y="Churn rate %", text="Churn rate %", hover_data={"size": True},
                 title=f"Churn rate by {column} (training set)")
    fig.update_xaxes(type="category")
    return fig
 
 
# TAB 3 — THRESHOLD AND BUSINESS VALUE
def threshold_view(model_name: str, threshold: float):
    probs = PROBS[model_name]
    contacted = probs >= threshold
    tn = int((~contacted & (Y_TEST == 0)).sum()); fp = int((contacted & (Y_TEST == 0)).sum())
    fn = int((~contacted & (Y_TEST == 1)).sum()); tp = int((contacted & (Y_TEST == 1)).sum())
 
    z = [[tn, fp], [fn, tp]]
    cm = go.Figure(go.Heatmap(z=z, text=z, texttemplate="%{text}", colorscale="Blues", showscale=False,
                              x=["Not contacted", "Contacted"], y=["Actually stayed", "Actually churned"]))
    cm.update_layout(title=f"{model_name} — confusion matrix @ {threshold:.2f}",
                     yaxis_autorange="reversed")
 
    gained = META["save_rate"] * (META["months_kept"] - META["offer_months"]) * BILLS[contacted & (Y_TEST == 1)].sum()
    wasted = META["offer_months"] * BILLS[contacted & (Y_TEST == 0)].sum()
    text = (
        f"### Expected net value of the campaign: ${gained - wasted:,.0f}\n"
        f"Contacted {tp + fp} of {len(Y_TEST)} customers. "
        f"Revenue kept from {tp} contacted churners: **${gained:,.0f}** · "
        f"free months given to {fp} customers who would have stayed: **${wasted:,.0f}** · "
        f"{fn} churners not contacted (no change vs. no campaign)."
    )
 
    grid = np.round(np.arange(0.0, 1.001, 0.01), 2)
    curve = go.Figure(go.Scatter(x=grid, y=[campaign_value(Y_TEST, probs, BILLS, t) for t in grid],
                                 mode="lines", name="Model"))
    for label, value in [("contact everyone", campaign_value(Y_TEST, np.ones(len(Y_TEST)), BILLS, 0.5)),
                         ("contact all month-to-month", campaign_value(Y_TEST, MONTH_TO_MONTH, BILLS, 0.5))]:
        curve.add_hline(y=value, line_dash="dot", annotation_text=label)
    curve.add_vline(x=THRESHOLD, line_dash="dash", annotation_text=f"chosen on validation ({THRESHOLD})")
    curve.add_vline(x=threshold, line_color="red", annotation_text="slider", annotation_position="bottom right")
    curve.update_layout(title="Expected net value across thresholds (test set)",
                        xaxis_title="Threshold", yaxis_title="Net value ($)", showlegend=False)
    return cm, text, curve
 

# LAYOUT
with gr.Blocks(title="Telco Churn Dashboard") as demo:
    gr.Markdown("# Telco churn — who should get a retention offer?")
    gr.Markdown(
        f"One logistic regression trained three ways on the same split, with the same L2 penalty "
        f"(C = {META['C']}). The threshold **{THRESHOLD}** was chosen on the validation set to "
        f"maximise the campaign's expected net value. Everything is loaded from `artifacts/`; nothing is retrained."
    )
 
    with gr.Tab("Model comparison"):
        gr.Dataframe(metrics_table(), label=f"Test-set metrics @ threshold {THRESHOLD}")
        gr.Plot(calibration_plot())
        gr.Plot(strip_plot())
        gr.Markdown("### Do the three implementations agree?")
        gr.Dataframe(AGREEMENT.assign(**{c: AGREEMENT[c].map("{:.1e}".format)
                                         for c in AGREEMENT.columns if c != "pair"}),
                     label="Largest differences between implementations")
        gr.Plot(loss_curve_plot())
        gr.Dataframe(COEFS.round(4).reset_index(names="feature"), label="Weights (standardised features)")
 
    with gr.Tab("Feature & target distributions"):
        column = gr.Dropdown(DISTRIBUTION_COLUMNS, value="Churn (target)", label="Column")
        dist = gr.Plot()
        column.change(distribution_plot, column, dist)
        demo.load(distribution_plot, column, dist)
        gr.Dataframe(ABLATION.round(4), label="Feature ablation (5-fold cross-validation on train + validation)")
 
    with gr.Tab("Threshold & business value"):
        gr.Markdown(
            f"Offer: **{META['offer_months']} free month**; it only costs revenue if the customer would otherwise have paid. "
            f"A saved churner stays **{META['months_kept']} more months**. "
            f"The offer keeps **{META['save_rate']:.0%}** of the churners it reaches. "
            f"Values use each customer's own monthly bill. Break-even probability: **{META['break_even']:.3f}**."
        )
        model = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
        slider = gr.Slider(0.0, 1.0, value=THRESHOLD, step=0.01, label="Decision threshold")
        with gr.Row():
            cm_plot = gr.Plot()
            value_curve = gr.Plot()
        value_md = gr.Markdown()
        outputs = [cm_plot, value_md, value_curve]
        model.change(threshold_view, [model, slider], outputs)
        slider.change(threshold_view, [model, slider], outputs)
        demo.load(threshold_view, [model, slider], outputs)
        gr.Dataframe(SENSITIVITY.round(3), label="If the save rate were different (test set, $)")
 
 
if __name__ == "__main__":
    demo.launch()