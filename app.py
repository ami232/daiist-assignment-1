# 1. LIBRARIES & LOAD SAVED ARTIFACTS

from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

import plotly.graph_objects as go


# 0. DASHBOARD DESIGN (just for visuals)

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue", # i chose blue in order to match with the teleco style
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="md"
)

custom_css = """
.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
    background: #FFFFFF !important;
}

/* Main dashboard header */
.dashboard-header {
    background: #102A43;
    padding: 32px 36px;
    border-radius: 16px;
    margin-bottom: 20px;
}

.dashboard-header h1 {
    color: #FFFFFF !important;
    font-size: 30px !important;
    margin-bottom: 8px !important;
}

.dashboard-header p {
    color: #DBEAFE !important;
    font-size: 14px !important;
}

/* KPI cards */
.kpi-card {
    background: #F1F6FD;
    border: 1px solid #D7E5F5;
    border-radius: 12px;
    padding: 20px;
}

.kpi-label {
    color: #526B87;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    color: #102A43;
    font-size: 29px;
    font-weight: 700;
    margin-top: 8px;
}

.kpi-description {
    color: #64748B;
    font-size: 12px;
    margin-top: 4px;
}

/* Section headings */
.section-heading h2 {
    color: #102A43 !important;
    font-size: 22px !important;
}

/* Tabs */
.tab-nav button.selected {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
}
"""

# ------------------------------------------------------------


ARTIFACT_DIR = Path("artifacts")

# Load trained scikit-learn model and preprocessor
sklearn_model = joblib.load(
    ARTIFACT_DIR / "sklearn_model.joblib"
)

preprocessor = joblib.load(
    ARTIFACT_DIR / "preprocessor.joblib"
)

# Load saved test data
X_test = pd.read_csv(
    ARTIFACT_DIR / "X_test.csv"
)

y_test = np.load(
    ARTIFACT_DIR / "y_test.npy"
)

# Load predicted probabilities from all three models
saved_predictions = np.load(
    ARTIFACT_DIR / "test_probabilities.npz"
)

model_probabilities = {
    "Scikit-learn": saved_predictions["sklearn"],
    "Manual PyTorch": saved_predictions["manual_pytorch"],
    "Standard PyTorch": saved_predictions["standard_pytorch"]
}

# Load business assumptions
business_settings = joblib.load(
    ARTIFACT_DIR / "business_settings.joblib"
)

CONTACT_COST = business_settings["contact_cost"]
MISSED_CHURN_COST = business_settings["missed_churn_cost"]
BUSINESS_THRESHOLD = business_settings["threshold"]

print("All saved artifacts loaded successfully.")
print("Test observations:", len(y_test))
print("Available models:", list(model_probabilities.keys()))

# the code until now recovers all the files we have generated during training 
# in that way, if someone changes the threshold in Gradio doesn't need to re-execute the models, we will just use the saved probabilities

#---------------------------------------------------------------------------------------
# 2. MODEL COMPARISON

# this code runs our 3 models and converts the probabilities into predictions usingthe 0.5 threshold and calculating the 5 metrics, creating afterwards a table with the results

def compare_models():

    results = []

    for model_name, probabilities in model_probabilities.items():

        predictions = (probabilities >= 0.5).astype(int)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Log Loss": log_loss(y_test, probabilities)
        })

    return pd.DataFrame(results).round(4)


comparison_df = compare_models()


#---------------------------------------------------------------------------------------
# 3. PREDICTED VS ACTUAL CHURN PLOT

def predicted_vs_actual_plot(model_name):

    probabilities = model_probabilities[model_name]

    plot_df = pd.DataFrame({
        "Predicted Probability": probabilities,
        "Actual Churn": y_test
    })

    # Divide customers into probability ranges
    plot_df["Probability Bin"] = pd.cut(
        plot_df["Predicted Probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True
    )

    summary = plot_df.groupby(
        "Probability Bin", observed=False
    ).agg(
        predicted=("Predicted Probability", "mean"),
        actual=("Actual Churn", "mean"),
        customers=("Actual Churn", "size")
    ).dropna()

    fig = go.Figure()

    # Actual observed churn rate
    fig.add_trace(go.Scatter(
        x=summary["predicted"],
        y=summary["actual"],
        mode="lines+markers",
        name="Actual churn rate",
        line=dict(color="#2563EB", width=3),
        marker=dict(size=8, color="#2563EB")
    ))

    # Perfect calibration reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line=dict(
            color="#102A43",
            dash="dash",
            width=2
        )
    ))


    fig.update_layout(
        title=f"Predicted vs Actual Churn - {model_name}",
        xaxis_title="Average predicted churn probability",
        yaxis_title="Observed churn rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )

    return fig

# the discontinued line represents the perfect callibration

#---------------------------------------------------------------------------------------
# 3. DATA EXPLORATION
import plotly.express as px

def feature_distribution_plot(feature):

    plot_df = X_test.copy()
    plot_df["Churn"] = np.where(y_test == 1, "Yes", "No")

    # Special case: target variable
    if feature == "Churn":
        fig = px.histogram(
            plot_df,
            x="Churn",
            color="Churn",
            title="Customer Churn Distribution",
            color_discrete_map={
                "No": "#2563EB",
                "Yes": "#102A43"
        })

    # Numerical variables
    elif feature in ["tenure", "MonthlyCharges", "TotalCharges"]:
        fig = px.histogram(
            plot_df,
            x=feature,
            color="Churn",
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            title=f"Distribution of {feature} by Churn",
            color_discrete_map={
                "No": "#2563EB",
                "Yes": "#102A43"
        }
        )

    # Categorical variables
    else:
        fig = px.histogram(
            plot_df,
            x=feature,
            color="Churn",
            barmode="group",
            title=f"Distribution of {feature} by Churn",
            color_discrete_map={
                "No": "#2563EB",
                "Yes": "#102A43"
        }
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=feature,
        yaxis_title="Number of customers"
    )

    return fig

# this creates different charts depending on the selected variable (ex. tenure)

#---------------------------------------------------------------------------------------
# 4. BUSINESS DECISION ANALYSIS

# this receives 2 parameters: model name and threshold and calculates the predictions, the confussion matrix, precision, recall and the total estimated cost
# the formula is the same as the one we used in train.py -> total cost = 10 · (TP + FP) + 100 · (FN)

from sklearn.metrics import confusion_matrix

def business_decision_analysis(model_name, threshold):

    # Load probabilities from the selected trained model
    probabilities = model_probabilities[model_name]

    # Apply the selected classification threshold
    predictions = (probabilities >= threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    # Calculate evaluation metrics
    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    # Calculate estimated business cost
    contact_cost = CONTACT_COST * (tp + fp)
    missed_churn_cost = MISSED_CHURN_COST * fn
    total_cost = contact_cost + missed_churn_cost

    # Create confusion matrix heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted No Churn", "Predicted Churn"],
        y=["Actual No Churn", "Actual Churn"],
        text=[[tn, fp], [fn, tp]],
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#EFF6FF"],
            [0.5, "#60A5FA"],
            [1.0, "#102A43"]
        ],
        showscale=False
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        yaxis_autorange="reversed",
        template="plotly_white"
        )

    # Business summary
    summary = (
        f"### Business Results\n"
        f"**Threshold:** {threshold:.2f}\n\n"
        f"**Customers contacted:** {tp + fp} / {len(y_test)} "
        f"({(tp + fp) / len(y_test) * 100:.1f}%)\n\n"
        f"**Precision:** {precision:.2%}\n\n"
        f"**Recall:** {recall:.2%}\n\n"
        f"**Contact cost:** €{contact_cost:,.0f}\n\n"
        f"**Missed churn cost:** €{missed_churn_cost:,.0f}\n\n"
        f"### Total Estimated Cost: €{total_cost:,.0f}"
    )

    return fig, summary


#---------------------------------------------------------------------------------------
# 3. INITIAL GRADIO INTERFACE

with gr.Blocks(
    title="Telco Customer Churn Dashboard",
    theme=custom_theme,
    css=custom_css
) as demo:

    # dashboard header
    gr.HTML("""
        <div class="dashboard-header">
            <h1>Telco Customer Churn Analytics</h1>
            <p>
                Customer Retention Intelligence |
                Predictive Modeling & Business Impact
            </p>
        </div>
    """)

    # key performance indicators
    with gr.Row():

        gr.HTML(f"""
            <div class="kpi-card">
                <div class="kpi-label">Test Customers</div>
                <div class="kpi-value">{len(y_test):,}</div>
                <div class="kpi-description">
                    Independent evaluation sample
                </div>
            </div>
        """)

        gr.HTML(f"""
            <div class="kpi-card">
                <div class="kpi-label">Observed Churn Rate</div>
                <div class="kpi-value">{np.mean(y_test):.1%}</div>
                <div class="kpi-description">
                    Percentage of customers who left
                </div>
            </div>
        """)

        gr.HTML(f"""
            <div class="kpi-card">
                <div class="kpi-label">Business Threshold</div>
                <div class="kpi-value">{BUSINESS_THRESHOLD:.0%}</div>
                <div class="kpi-description">
                    Selected on validation data
                </div>
            </div>
        """)

    # 4.1. MODEL COMPARISON TAB
    with gr.Tab("Model Comparison"):

        gr.Markdown("## Three-Model Performance Comparison")

        gr.Dataframe(
            value=comparison_df,
            interactive=False,
            label="Test Set Metrics (Threshold = 0.5)"
        )

        gr.Markdown("## Predicted vs Actual Churn")

        model_selector = gr.Dropdown(
            choices=list(model_probabilities.keys()),
            value="Scikit-learn",
            label="Select Model"
        )

        prediction_plot = gr.Plot(
            value=predicted_vs_actual_plot("Scikit-learn")
        )

        model_selector.change(
            fn=predicted_vs_actual_plot,
            inputs=model_selector,
            outputs=prediction_plot
        )
    # 4.1. MODEL COMPARISON TAB
    with gr.Tab("Data Exploration"):

        gr.Markdown("## Customer Data Exploration")

        gr.Markdown(
            "Explore customer characteristics and how they differ "
            "between churned and retained customers."
        )

        feature_selector = gr.Dropdown(
            choices=[
                "Churn",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "Contract",
                "InternetService",
                "PaymentMethod"
            ],
            value="Churn",
            label="Select Variable"
        )

        distribution_plot = gr.Plot(
            value=feature_distribution_plot("Churn")
        )

        feature_selector.change(
            fn=feature_distribution_plot,
            inputs=feature_selector,
            outputs=distribution_plot
        )

    # 4.3. BUSINESS DECISION 
    with gr.Tab("Business Decision"):

        gr.Markdown("## Retention Strategy & Business Cost")

        gr.Markdown(
            "Explore how changing the classification threshold "
            "affects customer retention decisions and estimated costs. "
            "All financial values are hypothetical assumptions."
        )

        business_model_selector = gr.Dropdown(
            choices=list(model_probabilities.keys()),
            value="Scikit-learn",
            label="Select Model"
        )

        threshold_slider = gr.Slider(
            minimum=0.01,
            maximum=0.99,
            value=BUSINESS_THRESHOLD,
            step=0.01,
            label="Classification Threshold"
        )

        with gr.Row():

            confusion_plot = gr.Plot(
                value=business_decision_analysis(
                    "Scikit-learn", BUSINESS_THRESHOLD
                )[0]
            )

            business_summary = gr.Markdown(
                value=business_decision_analysis(
                    "Scikit-learn", BUSINESS_THRESHOLD
                )[1]
            )

        # Update outputs when threshold changes
        threshold_slider.change(
            fn=business_decision_analysis,
            inputs=[business_model_selector, threshold_slider],
            outputs=[confusion_plot, business_summary]
        )

        # Update outputs when model changes
        business_model_selector.change(
            fn=business_decision_analysis,
            inputs=[business_model_selector, threshold_slider],
            outputs=[confusion_plot, business_summary]
        )

if __name__ == "__main__":
    demo.launch()