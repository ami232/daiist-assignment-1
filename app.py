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
import joblib
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

sk_model = joblib.load("models/sklearn_model.joblib")
scaler = joblib.load("models/scaler.joblib")

manual_weights = torch.load("models/manual_weights.pt")
manual_bias = torch.load("models/manual_bias.pt")

class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

n_features = manual_weights.shape[0]
nn_model = LogisticRegressionModel(n_features)
nn_model.load_state_dict(torch.load("models/nn_model_state.pt"))
nn_model.eval()

data = np.load("models/test_results.npz", allow_pickle=True)
ids_test = data["ids_test"]
y_test = data["y_test"]
sk_probs = data["sk_probs"]
manual_probs = data["manual_probs"]
nn_probs = data["nn_probs"]
tenure = data["tenure"]
monthly_charges = data["monthly_charges"]
total_services = data["total_services"]
contract_risk = data["contract_risk"]

print("Loaded all models and test data successfully")
print("Test set size:", len(y_test))
print("Sklearn probs sample:", sk_probs[:5])

def get_metrics(probs, y_true, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    return acc, auc

baseline_acc = 1 - y_test.mean()
sk_acc, sk_auc = get_metrics(sk_probs, y_test)
manual_acc, manual_auc = get_metrics(manual_probs, y_test)
nn_acc, nn_auc = get_metrics(nn_probs, y_test)

def plot_comparison():
    labels = ["Baseline", "Sklearn", "Manual PyTorch", "Standard PyTorch"]
    accs = [baseline_acc, sk_acc, manual_acc, nn_acc]

    fig, ax = plt.subplots()
    ax.bar(labels, accs, color=["gray", "orchid", "orchid", "orchid"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison: Accuracy on Test Set")
    return fig

def plot_prediction_vs_actual():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    model_probs = [("Sklearn", sk_probs), ("Manual PyTorch", manual_probs), ("Standard PyTorch", nn_probs)]

    for ax, (name, probs) in zip(axes, model_probs):
        ax.hist(probs[y_test == 0], bins=20, alpha=0.7, label="Actual: No Churn", color="orchid")
        ax.hist(probs[y_test == 1], bins=20, alpha=0.7, label="Actual: Churn", color="indigo")
        ax.set_title(name)
        ax.set_xlabel("Predicted Probability")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Number of Customers")
    fig.tight_layout()
    return fig

feature_map = {
    "Tenure (months)": tenure,
    "Monthly Charges ($)": monthly_charges,
    "Total Services Subscribed": total_services,
    "Contract Risk (0=month-to-month, 2=two-year)": contract_risk,
}

def plot_feature_distribution(feature_name):
    values = feature_map[feature_name]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values[y_test == 0], bins=20, alpha=0.7, label="No Churn", color="orchid")
    ax.hist(values[y_test == 1], bins=20, alpha=0.7, label="Churn", color="indigo")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Number of Customers")
    ax.set_title(f"{feature_name}, split by actual churn")
    ax.legend()
    return fig

from matplotlib.colors import LinearSegmentedColormap
purple_cmap = LinearSegmentedColormap.from_list("custom", ["white", "orchid", "indigo"])

def update_threshold(threshold):
    preds = (sk_probs >= threshold).astype(int)

    tp = ((preds == 1) & (y_test == 1)).sum()
    tn = ((preds == 0) & (y_test == 0)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap=purple_cmap, vmax=cm.max() * 1.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: No Churn", "Pred: Churn"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: No Churn", "Actual: Churn"])
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()

    offer_cost = 50
    avg_monthly = monthly_charges.mean()
    value_at_risk_cost = 3 * avg_monthly

    total_cost = fn * value_at_risk_cost + fp * offer_cost
    cost_text = f"<h2 style='font-size:32px'>Business cost at this threshold: ${total_cost:,.0f}</h2><p style='font-size:18px'>(FN={fn}, FP={fp})</p>"
    return fig, cost_text

def plot_cost_curve():
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []
    for t in thresholds:
        preds = (sk_probs >= t).astype(int)
        fn = ((preds == 0) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        cost = fn * (3 * monthly_charges.mean()) + fp * 50
        costs.append(cost)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, costs, color="indigo", linewidth=2)
    ax.axvline(best_threshold, color="orchid", linestyle="--", linewidth=2, label=f"Best: {best_threshold:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Business Cost ($)")
    ax.set_title("Cost vs. Threshold")
    ax.legend()
    return fig, best_threshold, best_cost

with gr.Blocks() as demo:
    gr.HTML("<h1 style='font-size:42px'>Customer Churn Prediction Dashboard</h1>")

    with gr.Tab("Model Comparison"):
        gr.Plot(value=plot_comparison())
        gr.HTML(
            f"""
            <table style='font-size:20px; border-collapse: collapse; width: 100%;'>
                <tr><th style='text-align:left; padding:8px;'>Model</th><th style='padding:8px;'>Accuracy</th><th style='padding:8px;'>ROC AUC</th></tr>
                <tr><td style='padding:8px;'>Baseline (always "no churn")</td><td style='padding:8px;'>{baseline_acc:.3f}</td><td style='padding:8px;'>—</td></tr>
                <tr><td style='padding:8px;'>Scikit-learn</td><td style='padding:8px;'>{sk_acc:.3f}</td><td style='padding:8px;'>{sk_auc:.3f}</td></tr>
                <tr><td style='padding:8px;'>Manual PyTorch</td><td style='padding:8px;'>{manual_acc:.3f}</td><td style='padding:8px;'>{manual_auc:.3f}</td></tr>
                <tr><td style='padding:8px;'>Standard PyTorch</td><td style='padding:8px;'>{nn_acc:.3f}</td><td style='padding:8px;'>{nn_auc:.3f}</td></tr>
            </table>
            """
        )
        gr.Plot(value=plot_prediction_vs_actual())
        gr.HTML(
            "<p style='font-size:17px'>Each panel shows one model's predicted probabilities, split by what actually happened. "
            "A model with real signal should show the two colors clearly separated, "
            "'No Churn' customers clustered toward the left, 'Churn' customers shifted right.</p>"
          )

    with gr.Tab("Distributions"):
        feature_dropdown = gr.Dropdown(
            choices=list(feature_map.keys()),
            value="Tenure (months)",
            label="Choose a feature to view",
        )
        dist_plot = gr.Plot()

        feature_dropdown.change(fn=plot_feature_distribution, inputs=feature_dropdown, outputs=dist_plot)
        demo.load(fn=plot_feature_distribution, inputs=feature_dropdown, outputs=dist_plot)

        gr.HTML(
            "<p style='font-size:17px'>Each feature's distribution, split by whether the customer actually churned. "
            "A useful feature should show the two colors shifted apart, not perfectly overlapping.</p>"
        )

    with gr.Tab("Threshold & Cost"):
        slider = gr.Slider(0, 1, value=0.5, step=0.01, label="Decision Threshold")
        cm_plot = gr.Plot()
        cost_display = gr.Markdown()

        slider.change(fn=update_threshold, inputs=slider, outputs=[cm_plot, cost_display])
        demo.load(fn=update_threshold, inputs=slider, outputs=[cm_plot, cost_display])
        
        gr.Markdown("---")
        cost_curve_plot = gr.Plot()
        best_info = gr.Markdown()

        def load_cost_curve():
            fig, best_t, best_c = plot_cost_curve()
            return fig, f"<h2 style='font-size:32px'>Optimal threshold: {best_t:.2f}</h2><p style='font-size:18px'>(lowest cost: ${best_c:,.0f})</p>"

        demo.load(fn=load_cost_curve, inputs=None, outputs=[cost_curve_plot, best_info])

if __name__ == "__main__":
    demo.launch()