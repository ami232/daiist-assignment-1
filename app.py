"""
Gradio dashboard for the "should an investor buy this house" classifier.

IMPORTANT: this file only LOADS artifacts saved by train.py and runs
inference (a forward pass) with them. Nothing here trains or fits
anything -- if train.py hasn't been run yet, this will fail loudly
rather than silently retrain.

Views:
  1. Model comparison  -- metrics table + predicted-probability-vs-actual
                           plot for scikit-learn / manual PyTorch / nn.Module
  2. Distributions      -- feature and target (buy/no-buy) distributions
  3. Threshold explorer -- slider over the decision threshold, live
                           confusion matrix + a business-cost estimate
"""

import json
import os
import sys

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "src")
from models import (  # noqa: E402
    predict_sklearn, predict_pytorch_manual, predict_pytorch_module,
    load_pytorch_manual, load_pytorch_module,
)

MODELS_DIR = "models"
DILIGENCE_COST = 3000  # assumed $ cost of investigating/appraising one flagged
                        # listing (inspection, comps pull, agent time). A
                        # documented assumption, not derived from the data.

MODEL_LABELS = {
    "sklearn": "scikit-learn",
    "pytorch_manual": "Manual PyTorch",
    "pytorch_module": "PyTorch nn.Module",
}


def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing '{path}'. Run `uv run python main.py train` first -- "
            "the app only loads saved artifacts, it never trains."
        )
    return path


def load_everything():
    for f in [
        "preprocessor.joblib", "sklearn_model.joblib", "pytorch_manual.pt",
        "pytorch_module.pt", "n_features.json", "metrics.json",
        "label_report.json", "train_raw.csv", "test_raw.csv",
    ]:
        _require(os.path.join(MODELS_DIR, f))

    preprocessor = joblib.load(f"{MODELS_DIR}/preprocessor.joblib")
    sklearn_model = joblib.load(f"{MODELS_DIR}/sklearn_model.joblib")

    with open(f"{MODELS_DIR}/n_features.json") as f:
        n_features = json.load(f)["n_features"]
    manual_params = load_pytorch_manual(f"{MODELS_DIR}/pytorch_manual.pt")
    module_model = load_pytorch_module(f"{MODELS_DIR}/pytorch_module.pt", n_features)

    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{MODELS_DIR}/label_report.json") as f:
        label_report = json.load(f)

    train_raw = pd.read_csv(f"{MODELS_DIR}/train_raw.csv", index_col=0)
    test_raw = pd.read_csv(f"{MODELS_DIR}/test_raw.csv", index_col=0)

    # Run inference once at startup (NOT training) so every view has
    # test-set probabilities ready to slice/plot.
    X_test = preprocessor.transform(test_raw)
    y_test = test_raw["buy"].to_numpy()
    probs = {
        "sklearn": predict_sklearn(sklearn_model, X_test),
        "pytorch_manual": predict_pytorch_manual(manual_params, X_test),
        "pytorch_module": predict_pytorch_module(module_model, X_test),
    }

    return {
        "preprocessor": preprocessor,
        "metrics": metrics,
        "label_report": label_report,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "y_test": y_test,
        "probs": probs,
    }


STATE = load_everything()


# ---------------------------------------------------------------------------
# View 1: model comparison
# ---------------------------------------------------------------------------
def metrics_table():
    rows = []
    for key, label in [("naive_baseline", "Naive baseline"), *MODEL_LABELS.items()]:
        m = STATE["metrics"][key]
        rows.append({
            "Model": label,
            "Accuracy": round(m["accuracy"], 3),
            "Precision": round(m["precision"], 3),
            "Recall": round(m["recall"], 3),
            "F1": round(m["f1"], 3),
            "ROC AUC": round(m["roc_auc"], 3),
        })
    return pd.DataFrame(rows)


def prediction_vs_actual_plot(model_key):
    y_true = STATE["y_test"]
    prob = STATE["probs"][model_key]
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=len(y_true))

    fig, ax = plt.subplots(figsize=(6, 4))
    correct = (prob >= 0.5).astype(int) == y_true
    ax.scatter(prob[correct], y_true[correct] + jitter[correct],
               alpha=0.5, s=14, color="#2a9d8f", label="Correctly classified (at 0.5)")
    ax.scatter(prob[~correct], y_true[~correct] + jitter[~correct],
               alpha=0.6, s=14, color="#e76f51", label="Misclassified (at 0.5)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No buy (0)", "Buy (1)"])
    ax.set_xlabel("Predicted probability of 'buy'")
    ax.set_title(f"Predicted probability vs. actual label -- {MODEL_LABELS[model_key]}")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# View 2: distributions
# ---------------------------------------------------------------------------
NUMERIC_FEATURE_CHOICES = [
    "SalePrice", "Gr Liv Area", "Overall Qual", "Overall Cond", "Year Built",
    "Lot Area", "Total Bsmt SF", "Garage Area", "house_age", "total_sqft",
    "total_bathrooms", "qual_x_cond",
]


def target_distribution_plot():
    combined = pd.concat([STATE["train_raw"]["buy"], STATE["test_raw"]["buy"]])
    counts = combined.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["No buy (0)", "Buy (1)"], counts.values, color=["#264653", "#e9c46a"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(v), ha="center")
    ax.set_title(f"Target distribution (buy rate = {combined.mean():.1%})")
    fig.tight_layout()
    return fig


def feature_distribution_plot(feature):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in [(0, "#264653"), (1, "#e9c46a")]:
        train = STATE["train_raw"]
        vals = train.loc[train["buy"] == label, feature].dropna()
        ax.hist(vals, bins=30, alpha=0.6, color=color,
                label=f"{'Buy' if label else 'No buy'} (n={len(vals)})")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count (training set)")
    ax.set_title(f"'{feature}' distribution by label")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# View 3: threshold explorer (confusion matrix + business cost)
# ---------------------------------------------------------------------------
def _potential_profit():
    test = STATE["test_raw"]
    profit = (test["expected_psf"] - test["price_per_sqft"]).clip(lower=0) * test["Gr Liv Area"]
    return profit.to_numpy()


def threshold_view(model_key, threshold):
    y_true = STATE["y_test"]
    prob = STATE["probs"][model_key]
    pred = (prob >= threshold).astype(int)
    profit = _potential_profit()

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: No buy", "Pred: Buy"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: No buy", "Actual: Buy"])
    ax.set_title(f"Confusion matrix @ threshold={threshold:.2f}")
    fig.tight_layout()

    tp_mask = (pred == 1) & (y_true == 1)
    fp_mask = (pred == 1) & (y_true == 0)
    fn_mask = (pred == 0) & (y_true == 1)

    n_flagged = int(tp_mask.sum() + fp_mask.sum())
    profit_captured = float(profit[tp_mask].sum())
    diligence_spent = n_flagged * DILIGENCE_COST
    net_value = profit_captured - diligence_spent
    missed_profit = float(profit[fn_mask].sum())

    summary = (
        f"### Estimated business impact @ threshold {threshold:.2f}\n\n"
        f"*(Assumes ${DILIGENCE_COST:,} in due-diligence cost per flagged listing "
        f"-- an assumption, not derived from the data.)*\n\n"
        f"| | |\n|---|---|\n"
        f"| Houses flagged as \"buy\" | {n_flagged} |\n"
        f"| Correctly identified deals (TP) | {int(tp_mask.sum())} |\n"
        f"| False alarms investigated for nothing (FP) | {int(fp_mask.sum())} |\n"
        f"| Real deals missed (FN) | {int(fn_mask.sum())} |\n"
        f"| Gross profit captured from correct calls | ${profit_captured:,.0f} |\n"
        f"| Due-diligence cost spent on all flagged listings | ${diligence_spent:,.0f} |\n"
        f"| **Net estimated value** | **${net_value:,.0f}** |\n"
        f"| Profit left on the table from missed deals (FN, informational) | ${missed_profit:,.0f} |\n"
    )
    return fig, summary


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks(title="Ames Investor Deal Classifier") as demo:
        gr.Markdown(
            "# Is this house a good deal?\n"
            "A logistic regression trained three ways (scikit-learn, a manual "
            "PyTorch loop, and `torch.nn.Module` + `optim`) to flag houses "
            "selling >10% below their neighborhood/quality/year comps. "
            "Trained on 2006-2008 sales, evaluated on 2009-2010 sales. "
            "All models below are loaded from disk -- nothing retrains here."
        )

        with gr.Tab("Model comparison"):
            gr.Markdown("### Test-set metrics (2009-2010 houses)")
            gr.Dataframe(metrics_table(), interactive=False)
            model_dd = gr.Dropdown(
                choices=[(label, key) for key, label in MODEL_LABELS.items()], value="sklearn",
                label="Model for the plot below",
            )
            pred_plot = gr.Plot()
            model_dd.change(prediction_vs_actual_plot, inputs=model_dd, outputs=pred_plot)
            demo.load(prediction_vs_actual_plot, inputs=model_dd, outputs=pred_plot)

        with gr.Tab("Distributions"):
            gr.Markdown("### Target distribution")
            gr.Plot(target_distribution_plot())
            gr.Markdown("### Feature distribution by label (training set)")
            feat_dd = gr.Dropdown(
                choices=NUMERIC_FEATURE_CHOICES, value="Overall Qual",
                label="Feature",
            )
            feat_plot = gr.Plot()
            feat_dd.change(feature_distribution_plot, inputs=feat_dd, outputs=feat_plot)
            demo.load(feature_distribution_plot, inputs=feat_dd, outputs=feat_plot)

        with gr.Tab("Threshold explorer"):
            gr.Markdown(
                "Move the threshold to see the trade-off between catching more "
                "deals (lower threshold, more false alarms) and being more "
                "conservative (higher threshold, more missed deals)."
            )
            with gr.Row():
                thresh_model_dd = gr.Dropdown(
                    choices=[(label, key) for key, label in MODEL_LABELS.items()], value="sklearn", label="Model",
                )
                thresh_slider = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Decision threshold")
            with gr.Row():
                cm_plot = gr.Plot()
                cost_md = gr.Markdown()

            def _update(model_key, threshold):
                fig, summary = threshold_view(model_key, threshold)
                return fig, summary

            thresh_model_dd.change(_update, inputs=[thresh_model_dd, thresh_slider], outputs=[cm_plot, cost_md])
            thresh_slider.change(_update, inputs=[thresh_model_dd, thresh_slider], outputs=[cm_plot, cost_md])
            demo.load(_update, inputs=[thresh_model_dd, thresh_slider], outputs=[cm_plot, cost_md])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()