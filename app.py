"""Gradio dashboard for the dead-stock classifier.

Loads the artefacts train.py wrote under models/ and serves four tabs:
  1. Data       — class balance and how cost / family / material / supplier relate to it
  2. Models     — how the three models score, rank (ROC) and agree (coefficients)
  3. Threshold  — move the decision threshold, watch the confusion matrix and the cost in DOP
  4. Scorer     — type a SKU, get three dead-stock probabilities side by side

This script NEVER trains. It loads from models/ and fails loudly if a file is missing.
The only raw data it reads is the CSV, and only to draw the Data tab's charts (the label is
rebuilt there for colouring the charts, never used as a feature).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend: we hand figures to Gradio, never display them
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_curve
import gradio as gr

from src import data
from src.data import FN_COST, FP_COST_CAP, FP_MARGIN_MULTIPLIER, fp_cost_per_sku
from src.features import build_features, family, material

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_CHOICES = ["LogReg", "Manual", "nn.Linear"]
# Column in test_predictions.csv holding each model's probability.
PROB_COL = {"LogReg": "prob_logreg", "Manual": "prob_manual", "nn.Linear": "prob_torch"}
# Key in metrics.json holding each model's block.
METRIC_KEY = {"LogReg": "logreg", "Manual": "manual", "nn.Linear": "torch_module"}


# ----------------------------------------------------------------------------- load artefacts
def _require(path: Path) -> Path:
    """Fail loudly: the dashboard loads trained models, it never retrains them."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run `uv run python main.py train` first.")
    return path


feature_fit = joblib.load(_require(MODELS_DIR / "feature_fit.joblib"))
logreg = joblib.load(_require(MODELS_DIR / "logreg.joblib"))
manual = torch.load(_require(MODELS_DIR / "manual.pt"))  # {"w": tensor, "b": tensor}
feature_names = json.loads(_require(MODELS_DIR / "feature_names.json").read_text())
metrics = json.loads(_require(MODELS_DIR / "metrics.json").read_text())
test_preds = pd.read_csv(_require(MODELS_DIR / "test_predictions.csv"))

# Rebuild the nn.Linear and load its saved weights.
torch_model = nn.Linear(len(feature_names), 1)
torch_model.load_state_dict(torch.load(_require(MODELS_DIR / "torch_module.pt")))
torch_model.eval()

# The cost-minimising threshold train.py picked on the validation slice.
THRESHOLD = float(metrics["threshold"]["chosen"])
Y_TEST = test_preds["y_true"].to_numpy(dtype=int)
# Per-SKU cost of a false positive, from the price and cost columns saved with the predictions.
FP_COST_TEST = fp_cost_per_sku(test_preds).to_numpy(dtype=float)

# Raw data, only for the Data tab's charts (reading, not retraining). The label is rebuilt
# here purely to colour the charts by class.
raw_df = data.load_raw()
raw_df["dead"] = data.make_target(raw_df)
raw_df["log_costo"] = np.log(raw_df["Costo"])
raw_df["family"] = raw_df["Descripción"].astype(str).map(family)
raw_df["material"] = raw_df["Descripción"].astype(str).map(material)


# ----------------------------------------------------------------------------- prediction
def _coefficients() -> dict[str, np.ndarray]:
    """The learned weight vector of each model, aligned to feature_names."""
    return {
        "LogReg": logreg.coef_.ravel(),
        "Manual": manual["w"].numpy(),
        "nn.Linear": torch_model.weight.detach().numpy().ravel(),
    }


def _intercepts() -> dict[str, float]:
    return {
        "LogReg": float(logreg.intercept_[0]),
        "Manual": float(manual["b"]),
        "nn.Linear": float(torch_model.bias.detach()),
    }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _predict_proba(X: np.ndarray) -> dict[str, np.ndarray]:
    """Each model's P(dead) for already-scaled feature rows."""
    coefs, biases = _coefficients(), _intercepts()
    out = {name: _sigmoid(X @ coefs[name] + biases[name]) for name in ("LogReg", "Manual")}
    with torch.no_grad():
        logits = torch_model(torch.tensor(X, dtype=torch.float32)).squeeze(1)
        out["nn.Linear"] = torch.sigmoid(logits).numpy()
    return out


def _confusion(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """(tn, fp, fn, tp) at a threshold. Same arithmetic as train.py, so numbers match."""
    y_pred = (prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tn, fp, fn, tp


def _cost(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> tuple[float, float]:
    """(false-positive cost, false-negative cost) in DOP at a threshold: each false positive
    costs that SKU's own FP_COST_TEST entry, each false negative the flat FN_COST."""
    y_pred = (prob >= threshold).astype(int)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return float(FP_COST_TEST[fp_mask].sum()), float(FN_COST * fn)


def _fig(figsize):
    """A figure NOT registered with pyplot, so slider/dropdown callbacks can make as many as
    they like without pyplot holding on to them (and warning after 20)."""
    fig = Figure(figsize=figsize)
    return fig, fig.subplots()


# ----------------------------------------------------------------------------- Tab 1: Data
def plot_class_balance():
    counts = raw_df["dead"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = _fig(figsize=(5, 4))
    bars = ax.bar(["live (0)", "dead (1)"], counts.values, color=["tab:blue", "tab:red"])
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}  ({c / counts.sum():.0%})",
                ha="center", va="bottom")
    ax.set_ylabel("SKUs")
    ax.set_title("Class balance (dead = no stock AND no reorder point)")
    fig.tight_layout()
    return fig


def plot_logcost_by_class():
    fig, ax = _fig(figsize=(6, 4))
    bins = np.linspace(raw_df["log_costo"].min(), raw_df["log_costo"].max(), 40)
    for label, name, color in ((0, "live", "tab:blue"), (1, "dead", "tab:red")):
        ax.hist(raw_df.loc[raw_df["dead"] == label, "log_costo"], bins=bins, alpha=0.5,
                label=name, color=color)
    ax.set_xlabel("log(Costo)")
    ax.set_ylabel("count")
    ax.set_title("Log cost by class")
    ax.legend()
    fig.tight_layout()
    return fig


def _rate_bar(col: str, title: str):
    """Dead rate per category, bars ordered by rate, count written on each bar."""
    g = raw_df.groupby(col)["dead"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    fig, ax = _fig(figsize=(7, 4))
    bars = ax.bar(g.index, g["mean"], color="tab:red", alpha=0.7)
    for b, n in zip(bars, g["size"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n}", ha="center",
                va="bottom", fontsize=7)
    ax.axhline(raw_df["dead"].mean(), color="k", linestyle="--", linewidth=1,
               label=f"overall {raw_df['dead'].mean():.0%}")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("dead rate")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_family_rate():
    return _rate_bar("family", "Dead rate by product family")


def plot_material_rate():
    return _rate_bar("material", "Dead rate by material")


def plot_supplier_scatter():
    g = raw_df.groupby("Suplidor (ID)")["dead"].agg(["mean", "size"])
    fig, ax = _fig(figsize=(6, 4))
    ax.scatter(g["size"], g["mean"], s=20, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("SKUs per supplier (log scale) — the freq_supplier feature")
    ax.set_ylabel("dead rate")
    ax.set_title("Supplier size vs dead rate (one point per supplier)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Tab 2: Models
def metrics_table() -> pd.DataFrame:
    def row(name, block, tag):
        r = block[tag]
        return {"model": name, "threshold": r["threshold"],
                "accuracy": round(r["accuracy"], 3), "precision": round(r["precision"], 3),
                "recall": round(r["recall"], 3), "F1": round(r["f1"], 3),
                "ROC-AUC": round(r["roc_auc"], 3), "expected cost (DOP)": int(r["expected_cost"])}
    blocks = [
        ("Baseline: majority", metrics["baseline_majority"]["test"]),
        ("Baseline: Costo > median", metrics["baseline_cost_median"]["test"]),
        ("LogReg", metrics["logreg"]["test"]),
        ("Manual", metrics["manual"]["test"]),
        ("nn.Linear", metrics["torch_module"]["test"]),
    ]
    rows = [row(n, b, "at_0.5") for n, b in blocks] + [row(n, b, "at_threshold") for n, b in blocks]
    return pd.DataFrame(rows)


def plot_roc():
    fig, ax = _fig(figsize=(5.5, 5))
    for name in MODEL_CHOICES:
        prob = test_preds[PROB_COL[name]].to_numpy()
        fpr, tpr, _ = roc_curve(Y_TEST, prob)
        auc = metrics[METRIC_KEY[name]]["test"]["at_0.5"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("false positive rate (live items flagged)")
    ax.set_ylabel("true positive rate (dead items caught)")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_prob_hist(model_name: str):
    prob = test_preds[PROB_COL[model_name]].to_numpy()
    fig, ax = _fig(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    for label, name, color in ((0, "live", "tab:blue"), (1, "dead", "tab:red")):
        ax.hist(prob[Y_TEST == label], bins=bins, alpha=0.5, label=f"true {name}", color=color)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="0.5")
    ax.axvline(THRESHOLD, color="k", linestyle="--", linewidth=1, label=f"chosen {THRESHOLD:.2f}")
    ax.set_xlabel("predicted P(dead)")
    ax.set_ylabel("count")
    ax.set_title(f"Predicted probability by true class — {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_coefficients():
    coefs = _coefficients()
    order = np.argsort(np.abs(coefs["LogReg"]))[::-1]  # most influential first
    names = [feature_names[i] for i in order]
    y = np.arange(len(names))
    fig, ax = _fig(figsize=(7, 11))
    for offset, name in zip((-0.25, 0.0, 0.25), MODEL_CHOICES):
        ax.barh(y + offset, coefs[name][order], height=0.25, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient on the logit (standardised features)")
    ax.set_title("Coefficients — do the three models agree?")
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Tab 3: Threshold
def threshold_view(threshold: float, model_name: str):
    """Confusion matrix heatmap + the cost figure, recomputed from the saved test
    probabilities every time the slider moves. Nothing is retrained."""
    prob = test_preds[PROB_COL[model_name]].to_numpy()
    tn, fp, fn, tp = _confusion(Y_TEST, prob, threshold)
    fp_cost, fn_cost = _cost(Y_TEST, prob, threshold)
    cost = fp_cost + fn_cost
    cost_at_chosen = sum(_cost(Y_TEST, prob, THRESHOLD))

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = _fig(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred live", "pred dead"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true live", "true dead"])
    ax.set_title(f"{model_name} @ threshold {threshold:.2f}")
    fig.tight_layout()

    flagged = tp + fp
    note = (
        f"### {flagged} of {len(Y_TEST)} test SKUs flagged · expected cost **{cost:,.0f} DOP**\n"
        f"- false positives: {fp} live items flagged, costing **{fp_cost:,.0f}** "
        f"({FP_MARGIN_MULTIPLIER:.0f} × each SKU's own margin, Precio detalle − Costo, "
        f"capped at {FP_COST_CAP:,.0f})\n"
        f"- false negatives: {fn} dead items missed, costing **{fn_cost:,.0f}** "
        f"({fn} × flat {FN_COST:,.0f})\n\n"
        f"Of the {flagged} flagged, {tp} are actually dead "
        f"(precision {tp / flagged if flagged else 0:.0%}).\n\n"
        f"Cost-minimising threshold chosen on the validation slice: **{THRESHOLD:.2f}** "
        f"(this model's test cost there: {cost_at_chosen:,.0f} DOP). "
        f"Cost rules are assumptions, set in `src/data.py`."
    )
    return fig, note


# ----------------------------------------------------------------------------- Tab 4: Scorer
def score(desc, prefix, dept, supplier, area, costo, precio) -> tuple[pd.DataFrame, str]:
    # Turn the chosen prefix back into a Código whose code_prefix() == prefix, because
    # build_features derives the prefix from Código, not from a prefix field.
    if prefix == "NUM":
        codigo = "99999"           # purely numeric → code_prefix returns "NUM"
    elif prefix == "OTHER":
        codigo = "ZZZZ9999"        # a prefix that won't be in the train top-N → OTHER
    else:
        codigo = prefix

    # One raw row with the SAME dtypes as the training data, so astype(str) inside
    # build_features produces frequency-table keys that match ("218", not "218.0").
    row = pd.DataFrame([{
        "Descripción": desc or "",
        "Código": codigo,
        "Dpto (ID)": int(dept),
        "Suplidor (ID)": int(supplier),
        "Área (ID)": int(area),
        # Price and cost are NOT features; they only price a false positive below.
        "Costo": float(costo),
        "Precio detalle": float(precio),
    }])

    X, _ = build_features(row, fit=feature_fit)   # reuse fit: learns nothing
    probs = _predict_proba(X.to_numpy())

    rows = []
    for name in MODEL_CHOICES:
        p = float(probs[name][0])
        rows.append({
            "model": name,
            "P(dead)": round(p, 3),
            f"decision @ {THRESHOLD:.2f}": "review for discontinuation" if p >= THRESHOLD else "keep",
        })
    fp_cost = float(fp_cost_per_sku(row).iloc[0])
    note = (f"If this SKU is live and gets flagged anyway, the assumed cost is "
            f"**{fp_cost:,.0f} DOP** ({FP_MARGIN_MULTIPLIER:.0f} × margin "
            f"{precio - costo:,.0f}, floored at 0, capped at {FP_COST_CAP:,.0f}). "
            f"Missing it if dead: {FN_COST:,.0f} DOP over the planning horizon.")
    return pd.DataFrame(rows), note


# ----------------------------------------------------------------------------- UI
PREFIX_CHOICES = feature_fit["prefix_levels"]        # top-N code prefixes + OTHER
DEPT_CHOICES = feature_fit["top_depts"] + ["(other)"]  # top-8 departments + a catch-all

with gr.Blocks(title="Display Internacional — dead-stock classifier") as demo:
    gr.Markdown("# Display Internacional — dead-stock classifier\n"
                "P(dead) per SKU, where dead = no stock AND no reorder point. "
                "A triage list for the discontinuation review, not a verdict.")

    with gr.Tab("Data"):
        with gr.Row():
            gr.Plot(plot_class_balance())
            gr.Plot(plot_logcost_by_class())
        with gr.Row():
            gr.Plot(plot_family_rate())
            gr.Plot(plot_material_rate())
        gr.Plot(plot_supplier_scatter())

    with gr.Tab("Models"):
        gr.Dataframe(metrics_table(), label=f"Test-set metrics at 0.5 and at the chosen threshold ({THRESHOLD:.2f})")
        with gr.Row():
            gr.Plot(plot_roc())
            with gr.Column():
                model_pick = gr.Dropdown(MODEL_CHOICES, value="LogReg", label="Model for the histogram")
                hist = gr.Plot(plot_prob_hist("LogReg"))
                model_pick.change(plot_prob_hist, model_pick, hist)
        gr.Plot(plot_coefficients())

    with gr.Tab("Threshold"):
        gr.Markdown("Move the threshold: a SKU is flagged when P(dead) ≥ threshold. "
                    "Counts and cost come from the saved test-set probabilities.")
        with gr.Row():
            thr = gr.Slider(0.0, 1.0, value=THRESHOLD, step=0.01, label="Decision threshold")
            thr_model = gr.Dropdown(MODEL_CHOICES, value="LogReg", label="Model")
        init_fig, init_note = threshold_view(THRESHOLD, "LogReg")
        with gr.Row():
            cm_plot = gr.Plot(init_fig)
            cost_note = gr.Markdown(init_note)
        thr.change(threshold_view, [thr, thr_model], [cm_plot, cost_note])
        thr_model.change(threshold_view, [thr, thr_model], [cm_plot, cost_note])

    with gr.Tab("Scorer"):
        gr.Markdown("Type what you know about the SKU. Unseen IDs are handled (encoded as 0).")
        with gr.Row():
            in_desc = gr.Textbox(label="Descripción", placeholder="e.g. GANCHO METAL 4 PLG")
            in_prefix = gr.Dropdown(PREFIX_CHOICES, value=PREFIX_CHOICES[0], label="Código prefix")
        with gr.Row():
            in_dept = gr.Dropdown(DEPT_CHOICES, value=DEPT_CHOICES[0], label="Dpto (ID)")
            in_supplier = gr.Number(value=0, precision=0, label="Suplidor (ID)")
            in_area = gr.Number(value=0, precision=0, label="Área (ID)")
        with gr.Row():
            in_costo = gr.Number(value=1000, label="Costo (DOP) — for the cost figure only")
            in_precio = gr.Number(value=2500, label="Precio detalle (DOP) — for the cost figure only")
        go = gr.Button("Score SKU", variant="primary")
        out = gr.Dataframe(label="P(dead) per model")
        out_cost = gr.Markdown()

        def _score(desc, prefix, dept, supplier, area, costo, precio):
            dept_val = -1 if dept == "(other)" else dept  # catch-all → unseen dept id
            return score(desc, prefix, dept_val, supplier, area, costo, precio)

        go.click(_score, [in_desc, in_prefix, in_dept, in_supplier, in_area, in_costo, in_precio],
                 [out, out_cost])


if __name__ == "__main__":
    demo.launch()
