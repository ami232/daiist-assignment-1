"""
Gradio dashboard for Ames Housing Ridge models.

Loads training artifacts only — never retrains. Views:
- Model comparison: all three models on one prediction vs actual plot,
  dollar metrics from training, and prediction agreement between them.
- Distributions: target and any feature, in original units, per split.
- Asking-price check: the report's decision rule applied to one house.
"""

import json
from itertools import combinations
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

ART = Path(__file__).resolve().parent / "artifacts"

SPLITS = {
    "train (2006–2008)": "train",
    "val (2009)": "val",
    "test (2010)": "test",
}
SPLIT_LABELS = {key: label for label, key in SPLITS.items()}
SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
TARGET = "SalePrice"
HOUSE_FACTS = [
    "Yr Sold", "Gr Liv Area", "Total Bsmt SF", "Overall Qual",
    "Overall Cond", "Age", "Garage Area", "Full Bath",
]


def _load_artifacts():
    X = {key: np.load(ART / f"X_{key}.npy") for key in SPLITS.values()}
    y_log = {key: np.load(ART / f"y_{key}.npy") for key in SPLITS.values()}
    feature_names = json.loads((ART / "feature_names.json").read_text(encoding="utf-8"))
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    scaler = joblib.load(ART / "scaler.joblib")

    sklearn_model = joblib.load(ART / "sklearn_model.joblib")

    manual = torch.load(ART / "pytorch_manual.pt", map_location="cpu", weights_only=False)
    w_manual = manual["w"].detach().cpu().numpy().ravel()
    b_manual = float(manual["b"])

    standard = torch.load(
        ART / "pytorch_standard.pt", map_location="cpu", weights_only=False
    )
    n_features = int(standard["n_features"])
    std_model = nn.Linear(n_features, 1, bias=True, dtype=torch.float64)
    std_model.load_state_dict(standard["state_dict"])
    std_model.eval()

    return (
        X, y_log, feature_names, metrics, scaler,
        sklearn_model, w_manual, b_manual, std_model,
    )


(
    X_SPLITS, Y_LOG, FEATURES, METRICS, SCALER,
    SK_MODEL, W_MANUAL, B_MANUAL, STD_MODEL,
) = _load_artifacts()

SK_NAME = f"sklearn {METRICS['chosen_family']}"
MODELS = [SK_NAME, "manual PyTorch", "standard PyTorch"]
MARKERS = {
    SK_NAME: {"marker": "o", "facecolors": "none", "edgecolors": "tab:blue", "s": 42},
    "manual PyTorch": {"marker": "x", "color": "tab:orange", "s": 22},
    "standard PyTorch": {"marker": "+", "color": "tab:green", "s": 30},
}


def predict_log(model_name: str, X: np.ndarray) -> np.ndarray:
    if model_name == SK_NAME:
        return SK_MODEL.predict(X)
    if model_name == "manual PyTorch":
        return X @ W_MANUAL + B_MANUAL
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float64)
        return STD_MODEL(xt).numpy().ravel()


ACTUAL = {key: np.exp(y) for key, y in Y_LOG.items()}
PRED = {
    model: {key: np.exp(predict_log(model, X)) for key, X in X_SPLITS.items()}
    for model in MODELS
}
RAW = {
    key: pd.DataFrame(SCALER.inverse_transform(X), columns=FEATURES)
    for key, X in X_SPLITS.items()
}
DOLLARS = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")


# ---------- Model comparison ----------


def plot_pred_vs_actual(models: list[str], split_label: str):
    split = SPLITS[split_label]
    actual = ACTUAL[split]

    fig = Figure(figsize=(6.5, 5.5))
    ax = fig.subplots()
    lo, hi = actual.min(), actual.max()
    for model in models:
        pred = PRED[model][split]
        ax.scatter(actual, pred, alpha=0.6, linewidths=1, label=model, **MARKERS[model])
        lo, hi = min(lo, pred.min()), max(hi, pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, label="perfect fit")
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")
    ax.xaxis.set_major_formatter(DOLLARS)
    ax.yaxis.set_major_formatter(DOLLARS)
    ax.set_title(split_label if models else "Select at least one model")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def metrics_table() -> pd.DataFrame:
    rows = [
        {
            "model": name,
            "split": split,
            "RMSE ($)": f"{m['rmse']:,.0f}",
            "MAE ($)": f"{m['mae']:,.0f}",
        }
        for name, splits in METRICS["final"].items()
        for split, m in splits.items()
    ]
    return pd.DataFrame(rows)


def agreement_table(split_label: str) -> pd.DataFrame:
    split = SPLITS[split_label]
    rows = []
    for a, b in combinations(MODELS, 2):
        gap = np.abs(PRED[a][split] - PRED[b][split])
        rows.append({
            "pair": f"{a.split()[0]} vs {b.split()[0]}",
            "max |Δ| ($)": f"{gap.max():.1e}",
            "mean |Δ| ($)": f"{gap.mean():.1e}",
        })
    return pd.DataFrame(rows)


# ---------- Distributions ----------


def _values(variable: str, split: str) -> np.ndarray:
    if variable == TARGET:
        return ACTUAL[split]
    return RAW[split][variable].to_numpy()


def _axis_label(variable: str) -> str:
    return "Sale price" if variable == TARGET else variable


def _split_hist(ax, values: dict[str, np.ndarray]) -> float:
    bins = np.histogram_bin_edges(np.concatenate(list(values.values())), bins=40)
    for key, v in values.items():
        ax.hist(
            v, bins=bins, weights=np.full(len(v), 100 / len(v)), histtype="step",
            linewidth=1.8, color=SPLIT_COLORS[key], label=SPLIT_LABELS[key],
        )
    ax.set_ylabel("% of houses in split")
    return pd.Series(values["train"]).skew()


def plot_distribution(variable: str):
    fig = Figure(figsize=(6.5, 4.5))
    ax = fig.subplots()
    skew = _split_hist(ax, {key: _values(variable, key) for key in SPLITS.values()})
    ax.set_xlabel(_axis_label(variable))
    if variable == TARGET:
        ax.xaxis.set_major_formatter(DOLLARS)
    ax.set_title(f"{variable} by split (train skew {skew:.2f})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_relationship(variable: str):
    fig = Figure(figsize=(6.5, 4.5))
    ax = fig.subplots()
    if variable == TARGET:
        skew = _split_hist(ax, Y_LOG)
        ax.set_xlabel("log(SalePrice)")
        ax.set_title(f"log(SalePrice), the training target (train skew {skew:.2f})")
    else:
        for key in SPLITS.values():
            ax.scatter(
                _values(variable, key), ACTUAL[key], s=12, alpha=0.35,
                edgecolors="none", color=SPLIT_COLORS[key], label=SPLIT_LABELS[key],
            )
        ax.set_xlabel(variable)
        ax.set_ylabel("Sale price")
        ax.yaxis.set_major_formatter(DOLLARS)
        ax.set_title(f"{variable} vs sale price")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------- Asking-price check ----------


def _level(house: pd.Series, prefix: str) -> str:
    hits = [c[len(prefix) + 1:] for c in FEATURES if c.startswith(f"{prefix}_") and house[c] > 0.5]
    return hits[0] if hits else "reference level"


def _signed(dollars: float) -> str:
    return f"{'+' if dollars >= 0 else '−'}${abs(dollars):,.0f}"


def check_ask(model: str, split_label: str, row: float, ask: float | None) -> str:
    split = SPLITS[split_label]
    row = int(row)
    estimate = PRED[model][split][row]
    actual = ACTUAL[split][row]
    threshold = METRICS["final"][model]["val"]["mae"]
    house = RAW[split].iloc[row]

    facts = "\n".join(
        f"| {c} | {house[c]:.0f} |" if c == "Yr Sold" else f"| {c} | {house[c]:,.0f} |"
        for c in HOUSE_FACTS
    )
    facts += f"\n| Neighborhood | {_level(house, 'Neighborhood')} |"

    if ask is None:
        verdict = "Enter an asking price."
        ask_rows = ""
    else:
        gap = ask - estimate
        if gap > threshold:
            verdict = (
                f"**Aggressive ask.** It sits ${gap:,.0f} above the estimate, "
                f"more than the typical miss of ${threshold:,.0f}."
            )
        else:
            verdict = (
                f"**Not flagged.** Ask − estimate is {_signed(gap)}; the flag "
                f"needs more than +${threshold:,.0f}."
            )
        ask_rows = (
            f"| Asking price | ${ask:,.0f} |\n"
            f"| Ask − estimate | {_signed(gap)} |\n"
        )

    return (
        f"### Estimated close: ${estimate:,.0f}\n\n"
        f"{verdict}\n\n"
        "| Check | Value |\n|---|---:|\n"
        f"{ask_rows}"
        f"| Flag threshold (2009 validation MAE, {model}) | ${threshold:,.0f} |\n"
        f"| Recorded close (unknown at offer time) | ${actual:,.0f} |\n\n"
        "| House | Value |\n|---|---:|\n"
        f"{facts}"
    )


def pick_split(split_label: str):
    n = len(ACTUAL[SPLITS[split_label]])
    return gr.Slider(minimum=0, maximum=n - 1, value=0, step=1)


def pick_house(model: str, split_label: str, row: float):
    ask = round(float(ACTUAL[SPLITS[split_label]][int(row)]), -3)
    return ask, check_ask(model, split_label, row, ask)


NOTE = (
    "Models are loaded from `artifacts/` only; nothing is retrained here. "
    "Validation (2009) chose α and the learning rate; test (2010) is the "
    "reported score. Predictions are dollars after undoing log(SalePrice)."
)
CHECK_NOTE = (
    "Decision rule from the report: flag an ask that beats the estimate by "
    "more than the typical dollar miss, taken as the model's 2009 validation "
    "MAE. The file has no asking prices, so the field starts at the recorded "
    "close; type any ask to test it."
)

with gr.Blocks(title="Ames Housing — model comparison") as demo:
    gr.Markdown("# Ames Housing dashboard")
    gr.Markdown(NOTE)

    with gr.Tabs(selected="compare"):
        with gr.Tab("Model comparison", id="compare"):
            with gr.Row():
                models_cb = gr.CheckboxGroup(MODELS, value=MODELS, label="Models")
                split_dd = gr.Dropdown(list(SPLITS), value="test (2010)", label="Split")
            with gr.Row():
                pred_plot = gr.Plot(show_label=False)
                with gr.Column():
                    gr.Markdown("**Dollar error by split**")
                    gr.Dataframe(metrics_table(), interactive=False)
                    gr.Markdown("**Agreement between implementations** on the selected split")
                    agree_df = gr.Dataframe(interactive=False)
            for control in (models_cb, split_dd):
                control.change(plot_pred_vs_actual, [models_cb, split_dd], pred_plot)
            split_dd.change(agreement_table, split_dd, agree_df)
            demo.load(plot_pred_vs_actual, [models_cb, split_dd], pred_plot)
            demo.load(agreement_table, split_dd, agree_df)

        with gr.Tab("Distributions", id="distributions"):
            var_dd = gr.Dropdown(
                [TARGET] + FEATURES, value=TARGET, label="Variable (original units)"
            )
            with gr.Row():
                dist_plot = gr.Plot(show_label=False)
                rel_plot = gr.Plot(show_label=False)
            var_dd.change(plot_distribution, var_dd, dist_plot)
            var_dd.change(plot_relationship, var_dd, rel_plot)
            demo.load(plot_distribution, var_dd, dist_plot)
            demo.load(plot_relationship, var_dd, rel_plot)

        with gr.Tab("Asking-price check", id="ask"):
            gr.Markdown(CHECK_NOTE)
            with gr.Row():
                ask_model = gr.Dropdown(MODELS, value=SK_NAME, label="Model")
                ask_split = gr.Dropdown(list(SPLITS), value="test (2010)", label="Split")
            with gr.Row():
                row_sl = gr.Slider(
                    0, len(ACTUAL["test"]) - 1, value=0, step=1,
                    label="House (row in split)",
                )
                ask_num = gr.Number(label="Asking price ($)", precision=0)
            check_md = gr.Markdown()
            ask_split.change(pick_split, ask_split, row_sl).then(
                pick_house, [ask_model, ask_split, row_sl], [ask_num, check_md]
            )
            row_sl.change(pick_house, [ask_model, ask_split, row_sl], [ask_num, check_md])
            for control in (ask_model, ask_num):
                control.change(check_ask, [ask_model, ask_split, row_sl, ask_num], check_md)
            demo.load(pick_house, [ask_model, ask_split, row_sl], [ask_num, check_md])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
