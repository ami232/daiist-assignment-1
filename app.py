"""
Assignment 1 — Gradio dashboard for the cloud-kitchen concept scorer.

What this app does
------------------
It LOADS the three models trained by train.ipynb (scikit-learn, manual
PyTorch loop, standard nn.Module + Adam/OneCycleLR) and everything else the
training stage saved in artifacts/. It NEVER trains or re-fits anything: no
model, no scaler, no feature statistic. If artifacts/ is missing, it stops
with a message telling you to run `uv run python main.py train` first.

Tabs
----
1. Overview          - the use case and the headline metrics table.
2. Data              - target distribution, class balance, and the
                       distribution of any feature split by class.
3. Model comparison  - predicted vs. actual for the three models
                       (probability distributions by actual outcome,
                       calibration) and how closely the models agree.
4. Threshold & cost  - threshold slider + editable business costs ->
                       live confusion matrix and rupee cost.
5. Try a concept     - describe a cloud-kitchen idea and get each model's
                       probability of success and a launch verdict.
6. Training          - loss curves and the OneCycleLR schedule.

Run with:  uv run python main.py app   (opens http://127.0.0.1:7860)
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

import pipeline as P  # the SAME feature code the models were trained with

# ---------------------------------------------------------------------------
# 1. Load the training stage's artifacts (no training happens here)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
REQUIRED = ["sklearn_model.joblib", "scaler.joblib", "manual_torch.pt", "standard_torch.pt",
            "feature_params.json", "predictions.csv", "train_table.csv", "summary.json"]
missing = [f for f in REQUIRED if not (ART / f).exists()]
if missing:
    raise SystemExit(f"Missing {missing} in {ART}. Run `uv run python main.py train` first.")

with open(ART / "summary.json") as f:
    SUMMARY = json.load(f)
with open(ART / "feature_params.json") as f:
    FEATURE_PARAMS = json.load(f)
FEATURES = FEATURE_PARAMS["feature_names"]

# The fitted scaler: the training split's mean/std per feature. We only call
# .transform() on it - never .fit().
SCALER = joblib.load(ART / "scaler.joblib")

# Model 1: scikit-learn LogisticRegression.
SK_MODEL = joblib.load(ART / "sklearn_model.joblib")

# Model 2: manual PyTorch loop - just the learned tensors w and b.
# weights_only=True: only load plain tensors, never arbitrary Python objects.
_manual = torch.load(ART / "manual_torch.pt", weights_only=True)
MANUAL_W, MANUAL_B = _manual["w"], _manual["b"]

# Model 3: standard workflow - rebuild the empty nn.Module, then load the
# trained weights (state_dict) into it.
STD_MODEL = P.LogisticRegressionModel(len(FEATURES))
STD_MODEL.load_state_dict(torch.load(ART / "standard_torch.pt", weights_only=True))
STD_MODEL.eval()

# Validation + test restaurants with all three models' saved probabilities,
# and the training restaurants (raw columns + engineered features).
PREDS = pd.read_csv(ART / "predictions.csv")
TRAIN = pd.read_csv(ART / "train_table.csv")

MODELS = ["scikit-learn", "Manual PyTorch", "Standard PyTorch"]
PROB_COL = {"scikit-learn": "p_sklearn", "Manual PyTorch": "p_manual", "Standard PyTorch": "p_standard"}

# ---------------------------------------------------------------------------
# 2. Chart style (same palette as the training notebook)
# ---------------------------------------------------------------------------
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
MODEL_COLORS = {"scikit-learn": BLUE, "Manual PyTorch": ORANGE, "Standard PyTorch": AQUA}
CLASS_COLORS = {"Success (≥ 3.8)": BLUE, "Not success": NEUTRAL}


def style(fig, title, height=420):
    """Apply one consistent look to every Plotly chart."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK_2, size=12), margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def class_label(y):
    """0/1 target -> readable label used in legends."""
    return np.where(np.asarray(y) == 1, "Success (≥ 3.8)", "Not success")


# ---------------------------------------------------------------------------
# 3. Scoring new restaurants with all three saved models
# ---------------------------------------------------------------------------
def predict_all(raw_rows: pd.DataFrame) -> dict:
    """
    Raw restaurant rows -> P(success) from each of the three models.
    Steps: build features with the SAVED training statistics -> scale with the
    SAVED scaler -> apply each model's SAVED weights.
    """
    X = SCALER.transform(P.build_features(raw_rows, FEATURE_PARAMS))
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        manual = torch.sigmoid(Xt @ MANUAL_W + MANUAL_B).numpy()
        standard = STD_MODEL.predict_proba(Xt).numpy()
    return {"scikit-learn": SK_MODEL.predict_proba(X)[:, 1],
            "Manual PyTorch": manual, "Standard PyTorch": standard}


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
def metrics_table(split_key):
    """Metrics of baseline + three models for 'metrics_val' or 'metrics_test'."""
    df = pd.DataFrame(SUMMARY[split_key]).T.round(4)
    return df.reset_index(names="model")


OVERVIEW_MD = f"""
**Use case.** A delivery-only (cloud-kitchen) company is choosing which **cuisine + price** concepts to launch in
Gurgaon and Noida. The models estimate the probability that a restaurant is rated **≥ {SUMMARY['success_rating']}**
on Zomato ("success"), from its cuisines, price, location and services.

**Data.** Zomato restaurants in India that have a rating and a valid price. Split by brand (all outlets of a
chain in the same split): {SUMMARY['split_sizes']['train']:,} train / {SUMMARY['split_sizes']['validation']:,}
validation / {SUMMARY['split_sizes']['test']:,} test restaurants.

**One model, three implementations** (logistic regression, L2 penalty with C = {SUMMARY['best_C']}):
- **scikit-learn** `LogisticRegression` (lbfgs solver)
- **Manual PyTorch** loop: full-batch gradient descent, lr = {SUMMARY['manual']['lr']}, {SUMMARY['manual']['epochs']:,} epochs
- **Standard PyTorch**: `nn.Module` + `BCEWithLogitsLoss` + {SUMMARY['standard']['optimizer']},
  max_lr = {SUMMARY['standard']['max_lr']}, {SUMMARY['standard']['epochs']:,} epochs

**Baseline:** predict the training success rate ({SUMMARY['baseline_rate']:.1%}) for every restaurant.

Everything shown here is loaded from `artifacts/`, which the training stage produced. Nothing is retrained.
"""


# ---------------------------------------------------------------------------
# Tab 2: Data distributions
# ---------------------------------------------------------------------------
def rating_histogram():
    """Target: the rating distribution, coloured by class, with the 3.8 cut-off."""
    fig = go.Figure()
    for label, color in CLASS_COLORS.items():
        r = TRAIN.loc[class_label(TRAIN["success"]) == label, "Aggregate rating"]
        fig.add_trace(go.Histogram(x=r, xbins=dict(start=1.75, end=5.0, size=0.1), marker_color=color,
                                   name=label, hovertemplate="rating %{x}<br>%{y} restaurants<extra></extra>"))
    fig.add_vline(x=SUMMARY["success_rating"] - 0.05, line_dash="dash", line_color=INK,
                  annotation_text="cut-off 3.8", annotation_position="top")
    fig.update_layout(barmode="stack", bargap=0.1)
    fig.update_xaxes(title="Zomato aggregate rating")
    fig.update_yaxes(title="Restaurants")
    return style(fig, "Target: rating distribution (training split)", 380)


def class_balance():
    """Target: how many restaurants are in each class."""
    counts = pd.Series(class_label(TRAIN["success"])).value_counts().reindex(list(CLASS_COLORS))
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                           marker_color=[CLASS_COLORS[k] for k in counts.index],
                           text=[f"{n:,} ({n / counts.sum():.1%})" for n in counts.values], textposition="outside",
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_xaxes(title="Restaurants", range=[0, counts.max() * 1.25])
    return style(fig, "Class balance (training split)", 260)


# Features you can inspect: raw price/rating columns plus every engineered feature.
DIST_CHOICES = ["Average Cost for two"] + FEATURES


def feature_distribution(feature):
    """Distribution of one feature, split by class. 0/1 features -> success rate per value."""
    values = TRAIN[feature]
    labels = class_label(TRAIN["success"])
    if set(values.dropna().unique()) <= {0.0, 1.0}:
        # Binary feature: show how many restaurants have it and their success rate.
        g = TRAIN.groupby(values)["success"].agg(rate="mean", n="size").reindex([0.0, 1.0])
        fig = go.Figure(go.Bar(x=["No (0)", "Yes (1)"], y=g["rate"], marker_color=BLUE,
                               text=[f"{r:.0%} of {int(n):,}" if pd.notna(n) else "" for r, n in zip(g["rate"], g["n"])],
                               textposition="outside", hovertemplate="%{x}: %{y:.1%} success<extra></extra>"))
        fig.add_hline(y=TRAIN["success"].mean(), line_dash="dot", line_color=INK_2,
                      annotation_text="overall success rate", annotation_position="top left")
        fig.update_yaxes(title="Share rated ≥ 3.8", tickformat=".0%", range=[0, max(g["rate"].max() * 1.3, 0.3)])
        return style(fig, f"{feature}: success rate by value (training split)")
    # Numeric feature: overlaid histograms, normalised so the two classes are comparable in shape.
    fig = go.Figure()
    log_x = feature == "Average Cost for two"
    for label, color in CLASS_COLORS.items():
        v = values[labels == label]
        fig.add_trace(go.Histogram(x=np.log10(v) if log_x else v, nbinsx=40, histnorm="probability density",
                                   name=label, marker_color=color, opacity=0.6))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=f"log10({feature})" if log_x else feature)
    fig.update_yaxes(title="Density")
    return style(fig, f"{feature}: distribution by class (training split)")


# ---------------------------------------------------------------------------
# Tab 3: Model comparison - predicted vs. actual
# ---------------------------------------------------------------------------
def split_frame(split):
    return PREDS[PREDS["split"] == split]


def prob_by_outcome(split):
    """For each model: predicted probabilities of restaurants that actually succeeded vs. those that didn't.
    A good model pushes the blue (success) mass right and the grey mass left."""
    d = split_frame(split)
    fig = make_subplots(rows=1, cols=3, subplot_titles=MODELS, shared_yaxes=True)
    for i, m in enumerate(MODELS, start=1):
        for label, color in CLASS_COLORS.items():
            p = d.loc[class_label(d["success"]) == label, PROB_COL[m]]
            fig.add_trace(go.Histogram(x=p, xbins=dict(start=0, end=1, size=0.05), histnorm="probability",
                                       marker_color=color, opacity=0.6, name=label, legendgroup=label,
                                       showlegend=(i == 1)), row=1, col=i)
        fig.update_xaxes(title="Predicted P(success)", range=[0, 1], row=1, col=i)
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title="Share of restaurants in class", row=1, col=1)
    return style(fig, f"Predicted probability vs. actual outcome ({split})", 400)


def calibration(split):
    """Mean predicted probability vs. actual success rate in 10 equal-sized groups."""
    d = split_frame(split)
    fig = go.Figure()
    for m in MODELS:
        groups = pd.qcut(d[PROB_COL[m]], 10, labels=False, duplicates="drop")
        cal = d.assign(g=groups).groupby("g").agg(p=(PROB_COL[m], "mean"), y=("success", "mean"), n=("success", "size"))
        fig.add_trace(go.Scatter(x=cal["p"], y=cal["y"], mode="lines+markers", name=m, marker=dict(size=8),
                                 line=dict(color=MODEL_COLORS[m], width=2), customdata=cal["n"],
                                 hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>n=%{customdata}<extra>" + m + "</extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(color=INK_2, dash="dot", width=1)))
    fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="Actual success rate", range=[0, 1])
    return style(fig, f"Calibration: predicted vs. actual success rate ({split})", 430)


def agreement(split):
    """Each PyTorch model's probability against scikit-learn's, per restaurant. On the diagonal = identical."""
    d = split_frame(split)
    fig = go.Figure()
    for m in ["Manual PyTorch", "Standard PyTorch"]:
        diff = np.abs(d[PROB_COL[m]] - d["p_sklearn"])
        fig.add_trace(go.Scattergl(x=d["p_sklearn"], y=d[PROB_COL[m]], mode="markers", name=f"{m} (max |Δ| {diff.max():.4f})",
                                   marker=dict(color=MODEL_COLORS[m], size=6, opacity=0.6),
                                   hovertemplate="sklearn %{x:.4f}<br>" + m + " %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical", line=dict(color=INK_2, dash="dot", width=1)))
    fig.update_xaxes(title="scikit-learn P(success)", range=[0, 1])
    fig.update_yaxes(title="PyTorch P(success)", range=[0, 1])
    return style(fig, f"Do the three implementations agree? ({split})", 430)


def comparison_tab(split):
    key = "metrics_val" if split == "validation" else "metrics_test"
    return metrics_table(key), prob_by_outcome(split), calibration(split), agreement(split)


# ---------------------------------------------------------------------------
# Tab 4: Threshold & business cost
# ---------------------------------------------------------------------------
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def total_cost(y, p, t, flop, missed):
    """Rupee cost of launching every concept with p >= t: flops x flop cost + missed successes x missed cost."""
    launch = p >= t
    return np.sum(launch & (y == 0)) * flop + np.sum(~launch & (y == 1)) * missed


def threshold_tab(model, split, t, flop_lakh, missed_lakh):
    d = split_frame(split)
    y, p = d["success"].to_numpy(), d[PROB_COL[model]].to_numpy()
    flop, missed = flop_lakh * 1e5, missed_lakh * 1e5   # lakh -> rupees

    # Confusion matrix at the slider's threshold. Rows = actual, columns = decision.
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    z = [[tn, fp], [fn, tp]]
    cm = go.Figure(go.Heatmap(z=z, x=["Skip", "Launch"], y=["Not success", "Success"], colorscale=[[0, "#cde2fb"], [1, "#1c5cab"]],
                              showscale=False, text=[[f"{tn:,}<br>correct skip", f"{fp:,}<br>FLOP"],
                                                     [f"{fn:,}<br>MISSED", f"{tp:,}<br>good launch"]],
                              texttemplate="%{text}", textfont=dict(size=15), hoverinfo="skip"))
    cm.update_xaxes(title="Model decision")
    cm.update_yaxes(title="Actual outcome", autorange="reversed")
    style(cm, f"Confusion matrix — {model}, t = {t:.2f} ({split})", 380)

    # Cost at every threshold, with the current one and the cheapest one marked.
    costs = np.array([total_cost(y, p, th, flop, missed) for th in THRESHOLDS])
    best_t = THRESHOLDS[costs.argmin()]
    never = y.sum() * missed   # never launching = every success is missed
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=THRESHOLDS, y=costs / 1e5, mode="lines", name=model,
                               line=dict(color=MODEL_COLORS[model], width=2),
                               hovertemplate="t = %{x:.2f}<br>Rs %{y:,.0f} lakh<extra></extra>"))
    curve.add_hline(y=never / 1e5, line_dash="dot", line_color=INK_2, annotation_text="never launch",
                    annotation_position="top left")
    curve.add_vline(x=t, line_color=INK, line_width=1, annotation_text=f"current t = {t:.2f}")
    curve.add_trace(go.Scatter(x=[best_t], y=[costs.min() / 1e5], mode="markers", name=f"cheapest t = {best_t:.2f}",
                               marker=dict(size=11, color=MODEL_COLORS[model], line=dict(color=SURFACE, width=2))))
    curve.update_xaxes(title="Threshold t (launch if P(success) ≥ t)")
    curve.update_yaxes(title="Total cost (Rs lakh)")
    style(curve, f"Business cost vs. threshold ({split})", 380)

    cost_now = total_cost(y, p, t, flop, missed)
    theory_t = flop / (flop + missed) if flop + missed > 0 else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    md = f"""
### Total cost at t = {t:.2f}: **Rs {cost_now / 1e5:,.0f} lakh**
vs. **Rs {never / 1e5:,.0f} lakh** if we never launch → saving **Rs {(never - cost_now) / 1e5:,.0f} lakh**

| | |
|---|---|
| Concepts launched | {tp + fp:,} of {len(y):,} |
| Flops (FP) × Rs {flop_lakh:g} lakh | {fp:,} → Rs {fp * flop_lakh:,.0f} lakh |
| Missed successes (FN) × Rs {missed_lakh:g} lakh | {fn:,} → Rs {fn * missed_lakh:,.0f} lakh |
| Precision (launches that succeed) | {precision:.1%} |
| Recall (successes we launch) | {recall:.1%} |
| Cheapest threshold on this split | {best_t:.2f} |
| Theory threshold flop ÷ (flop + missed) | {theory_t:.2f} |
| Threshold chosen in training (validation, Rs 15/5 lakh) | {SUMMARY['chosen_thresholds'][model]:.2f} |
"""
    return cm, curve, md


# ---------------------------------------------------------------------------
# Tab 5: Try a concept
# ---------------------------------------------------------------------------
# All localities the model knows about, grouped by city ("City|Locality" keys).
LOCALITIES = {}
for key in FEATURE_PARAMS["locality_restaurant_count"]:
    city, loc = key.split("|", 1)
    LOCALITIES.setdefault(city, []).append(loc)
for city in LOCALITIES:
    # Most-listed localities first in the dropdown.
    LOCALITIES[city].sort(key=lambda l: -FEATURE_PARAMS["locality_restaurant_count"][f"{city}|{l}"])
CITIES = sorted(LOCALITIES, key=lambda c: -len(LOCALITIES[c]))
CUISINES = sorted(FEATURE_PARAMS["cuisine_median_cost"])
# The chain feature is looked up by brand name. To mark a concept as "part of an existing chain" we pass the
# name of any brand that has several outlets in the data (only the yes/no chain flag is used, not the brand).
CHAIN_EXAMPLE = next(name for name, n in FEATURE_PARAMS["brand_outlet_count"].items() if n > 1)
PRICE_GRID = np.arange(100, 3001, 25)


def concept_rows(cuisines, city, locality, costs, delivery, table_booking, chain):
    """Describe a hypothetical restaurant in the dataset's raw column format (one row per price)."""
    return pd.DataFrame({
        # Unknown name = a new single-outlet brand (is_chain = 0); CHAIN_EXAMPLE gives is_chain = 1.
        "Restaurant Name": CHAIN_EXAMPLE if chain else "__new_cloud_kitchen_brand__",
        "City": city, "Locality": locality, "Cuisines": ", ".join(cuisines),
        "Average Cost for two": costs, "Has Online delivery": "Yes" if delivery else "No",
        "Has Table booking": "Yes" if table_booking else "No",
    })


def update_localities(city):
    locs = LOCALITIES.get(city, [])
    return gr.update(choices=locs, value=locs[0] if locs else None)


def concept_tab(cuisines, city, locality, cost, delivery, table_booking, chain, t):
    if not cuisines:
        return "Pick at least one cuisine.", None
    # 1) Probability at the chosen price, from all three models.
    probs = predict_all(concept_rows(cuisines, city, locality, [cost], delivery, table_booking, chain))
    lines = [f"### Concept: {', '.join(cuisines)} · {locality}, {city} · Rs {cost:,.0f} for two"
             f"{' · delivery' if delivery else ''}\n", "| Model | P(rating ≥ 3.8) | Decision at t = " + f"{t:.2f} |", "|---|---|---|"]
    for m in MODELS:
        p = float(probs[m][0])
        lines.append(f"| {m} | **{p:.1%}** | {'✅ launch' if p >= t else '❌ skip'} |")
    norm = np.mean([FEATURE_PARAMS["cuisine_median_cost"].get(c, FEATURE_PARAMS["overall_median_cost"]) for c in cuisines])
    lines.append(f"\nTypical price for these cuisines (training data): Rs {norm:,.0f} → this concept is priced at "
                 f"**{cost / norm:.2f}×** the norm.")

    # 2) The same concept at every price: where does the model put the sweet spot?
    curve_probs = predict_all(concept_rows(cuisines, city, locality, PRICE_GRID, delivery, table_booking, chain))
    fig = go.Figure()
    for m in MODELS:
        fig.add_trace(go.Scatter(x=PRICE_GRID, y=curve_probs[m], mode="lines", name=m,
                                 line=dict(color=MODEL_COLORS[m], width=2),
                                 hovertemplate="Rs %{x}<br>%{y:.1%}<extra>" + m + "</extra>"))
    fig.add_hline(y=t, line_dash="dot", line_color=INK_2, annotation_text=f"launch threshold {t:.2f}",
                  annotation_position="top left")
    fig.add_vline(x=cost, line_color=INK, line_width=1, annotation_text="this concept")
    if delivery:
        fig.add_vrect(x0=1500, x1=PRICE_GRID[-1], fillcolor=GRID, opacity=0.5, line_width=0,
                      annotation_text="few delivery restaurants in data", annotation_position="bottom right")
    fig.update_xaxes(title="Average cost for two (Rs)")
    fig.update_yaxes(title="P(rating ≥ 3.8)", tickformat=".0%", range=[0, 1])
    style(fig, "Same concept at every price (model lines overlap when the models agree)", 420)
    return "\n".join(lines), fig


# ---------------------------------------------------------------------------
# Tab 6: Training curves (loaded from summary.json)
# ---------------------------------------------------------------------------
def training_curves():
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Manual PyTorch (gradient descent, lr = {SUMMARY['manual']['lr']})",
        f"Standard PyTorch ({SUMMARY['standard']['optimizer']}, max_lr = {SUMMARY['standard']['max_lr']})"],
        shared_yaxes=True)
    for col, key in [(1, "manual"), (2, "standard")]:
        h = SUMMARY[key]
        for name, series, color in [("train objective (BCE + L2)", h["train_objective"], BLUE),
                                    ("validation BCE", h["val_bce"], ORANGE)]:
            fig.add_trace(go.Scatter(y=series, mode="lines", name=name, line=dict(color=color, width=2),
                                     legendgroup=name, showlegend=(col == 1)), row=1, col=col)
        fig.add_hline(y=SUMMARY["sklearn_objective"], line_dash="dash", line_color=INK, line_width=1, row=1, col=col)
        fig.update_xaxes(title="Epoch", row=1, col=col)
    fig.update_yaxes(title="Loss", range=[SUMMARY["sklearn_objective"] - 0.02, 0.6], row=1, col=1)
    style(fig, "Final training runs (dashed = scikit-learn optimum of the training objective)", 420)

    lr = go.Figure(go.Scatter(y=SUMMARY["standard"]["lr_schedule"], mode="lines", line=dict(color=AQUA, width=2),
                              hovertemplate="epoch %{x}<br>lr %{y:.2e}<extra></extra>", name="learning rate"))
    lr.update_yaxes(type="log", title="Learning rate (log scale)")
    lr.update_xaxes(title="Epoch")
    style(lr, "OneCycleLR schedule used by the standard workflow: warm-up, then cosine decay", 340)
    return fig, lr


# ---------------------------------------------------------------------------
# 4. Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Cloud-kitchen concept scorer") as demo:
    gr.Markdown("# Cloud-kitchen concept scorer — Zomato India")

    with gr.Tab("Overview"):
        gr.Markdown(OVERVIEW_MD)
        gr.Markdown("#### Test set (touched once, final numbers)")
        gr.Dataframe(metrics_table("metrics_test"), interactive=False)
        gr.Markdown("#### Validation set")
        gr.Dataframe(metrics_table("metrics_val"), interactive=False)

    with gr.Tab("Data"):
        with gr.Row():
            gr.Plot(rating_histogram())
            gr.Plot(class_balance())
        feat = gr.Dropdown(DIST_CHOICES, value="rel_price", label="Feature to inspect (raw or engineered)")
        feat_plot = gr.Plot(feature_distribution("rel_price"))
        feat.change(feature_distribution, inputs=feat, outputs=feat_plot)

    with gr.Tab("Model comparison"):
        split_cmp = gr.Radio(["test", "validation"], value="test", label="Split")
        cmp_table = gr.Dataframe(interactive=False)
        cmp_hist = gr.Plot()
        with gr.Row():
            cmp_cal = gr.Plot()
            cmp_agree = gr.Plot()
        split_cmp.change(comparison_tab, inputs=split_cmp, outputs=[cmp_table, cmp_hist, cmp_cal, cmp_agree])
        demo.load(comparison_tab, inputs=split_cmp, outputs=[cmp_table, cmp_hist, cmp_cal, cmp_agree])

    with gr.Tab("Threshold & cost"):
        with gr.Row():
            th_model = gr.Dropdown(MODELS, value="scikit-learn", label="Model")
            th_split = gr.Radio(["test", "validation"], value="test", label="Split")
        threshold = gr.Slider(0.01, 0.99, value=SUMMARY["chosen_thresholds"]["scikit-learn"], step=0.01,
                              label="Decision threshold t: launch a concept if P(success) ≥ t")
        with gr.Row():
            flop_box = gr.Number(value=SUMMARY["cost_flop"] / 1e5, label="Cost of a flop (Rs lakh)", minimum=0)
            missed_box = gr.Number(value=SUMMARY["cost_missed"] / 1e5, label="Cost of a missed success (Rs lakh)", minimum=0)
        with gr.Row():
            th_cm = gr.Plot()
            th_md = gr.Markdown()
        th_curve = gr.Plot()
        th_inputs = [th_model, th_split, threshold, flop_box, missed_box]
        for comp in th_inputs:
            comp.change(threshold_tab, inputs=th_inputs, outputs=[th_cm, th_curve, th_md])
        demo.load(threshold_tab, inputs=th_inputs, outputs=[th_cm, th_curve, th_md])

    with gr.Tab("Try a concept"):
        gr.Markdown("Describe a cloud-kitchen idea. The decision uses the threshold set in the *Threshold & cost* tab.")
        with gr.Row():
            c_cuisines = gr.Dropdown(CUISINES, value=["North Indian"], multiselect=True, label="Cuisine(s)")
            c_city = gr.Dropdown(CITIES, value="Gurgaon", label="City")
            c_loc = gr.Dropdown(LOCALITIES["Gurgaon"], value=LOCALITIES["Gurgaon"][0], label="Locality")
        c_cost = gr.Slider(100, 3000, value=600, step=25, label="Average cost for two (Rs)")
        with gr.Row():
            c_delivery = gr.Checkbox(value=True, label="Online delivery")
            c_table = gr.Checkbox(value=False, label="Table booking")
            c_chain = gr.Checkbox(value=False, label="Part of an existing chain")
        c_md = gr.Markdown()
        c_plot = gr.Plot()
        c_city.change(update_localities, inputs=c_city, outputs=c_loc)
        c_inputs = [c_cuisines, c_city, c_loc, c_cost, c_delivery, c_table, c_chain, threshold]
        for comp in c_inputs:
            comp.change(concept_tab, inputs=c_inputs, outputs=[c_md, c_plot])
        demo.load(concept_tab, inputs=c_inputs, outputs=[c_md, c_plot])

    with gr.Tab("Training"):
        curves, schedule = training_curves()
        gr.Plot(curves)
        gr.Plot(schedule)


if __name__ == "__main__":
    # 127.0.0.1:7860 is what the GitHub check (verify.yml) expects. inbrowser=True opens a browser tab.
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
