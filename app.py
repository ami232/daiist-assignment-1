"""
Assignment 1 — Gradio dashboard for the Ames appraisal model.

Run with:  uv run python main.py app   (after `uv run python main.py train`)

Everything shown here comes from artifacts/, written by train.py. This file
never fits or trains anything: the models, the fitted preprocessor and the
Lot Frontage medians are only LOADED, and predictions for new inputs only
call .predict() / a forward pass.

Tabs:
  1. Model comparison   predicted vs actual price for the three implementations
                        and the baseline, metrics, agreement between methods
  2. Distributions      any feature or the target, train vs val vs test years
  3. Training           PyTorch loss curves vs sklearn's exact optimum, coefficients
  4. Appraiser          pick a 2010 house, change some characteristics, get a price
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from features import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, TEST_YEAR,
    engineer_features, filter_population, impute_lot_frontage, load_data,
)
from models import ARTIFACTS_DIR, MODEL_NAMES, load_models, predict_log_price

# ---------------------------------------------------------------------------
# Load artifacts (no training)
# ---------------------------------------------------------------------------
if not (ARTIFACTS_DIR / "metrics.json").exists():
    raise SystemExit("artifacts/ not found: run `uv run python main.py train` first.")

with open(ARTIFACTS_DIR / "metrics.json") as f:
    SUMMARY = json.load(f)
with open(ARTIFACTS_DIR / "frontage_medians.json") as f:
    FRONTAGE_MEDIANS = json.load(f)
PREPROCESSOR = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
MODELS = load_models(SUMMARY["n_features"])
PREDICTIONS = pd.read_csv(ARTIFACTS_DIR / "predictions.csv")
LOSS = pd.read_csv(ARTIFACTS_DIR / "loss_history.csv")
COEFS = pd.read_csv(ARTIFACTS_DIR / "coefficients.csv", index_col="feature")

LABELS = {
    "sklearn": "scikit-learn Ridge",
    "manual_torch": "PyTorch manual loop",
    "module_torch": "PyTorch nn.Module + optim",
    "baseline": "Naive baseline (train median)",
}
SPLIT_LABELS = {"train": "Train 2006-08", "val": "Validation 2009", "test": "Test 2010"}

# Colors: reference categorical palette (dataviz skill), fixed order.
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
MODEL_COLORS = {"sklearn": "#2a78d6", "manual_torch": "#eb6834", "module_torch": "#1baf7a"}
POINT_COLOR = "#2a78d6"
REFERENCE_COLOR = "#8a8984"  # neutral gray for reference lines / baseline
LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#0b0b0b"),
    margin=dict(l=60, r=20, t=60, b=50),
    hoverlabel=dict(bgcolor="white"),
)


def money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------
def metrics_table(split: str) -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES + ["baseline"]:
        m = SUMMARY["metrics"][split][name]
        rows.append({
            "Model": LABELS[name],
            "MAE": money(m["MAE ($)"]),
            "RMSE": money(m["RMSE ($)"]),
            "MAPE": f"{m['MAPE (%)']:.1f}%",
            "Within ±10%": f"{m['Within 10% (%)']:.1f}%",
            "Bias": money(m["Bias ($)"]),
        })
    return pd.DataFrame(rows)


def agreement_table() -> pd.DataFrame:
    rows = [
        {"Comparison": pair.replace("_", " "),
         "Largest prediction gap": f"{v['max prediction gap (%)']:.4f}%",
         "Largest coefficient gap": f"{v['max coefficient gap']:.2e}"}
        for pair, v in SUMMARY["agreement_test"].items()
    ]
    return pd.DataFrame(rows)


def pred_vs_actual_plot(split: str) -> go.Figure:
    """One panel per model (small multiples): the three implementations
    produce near-identical predictions, so overlaying them would hide two of
    the three clouds. The diagonal is a perfect prediction."""
    df = PREDICTIONS[PREDICTIONS["split"] == split]
    names = MODEL_NAMES + ["baseline"]
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=[LABELS[n] for n in names], horizontal_spacing=0.03)
    lo, hi = df[TARGET].min() * 0.9, df[TARGET].max() * 1.05
    for i, name in enumerate(names, start=1):
        mae = SUMMARY["metrics"][split][name]["MAE ($)"]
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=REFERENCE_COLOR, width=1.5, dash="dash"),
        ), row=1, col=i)
        fig.add_trace(go.Scatter(
            x=df[TARGET], y=df[f"pred_{name}"], mode="markers", showlegend=False,
            marker=dict(size=8, color=REFERENCE_COLOR if name == "baseline" else POINT_COLOR,
                        opacity=0.55, line=dict(width=1, color="white")),
            customdata=np.c_[df["Neighborhood"], df["PID"]],
            hovertemplate=("Actual %{x:$,.0f}<br>Predicted %{y:$,.0f}"
                           "<br>%{customdata[0]} · PID %{customdata[1]}<extra></extra>"),
        ), row=1, col=i)
        axis = "" if i == 1 else i  # plotly names the first subplot's axes "x"/"y", not "x1"/"y1"
        fig.add_annotation(text=f"MAE {money(mae)}", xref=f"x{axis} domain", yref=f"y{axis} domain",
                           x=0.03, y=0.97, showarrow=False, xanchor="left",
                           font=dict(size=12, color="#52514e"))
    fig.update_layout(**LAYOUT, height=430,
                      title=f"Predicted vs actual sale price — {SPLIT_LABELS[split]}")
    fig.update_xaxes(title_text="Actual price", tickformat="$~s", range=[lo, hi])
    fig.update_yaxes(tickformat="$~s", range=[lo, hi])
    fig.update_yaxes(title_text="Predicted price", row=1, col=1)
    return fig


def residual_plot(split: str) -> go.Figure:
    """Percentage error vs actual price for sklearn (the three are identical):
    shows WHERE the model struggles (cheap houses tend to be over-valued)."""
    df = PREDICTIONS[PREDICTIONS["split"] == split]
    pct = 100 * (df["pred_sklearn"] - df[TARGET]) / df[TARGET]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=REFERENCE_COLOR, width=1.5, dash="dash"))
    for band in (10, -10):
        fig.add_hline(y=band, line=dict(color=REFERENCE_COLOR, width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=df[TARGET], y=pct, mode="markers",
        marker=dict(size=8, color=POINT_COLOR, opacity=0.55, line=dict(width=1, color="white")),
        customdata=np.c_[df["Neighborhood"], df["pred_sklearn"]],
        hovertemplate=("Actual %{x:$,.0f}<br>Predicted %{customdata[1]:$,.0f}"
                       "<br>Error %{y:+.1f}%<br>%{customdata[0]}<extra></extra>"),
    ))
    fig.update_layout(**LAYOUT, height=360, showlegend=False,
                      title=f"Percentage error by price (scikit-learn) — dotted lines = ±10%")
    fig.update_xaxes(title_text="Actual price", tickformat="$~s")
    fig.update_yaxes(title_text="Error (% of actual)", ticksuffix="%")
    return fig


def update_comparison(split: str):
    return pred_vs_actual_plot(split), residual_plot(split), metrics_table(split)


# ---------------------------------------------------------------------------
# Tab 2: distributions
# ---------------------------------------------------------------------------
DIST_CHOICES = [TARGET, "log(SalePrice)"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES


def distribution_plot(column: str) -> go.Figure:
    df = PREDICTIONS.copy()
    df["log(SalePrice)"] = np.log(df[TARGET])
    fig = go.Figure()
    if column in CATEGORICAL_FEATURES:
        # Share of each split's sales per category (shares, not counts, because
        # the splits have different sizes).
        shares = (df.groupby("split")[column].value_counts(normalize=True)
                  .rename("share").reset_index())
        order = df[column].value_counts().index
        for split in ("train", "val", "test"):
            s = shares[shares["split"] == split].set_index(column).reindex(order)["share"].fillna(0)
            fig.add_trace(go.Bar(x=list(order), y=100 * s.values, name=SPLIT_LABELS[split],
                                 marker=dict(color=SPLIT_COLORS[split], line=dict(width=2, color="white")),
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + SPLIT_LABELS[split] + "</extra>"))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="Share of sales in split", ticksuffix="%")
    else:
        # Same bins for the three splits, each shown as the % of ITS OWN sales
        # (splits have different sizes). Step outlines instead of filled bars so
        # three overlapping distributions stay readable.
        values = df[column].dropna()
        edges = np.histogram_bin_edges(values, bins=40)
        for split in ("train", "val", "test"):
            counts, _ = np.histogram(df.loc[df["split"] == split, column].dropna(), bins=edges)
            pct = 100 * counts / max(counts.sum(), 1)
            fig.add_trace(go.Scatter(
                x=edges, y=np.r_[pct, pct[-1]], mode="lines", line_shape="hv",
                name=SPLIT_LABELS[split], line=dict(color=SPLIT_COLORS[split], width=2),
                hovertemplate="From %{x:,.3~f}: %{y:.1f}% of sales<extra>" + SPLIT_LABELS[split] + "</extra>",
            ))
        fig.update_yaxes(title_text="Share of sales in split", ticksuffix="%", rangemode="tozero")
        if column == TARGET:
            fig.update_xaxes(tickformat="$~s")
    fig.update_layout(**LAYOUT, height=440, title=f"{column}: train vs validation vs test years",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    fig.update_xaxes(title_text=column)
    return fig


def distribution_stats(column: str) -> pd.DataFrame:
    df = PREDICTIONS.copy()
    df["log(SalePrice)"] = np.log(df[TARGET])
    if column in CATEGORICAL_FEATURES:
        out = df.groupby("split")[column].agg(
            n="size", categories="nunique", most_common=lambda s: s.mode().iat[0])
    else:
        out = df.groupby("split")[column].describe()[["count", "mean", "std", "min", "50%", "max"]]
    out = out.reindex(["train", "val", "test"]).rename(index=SPLIT_LABELS).reset_index()
    return out.round(3)


# ---------------------------------------------------------------------------
# Tab 3: training diagnostics
# ---------------------------------------------------------------------------
def loss_plot() -> go.Figure:
    """Distance of each PyTorch run's training loss from sklearn's exact
    optimum, on a log scale: a straight-ish downward line = steady convergence."""
    optimum = SUMMARY["train_objective"]["sklearn"]
    fig = go.Figure()
    for name, dash in (("manual_torch", "solid"), ("module_torch", "dash")):
        gap = (LOSS[name] - optimum).clip(lower=1e-9)
        fig.add_trace(go.Scatter(
            x=LOSS["epoch"], y=gap, mode="lines", name=f"{LABELS[name]} (lr={SUMMARY['lr'][name]})",
            line=dict(color=MODEL_COLORS[name], width=2, dash=dash),
            hovertemplate="Epoch %{x:,}<br>Loss − optimum %{y:.2e}<extra></extra>",
        ))
    fig.update_layout(**LAYOUT, height=420, hovermode="x unified",
                      title="Training loss minus scikit-learn's exact optimum (log scale)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    fig.update_xaxes(title_text="Epoch (full-batch gradient step)")
    fig.update_yaxes(title_text="Loss − optimum", type="log", exponentformat="e")
    return fig


def coefficient_plot(top_n: int) -> go.Figure:
    coefs = COEFS.drop(index="intercept")
    top = coefs.reindex(coefs["sklearn"].abs().sort_values(ascending=False).index).head(int(top_n))
    top = top.iloc[::-1]  # largest at the top of a horizontal bar chart
    labels = [f.split("__", 1)[1] for f in top.index]
    fig = go.Figure()
    for name in MODEL_NAMES:
        fig.add_trace(go.Bar(
            y=labels, x=top[name], name=LABELS[name], orientation="h",
            marker=dict(color=MODEL_COLORS[name], line=dict(width=2, color="white")),
            hovertemplate="%{y}: %{x:+.4f}<extra>" + LABELS[name] + "</extra>",
        ))
    fig.update_layout(**LAYOUT, barmode="group", height=max(420, 34 * len(top) + 120),
                      title=f"Top {int(top_n)} coefficients by size (standardized features, log-price scale)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    fig.update_xaxes(title_text="Coefficient (change in log price per 1 std of the feature)")
    return fig


def tuning_table() -> pd.DataFrame:
    t = SUMMARY["tuning"]
    rows = [{"Search": "Ridge alpha (sklearn)", "Value": k, "Validation MAE": money(v)}
            for k, v in t["alpha_val_mae"].items()]
    for key, name in (("manual_lr_val_mae", "manual_torch"), ("module_lr_val_mae", "module_torch")):
        rows += [{"Search": f"Learning rate ({LABELS[name]})", "Value": k,
                  "Validation MAE": "diverged" if v is None else money(v)}
                 for k, v in t[key].items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tab 4: appraiser — price a 2010 house after editing a few characteristics
# ---------------------------------------------------------------------------
# Reading the raw CSV and re-applying the ROW-WISE feature engineering is not
# training: no statistic is learned. Everything learned (frontage medians,
# preprocessor, model weights) comes from artifacts/.
_raw = filter_population(load_data())
TEST_HOUSES = _raw[_raw["Yr Sold"] == TEST_YEAR].reset_index(drop=True)
HOUSE_CHOICES = [
    f"{i}: {r['Neighborhood']}, {int(r['Gr Liv Area'])} sq ft, sold {money(r[TARGET])}"
    for i, r in TEST_HOUSES.iterrows()
]
QUALITY_LEVELS = ["Ex", "Gd", "TA", "Fa", "Po"]
NEIGHBORHOODS = sorted(_raw["Neighborhood"].unique())


def house_defaults(choice: str):
    r = TEST_HOUSES.loc[int(choice.split(":")[0])].fillna(0)
    return (int(r["Overall Qual"]), int(r["Gr Liv Area"]), int(r["Total Bsmt SF"]),
            int(r["Year Built"]), r["Neighborhood"], int(r["Garage Cars"]),
            int(r["Full Bath"]), r["Kitchen Qual"])


def appraise(choice, overall_qual, gr_liv_area, bsmt_sf, year_built, neighborhood,
             garage_cars, full_bath, kitchen_qual):
    house = TEST_HOUSES.loc[[int(choice.split(":")[0])]].copy()
    actual = float(house[TARGET].iat[0])
    house["Overall Qual"] = overall_qual
    house["Gr Liv Area"] = gr_liv_area
    house["Total Bsmt SF"] = bsmt_sf
    house["Year Built"] = year_built
    house["Year Remod/Add"] = max(year_built, int(house["Year Remod/Add"].iat[0]))
    house["Neighborhood"] = neighborhood
    house["Garage Cars"] = garage_cars
    house["Full Bath"] = full_bath
    house["Kitchen Qual"] = kitchen_qual

    features = engineer_features(house)
    features = impute_lot_frontage(features, FRONTAGE_MEDIANS)
    X = PREPROCESSOR.transform(features[FEATURES])   # transform only, never fit
    preds = {k: float(np.exp(v[0])) for k, v in predict_log_price(MODELS, X).items()}
    preds["baseline"] = SUMMARY["baseline_price"]

    table = pd.DataFrame([{"Model": LABELS[k], "Estimated price": money(v)} for k, v in preds.items()])
    mae = SUMMARY["metrics"]["test"]["sklearn"]["MAE ($)"]
    note = (f"**Estimate: {money(preds['sklearn'])}** (typical error on 2010 sales: ±{money(mae)}).  \n"
            f"Recorded 2010 sale price of the original house: {money(actual)}"
            " — only comparable if you have not edited its characteristics.")
    return note, table


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
INTRO = f"""
# Ames home appraisal — one ridge regression, three implementations

Predicts the sale price of an ordinary (arm's-length) residential sale in Ames, Iowa.
Trained on {SUMMARY['split_sizes']['train']:,} sales from 2006-08, tuned on
{SUMMARY['split_sizes']['val']:,} sales from 2009, tested once on
{SUMMARY['split_sizes']['test']:,} sales from 2010. Target: log(price); decision metric:
MAE in dollars. Ridge alpha = {SUMMARY['alpha']}, PyTorch learning rate = {SUMMARY['lr']['manual_torch']},
{SUMMARY['epochs']:,} full-batch epochs. All views load saved artifacts — nothing is retrained here.
"""

with gr.Blocks(title="Ames appraisal model") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("1 · Model comparison"):
        split_radio = gr.Radio(choices=[("Validation 2009", "val"), ("Test 2010", "test")],
                               value="test", label="Evaluation year")
        pred_plot = gr.Plot()
        metrics_df = gr.Dataframe(label="Metrics (dollars; predictions converted back with exp)",
                                  interactive=False)
        resid_plot = gr.Plot()
        gr.Markdown("### Do the three implementations agree? (test year)")
        gr.Dataframe(value=agreement_table(), interactive=False)
        split_radio.change(update_comparison, split_radio, [pred_plot, resid_plot, metrics_df])

    with gr.Tab("2 · Distributions"):
        column_dd = gr.Dropdown(choices=DIST_CHOICES, value=TARGET, label="Feature or target")
        dist_plot = gr.Plot()
        dist_df = gr.Dataframe(label="Summary by split", interactive=False)
        gr.Markdown("Engineered features are shown as the model sees them before scaling "
                    "(e.g. `log_TotalSF`, quality ratings mapped to 0-5).")
        column_dd.change(lambda c: (distribution_plot(c), distribution_stats(c)),
                         column_dd, [dist_plot, dist_df])

    with gr.Tab("3 · Training"):
        gr.Plot(value=loss_plot())
        gr.Markdown(
            "The two PyTorch versions run the same full-batch gradient descent from the same "
            "starting point, so their curves overlap. Both approach scikit-learn's closed-form "
            "solution; the remaining gap is float32 rounding."
        )
        top_slider = gr.Slider(5, 40, value=15, step=1, label="Number of coefficients to show")
        coef_plot = gr.Plot()
        gr.Markdown(
            "Coefficients of strongly correlated features offset each other "
            "(e.g. `Qual_x_logSF` positive, `Overall Qual` negative): read them together, not one by one."
        )
        top_slider.change(coefficient_plot, top_slider, coef_plot)
        gr.Markdown("### Tuning on the validation year (2009)")
        gr.Dataframe(value=tuning_table(), interactive=False)

    with gr.Tab("4 · Appraiser"):
        gr.Markdown("Pick a house sold in 2010, edit some of its characteristics, and price it "
                    "with the three saved models. Other characteristics stay as recorded.")
        house_dd = gr.Dropdown(choices=HOUSE_CHOICES, value=HOUSE_CHOICES[0], label="Starting house")
        with gr.Row():
            in_qual = gr.Slider(1, 10, step=1, label="Overall quality (1-10)")
            in_area = gr.Number(label="Above-ground living area (sq ft)", precision=0, minimum=300)
            in_bsmt = gr.Number(label="Basement area (sq ft)", precision=0, minimum=0)
            in_year = gr.Number(label="Year built", precision=0, minimum=1870, maximum=TEST_YEAR)
        with gr.Row():
            in_nbhd = gr.Dropdown(choices=NEIGHBORHOODS, label="Neighborhood")
            in_garage = gr.Slider(0, 4, step=1, label="Garage capacity (cars)")
            in_bath = gr.Slider(0, 4, step=1, label="Full bathrooms (above ground)")
            in_kitchen = gr.Dropdown(choices=QUALITY_LEVELS, label="Kitchen quality (Ex > Gd > TA > Fa > Po)")
        inputs = [in_qual, in_area, in_bsmt, in_year, in_nbhd, in_garage, in_bath, in_kitchen]
        appraise_btn = gr.Button("Estimate price", variant="primary")
        estimate_md = gr.Markdown()
        estimate_df = gr.Dataframe(interactive=False)
        house_dd.change(house_defaults, house_dd, inputs)
        appraise_btn.click(appraise, [house_dd] + inputs, [estimate_md, estimate_df])

    # Fill every view on page load.
    demo.load(update_comparison, split_radio, [pred_plot, resid_plot, metrics_df])
    demo.load(lambda c: (distribution_plot(c), distribution_stats(c)), column_dd, [dist_plot, dist_df])
    demo.load(coefficient_plot, top_slider, coef_plot)
    demo.load(house_defaults, house_dd, inputs)


if __name__ == "__main__":
    demo.launch()
