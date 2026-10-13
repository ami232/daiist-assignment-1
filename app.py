"""
Assignment 1 - Gradio dashboard.  Run with:  uv run python main.py app

Everything shown here comes from the files train.py wrote to artifacts/:
the three saved models, the preprocessing constants, the table of
inspections and results.json. At startup the app loads the models and asks
each one for its predictions. It never fits anything.

Tabs
  Overview            headline numbers, actual vs predicted over time, results table
  Three models        prediction-vs-actual plot, agreement between implementations
  Features            what each feature group adds, distributions of features and target
  Threshold and cost  threshold slider, confusion matrix, cost of mistakes
  Priority list       the ranked list an inspection team would actually receive
  Score a restaurant  what-if scoring with all three models
"""

import html
import json
import os
import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

import charts
import features as F
import models as M

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# --- Load what training produced ----------------------------------------------------
if not (ARTIFACTS / "results.json").exists():
    raise SystemExit("No trained models found in artifacts/. Run `uv run python main.py train` first.")

RESULTS = json.loads((ARTIFACTS / "results.json").read_text())
PREP = joblib.load(ARTIFACTS / "preprocessing.joblib")
TABLE = pd.read_csv(ARTIFACTS / "model_table.csv", parse_dates=["inspection_date"], dtype={"zip": "string"})
MODELS = M.load_models(ARTIFACTS, n_features=len(F.FEATURE_NAMES))

MODEL_FEATURES = F.to_model_features(TABLE, PREP["constants"])
X = PREP["scaler"].transform(MODEL_FEATURES)
for kind, model in MODELS.items():  # inference only
    TABLE[f"p_{kind}"] = M.predict(kind, model, X)
TABLE["fail_rate_3y"] = MODEL_FEATURES["fail_rate_3y"]

KINDS = ["sklearn", "manual", "module"]
SPLITS = RESULTS["data"]["splits"]
DECISION = RESULTS["decision"]
PERIODS = {"Validation year": "validation", "Test year": "test"}
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def rows_of(split: str) -> pd.DataFrame:
    return TABLE if split == "all" else TABLE[TABLE["split"] == split]


def period_text(split: str) -> str:
    start, end = pd.Timestamp(SPLITS[split]["start"]), pd.Timestamp(SPLITS[split]["end"])
    return f"{start:%b %Y} to {end:%b %Y}"


# --- Small HTML helpers ----------------------------------------------------------------
def tiles(items: list[tuple[str, str, str]], compact: bool = False) -> str:
    """A row of stat tiles: (label, value, note)."""
    cells = "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div><div class="tile-note">{note}</div></div>'
        for label, value, note in items
    )
    return f'<div class="tiles{" compact" if compact else ""}">{cells}</div>'


def money(value: float) -> str:
    return f"${value:,.0f}"


NUMBER_LIKE = re.compile(r"[$+\-]?[\d,.]+(e[+\-]\d+)?%?")


def html_table(frame: pd.DataFrame, raw_columns: tuple[str, ...] = (), scroll: bool = False) -> str:
    """A plain HTML table. Columns that only hold numbers are right-aligned.

    Every cell is escaped, because restaurant names come straight from the
    city's data. `raw_columns` are the exception: markup this file built itself.
    """
    numeric = {
        column for column in frame.columns
        if column not in raw_columns and all(NUMBER_LIKE.fullmatch(str(value)) for value in frame[column])
    }
    css = lambda column: ' class="num"' if column in numeric else ""
    head = "".join(f"<th{css(column)}>{html.escape(str(column))}</th>" for column in frame.columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td{css(column)}>{value if column in raw_columns else html.escape(str(value))}</td>"
            for column, value in zip(frame.columns, row)
        ) + "</tr>"
        for row in frame.itertuples(index=False)
    )
    classes = "table-wrap" + (" scroll" if scroll else "") + (" narrow" if len(frame.columns) <= 3 else "")
    return f'<div class="{classes}"><table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# --- Overview ----------------------------------------------------------------------------
def overview_tiles() -> str:
    test = RESULTS["models"]["sklearn"]["metrics"]["test"]
    validation = RESULTS["models"]["sklearn"]["metrics"]["validation"]
    n, fails = len(rows_of("test")), int(rows_of("test")["failed"].sum())
    no_model = 1000 * min(DECISION["cost_false_negative"] * fails, DECISION["cost_false_positive"] * (n - fails)) / n
    saving = 1 - test["cost_per_1000"] / no_model
    gap = max(RESULTS["models"][kind]["max_probability_gap_to_sklearn"] for kind in KINDS)
    return tiles([
        ("Test-year ROC AUC", f"{test['roc_auc']:.3f}", f"Validation year {validation['roc_auc']:.3f}. A coin flip scores 0.500."),
        ("Failures found in the riskiest 20%", f"{test['capture_top20']:.0%}", "Inspecting in random order would find 20%."),
        ("Cost of mistakes per 1,000 inspections", money(test["cost_per_1000"]),
         f"{saving:.0%} below the best policy without a model ({money(no_model)})."),
        ("Largest gap between the three implementations", f"{100 * gap:.2f} pp",
         "Biggest difference in any test-year probability, in percentage points."),
    ])


def results_table(period: str) -> str:
    split = PERIODS[period]
    entries = [(label, values[split]) for label, values in RESULTS["baselines"].items()]
    entries += [(RESULTS["models"][kind]["name"], RESULTS["models"][kind]["metrics"][split]) for kind in KINDS]
    return html_table(pd.DataFrame([{
        "Model": label,
        "ROC AUC": f"{m['roc_auc']:.4f}",
        "PR AUC": f"{m['pr_auc']:.4f}",
        "Log-loss": f"{m['log_loss']:.4f}",
        "Brier": f"{m['brier']:.4f}",
        "Fails in top 20%": f"{m['capture_top20']:.1%}",
        "Threshold": f"{m['threshold']:.2f}",
        "Prioritised": f"{m['flagged_share']:.1%}",
        "Precision": f"{m['precision']:.1%}",
        "Recall": f"{m['recall']:.1%}",
        "Cost per 1,000": money(m["cost_per_1000"]),
    } for label, m in entries]))


# --- Three models --------------------------------------------------------------------------
def comparison_plots(period: str):
    rows = rows_of(PERIODS[period])
    return charts.calibration(rows, KINDS), charts.prediction_gap(rows)


def agreement_table() -> pd.DataFrame:
    tuning = RESULTS["tuning"]
    final_gap = {
        "sklearn": 0.0,
        "manual": next(run["objective_gap"] for run in tuning["manual"]["search"] if run["lr"] == tuning["manual"]["lr"]),
        "module": next(run["objective_gap"] for run in tuning["module"]["search"] if run["lr"] == tuning["module"]["lr"]),
    }
    how = {
        "sklearn": "L-BFGS solver, run to convergence",
        "manual": f"Full-batch gradient descent, lr {tuning['manual']['lr']:g}, {tuning['manual']['epochs']} epochs",
        "module": f"Mini-batch SGD ({tuning['module']['batch_size']} rows), lr {tuning['module']['lr']:g} "
                  f"x {tuning['module']['lr_decay']} per epoch, {tuning['module']['epochs']} epochs",
    }
    return pd.DataFrame([{
        "Implementation": RESULTS["models"][kind]["name"],
        "How it was optimised": how[kind],
        "Objective above scikit-learn": f"{max(final_gap[kind], 0):.1e}",
        "Largest coefficient gap": f"{RESULTS['models'][kind]['max_coef_gap_to_sklearn']:.1e}",
        "Largest probability gap": f"{RESULTS['models'][kind]['max_probability_gap_to_sklearn']:.1e}",
        "Test AUC": f"{RESULTS['models'][kind]['metrics']['test']['roc_auc']:.4f}",
        "Test log-loss": f"{RESULTS['models'][kind]['metrics']['test']['log_loss']:.4f}",
    } for kind in KINDS])


def investigation_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "nn.Module run": row["variant"],
        "Largest coefficient gap to scikit-learn": f"{row['max_coef_gap']:.1e}",
    } for row in RESULTS["tuning"]["module"]["investigation"]])


def regularisation_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "C": f"{row['C']:g}" + ("  (chosen)" if row["C"] == RESULTS["tuning"]["C"] else ""),
        "Validation log-loss": f"{row['log_loss']:.4f}",
        "Validation ROC AUC": f"{row['roc_auc']:.4f}",
    } for row in RESULTS["tuning"]["C_grid"]])


# --- Features --------------------------------------------------------------------------------
# label -> (column, shown as a percentage?)
EXPLORABLE = {
    "ZIP fail rate, last 12 months": ("zip_fail_rate_12m", True),
    "Own fail rate, last 3 years (smoothed)": ("fail_rate_3y", True),
    "Average violations per past inspection": ("mean_viol_prior", False),
    "Violations at the last inspection": ("prev_n_viol", False),
    "Past inspections in the last 3 years": ("n_prior", False),
    "Days since the last inspection": ("days_since_last", False),
    "Complaint visits in the last 3 years": ("n_complaints", False),
    "Chain size (restaurants sharing the name)": ("chain_bucket", False),
    "Month of the inspection": ("month", False),
    "City risk class": ("risk", False),
    "Last routine inspection failed (1 = yes)": ("prev_canvass_fail", False),
    "Pests cited before (1 = yes)": ("pest_any", False),
    "Temperature control cited before (1 = yes)": ("temp_any", False),
    "Management issues cited before (1 = yes)": ("mgmt_any", False),
}
TABLE["chain_bucket"] = pd.cut(
    TABLE["chain_size"], [0, 1, 2, 4, 9, 29, 10_000], labels=["1", "2", "3-4", "5-9", "10-29", "30+"]
).astype(str)
EXPLORE_PERIODS = {"Training years": "train", "Validation year": "validation", "Test year": "test", "All periods": "all"}


def feature_plots(label: str, period: str):
    column, percent = EXPLORABLE[label]
    rows = rows_of(EXPLORE_PERIODS[period])
    known = rows[rows[column].notna()]
    missing = len(rows) - len(known)
    note = (
        f"{len(known):,} inspections shown. {missing:,} more have no inspection in the previous 3 years to compute this from; "
        "the model fills those in with a typical value and switches on the no-history flag."
        if missing else f"{len(known):,} inspections shown."
    )
    return (
        charts.feature_distribution(known, column, label, percent),
        charts.feature_fail_rate(known, column, label, percent),
        note,
    )


def feature_dictionary() -> pd.DataFrame:
    coefficient = RESULTS["models"]["sklearn"]["coefficients"]
    return pd.DataFrame([{
        "Group": group, "Feature": label, "What it is": description, "Coefficient": f"{coefficient[name]:+.3f}",
    } for name, group, label, description in F.FEATURES])


# --- Threshold and cost ------------------------------------------------------------------------
def confusion_counts(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    flagged = p >= threshold
    return (int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum()),
            int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum()))


def threshold_view(model_name: str, period: str, threshold: float, cost_fn: float, cost_fp: float):
    kind = next(kind for kind in KINDS if charts.MODEL_NAMES[kind] == model_name)
    rows = rows_of(PERIODS[period])
    y, p = rows["failed"].to_numpy(), rows[f"p_{kind}"].to_numpy()
    cost_fn, cost_fp = float(cost_fn or 0), float(cost_fp or 0)

    tp, fp, fn, tn = confusion_counts(y, p, threshold)
    cost = cost_fn * fn + cost_fp * fp
    cost_all, cost_none = cost_fp * (fp + tn), cost_fn * (tp + fn)  # prioritise everyone / no one
    no_model = min(cost_all, cost_none)
    difference = cost - no_model
    versus = "No model policy to compare" if no_model == 0 else (
        f"{abs(difference) / no_model:.0%} {'below' if difference <= 0 else 'above'} the cheaper no-model policy ({money(no_model)})"
    )

    curve = []
    for t in THRESHOLDS:
        _, fp_t, fn_t, _ = confusion_counts(y, p, t)
        curve.append(cost_fn * fn_t + cost_fp * fp_t)

    tile_html = tiles([
        ("Cost of mistakes", money(cost), f"{money(1000 * cost / len(y))} per 1,000 inspections. {versus}."),
        ("Restaurants prioritised", f"{(tp + fp) / len(y):.0%}", f"{tp + fp:,} of {len(y):,} inspections in the period."),
        ("Precision", f"{tp / (tp + fp):.0%}" if tp + fp else "n/a", "Of the prioritised restaurants, the share that really failed."),
        ("Recall", f"{tp / (tp + fn):.0%}" if tp + fn else "n/a", "Of the restaurants that failed, the share that was prioritised."),
    ])
    return (
        tile_html,
        charts.confusion_matrix(tp, fp, fn, tn),
        charts.cost_curve(THRESHOLDS, curve, threshold, cost, DECISION["threshold"], cost_all, cost_none),
    )


# --- Priority list -----------------------------------------------------------------------------
TEST_MONTHS = sorted(rows_of("test")["inspection_date"].dt.to_period("M").unique())
MONTH_CHOICES = [month.strftime("%B %Y") for month in TEST_MONTHS]


def priority_list(month_label: str, capacity: int):
    month = TEST_MONTHS[MONTH_CHOICES.index(month_label)]
    due = rows_of("test")
    due = due[due["inspection_date"].dt.to_period("M") == month].sort_values("p_sklearn", ascending=False)
    capacity = int(min(capacity, len(due)))
    chosen = due.head(capacity)

    found, total_fails = int(chosen["failed"].sum()), int(due["failed"].sum())
    by_chance = capacity * due["failed"].mean()
    tile_html = tiles([
        ("Failing restaurants in the priority list", f"{found} of {capacity}", f"A list drawn at random would hold about {by_chance:.0f}."),
        ("Share of the month's failures reached first", f"{found / total_fails:.0%}" if total_fails else "n/a",
         f"{total_fails} of the {len(due):,} restaurants inspected that month failed."),
        ("Hit rate of the list", f"{found / capacity:.0%}" if capacity else "n/a", f"The month's overall fail rate was {due['failed'].mean():.0%}."),
    ])
    listing = pd.DataFrame({
        "Rank": np.arange(1, capacity + 1),
        "Restaurant": chosen["dba_name"].str.title().to_numpy(),
        "ZIP": chosen["zip"].to_numpy(),
        "Predicted risk": [f"{value:.0%}" for value in chosen["p_sklearn"]],
        "ZIP fail rate (12 m)": [f"{value:.0%}" for value in chosen["zip_fail_rate_12m"]],
        "Avg violations": ["no history" if pd.isna(value) else f"{value:.1f}" for value in chosen["mean_viol_prior"]],
        "Past inspections": chosen["n_prior"].astype(int).to_numpy(),
        "What happened": [
            f'<span class="outcome {"fail" if result == "Fail" else "ok"}"></span>{html.escape(result)}'
            for result in chosen["results"]
        ],
    })
    return tile_html, html_table(listing, raw_columns=("What happened",), scroll=True)


# --- Score a restaurant -------------------------------------------------------------------------
def pick_examples() -> dict[str, pd.Series]:
    """Real test-year inspections at fixed points of the risk ranking."""
    ranked = rows_of("test").sort_values("p_sklearn").reset_index(drop=True)
    points = {"Highest predicted risk in the test year": 1.0, "Riskier than 90% of restaurants": 0.9,
              "A typical restaurant (median risk)": 0.5, "Safer than 90% of restaurants": 0.1,
              "Lowest predicted risk in the test year": 0.0}
    return {label: ranked.iloc[int(round(q * (len(ranked) - 1)))] for label, q in points.items()}


EXAMPLES = pick_examples()


def example_values(label: str) -> list:
    row = EXAMPLES[label]
    number = lambda column, default: float(default if pd.isna(row[column]) else row[column])
    return [
        row["risk"], int(row["chain_size"]), MONTH_NAMES[int(row["month"]) - 1], round(100 * float(row["zip_fail_rate_12m"]), 1),
        int(row["n_prior"]), int(row["n_fail_prior"]), bool(number("prev_canvass_fail", 0)),
        int(number("days_since_last", PREP["constants"]["medians"]["days_since_last"])), int(row["n_complaints"]),
        round(number("mean_viol_prior", PREP["constants"]["medians"]["mean_viol_prior"]), 1),
        int(number("prev_n_viol", PREP["constants"]["medians"]["prev_n_viol"])),
        bool(number("pest_any", 0)), bool(number("temp_any", 0)), bool(number("mgmt_any", 0)),
    ]


def score(risk, chain_size, month, zip_rate, n_prior, n_fail_prior, prev_canvass_fail, days_since_last,
          n_complaints, mean_viol_prior, prev_n_viol, pest_any, temp_any, mgmt_any):
    n_prior = int(n_prior or 0)
    has_history = n_prior > 0
    # With no earlier inspection the history fields do not exist. Leaving
    # them missing sends the restaurant through the same imputation as in training.
    raw = pd.DataFrame([{
        "risk": risk,
        "chain_size": max(int(chain_size or 1), 1),
        "month": MONTH_NAMES.index(month) + 1,
        "zip_fail_rate_12m": float(zip_rate or 0) / 100,
        "n_prior": n_prior,
        "n_fail_prior": min(int(n_fail_prior or 0), n_prior),
        "prev_canvass_fail": float(prev_canvass_fail) if has_history else np.nan,
        "days_since_last": float(days_since_last or 0) if has_history else np.nan,
        "n_complaints": int(n_complaints or 0),
        "mean_viol_prior": float(mean_viol_prior or 0) if has_history else np.nan,
        "prev_n_viol": float(prev_n_viol or 0) if has_history else np.nan,
        "pest_any": float(pest_any) if has_history else np.nan,
        "temp_any": float(temp_any) if has_history else np.nan,
        "mgmt_any": float(mgmt_any) if has_history else np.nan,
    }])
    model_features = F.to_model_features(raw, PREP["constants"])
    standardised = PREP["scaler"].transform(model_features)
    probabilities = {kind: float(M.predict(kind, MODELS[kind], standardised)[0]) for kind in KINDS}

    threshold = DECISION["threshold"]
    decision = "Priority queue" if probabilities["sklearn"] >= threshold else "Normal queue"
    tile_html = tiles(
        [(charts.MODEL_NAMES[kind], f"{probabilities[kind]:.1%}", "Predicted probability of failing.") for kind in KINDS]
        + [("Decision", decision, f"Threshold {threshold:.2f}, chosen on the validation year.")],
        compact=True,
    )
    # In a logistic regression each feature adds coefficient x standardised
    # value to the log-odds, and a standardised value of 0 is the training average.
    coefficient = pd.Series(RESULTS["models"]["sklearn"]["coefficients"])
    contribution = coefficient * pd.Series(standardised[0], index=F.FEATURE_NAMES)
    return tile_html, charts.contributions(contribution, F.FEATURE_LABELS)


# --- Layout --------------------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
#header h1 { font-size: 30px; font-weight: 650; letter-spacing: -0.02em; margin: 6px 0 6px; color: #0b0b0b; }
#header p { font-size: 15.5px; line-height: 1.55; color: #52514e; max-width: 78ch; margin: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.tile { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 16px 18px; }
.tile-label { font-size: 13px; color: #52514e; }
.tile-value { font-size: 30px; font-weight: 650; letter-spacing: -0.02em; color: #0b0b0b; margin: 6px 0 4px; }
.tile-note { font-size: 12.5px; line-height: 1.45; color: #898781; }
.section h3 { font-size: 17px; font-weight: 650; color: #0b0b0b; margin: 18px 0 2px; }
.section p { font-size: 14px; line-height: 1.55; color: #52514e; max-width: 86ch; margin: 2px 0 0; }
.caption p { font-size: 13px; color: #898781; }
.table-wrap { overflow-x: auto; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; background: #fcfcfb; }
.table-wrap.scroll { max-height: 600px; overflow-y: auto; }
table.data { width: 100%; border-collapse: collapse; font-size: 13.5px; margin: 0; }
.table-wrap.narrow { max-width: 720px; }
table.data th, table.data td { border: none !important; }
table.data th { text-align: left; font-weight: 600; color: #52514e; padding: 10px 14px; vertical-align: bottom;
                background: #f4f3ef; border-bottom: 1px solid #e1e0d9 !important; position: sticky; top: 0; }
table.data td { padding: 9px 14px; color: #0b0b0b; border-bottom: 1px solid #f0efec !important; line-height: 1.45; }
table.data tr:last-child td { border-bottom: none !important; }
.tiles.compact { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.tiles.compact .tile-value { font-size: 23px; }
table.data .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.outcome { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; background: #b4b2a9; }
.outcome.fail { background: #d03b3b; }
footer { display: none !important; }
"""

# Runs once in the browser when the page loads.
# 1. The charts are designed for a light surface, so the page stays light
#    even when the browser asks for a dark theme.
# 2. Plotly only re-fits a chart when the window itself is resized. This
#    also re-fits when the page layout changes width or a tab is opened.
PAGE_SCRIPT = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get("__theme") !== "light") {
        url.searchParams.set("__theme", "light");
        window.location.replace(url);
        return;
    }
    let timer;
    const refit = () => {
        clearTimeout(timer);
        timer = setTimeout(() => window.dispatchEvent(new Event("resize")), 120);
    };
    let lastWidth = 0;
    new ResizeObserver((entries) => {
        const width = Math.round(entries[0].contentRect.width);
        if (width !== lastWidth) { lastWidth = width; refit(); }
    }).observe(document.body);
    document.addEventListener("click", (event) => {
        if (event.target.closest('[role="tab"]')) refit();
    });
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.stone,
    # Five names on purpose: with exactly four, Gradio 6.28 crashes at launch
    # while comparing this list with its built-in themes' font lists.
    font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
).set(
    body_background_fill="#f9f9f7",
    block_background_fill="#fcfcfb",
    block_border_color="rgba(11,11,11,0.10)",
    block_shadow="none",
    block_radius="10px",
)


def section(title: str, text: str) -> gr.Markdown:
    return gr.Markdown(f"### {title}\n{text}", elem_classes="section")


def plot(figure=None) -> gr.Plot:
    return gr.Plot(value=figure, show_label=False)


def table(content: pd.DataFrame | str) -> gr.HTML:
    return gr.HTML(content if isinstance(content, str) else html_table(content), padding=False)


default_period = "Test year"

with gr.Blocks(title="Which restaurants should be inspected first?") as demo:
    gr.HTML(
        '<div id="header"><h1>Which restaurants should be inspected first?</h1>'
        "<p>Chicago inspects every restaurant sooner or later. This model ranks the restaurants due for a "
        "routine inspection by how likely they are to fail, so the ones with problems are reached earlier. "
        f"It is a logistic regression on {len(F.FEATURE_NAMES)} features built from each restaurant's past inspections, "
        f"trained three ways on {SPLITS['train']['rows']:,} inspections and tested on the most recent year "
        f"({period_text('test')}, {SPLITS['test']['rows']:,} inspections).</p></div>"
    )

    with gr.Tabs():
        # ------------------------------------------------------------------ Overview
        with gr.Tab("Overview"):
            gr.HTML(overview_tiles())
            section(
                "Actual and predicted fail rate, month by month",
                "The shaded bands are the three periods. The model only ever learned from the training band; "
                "validation was used to choose the regularisation strength and the threshold, and the test year was "
                "looked at once, at the end.",
            )
            plot(charts.timeline(TABLE, SPLITS))
            section("Results on one period", "The same split for every row. Each baseline uses its own cheapest threshold from the validation year.")
            overview_period = gr.Radio(list(PERIODS), value=default_period, show_label=False, container=False)
            overview_table = table(results_table(default_period))
            section(
                "How steady is the model from quarter to quarter?",
                "One AUC per quarter. Training quarters are scored on data the model has seen, so they flatter it slightly. "
                "The validation year turned out to be the easiest stretch; the test year is back in the range of the "
                "training quarters rather than on a downward slope.",
            )
            plot(charts.stability(RESULTS["stability"], SPLITS))
            overview_period.change(results_table, overview_period, overview_table)

        # ------------------------------------------------------------------ Three models
        with gr.Tab("Three models"):
            section(
                "Prediction versus actual",
                "Inspections are sorted by predicted probability and cut into ten equal groups. Each marker compares a "
                "group's average prediction with the share that really failed; the dashed diagonal is a perfect match. "
                "The three implementations use markers of different sizes, so where they agree they stack into one bullseye.",
            )
            models_period = gr.Radio(list(PERIODS), value=default_period, show_label=False, container=False)
            initial_calibration, initial_gap = comparison_plots(default_period)
            with gr.Row():
                calibration_plot = plot(initial_calibration)
                gap_plot = plot(initial_gap)
            gr.Markdown(
                "Right: every dot is one inspection, placed by how far a PyTorch prediction is from scikit-learn's. "
                "The manual loop sits on the zero line; the nn.Module version scatters within a fraction of a percentage point.",
                elem_classes="caption",
            )
            models_period.change(comparison_plots, models_period, [calibration_plot, gap_plot])

            section("Do the three implementations agree?", "All three minimise the same objective on the same training matrix, so they should land on the same weights.")
            table(agreement_table())
            section(
                "Coefficients",
                "Bars are scikit-learn; the diamond is the manual loop and the dot is nn.Module. Features are standardised, "
                "so a bar's length is the change in log-odds for one standard deviation of that feature.",
            )
            plot(charts.coefficients(RESULTS["models"], F.FEATURE_LABELS))

            section(
                "Learning rate: how each PyTorch run approached scikit-learn's minimum",
                "Vertical axis: how far the training objective still is from the value scikit-learn reached (log scale). "
                "Left: the manual loop. Right: nn.Module with mini-batches. Too small a learning rate never arrives; "
                "too large a one overshoots in the manual loop.",
            )
            tuning = RESULTS["tuning"]
            with gr.Row():
                plot(charts.convergence(tuning["manual"]["search"], tuning["sklearn_objective"], tuning["manual"]["lr"], charts.MODEL_COLORS["manual"]))
                plot(charts.convergence(tuning["module"]["search"], tuning["sklearn_objective"], tuning["module"]["lr"], charts.MODEL_COLORS["module"]))
            section(
                "Why the nn.Module version is close but not identical",
                "Its only algorithmic difference from the manual loop is mini-batching. Without the learning-rate decay the gap is "
                "larger; with the manual loop's recipe (full batch, 500 epochs) it disappears.",
            )
            table(investigation_table())
            section("Regularisation strength", "Chosen on validation log-loss. Among values that tie at three decimals, the strongest regularisation wins.")
            table(regularisation_table())

        # ------------------------------------------------------------------ Features
        with gr.Tab("Features"):
            section(
                "What each feature group adds",
                "Left: validation AUC as the groups are added in order. Right: what the full model loses when one group is removed.",
            )
            with gr.Row():
                plot(charts.ablation_ladder(RESULTS["ablation"]["cumulative"]))
                plot(charts.ablation_drop(RESULTS["ablation"]["leave_one_out"]))

            section("Explore one feature", "Left: where passed and failed inspections sit along the feature. Right: the fail rate across its range.")
            with gr.Row():
                feature_choice = gr.Dropdown(list(EXPLORABLE), value=next(iter(EXPLORABLE)), label="Feature", scale=2)
                feature_period = gr.Radio(list(EXPLORE_PERIODS), value="Training years", label="Period", scale=3)
            initial_distribution, initial_rate, initial_note = feature_plots(next(iter(EXPLORABLE)), "Training years")
            with gr.Row():
                distribution_plot = plot(initial_distribution)
                rate_plot = plot(initial_rate)
            feature_note = gr.Markdown(initial_note, elem_classes="caption")
            gr.on(
                [feature_choice.change, feature_period.change], feature_plots,
                [feature_choice, feature_period], [distribution_plot, rate_plot, feature_note],
            )

            section(
                "The target",
                "A routine inspection ends in Pass, Pass with Conditions or Fail. Only Fail counts as the positive class. "
                "Visits with no outcome (business closed, no entry) are left out.",
            )
            plot(charts.result_mix(TABLE))
            section("Feature dictionary", "Coefficients are from the scikit-learn model, per standard deviation of the feature.")
            table(feature_dictionary())

        # ------------------------------------------------------------------ Threshold and cost
        with gr.Tab("Threshold and cost"):
            section(
                "Where to draw the line",
                "A restaurant goes into the priority queue when its predicted probability is at or above the threshold. "
                "A missed fail leaves violations unnoticed for longer; a wasted priority visit uses a slot on a restaurant "
                f"that would have passed. The validation year picked a threshold of {DECISION['threshold']:.2f}; for a perfectly "
                f"calibrated model the theory says {DECISION['theoretical_threshold']:.2f}.",
            )
            with gr.Row():
                threshold_model = gr.Radio([charts.MODEL_NAMES[kind] for kind in KINDS], value=charts.MODEL_NAMES["sklearn"], label="Model", scale=3)
                threshold_period = gr.Radio(list(PERIODS), value=default_period, label="Period", scale=2)
            with gr.Row():
                threshold_slider = gr.Slider(0.01, 0.99, value=DECISION["threshold"], step=0.01, label="Decision threshold", scale=3)
                cost_fn_input = gr.Number(DECISION["cost_false_negative"], label="Cost of a missed fail ($)", minimum=0, scale=1)
                cost_fp_input = gr.Number(DECISION["cost_false_positive"], label="Cost of a wasted priority visit ($)", minimum=0, scale=1)
            threshold_inputs = [threshold_model, threshold_period, threshold_slider, cost_fn_input, cost_fp_input]
            initial_tiles, initial_confusion, initial_curve = threshold_view(
                charts.MODEL_NAMES["sklearn"], default_period, DECISION["threshold"],
                DECISION["cost_false_negative"], DECISION["cost_false_positive"],
            )
            threshold_tiles = gr.HTML(initial_tiles)
            with gr.Row():
                confusion_plot = plot(initial_confusion)
                curve_plot = plot(initial_curve)
            gr.Markdown(
                "Left: the confusion matrix at your threshold. Right: the cost at every threshold; the dark dot is yours, "
                "and the dashed lines are the two policies that need no model.",
                elem_classes="caption",
            )
            gr.on(
                [component.change for component in threshold_inputs], threshold_view,
                threshold_inputs, [threshold_tiles, confusion_plot, curve_plot],
            )

        # ------------------------------------------------------------------ Priority list
        with gr.Tab("Priority list"):
            section(
                "What the inspection team would receive",
                "Pick a month of the test year and how many priority visits are available. The list ranks that month's "
                "restaurants by predicted risk; the last column shows what the inspector then found.",
            )
            with gr.Row():
                month_choice = gr.Dropdown(MONTH_CHOICES, value=MONTH_CHOICES[-1], label="Month", scale=1)
                capacity_slider = gr.Slider(10, 300, value=100, step=10, label="Priority visits available", scale=3)
            initial_list_tiles, initial_listing = priority_list(MONTH_CHOICES[-1], 100)
            list_tiles = gr.HTML(initial_list_tiles)
            listing = table(initial_listing)
            gr.on([month_choice.change, capacity_slider.change], priority_list, [month_choice, capacity_slider], [list_tiles, listing])

        # ------------------------------------------------------------------ Score a restaurant
        with gr.Tab("Score a restaurant"):
            section(
                "What would the model say?",
                "Start from a real test-year restaurant or type your own values. The same saved preprocessing and the three saved models score it live.",
            )
            first_example = "A typical restaurant (median risk)"
            defaults = example_values(first_example)
            example_choice = gr.Dropdown(list(EXAMPLES), value=first_example, label="Start from a real inspection")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        risk_input = gr.Radio(["Risk 1 (High)", "Risk 2 (Medium)", "Risk 3 (Low)"], value=defaults[0], label="City risk class")
                    with gr.Row():
                        chain_input = gr.Number(defaults[1], label="Restaurants sharing the name", precision=0, minimum=1)
                        month_input = gr.Dropdown(MONTH_NAMES, value=defaults[2], label="Month of the inspection")
                    zip_input = gr.Slider(0, 70, value=defaults[3], step=0.5, label="ZIP fail rate over the last 12 months (%)")
                    with gr.Row():
                        prior_input = gr.Number(defaults[4], label="Past inspections (3 years)", precision=0, minimum=0)
                        fails_input = gr.Number(defaults[5], label="Of which failed", precision=0, minimum=0)
                    with gr.Row():
                        days_input = gr.Number(defaults[7], label="Days since last inspection", precision=0, minimum=1)
                        complaints_input = gr.Number(defaults[8], label="Complaint visits (3 years)", precision=0, minimum=0)
                    with gr.Row():
                        mean_viol_input = gr.Number(defaults[9], label="Avg violations per inspection", minimum=0)
                        prev_viol_input = gr.Number(defaults[10], label="Violations at last inspection", precision=0, minimum=0)
                    canvass_input = gr.Checkbox(defaults[6], label="Last routine inspection failed")
                    pest_input = gr.Checkbox(defaults[11], label="Pests cited in the last 3 years")
                    temp_input = gr.Checkbox(defaults[12], label="Temperature control cited in the last 3 years")
                    mgmt_input = gr.Checkbox(defaults[13], label="Management issues cited in the last 3 years")
                    gr.Markdown("With zero past inspections the history fields are ignored and filled in as in training.", elem_classes="caption")
                score_inputs = [
                    risk_input, chain_input, month_input, zip_input, prior_input, fails_input, canvass_input,
                    days_input, complaints_input, mean_viol_input, prev_viol_input, pest_input, temp_input, mgmt_input,
                ]
                initial_score_tiles, initial_contributions = score(*defaults)
                with gr.Column(scale=3):
                    score_tiles = gr.HTML(initial_score_tiles)
                    gr.Markdown(
                        "Why: each bar is one feature's push on the log-odds, compared with an average training-period restaurant.",
                        elem_classes="caption",
                    )
                    contribution_plot = plot(initial_contributions)
            example_choice.change(example_values, example_choice, score_inputs)
            gr.on([component.change for component in score_inputs], score, score_inputs, [score_tiles, contribution_plot])

if __name__ == "__main__":
    # GitHub Actions sets CI=true; there is no browser to open there.
    demo.launch(theme=THEME, css=CSS, js=PAGE_SCRIPT, inbrowser=not os.environ.get("CI"))
