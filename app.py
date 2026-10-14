"""
Assignment 1 - Gradio dashboard (run with `uv run python main.py app`).

Everything shown here comes from artifacts written by train.py. The app loads
the fitted preprocessor and the three saved models, re-applies them to the
saved test quarter to prove the artifacts work, and never trains anything.

Tabs
  1. Overview            business framing, split timeline, headline numbers
  2. Data explorer       target and feature distributions per split
  3. Model comparison    metrics table, predicted-vs-actual, test weeks, residuals
  4. Three implementations  coefficient agreement, loss curves, objective values
  5. Dispatch decision   threshold slider -> confusion matrix + business cost
  6. What-if forecast    type a scenario, get all three models' forecasts
"""

from __future__ import annotations

import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from business import COST_MISSED_SURGE, COST_UNNEEDED_CREW, SURGE_THRESHOLD_RIDES, cost_curve, surge_decision
from models import ARTIFACTS, MODEL_LABELS, load_json, load_models, to_counts
from pipeline import LOG_TARGET, SPLITS, TARGET, derive_columns, make_whatif_row

# ---------------------------------------------------------------------------
# Load artifacts (no training here)
# ---------------------------------------------------------------------------
METRICS = load_json(ARTIFACTS / "metrics.json")
LOSS = load_json(ARTIFACTS / "loss_curves.json")
PREPROCESSOR = joblib.load(ARTIFACTS / "preprocessor.joblib")
MODELS = load_models()
TABLE = pd.read_csv(ARTIFACTS / "model_table.csv", parse_dates=["dteday"])
SAVED_PRED = pd.read_csv(ARTIFACTS / "test_predictions.csv", parse_dates=["dteday"])
ABLATION = pd.read_csv(ARTIFACTS / "ablation.csv")
TUNING = pd.read_csv(ARTIFACTS / "tuning.csv")

MODEL_KEYS = list(MODELS)  # sklearn, torch_manual, torch_standard
BASELINE_KEYS = ["baseline_mean", "baseline_hour_profile", "baseline_last_week_level"]
LABELS = {
    **MODEL_LABELS,
    "baseline_mean": "Baseline: training mean",
    "baseline_hour_profile": "Baseline: hour x day-type profile",
    "baseline_last_week_level": "Baseline: last week's same-hour level",
}
COLORS = {
    "actual": "#1f1f1f",
    "sklearn": "#2f6db5",
    "torch_manual": "#e07b2a",
    "torch_standard": "#2a9d5c",
    "baseline_mean": "#b8b8b8",
    "baseline_hour_profile": "#8c8c8c",
    "baseline_last_week_level": "#c9a227",
}
TEMPLATE = "plotly_white"

# Re-apply the saved preprocessor + models to the test quarter. This is
# inference only; it also checks the loaded artifacts reproduce what train.py
# saved, which is the whole point of saving them.
TEST = derive_columns(TABLE[TABLE["split"] == "test"].dropna(subset=["lag_level"]).reset_index(drop=True))
X_TEST = PREPROCESSOR.transform(TEST).astype(np.float32)
PRED = {k: to_counts(m.predict(X_TEST)) for k, m in MODELS.items()}
for k in BASELINE_KEYS:
    PRED[k] = SAVED_PRED[f"pred_{k}"].to_numpy()
Y_TEST = TEST[TARGET].to_numpy(dtype=float)
_max_diff = max(float(np.abs(PRED[k] - SAVED_PRED[f"pred_{k}"].to_numpy()).max()) for k in MODEL_KEYS)
print(f"[app] loaded 3 models; max |live prediction - saved prediction| on test = {_max_diff:.6f} rides", flush=True)
TEST_T = TEST["dteday"] + pd.to_timedelta(TEST["hr"], unit="h")

BIZ = METRICS["business"]
FEATURE_NAMES = METRICS["feature_names"]
CATEGORICAL = {"hr": "hour of day", "weekday": "weekday (0=Sun)", "mnth": "month", "season": "season",
               "weathersit": "weather (1 clear, 2 mist, 3 rain/snow)", "workingday": "working day",
               "holiday": "holiday", "yr": "year (0=2011, 1=2012)"}
NUMERIC = {"temp_c": "temperature (C)", "hum": "relative humidity (0-1)", "windspeed": "wind speed (normalised)",
           "lag_level": "recent same-hour level, log1p(rides)"}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _subset(split: str) -> pd.DataFrame:
    return TABLE if split == "all" else TABLE[TABLE["split"] == split]


def fig_split_timeline() -> go.Figure:
    daily = TABLE.groupby(["dteday", "split"], as_index=False)[TARGET].sum()
    fig = go.Figure()
    for split, color in (("train", "#2f6db5"), ("val", "#e07b2a"), ("test", "#2a9d5c")):
        d = daily[daily["split"] == split]
        a, b = SPLITS[split]
        fig.add_trace(go.Scatter(x=d["dteday"], y=d[TARGET], mode="lines", name=f"{split} ({a} .. {b})",
                                 line=dict(color=color, width=1)))
    fig.add_vline(x=pd.Timestamp("2012-10-29").timestamp() * 1000, line=dict(color="grey", dash="dot"))
    fig.add_annotation(x=pd.Timestamp("2012-10-29"), y=daily[TARGET].max(), text="Hurricane Sandy", showarrow=False,
                       xanchor="right", font=dict(size=10, color="grey"))
    fig.update_layout(template=TEMPLATE, title="Daily rides and the time-based split", yaxis_title="rides / day",
                      legend=dict(orientation="h", y=-0.2), height=360, margin=dict(t=50, b=40))
    return fig


def fig_target_hist(split: str, scale: str) -> go.Figure:
    d = _subset(split)
    col = TARGET if scale == "rides / hour" else LOG_TARGET
    fig = go.Figure(go.Histogram(x=d[col], nbinsx=60, marker_color="#2f6db5"))
    fig.update_layout(template=TEMPLATE, title=f"Target distribution: {scale} ({split}, n={len(d)})",
                      xaxis_title=scale, yaxis_title="hours", height=340, margin=dict(t=50, b=40))
    return fig


def fig_feature(feature: str, split: str) -> go.Figure:
    d = _subset(split)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("distribution", "mean rides / hour by value"))
    if feature in CATEGORICAL:
        counts = d[feature].value_counts().sort_index()
        means = d.groupby(feature)[TARGET].mean()
        fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color="#2f6db5", name="hours"), 1, 1)
        fig.add_trace(go.Bar(x=means.index.astype(str), y=means.values, marker_color="#e07b2a", name="mean rides"), 1, 2)
    else:
        fig.add_trace(go.Histogram(x=d[feature], nbinsx=40, marker_color="#2f6db5", name="hours"), 1, 1)
        bins = pd.cut(d[feature], bins=20)
        means = d.groupby(bins, observed=True)[TARGET].mean()
        centers = [iv.mid for iv in means.index]
        fig.add_trace(go.Scatter(x=centers, y=means.values, mode="lines+markers", marker_color="#e07b2a",
                                 name="mean rides"), 1, 2)
    label = CATEGORICAL.get(feature, NUMERIC.get(feature, feature))
    fig.update_layout(template=TEMPLATE, title=f"{label} ({split})", showlegend=False, height=360,
                      margin=dict(t=60, b=40))
    return fig


def fig_hour_profile(split: str) -> go.Figure:
    d = _subset(split)
    fig = go.Figure()
    for wd, name, color in ((1, "working day", "#2f6db5"), (0, "weekend / holiday", "#e07b2a")):
        prof = d[d["workingday"] == wd].groupby("hr")[TARGET].mean()
        fig.add_trace(go.Scatter(x=prof.index, y=prof.values, mode="lines+markers", name=name, line=dict(color=color)))
    fig.update_layout(template=TEMPLATE, title="Hourly profile by day type (why the model uses hour x working-day dummies)",
                      xaxis_title="hour of day", yaxis_title="mean rides / hour", height=340, margin=dict(t=50, b=40))
    return fig


def metrics_df(split: str) -> pd.DataFrame:
    rows = []
    for k in BASELINE_KEYS + MODEL_KEYS:
        m = METRICS["results"][k][split]
        rows.append({"model": LABELS[k], "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"], "RMSLE": m["rmsle"],
                     "bias (pred - actual)": m["bias"], "MAE on peak hours": m["mae_peak_hours"]})
    return pd.DataFrame(rows).round(3)


def fig_pred_vs_actual(selected: list[str], log_axes: bool) -> go.Figure:
    fig = go.Figure()
    lim = max(Y_TEST.max(), max(PRED[k].max() for k in selected)) * 1.03 if selected else Y_TEST.max()
    fig.add_trace(go.Scatter(x=[1, lim], y=[1, lim], mode="lines", line=dict(color="#1f1f1f", dash="dash", width=1),
                             name="perfect forecast"))
    for k in selected:
        fig.add_trace(go.Scatter(x=Y_TEST, y=PRED[k], mode="markers", name=LABELS[k],
                                 marker=dict(size=5, opacity=0.45, color=COLORS[k]),
                                 customdata=np.stack([TEST["dteday"].dt.strftime("%Y-%m-%d"), TEST["hr"]], axis=1),
                                 hovertemplate="%{customdata[0]} %{customdata[1]}h<br>actual %{x:.0f}<br>pred %{y:.0f}"))
    axis = dict(type="log") if log_axes else {}
    fig.update_layout(template=TEMPLATE, title="Predicted vs actual rides / hour (test quarter, Oct-Dec 2012)",
                      xaxis=dict(title="actual", **axis), yaxis=dict(title="predicted", **axis), height=480,
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=40))
    return fig


WEEK_STARTS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2012-10-01", "2012-12-24", freq="7D")]


def fig_week(week_start: str, selected: list[str]) -> go.Figure:
    start = pd.Timestamp(week_start)
    mask = ((TEST["dteday"] >= start) & (TEST["dteday"] < start + pd.Timedelta(days=7))).to_numpy()
    fig = go.Figure(go.Scatter(x=TEST_T[mask], y=Y_TEST[mask], mode="lines", name="actual",
                               line=dict(color=COLORS["actual"], width=2)))
    for k in selected:
        fig.add_trace(go.Scatter(x=TEST_T[mask], y=PRED[k][mask], mode="lines", name=LABELS[k],
                                 line=dict(color=COLORS[k], width=1.3)))
    fig.update_layout(template=TEMPLATE, title=f"Hourly forecast vs actual, week of {week_start}",
                      yaxis_title="rides / hour", height=380, legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=50, b=40))
    return fig


def fig_residuals(selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for k in selected:
        fig.add_trace(go.Histogram(x=PRED[k] - Y_TEST, nbinsx=80, name=LABELS[k], opacity=0.55,
                                   marker_color=COLORS[k]))
    fig.update_layout(template=TEMPLATE, barmode="overlay", title="Residuals (predicted - actual), test quarter",
                      xaxis_title="rides / hour", yaxis_title="hours", height=340, legend=dict(orientation="h", y=-0.25),
                      margin=dict(t=50, b=40))
    return fig


def fig_coefficients(top_n: int) -> go.Figure:
    coefs = {k: np.asarray(METRICS["coefficients"][k]) for k in MODEL_KEYS}
    order = np.argsort(-np.abs(coefs["sklearn"]))[:top_n]
    names = [FEATURE_NAMES[i] for i in order]
    fig = go.Figure()
    for k in MODEL_KEYS:
        fig.add_trace(go.Bar(y=names, x=coefs[k][order], orientation="h", name=LABELS[k], marker_color=COLORS[k]))
    fig.update_layout(template=TEMPLATE, barmode="group", title=f"Top {top_n} coefficients (standardised features, log1p scale)",
                      xaxis_title="coefficient", height=max(380, 22 * top_n), yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=-0.08), margin=dict(t=50, b=40, l=10))
    return fig


def agreement_df() -> pd.DataFrame:
    rows = []
    for pair, a in METRICS["agreement"].items():
        rows.append({"pair": pair.replace("_vs_", "  vs  "), "max |Δcoef|": a["max_abs_coef_diff"],
                     "mean |Δcoef|": a["mean_abs_coef_diff"], "coef correlation": a["coef_correlation"],
                     "Δintercept": a["intercept_diff"], "RMSE between predictions (rides)": a["rmse_between_predictions_counts"],
                     "max |Δprediction| (rides)": a["max_abs_pred_diff_counts"]})
    return pd.DataFrame(rows).round(5)


def objective_df() -> pd.DataFrame:
    hp = METRICS["hyperparameters"]
    rows = [{"implementation": LABELS[k], "train objective  MSE + α/n·‖w‖²": METRICS["objective_train"][k],
             "intercept": METRICS["intercepts"][k]} for k in MODEL_KEYS]
    df = pd.DataFrame(rows)
    df["hyperparameters"] = [f"alpha={hp['alpha']}",
                             f"alpha={hp['alpha']}, lr={hp['torch_manual']['lr']}, {hp['torch_manual']['n_steps']} full-batch steps",
                             f"weight_decay=2α/n={hp['torch_standard']['weight_decay']:.2e}, lr={hp['torch_standard']['lr']}, "
                             f"{hp['torch_standard']['n_epochs']} epochs x batch {hp['torch_standard']['batch_size']}"]
    return df.round(6)


def fig_loss_curves() -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Manual loop: full-batch gradient descent",
                                                        "nn.Module + Adam: per epoch"))
    h = pd.DataFrame(LOSS["torch_manual"])
    fig.add_trace(go.Scatter(x=h["step"], y=h["train_mse"], name="manual train", line=dict(color=COLORS["torch_manual"])), 1, 1)
    fig.add_trace(go.Scatter(x=h["step"], y=h["val_mse"], name="manual val", line=dict(color=COLORS["torch_manual"], dash="dot")), 1, 1)
    h = pd.DataFrame(LOSS["torch_standard"])
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_mse"], name="standard train", line=dict(color=COLORS["torch_standard"])), 1, 2)
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_mse"], name="standard val", line=dict(color=COLORS["torch_standard"], dash="dot")), 1, 2)
    for c in (1, 2):
        fig.add_hline(y=LOSS["sklearn_train_mse"], line=dict(color=COLORS["sklearn"], dash="dash", width=1), row=1, col=c)
    fig.update_xaxes(type="log", title_text="gradient step", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_yaxes(type="log", title_text="MSE on log1p(rides)", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_layout(template=TEMPLATE, height=380, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.2),
                      title="Training curves (dashed blue = sklearn's closed-form train MSE)")
    return fig


# ---------------------------------------------------------------------------
# Dispatch decision (threshold slider -> confusion matrix + cost)
# ---------------------------------------------------------------------------
def decision_view(model_key: str, forecast_thr: float, surge_thr: float, cost_missed: float, cost_unneeded: float):
    pred = PRED[model_key]
    d = surge_decision(Y_TEST, pred, forecast_thr, surge_thr, cost_missed, cost_unneeded)

    z = [[d["tp"], d["fn"]], [d["fp"], d["tn"]]]
    cm = go.Figure(go.Heatmap(z=z, x=["dispatch crew", "no crew"], y=["surge hour", "normal hour"],
                              text=[[f"TP {d['tp']}", f"FN {d['fn']}<br>missed surge"], [f"FP {d['fp']}<br>unneeded crew", f"TN {d['tn']}"]],
                              texttemplate="%{text}", colorscale=[[0, "#f4f6fa"], [1, "#2f6db5"]], showscale=False))
    cm.update_layout(template=TEMPLATE, title=f"Confusion matrix, test quarter ({len(Y_TEST)} hours)",
                     xaxis_title="decision from forecast", yaxis_title="what actually happened", height=340,
                     margin=dict(t=50, b=40), yaxis=dict(autorange="reversed"))

    thresholds = np.arange(100, 801, 10)
    curve = cost_curve(Y_TEST, pred, thresholds, surge_threshold=surge_thr, cost_missed=cost_missed, cost_unneeded=cost_unneeded)
    costs = [c["total_cost"] for c in curve]
    best = curve[int(np.argmin(costs))]
    always = surge_decision(Y_TEST, np.full_like(Y_TEST, 1e9), 0, surge_thr, cost_missed, cost_unneeded)["total_cost"]
    never = surge_decision(Y_TEST, np.zeros_like(Y_TEST), 1e9, surge_thr, cost_missed, cost_unneeded)["total_cost"]
    cc = go.Figure()
    cc.add_trace(go.Scatter(x=thresholds, y=costs, mode="lines", name=LABELS[model_key], line=dict(color=COLORS[model_key])))
    cc.add_hline(y=always, line=dict(color="grey", dash="dot"), annotation_text=f"always dispatch: {always:,.0f}", annotation_position="top left")
    cc.add_hline(y=never, line=dict(color="grey", dash="dash"), annotation_text=f"never dispatch: {never:,.0f}", annotation_position="bottom left")
    cc.add_vline(x=forecast_thr, line=dict(color="#1f1f1f"), annotation_text="current", annotation_position="top")
    cc.add_trace(go.Scatter(x=[best["forecast_threshold"]], y=[best["total_cost"]], mode="markers", name="cheapest threshold",
                            marker=dict(color="#d62728", size=10, symbol="star")))
    cc.update_layout(template=TEMPLATE, title="Total cost vs forecast threshold", xaxis_title="dispatch if forecast ≥ threshold (rides/h)",
                     yaxis_title="cost on test quarter (USD)", height=340, legend=dict(orientation="h", y=-0.25),
                     margin=dict(t=50, b=40))

    summary = (
        f"### Total cost: **{d['total_cost']:,.0f} USD** over the test quarter "
        f"({d['cost_per_hour']:.1f} USD per hour)\n\n"
        f"- Surge hours (actual ≥ {surge_thr:.0f} rides): **{d['n_surge_hours']}** of {len(Y_TEST)}\n"
        f"- Missed surges (FN): **{d['fn']}** × {cost_missed:.0f} USD = {d['fn'] * cost_missed:,.0f} USD\n"
        f"- Unneeded crews (FP): **{d['fp']}** × {cost_unneeded:.0f} USD = {d['fp'] * cost_unneeded:,.0f} USD\n"
        f"- Recall {d['recall']:.2f}, precision {d['precision']:.2f}\n"
        f"- Cheapest threshold for this model on the test quarter: **{best['forecast_threshold']:.0f}** "
        f"({best['total_cost']:,.0f} USD). Chosen on validation during training: "
        f"**{METRICS['decision'][model_key]['val_best_forecast_threshold']:.0f}**.\n"
        f"- Reference: always dispatch {always:,.0f} USD, never dispatch {never:,.0f} USD."
    )
    return cm, summary, cc


# ---------------------------------------------------------------------------
# What-if forecast
# ---------------------------------------------------------------------------
WEATHER = {"1 - clear / few clouds": 1, "2 - mist / cloudy": 2, "3 - light rain or snow": 3}
DAYTYPE = {"working day": (1, 0), "weekend": (0, 0), "public holiday": (0, 1)}
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def whatif(hr, daytype, weekday, month, year, weather, temp_c, hum_pct, wind_kmh, recent_level, forecast_thr):
    workingday, holiday = DAYTYPE[daytype]
    row = make_whatif_row(hr=hr, workingday=workingday, holiday=holiday, weekday=WEEKDAYS.index(weekday),
                          mnth=MONTHS.index(month) + 1, yr=0 if year == "2011" else 1, weathersit=WEATHER[weather],
                          temp_c=temp_c, hum=hum_pct / 100.0, windspeed=wind_kmh / 67.0, lag_level_counts=recent_level)
    X = PREPROCESSOR.transform(row).astype(np.float32)
    rows = []
    for k, m in MODELS.items():
        rides = float(to_counts(m.predict(X))[0])
        rows.append({"model": LABELS[k], "forecast rides / hour": round(rides, 1),
                     "dispatch crew?": "yes" if rides >= forecast_thr else "no"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
HP = METRICS["hyperparameters"]
TEST_RES = METRICS["results"]
OVERVIEW_MD = f"""
# Day-ahead hourly bike demand for rebalancing crews

**Dataset.** UCI Bike Sharing (Capital Bikeshare, Washington DC): one row per hour, 2011-2012, {len(TABLE):,} hours
with calendar, weather and the number of rides. Committed in `data/hour.csv`.

**Decision supported.** Every evening the operations team receives tomorrow's hourly forecast and decides, hour by hour,
whether to put an extra rebalancing crew on the street. A crew is needed in a *surge hour* (≥ {BIZ['surge_threshold']} rides). Missing
a surge costs about {BIZ['cost_missed_surge']:.0f} USD (empty stations, lost rides), an unneeded crew about {BIZ['cost_unneeded_crew']:.0f} USD.

**What that framing changed in the pipeline.**
- **Time-based split**: train Jan 2011 - Jun 2012, validate Jul - Sep 2012, test Oct - Dec 2012. A random split would
  leak the 2012 growth and neighbouring hours into training.
- **Target**: `log1p(rides)`, because demand is multiplicative (rain removes a share of rides, not a fixed number).
- **Lag feature** uses only days *d-2 .. d-8*, the latest complete days available when the forecast is produced.
- **Threshold metric**: expected dispatch cost (missed surges × {BIZ['cost_missed_surge']:.0f} + unneeded crews × {BIZ['cost_unneeded_crew']:.0f}), not accuracy.

**Model.** Ridge-regularised linear regression (alpha = {HP['alpha']}) on {len(FEATURE_NAMES)} standardised features, trained three ways on the
identical design matrix. Test-quarter RMSE: scikit-learn {TEST_RES['sklearn']['test']['rmse']:.1f}, manual PyTorch loop
{TEST_RES['torch_manual']['test']['rmse']:.1f}, nn.Module + optim {TEST_RES['torch_standard']['test']['rmse']:.1f} rides/hour
(hour-profile baseline {TEST_RES['baseline_hour_profile']['test']['rmse']:.1f}). See REPORT.md for the full story.
"""

DEFAULT_MODELS = MODEL_KEYS

with gr.Blocks(title="Bike demand: one model, three implementations") as demo:
    with gr.Tab("1. Overview"):
        gr.Markdown(OVERVIEW_MD)
        gr.Plot(value=fig_split_timeline())

    with gr.Tab("2. Data explorer"):
        with gr.Row():
            split_sel = gr.Radio(["train", "val", "test", "all"], value="train", label="split")
            scale_sel = gr.Radio(["rides / hour", "log1p(rides)"], value="rides / hour", label="target scale")
        target_plot = gr.Plot(value=fig_target_hist("train", "rides / hour"))
        feat_sel = gr.Dropdown(choices=list(CATEGORICAL) + list(NUMERIC), value="hr", label="feature")
        feat_plot = gr.Plot(value=fig_feature("hr", "train"))
        profile_plot = gr.Plot(value=fig_hour_profile("train"))
        gr.Markdown("### Feature-set ablation (validation quarter, same ridge model)")
        gr.Dataframe(value=ABLATION.round(3), interactive=False, wrap=True)
        gr.Markdown("### Hyperparameter sweeps (validation quarter)")
        gr.Dataframe(value=TUNING.round(4), interactive=False)
        split_sel.change(fig_target_hist, [split_sel, scale_sel], target_plot)
        scale_sel.change(fig_target_hist, [split_sel, scale_sel], target_plot)
        split_sel.change(fig_feature, [feat_sel, split_sel], feat_plot)
        feat_sel.change(fig_feature, [feat_sel, split_sel], feat_plot)
        split_sel.change(fig_hour_profile, split_sel, profile_plot)

    with gr.Tab("3. Model comparison"):
        with gr.Row():
            metric_split = gr.Radio(["test", "val"], value="test", label="metrics on")
        metrics_table = gr.Dataframe(value=metrics_df("test"), interactive=False)
        metric_split.change(metrics_df, metric_split, metrics_table)
        with gr.Row():
            model_sel = gr.CheckboxGroup(choices=[(LABELS[k], k) for k in MODEL_KEYS + BASELINE_KEYS[1:]],
                                         value=DEFAULT_MODELS, label="models to show")
            log_axes = gr.Checkbox(value=False, label="log axes")
        pva_plot = gr.Plot(value=fig_pred_vs_actual(DEFAULT_MODELS, False))
        model_sel.change(fig_pred_vs_actual, [model_sel, log_axes], pva_plot)
        log_axes.change(fig_pred_vs_actual, [model_sel, log_axes], pva_plot)
        week_sel = gr.Dropdown(choices=WEEK_STARTS, value="2012-11-05", label="test week (Hurricane Sandy: 2012-10-29)")
        week_plot = gr.Plot(value=fig_week("2012-11-05", DEFAULT_MODELS))
        week_sel.change(fig_week, [week_sel, model_sel], week_plot)
        model_sel.change(fig_week, [week_sel, model_sel], week_plot)
        resid_plot = gr.Plot(value=fig_residuals(DEFAULT_MODELS))
        model_sel.change(fig_residuals, model_sel, resid_plot)

    with gr.Tab("4. Three implementations"):
        gr.Markdown(
            "All three minimise **MSE(log1p rides) + α/n·‖w‖²** on the same standardised matrix. scikit-learn solves it in "
            "closed form; the manual loop runs full-batch gradient descent with autograd and a hand-written update; the standard "
            "workflow uses `nn.Linear` + `nn.MSELoss` + `torch.optim.Adam` with `weight_decay = 2α/n` on the weights only."
        )
        gr.Dataframe(value=objective_df(), interactive=False, wrap=True)
        gr.Dataframe(value=agreement_df(), interactive=False)
        topn = gr.Slider(5, len(FEATURE_NAMES), value=20, step=1, label="number of coefficients to show")
        coef_plot = gr.Plot(value=fig_coefficients(20))
        topn.change(fig_coefficients, topn, coef_plot)
        gr.Plot(value=fig_loss_curves())

    with gr.Tab("5. Dispatch decision"):
        gr.Markdown(
            "Move the **forecast threshold** to see how the dispatch rule *\"send a crew if the forecast ≥ threshold\"* "
            "trades missed surges against unneeded crews. Because a missed surge is more expensive, the cheapest threshold sits "
            "below the surge level itself."
        )
        with gr.Row():
            dec_model = gr.Dropdown(choices=[(LABELS[k], k) for k in MODEL_KEYS + BASELINE_KEYS[1:]], value="sklearn", label="model")
            fthr = gr.Slider(100, 800, value=METRICS["decision"]["sklearn"]["val_best_forecast_threshold"], step=10,
                             label="forecast threshold (rides / hour)")
        with gr.Row():
            sthr = gr.Slider(200, 700, value=BIZ["surge_threshold"], step=10, label="surge definition: actual rides ≥")
            cmiss = gr.Number(value=BIZ["cost_missed_surge"], label="cost of a missed surge (USD)")
            cunn = gr.Number(value=BIZ["cost_unneeded_crew"], label="cost of an unneeded crew (USD)")
        _cm, _txt, _cc = decision_view("sklearn", METRICS["decision"]["sklearn"]["val_best_forecast_threshold"],
                                       BIZ["surge_threshold"], BIZ["cost_missed_surge"], BIZ["cost_unneeded_crew"])
        cost_md = gr.Markdown(_txt)
        with gr.Row():
            cm_plot = gr.Plot(value=_cm)
            cc_plot = gr.Plot(value=_cc)
        for comp in (dec_model, fthr, sthr, cmiss, cunn):
            comp.change(decision_view, [dec_model, fthr, sthr, cmiss, cunn], [cm_plot, cost_md, cc_plot])

    with gr.Tab("6. What-if forecast"):
        gr.Markdown("Describe tomorrow's hour and get the forecast from each saved model (inference only).")
        with gr.Row():
            w_hr = gr.Slider(0, 23, value=17, step=1, label="hour")
            w_day = gr.Radio(list(DAYTYPE), value="working day", label="day type")
            w_wd = gr.Dropdown(WEEKDAYS, value="Tuesday", label="weekday")
            w_mo = gr.Dropdown(MONTHS, value="Oct", label="month")
            w_yr = gr.Radio(["2011", "2012"], value="2012", label="year")
        with gr.Row():
            w_wx = gr.Dropdown(list(WEATHER), value="1 - clear / few clouds", label="weather")
            w_t = gr.Slider(-10, 40, value=18, step=0.5, label="temperature (°C)")
            w_h = gr.Slider(10, 100, value=55, step=1, label="relative humidity (%)")
            w_w = gr.Slider(0, 60, value=12, step=1, label="wind (km/h)")
            w_lag = gr.Number(value=450, label="recent same-hour demand (mean rides, days d-2..d-8)")
            w_thr = gr.Number(value=METRICS["decision"]["sklearn"]["val_best_forecast_threshold"], label="dispatch if forecast ≥")
        w_out = gr.Dataframe(value=whatif(17, "working day", "Tuesday", "Oct", "2012", "1 - clear / few clouds", 18, 55, 12, 450,
                                          METRICS["decision"]["sklearn"]["val_best_forecast_threshold"]), interactive=False)
        w_inputs = [w_hr, w_day, w_wd, w_mo, w_yr, w_wx, w_t, w_h, w_w, w_lag, w_thr]
        for comp in w_inputs:
            comp.change(whatif, w_inputs, w_out)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=os.environ.get("CI") is None, show_error=True)
