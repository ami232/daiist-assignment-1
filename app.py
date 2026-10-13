from datetime import date, timedelta

import gradio as gr
import polars as pl

from src.model import predict
from src.model.predict import MODELS, cost
from src.plotting import dashboard as dash
from src.stats.distribution_shift import psi_table

Z_WINDOW = 90
KPIS = {
    "y_spread": "Realised spread",
    "spread_1d": "Spread D-1",
    "price_es": "Price D-1",
    "curve_implied_price": "Curve implied price",
    "demand_forecast_d1": "Demand fcst",
    "wind_forecast_d1": "Wind fcst",
    "solar_pv_forecast_d1": "Solar fcst",
    "residual_load": "Residual load",
    "renewable_share": "Renewable share",
    "generation_actual|B04|in": "Gas gen D-2",
    "generation_actual|B12|in": "Hydro reservoir D-2",
    "imbalance_price_up": "Imbalance up D-2",
    "transfer_capacity_da|FR|ES": "FR to ES capacity",
}
DEFAULT_PRESET = "3M"
PRESETS = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365, "Test": None, "All": None}
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]

DATA = predict.load_predictions()
PRICES = predict.hourly_zscores(predict.load_hourly_prices(), Z_WINDOW)
Z = predict.zscores(DATA, list(KPIS), Z_WINDOW)
WEIGHTS = predict.weights()
FEATURES = predict.feature_names()
TRAIN = DATA.filter(split="train")
TEST_START = DATA.filter(split="test")["day"].min()
FIRST, LAST = DATA["day"].min(), DATA["day"].max()


def preset_window(preset: str) -> list[str]:
    start = {"Test": TEST_START, "All": FIRST}.get(preset) or LAST - timedelta(days=PRESETS[preset])
    return [str(start), str(LAST)]


def typed_window(start: str, end: str) -> list[str]:
    """Clamps typed dates into the data range."""
    days = sorted(min(max(date.fromisoformat(str(d)[:10]), FIRST), LAST) for d in (start, end))
    return [str(d) for d in days]


def window_frame(window: list[str]) -> pl.DataFrame:
    df = DATA.filter(pl.col("day").is_between(date.fromisoformat(window[0]), date.fromisoformat(window[1])))
    if df.is_empty():
        raise gr.Error("No days in this range")
    return df


def refresh(window: list[str], model: str, threshold: float) -> tuple:
    df = window_frame(window)
    last = df.row(-1, named=True)
    p = last[f"p_{model}"]
    edge = predict.edge(df, model, threshold)
    lo, hi = predict.edge_interval(edge["edge"])
    score = predict.scorecard(df, threshold)
    sample = ", incl. train days" if df["day"].min() < TEST_START else ""
    selected = score.filter(model=model).row(0, named=True)
    out = predict.outcomes(df, model, threshold)
    saved = out.filter(cell="saved")["eur"].sum()
    tiles = dash.stat_tiles(
        [
            (f"Call for {last['day']:%d %b %Y}", "CYCLE" if p >= threshold else "REST", dash.LABELS[model]),
            ("Probability", f"{p:.0%}", f"threshold {threshold:.0%}"),
            ("Realised spread", f"{last['y_spread']:.1f}", f"EUR/MWh, cost {cost:.0f}"),
            ("P&L on range", f"{selected['profit_eur_per_mw']:,.0f}", f"EUR/MW, {selected['capture'] or 0:.0%} of perfect"),
            ("Saved by resting", f"{saved:,.0f}", "EUR/MW"),
            ("Edge vs naive", f"{edge['edge'].sum():+,.0f}", f"EUR/MW, 90% CI {lo:+,.0f} to {hi:+,.0f}{sample}"),
        ],
    )
    terms, bias = predict.contributions(df.tail(1), model)
    prices = PRICES.filter(pl.col("day").is_between(df["day"].min(), last["day"]))
    return (
        tiles,
        dash.earnings_chart(out, model, TEST_START),
        dash.day_price_chart(prices.filter(day=last["day"]), last["day"], Z_WINDOW),
        dash.hour_roles_chart(prices),
        dash.hourly_z_heatmap(prices, Z_WINDOW),
        dash.zscore_grid(Z.filter(pl.col("day").is_between(df["day"].min(), last["day"])).tail(7), KPIS, Z_WINDOW),
        dash.spread_chart(df, model, threshold, TEST_START),
        dash.probability_chart(df, model, threshold, TEST_START),
        dash.confusion_chart(out, threshold),
        dash.threshold_chart(predict.threshold_sweep(df, THRESHOLDS), threshold),
        dash.scorecard_table(score, model),
        dash.pnl_chart(df, threshold),
        dash.calibration_chart(predict.calibration(df)),
        dash.edge_table(edge, model),
        dash.contribution_chart(terms, bias, f"Why {dash.LABELS[model]} says {p:.0%} for {last['day']:%d %b %Y}"),
        dash.distribution_chart(df, TRAIN, "y_spread", "Daily spread vs training period (EUR/MWh)", (cost, "cycle cost")),
        dash.drift_chart(psi_table(TRAIN, df, FEATURES)),
    )


def feature_chart(window: list[str], feature: str) -> object:
    df = window_frame(window)
    return dash.distribution_chart(df, TRAIN, feature, f"Distribution of {feature}", (df[feature][-1], "latest"))


with gr.Blocks(title="Battery Spread Desk") as demo:
    gr.Markdown("# Battery Spread Desk\nShould a 4h battery cycle tomorrow in the Spanish day-ahead market?")
    with gr.Row(equal_height=True):
        presets = [gr.Button(name, size="sm", min_width=60, scale=0) for name in PRESETS]
        start = gr.DateTime(preset_window(DEFAULT_PRESET)[0], include_time=False, type="string", label="From", scale=1)
        end = gr.DateTime(str(LAST), include_time=False, type="string", label="To", scale=1)
        model = gr.Dropdown([(dash.LABELS[m], m) for m in MODELS], value=MODELS[0], label="Model", scale=1)
        threshold = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Decision threshold", scale=2)
    window = dash.DateRange(FIRST, LAST, preset_window(DEFAULT_PRESET))
    tiles = gr.HTML()
    with gr.Tabs():
        with gr.Tab("Battery desk"):
            earnings = gr.Plot(show_label=False)
            with gr.Row():
                day_price = gr.Plot(show_label=False, scale=3)
                hour_roles = gr.Plot(show_label=False, scale=2)
            hourly_z = gr.Plot(show_label=False)
            grid = gr.HTML()
            spread = gr.Plot(show_label=False)
        with gr.Tab("Model"):
            gr.Markdown("### Predictions vs outcome")
            probability = gr.Plot(show_label=False)
            with gr.Row():
                confusion = gr.Plot(show_label=False)
                sweep = gr.Plot(show_label=False)
            table = gr.HTML()
            with gr.Row():
                pnl = gr.Plot(show_label=False)
                calibration = gr.Plot(show_label=False)
            edges = gr.HTML()
            gr.Markdown("### Why this call")
            contribution = gr.Plot(show_label=False)
            gr.Plot(dash.weights_chart(WEIGHTS), show_label=False)
            gr.Markdown("### Target and feature distributions")
            with gr.Row():
                target_hist = gr.Plot(show_label=False)
                with gr.Column():
                    feature = gr.Dropdown(FEATURES, value="wind_forecast_d1", label="Feature")
                    feature_hist = gr.Plot(show_label=False)
            drift = gr.Plot(show_label=False)

    inputs = [window, model, threshold]
    outputs = [
        tiles,
        earnings,
        day_price,
        hour_roles,
        hourly_z,
        grid,
        spread,
        probability,
        confusion,
        sweep,
        table,
        pnl,
        calibration,
        edges,
        contribution,
        target_hist,
        drift,
    ]
    for button in presets:
        button.click(lambda name=button.value: preset_window(name), None, window)
    for typed in (start, end):
        typed.change(typed_window, [start, end], window)
    window.change(lambda w: (w[0], w[1]), window, [start, end])
    for control in inputs:
        control.change(refresh, inputs, outputs, trigger_mode="always_last")
    for control in (window, feature):
        control.change(feature_chart, [window, feature], feature_hist, trigger_mode="always_last")
    demo.load(refresh, inputs, outputs).then(feature_chart, [window, feature], feature_hist)

if __name__ == "__main__":
    demo.launch(theme=dash.THEME, css=dash.CSS)
