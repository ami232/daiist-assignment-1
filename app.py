"""Gradio dashboard: run with `uv run python main.py app`.

Loads only what train.py saved in artifacts/: the preprocessor, the three trained models and
the dataset. It runs the models forward to get probabilities (inference) and never fits anything.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from volmodel import config, evaluation as ev, models as M
from volmodel.features import FEATURE_DESCRIPTIONS
from volmodel.preprocess import Preprocessor

ART = config.ARTIFACTS_DIR
if not (ART / "metrics.json").exists():
    raise SystemExit("artifacts/ is empty. Run `uv run python main.py train` first.")

# Load artifacts (no training anywhere in this file) -----------------------------------------
METRICS = json.loads((ART / "metrics.json").read_text())
PREP = Preprocessor.load(ART / "preprocessor.json")
FEATURES = PREP.features
SK, MANUAL, MODULE = M.load_models(ART, len(FEATURES))
DATA = pd.read_csv(ART / "dataset.csv", parse_dates=["date"], index_col="date")
CURVES = pd.read_csv(ART / "training_curves.csv", index_col="epoch")

BLOCKS = {s: DATA[DATA["split"] == s] for s in ("train", "val", "test", "live")}
PROBS = {s: M.predict_all(SK, MANUAL, MODULE, PREP.transform(BLOCKS[s])) for s in BLOCKS}
Y = {s: BLOCKS[s]["vol_rises"].to_numpy() for s in ("train", "val", "test")}
C_FP = METRICS["costs"]["false_alarm_bps"]
DEFAULT_K = METRICS["config"]["miss_multiple"]
DEFAULT_T = METRICS["thresholds"]["chosen_on_val"]["sklearn"]

NAMES = {"sklearn": "scikit-learn", "torch_manual": "PyTorch manual loop", "torch_module": "PyTorch nn.Module"}
# Two hues only, blue and forest green, checked for colour-blind separation in light and dark mode.
# scikit-learn (the exact reference) is blue. Both PyTorch versions are green and are told apart
# by line style (dashed vs dotted) and, in bar charts, by a hatch pattern, never by colour alone.
BLUE, GREEN = "#2a78d6", "#2b8a3e"
COLORS = {"sklearn": BLUE, "torch_manual": GREEN, "torch_module": GREEN}
DASHES = {"sklearn": "solid", "torch_manual": "dash", "torch_module": "dot"}
PATTERNS = {"sklearn": "", "torch_manual": "", "torch_module": "/"}
FOREST = gr.themes.Color(c50="#ecf7ef", c100="#d3eedb", c200="#a9dcb8", c300="#79c493", c400="#4aa86b",
                         c500="#2b8a3e", c600="#237535", c700="#1d5e2f", c800="#1b4332", c900="#143326",
                         c950="#0b1f17", name="forest")
THEME = gr.themes.Default(primary_hue=FOREST, secondary_hue="blue")
GRAY, INK = "#8a8985", "#7a7974"
BLOCK_LABELS = {"Validation (2015 to 2019)": "val", "Test (2020 to 2026)": "test"}


def style(fig: go.Figure, title: str, y_title: str = "", x_title: str = "", height: int = 380) -> go.Figure:
    """One recessive look for every chart: transparent background, thin grid, legend on top."""
    fig.update_layout(
        title={"text": title, "x": 0, "y": 0.98, "yanchor": "top", "font": {"size": 15}}, height=height, template="none",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": INK, "size": 12},
        margin={"l": 60, "r": 20, "t": 95, "b": 50}, hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        modebar={"orientation": "v", "bgcolor": "rgba(0,0,0,0)"},   # toolbar on the side, off the title
    )
    grid = {"gridcolor": "rgba(128,128,128,0.18)", "zerolinecolor": "rgba(128,128,128,0.3)", "linecolor": "rgba(128,128,128,0.4)"}
    fig.update_xaxes(title=x_title, **grid)
    fig.update_yaxes(title=y_title, **grid)
    return fig


# Tab 1: model comparison --------------------------------------------------------------------

def metrics_table(block: str) -> pd.DataFrame:
    res = METRICS["results"][block]
    rows = []
    labels = {**NAMES, "baseline_majority": "Baseline: never cut (majority)", "baseline_mean_reversion": "Baseline: PM mean reversion rule"}
    for key, label in labels.items():
        r = res[key]
        fmt = lambda v, d=3: "n/a" if v is None else round(v, d)
        rows.append({"Model": label, "AUC": fmt(r["auc"]), "Log loss": fmt(r["log_loss"]), "Brier": fmt(r["brier"]),
                     "Accuracy": fmt(r["accuracy"]), "Precision": fmt(r["precision"]), "Recall": fmt(r["recall"]),
                     "Days flagged": f"{r['share_flagged']:.0%}", "Cost per decision (bps)": round(r["cost_per_decision_bps"], 1)})
    always = res["policy_always_cut"]
    rows.append({"Model": "Policy: always cut", "AUC": "n/a", "Log loss": "n/a", "Brier": "n/a", "Accuracy": "n/a",
                 "Precision": "n/a", "Recall": 1.0, "Days flagged": "100%", "Cost per decision (bps)": round(always["cost_per_decision_bps"], 1)})
    return pd.DataFrame(rows)


def prediction_vs_actual(block: str) -> go.Figure:
    """Rolling 3-month average of each model's predicted probability against how often vol actually rose.

    Three months, not one: consecutive labels share 20 of 21 future returns, so a 21-day
    window of labels swings between 0% and 100% and hides the comparison."""
    idx = BLOCKS[block].index
    fig = go.Figure()
    actual = pd.Series(Y[block], index=idx).rolling(63).mean()
    fig.add_trace(go.Scatter(x=idx, y=actual, name="Actual: share of days vol rose", line={"color": GRAY, "width": 2}))
    for m in NAMES:
        p = pd.Series(PROBS[block][m], index=idx).rolling(63).mean()
        fig.add_trace(go.Scatter(x=idx, y=p, name=NAMES[m], line={"color": COLORS[m], "width": 2, "dash": DASHES[m]}))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", hoverformat=".1%")
    return style(fig, "Predicted probability vs actual outcome (3-month rolling average)", "Probability vol rises")


def calibration(block: str) -> go.Figure:
    """Predicted probability (binned by decile) against the realized frequency in that bin."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", mode="lines",
                             line={"color": GRAY, "width": 1, "dash": "dash"}, hoverinfo="skip"))
    for m in NAMES:
        p = PROBS[block][m]
        bins = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": Y[block]}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], name=NAMES[m], mode="lines+markers",
                                 line={"color": COLORS[m], "width": 2, "dash": DASHES[m]}, marker={"size": 8}))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(hovermode="closest")
    return style(fig, "Calibration: predicted vs realized frequency, by decile", "Share of days vol actually rose", "Mean predicted probability")


def coefficient_chart() -> go.Figure:
    coefs = METRICS["coefficients"]
    order = sorted(FEATURES, key=lambda c: coefs["sklearn"][c])
    fig = go.Figure()
    for m in NAMES:
        fig.add_trace(go.Bar(y=order, x=[coefs[m][c] for c in order], name=NAMES[m], orientation="h",
                             marker={"color": COLORS[m], "line": {"width": 0},
                                     "pattern": {"shape": PATTERNS[m], "fgcolor": "#ffffff", "solidity": 0.35}}))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="y unified")
    return style(fig, "Coefficients on standardized features (same sign and size in all three)", "", "Coefficient", height=520)


def training_curve_chart() -> go.Figure:
    fig = go.Figure()
    optimum = METRICS["objective_on_train"]["sklearn"]
    for col in CURVES.columns:
        kind, lr = col.split("_lr_")
        name = f"{'Manual loop' if kind == 'manual' else 'nn.Module'}, lr {lr}"
        color = GREEN                                    # both are PyTorch: line style tells them apart
        width = 2.5 if float(lr) == METRICS["tuning"][f"lr_{kind}"] else 1
        fig.add_trace(go.Scatter(x=CURVES.index, y=CURVES[col] - optimum + 1e-9, name=name,
                                 line={"color": color, "width": width, "dash": "solid" if kind == "manual" else "dot"}))
    fig.update_yaxes(type="log", exponentformat="power")
    fig = style(fig, "Gradient descent: distance to the scikit-learn optimum (thick = chosen learning rate)",
                "Training objective minus optimum", "Epoch", height=460)
    fig.update_layout(legend={"orientation": "v", "x": 1.01, "y": 1, "yanchor": "top"}, margin={"r": 170})
    return fig


def agreement_text() -> str:
    a = METRICS["agreement_test"]
    obj = METRICS["objective_on_train"]
    lines = ["| Pair | Max prob. difference | Mean prob. difference | Days with a different decision |", "|---|---|---|---|"]
    for pair, v in a.items():
        x, y = pair.split("_vs_")
        lines.append(f"| {NAMES[x]} vs {NAMES[y]} | {v['max_abs_prob_diff']:.1e} | {v['mean_abs_prob_diff']:.1e} | {v['decision_disagreements']} of {len(Y['test'])} |")
    lines.append("")
    lines.append("Training objective reached: " + ", ".join(f"{NAMES[m]} {v:.8f}" for m, v in obj.items()))
    return "\n".join(lines)


# Tab 2: features and target ---------------------------------------------------------------

def feature_distribution(feature: str) -> tuple[go.Figure, str]:
    train = BLOCKS["train"]
    fig = go.Figure()
    for label, value, color in (("Vol did not rise", 0, GRAY), ("Vol rose", 1, COLORS["sklearn"])):
        fig.add_trace(go.Histogram(x=train.loc[train["vol_rises"] == value, feature], name=label, histnorm="probability density",
                                   opacity=0.6, marker={"color": color, "line": {"width": 0}}, nbinsx=60))
    fig.update_layout(barmode="overlay", hovermode="closest")
    return style(fig, f"{feature}: distribution on the training years by outcome", "Density", feature), FEATURE_DESCRIPTIONS[feature]


def target_by_year() -> go.Figure:
    labelled = DATA[DATA["split"].isin(["train", "val", "test"])]
    by_year = labelled.groupby(labelled.index.year)
    enough = by_year.size() >= 100          # 1990 has a single labelled day after the 1-year warm-up
    rate = by_year["vol_rises"].mean()[enough]
    split_of_year = by_year["split"].agg(lambda s: s.mode()[0])[enough]
    block_colors = {"train": BLUE, "val": GRAY, "test": GREEN}
    colors = split_of_year.map(block_colors)
    fig = go.Figure(go.Bar(x=rate.index, y=rate.values, marker={"color": list(colors), "line": {"width": 0}},
                           hovertemplate="%{x}: %{y:.0%} of days<extra></extra>", showlegend=False))
    for name, key in (("Train", "train"), ("Validation", "val"), ("Test", "test")):
        fig.add_trace(go.Bar(x=[None], y=[None], name=name, marker={"color": block_colors[key]}))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(hovermode="closest", bargap=0.2)
    return style(fig, "Target base rate by year: share of days where next month's vol ends up higher", "Share of days")


def vol_ratio_hist() -> go.Figure:
    labelled = DATA[DATA["split"].isin(["train", "val", "test"])]
    ratio = np.log(labelled["fwd_vol_21d"] / labelled["rv_21d"])
    fig = go.Figure(go.Histogram(x=ratio, nbinsx=80, marker={"color": COLORS["sklearn"], "line": {"width": 0}}, name="All labelled days"))
    fig.add_vline(x=0, line={"color": GRAY, "dash": "dash"})
    fig.update_layout(hovermode="closest", showlegend=False)
    return style(fig, "What the label thresholds: log of next-month vol over trailing vol (right of 0 = label 1)", "Days", "log(next 21d vol / trailing 21d vol)")


def ablation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = METRICS["feature_ablation_val"]
    cum = pd.DataFrame(a["cumulative"]).rename(columns={"added": "Feature group added", "val_auc": "Validation AUC", "val_log_loss": "Validation log loss"})
    loo = pd.DataFrame(a["leave_one_group_out"]).rename(columns={"removed": "Feature group removed", "val_auc": "Validation AUC", "val_log_loss": "Validation log loss"})
    return cum.round(4), loo.round(4)


# Tab 3: threshold and business cost -------------------------------------------------------

def confusion_figure(tp, fp, tn, fn) -> go.Figure:
    z = [[tp, fn], [fp, tn]]
    text = [[f"Correct cut<br><b>{tp}</b>", f"Miss<br><b>{fn}</b>"], [f"False alarm<br><b>{fp}</b>", f"Correct hold<br><b>{tn}</b>"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Model says cut", "Model says hold"], y=["Vol rose", "Vol did not rise"],
                               text=text, texttemplate="%{text}", textfont={"color": "#0b0b0b", "size": 14},
                               colorscale=[[0, "#e6effb"], [1, "#86b6ef"]],
                               showscale=False, xgap=2, ygap=2, hovertemplate="%{y}, %{x}: %{z} days<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    fig = style(fig, "Confusion matrix", "Actual", "Decision", height=340)
    fig.update_layout(margin={"l": 140})
    return fig


def cost_curve(block: str, model: str, threshold: float, k: float, t_opt: float) -> go.Figure:
    p, y = PROBS[block][model], Y[block]
    grid = ev.THRESHOLD_GRID
    costs = [ev.business_cost(y, p >= t, C_FP, k)["cost_per_decision_bps"] for t in grid]
    fig = go.Figure(go.Scatter(x=grid, y=costs, name=f"{NAMES[model]} cost", line={"color": COLORS[model], "width": 2}))
    for name, dec in (("Always cut", np.ones(len(y), bool)), ("Never cut", np.zeros(len(y), bool))):
        c = ev.business_cost(y, dec, C_FP, k)["cost_per_decision_bps"]
        fig.add_hline(y=c, line={"color": GRAY, "width": 1, "dash": "dash"}, annotation_text=name, annotation_font_color=INK)
    fig.add_vline(x=threshold, line={"color": INK, "width": 1.5}, annotation_text="Current", annotation_font_color=INK)
    fig.add_vline(x=t_opt, line={"color": COLORS[model], "width": 1, "dash": "dot"}, annotation_text="Best on validation",
                  annotation_position="bottom right", annotation_font_color=INK)
    fig.update_xaxes(range=[0, 1])
    return style(fig, "Cost per decision as the threshold moves", "bps per daily decision", "Threshold")


def backtest_figure(block: str, decision: pd.Series) -> tuple[go.Figure, pd.DataFrame]:
    bt = ev.overlay_backtest(DATA, block, decision)
    always = ev.overlay_backtest(DATA, block, pd.Series(True, index=decision.index))
    eq = bt["equity"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq["vol_target_only"], name="Vol target only", line={"color": GRAY, "width": 2}))
    fig.add_trace(go.Scatter(x=eq.index, y=always["equity"]["with_overlay"], name="Always 25% less risk",
                             line={"color": GRAY, "width": 1.5, "dash": "dot"}))
    fig.add_trace(go.Scatter(x=eq.index, y=eq["with_overlay"], name="With model overlay", line={"color": COLORS["sklearn"], "width": 2}))
    rows = []
    for name, s in (("Vol target only", bt["vol_target_only"]), ("Always 25% less risk", always["with_overlay"]), ("With model overlay", bt["with_overlay"])):
        rows.append({"Strategy": name, "Annual return": f"{s['annual_return']:.1%}", "Annual vol": f"{s['annual_vol']:.1%}",
                     "Sharpe": round(s["sharpe"], 2), "Max drawdown": f"{s['max_drawdown']:.1%}",
                     "Worst month": f"{s['worst_month']:.1%}", "Months below -5%": s["months_below_minus5"]})
    title = f"Overlay backtest, monthly rebalancing ({bt['months_cut']} of {bt['months']} months cut)"
    return style(fig, title, "Growth of 1", height=360), pd.DataFrame(rows)


def threshold_view(block_label: str, model: str, threshold: float, k: float):
    block = BLOCK_LABELS[block_label]
    p, y = PROBS[block][model], Y[block]
    t_opt = ev.best_threshold(Y["val"], PROBS["val"][model], C_FP, k)
    m = ev.classification_metrics(y, p, p >= threshold, C_FP, k)
    never = ev.business_cost(y, np.zeros(len(y), bool), C_FP, k)
    always = ev.business_cost(y, np.ones(len(y), bool), C_FP, k)
    rule = ev.business_cost(y, ev.mean_reversion_rule(BLOCKS[block]), C_FP, k)
    summary = (
        f"### Business cost: **{m['cost_per_decision_bps']:.1f} bps per daily decision**  \n"
        f"{m['false_alarms']} false alarms x {C_FP:.1f} bps + {m['misses']} misses x {C_FP * k:.1f} bps "
        f"= {m['total_cost_bps']:,.0f} bps over {len(y):,} days  \n"
        f"Compare: never cut {never['cost_per_decision_bps']:.1f}, always cut {always['cost_per_decision_bps']:.1f}, "
        f"PM mean reversion rule {rule['cost_per_decision_bps']:.1f} bps per decision  \n"
        f"Precision {m['precision'] or 0:.0%}, recall {m['recall']:.0%}, days flagged {m['share_flagged']:.0%}. "
        f"Cost-optimal threshold on validation at {k:g}x: **{t_opt:.2f}** "
        f"(theory for calibrated probabilities: {1 / (1 + k):.2f})"
    )
    decision = pd.Series(p >= threshold, index=BLOCKS[block].index)
    bt_fig, bt_table = backtest_figure(block, decision)
    return (confusion_figure(m["tp"], m["fp"], m["tn"], m["fn"]), summary, cost_curve(block, model, threshold, k, t_opt), bt_fig, bt_table)


def live_signal_text() -> str:
    live = BLOCKS["live"]
    p = PROBS["live"]["sklearn"][-1]
    date = live.index[-1].strftime("%d %b %Y")
    action = "CUT gross exposure by 25%" if p >= DEFAULT_T else "HOLD full vol-target exposure"
    return (f"**Latest signal, {date}:** probability that next month's vol exceeds trailing vol = **{p:.0%}** "
            f"(threshold {DEFAULT_T:.2f}) -> **{action}**. Trailing 21d vol {live['rv_21d'].iloc[-1]:.1%}, VIX {live['vix'].iloc[-1]:.1f}.")


# Layout -------------------------------------------------------------------------------------

def build() -> gr.Blocks:
    q = METRICS["data_quality"]
    tune = METRICS["tuning"]
    with gr.Blocks(title="Vol-rise overlay", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Vol-rise overlay for a vol-target book\n"
            "Each day the model estimates the probability that realized S&P 500 vol over the next 21 trading days "
            "ends up **above** the trailing 21-day vol the book is sized on. Above the threshold, the book cuts gross "
            f"exposure by {config.HAIRCUT:.0%} for the month. Logistic regression trained three ways on 1990 to 2014, "
            f"tuned on 2015 to 2019 (C = {tune['C']}, PyTorch learning rate {tune['lr_manual']}), tested once on 2020 to 2026."
        )
        gr.Markdown(live_signal_text())

        with gr.Tab("Model comparison"):
            block1 = gr.Radio(list(BLOCK_LABELS), value="Test (2020 to 2026)", label="Block")
            table = gr.Dataframe(metrics_table("test"), label="Three implementations against the baselines", interactive=False)
            pva = gr.Plot(prediction_vs_actual("test"), show_label=False)
            cal = gr.Plot(calibration("test"), show_label=False)
            gr.Markdown("### Do the three implementations agree?\n" + agreement_text())
            gr.Plot(coefficient_chart(), show_label=False)
            gr.Plot(training_curve_chart(), show_label=False)
            block1.change(lambda b: (metrics_table(BLOCK_LABELS[b]), prediction_vs_actual(BLOCK_LABELS[b]), calibration(BLOCK_LABELS[b])),
                          block1, [table, pva, cal])

        with gr.Tab("Features and target"):
            feat = gr.Dropdown(FEATURES, value="vix_gap", label="Feature")
            fig0, desc0 = feature_distribution("vix_gap")
            desc = gr.Markdown(desc0)
            dist = gr.Plot(fig0, show_label=False)
            feat.change(feature_distribution, feat, [dist, desc])
            with gr.Row():
                gr.Plot(target_by_year(), show_label=False)
                gr.Plot(vol_ratio_hist(), show_label=False)
            cum, loo = ablation_tables()
            final = METRICS["feature_ablation_val"]["final_feature_set"]
            gr.Markdown("### Which feature groups earn their place (validation, scikit-learn)\n"
                        "Both tables use all 11 candidate features. The window shape group lowered validation AUC and had "
                        f"coefficients near zero, so it was dropped: the final model uses {len(final['features'])} features "
                        f"(validation AUC {final['val_auc']:.3f}, log loss {final['val_log_loss']:.3f}).")
            with gr.Row():
                gr.Dataframe(cum, label="Adding candidate groups one at a time", interactive=False)
                gr.Dataframe(loo, label="Removing one group from the 11-feature candidate model", interactive=False)
            gr.Markdown(
                "### Data-quality decisions\n"
                f"1. S&P open dropped: on {q['share_fake_open_1990_2005']:.0%} of days from 1990 to 2005 it equals the previous close.\n"
                f"2. VIX open, high and low ignored: flat on {q['vix_flat_ohlc_days_1990_1991']} days in 1990 and 1991. Close only.\n"
                f"3. VIX rows on US holidays dropped by the join: {', '.join(q['vix_dates_dropped_no_spx_trading'])}.\n"
                f"4. Rows by block: {q['rows_by_split']}. 'purged' = last 21 rows before each boundary, "
                "'warmup' = first year used by the 1-year features, 'live' = last 21 days with no label yet."
            )

        with gr.Tab("Threshold and business cost"):
            gr.Markdown(
                f"A false alarm costs **{C_FP:.1f} bps**: the return given up by cutting {config.HAIRCUT:.0%} in a month where vol did not rise "
                "(average on the training years). A miss is priced as a multiple of that, because its real cost is tail risk, not average P&L."
            )
            with gr.Row():
                block3 = gr.Radio(list(BLOCK_LABELS), value="Test (2020 to 2026)", label="Block")
                model3 = gr.Dropdown([(v, k) for k, v in NAMES.items()], value="sklearn", label="Model")
            with gr.Row():
                thr = gr.Slider(0.01, 0.99, value=DEFAULT_T, step=0.01, label="Decision threshold")
                mult = gr.Slider(0.5, 10, value=DEFAULT_K, step=0.5, label="Cost of a miss, as a multiple of a false alarm")
            init = threshold_view("Test (2020 to 2026)", "sklearn", DEFAULT_T, DEFAULT_K)
            with gr.Row():
                cm = gr.Plot(init[0], show_label=False)
                with gr.Column():
                    summary = gr.Markdown(init[1])
            curve = gr.Plot(init[2], show_label=False)
            bt_plot = gr.Plot(init[3], show_label=False)
            bt_table = gr.Dataframe(init[4], interactive=False, label="Backtest statistics (S&P price return, no costs)")
            for control in (block3, model3, thr, mult):
                control.change(threshold_view, [block3, model3, thr, mult], [cm, summary, curve, bt_plot, bt_table])
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=THEME)
