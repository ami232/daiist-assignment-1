"""
Assignment 1 — Gradio dashboard.

Loads only what `train.py` saved in artifacts/ (three trained models, scaler,
features, predictions, validation sweep, bootstrap interval). It runs
inference with the saved models and never trains anything.

    uv run python main.py app
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from sklearn.metrics import log_loss

import gradio as gr

ART = Path(__file__).resolve().parent / "artifacts"
REQUIRED = ["sklearn_model.joblib", "scaler.joblib", "manual_model.pt", "nn_model.pt", "meta.json",
            "predictions.csv", "features.csv", "val_threshold_sweep.csv", "bootstrap.json"]
missing = [f for f in REQUIRED if not (ART / f).exists()]
if missing:
    raise SystemExit(
        f"Missing training artifacts in {ART}: {', '.join(missing)}.\n"
        "The dashboard only loads saved models and never retrains. Run "
        "`uv run python main.py train` first."
    )

# --------------------------------------------------------------------------
# Load artifacts and run inference with the saved models
# --------------------------------------------------------------------------
meta = json.loads((ART / "meta.json").read_text())
boot_file = json.loads((ART / "bootstrap.json").read_text())
boot, BOOT_DRAWS = boot_file["test_profit"], int(boot_file["draws"])
THR = float(meta["threshold"])
FEATURES = meta["features"]
sweep_val = pd.read_csv(ART / "val_threshold_sweep.csv")
feats = pd.read_csv(ART / "features.csv")
saved_pred = pd.read_csv(ART / "predictions.csv")

scaler = joblib.load(ART / "scaler.joblib")
sk_model = joblib.load(ART / "sklearn_model.joblib")
man = torch.load(ART / "manual_model.pt", weights_only=True)
nn_model = nn.Linear(len(FEATURES), 1).double()
nn_model.load_state_dict(torch.load(ART / "nn_model.pt", weights_only=True))

Xs = scaler.transform(feats[FEATURES].to_numpy(float))
with torch.no_grad():
    Xt = torch.tensor(Xs, dtype=torch.float64)
    live = {
        "sklearn": sk_model.predict_proba(Xs)[:, 1],
        "manual": torch.sigmoid(Xt @ man["w"].double() + man["b"].double()).numpy(),
        "nn": torch.sigmoid(nn_model(Xt).squeeze(1)).numpy(),
    }
live = pd.DataFrame({"rid": feats["rid"], **live})
chk = saved_pred.merge(live, on="rid", suffixes=("", "_live"))
for k in live.columns[1:]:
    if np.abs(chk[f"p_{k}"] - chk[k]).max() > 1e-6:
        raise SystemExit("Saved predictions don't match the saved models; re-run "
                         "`uv run python main.py train`.")
test = saved_pred[saved_pred.split == "test"].merge(live, on="rid", suffixes=("_saved", "")).reset_index(drop=True)
Y = test["y_home_win"].to_numpy()
ODDS = test["B365H"].to_numpy(float)
MODELS = [("sklearn", "scikit-learn"), ("manual", "Manual PyTorch loop"), ("nn", "nn.Module + optim")]
P = {k: test[k].to_numpy() for k, _ in MODELS}
PRIOR = float(meta["train_home_win_rate"])

# --------------------------------------------------------------------------
# Style: one light theme for the page and every chart. Model colours are the
# validated categorical slots 1-3, with marker and dash styles as a second cue.
# --------------------------------------------------------------------------
PAGE, SURFACE, INK, INK2, GRID = "#f4f4f1", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
COL = {"sklearn": "#2a78d6", "manual": "#eb6834", "nn": "#1baf7a"}
DASH = {"sklearn": "solid", "manual": "dash", "nn": "dot"}
SYM = {"sklearn": "circle", "manual": "square", "nn": "diamond"}
WIN, LOSE = "#2a78d6", "#eb6834"
MARK = "#333333"  # threshold marker: dark grey, not used by any model
NAME = dict(MODELS)

THEME = gr.themes.Default(
    primary_hue="slate", secondary_hue="slate", neutral_hue="stone",
    font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "Arial", "sans-serif"],
    font_mono=["system-ui", "-apple-system", "Segoe UI", "Roboto", "Arial", "sans-serif"],
    radius_size="md",
).set(
    body_background_fill=PAGE, body_text_color=INK, body_text_color_subdued=INK2,
    block_background_fill=SURFACE, block_border_color=GRID, block_border_width="1px", block_shadow="none",
    block_title_text_color=INK2, block_label_text_color=INK2,
    button_primary_background_fill="#2b3a4f", button_primary_background_fill_hover="#1f2b3b",
    button_primary_text_color="#ffffff", button_secondary_background_fill=SURFACE,
    button_secondary_border_color="#b9b8b2", button_secondary_text_color=INK,
    slider_color="#2b3a4f",
)

CSS = f"""
:root {{ color-scheme: light; }}
.gradio-container {{ max-width: 1180px !important; margin: 0 auto !important; font-family: {FONT}; }}
.gradio-container h1 {{ font-size: 26px; font-weight: 700; letter-spacing: -0.01em; margin: 8px 0 4px; }}
.gradio-container h2 {{ font-size: 19px; font-weight: 650; letter-spacing: -0.005em; margin: 22px 0 6px; }}
.caption p {{ font-size: 13.5px; line-height: 1.5; color: {INK2}; margin: 2px 0 14px; }}
.note {{ border-left: 3px solid {MARK}; padding: 2px 14px; background: {SURFACE}; }}
.note p {{ font-size: 14px; line-height: 1.55; color: {INK}; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 6px 0 10px; }}
.cards.secondary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.card {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 12px; padding: 18px 20px; }}
.card .label {{ font-size: 13px; font-weight: 600; color: {INK2}; }}
.card .value {{ font-size: 32px; font-weight: 700; color: {INK}; line-height: 1.15; margin: 6px 0 4px;
               font-variant-numeric: tabular-nums; }}
.card .sub {{ font-size: 13px; line-height: 1.4; color: {INK2}; }}
.card .value.compact {{ font-size: 24px; margin: 9px 0 6px; }}
.card.small {{ padding: 12px 16px; }}
.card.small .value {{ font-size: 20px; margin: 4px 0 2px; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 560px) {{ .cards, .cards.secondary {{ grid-template-columns: minmax(0, 1fr); }} }}
"""

# Keep the page light even when the browser or OS prefers dark.
JS = """() => {
  const f = () => document.body.classList.remove('dark');
  f(); new MutationObserver(f).observe(document.body, {attributes: true, attributeFilter: ['class']});
}"""


def style(fig, title, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK, family=FONT), x=0.0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(color=INK2, size=12, family=FONT), margin=dict(l=64, r=24, t=64, b=56),
        legend=dict(orientation="h", y=-0.22, font=dict(color=INK2)), hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def profit(p, y, odds, thr):
    mask = (p >= thr) & ~np.isnan(odds)
    return int(mask.sum()), int(((y == 1) & mask).sum()), float(np.where(y[mask] == 1, odds[mask] - 1, -1.0).sum())


def roi(pr, bets):
    return f"{100 * pr / bets:+.1f}%" if bets else "n/a"


def every_home(y, odds):
    return profit(np.ones(len(y)), y, odds, 0.5)


def caption(text):
    return gr.Markdown(text, elem_classes="caption")


# --------------------------------------------------------------------------
# Headline tab
# --------------------------------------------------------------------------
def card(label, value, sub, small=False, compact=False):
    return (f'<div class="card{" small" if small else ""}"><div class="label">{label}</div>'
            f'<div class="value{" compact" if compact else ""}">{value}</div><div class="sub">{sub}</div></div>')


def headline_cards():
    b, w, pr = profit(P["sklearn"], Y, ODDS, THR)
    ci = boot["sklearn"]
    bet_mask = (P["sklearn"] >= THR) & ~np.isnan(ODDS)
    avg_odds = float(ODDS[bet_mask].mean())
    spans = ci["ci_low"] < 0 < ci["ci_high"]
    main = "".join([
        card("Test profit", f"{pr:+.2f} units", f"ROI {roi(pr, b)} at threshold {THR:.3f}; same for all three models"),
        card("Number of bets", f"{b}", f"out of {len(Y):,} test matches"),
        card("Win rate of bets", f"{100 * w / b:.1f}%", f"{w} of {b} bets placed were won (not model accuracy); average odds {avg_odds:.2f}"),
        card("95% bootstrap interval", f"[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]",
             "includes zero" if spans else "does not include zero", compact=True),
    ])
    bh, wh, prh = every_home(Y, ODDS)
    cih = boot["every_home"]
    base = "".join([
        card("Bet on every home team", f"{prh:+.2f} units",
             f"{bh:,} bets, win rate {100 * wh / bh:.1f}%, interval [{cih['ci_low']:+.2f}, {cih['ci_high']:+.2f}]", True),
        card("Majority class (never bets)", "0.00 units",
             f"no bets placed; accuracy {np.mean(Y == 0):.3f}, the accuracy bar to beat", True),
    ])
    return f'<div class="cards">{main}</div>', f'<div class="cards secondary">{base}</div>'


def headline_table():
    rows = []
    for k, label in MODELS:
        b, w, pr = profit(P[k], Y, ODDS, THR)
        ci = boot[k]
        rows.append([label, b, f"{pr:+.2f}", roi(pr, b), f"[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]",
                     f"{np.mean((P[k] >= THR) == (Y == 1)):.4f}", f"{np.mean((P[k] >= 0.5) == (Y == 1)):.4f}",
                     f"{log_loss(Y, P[k]):.4f}"])
    rows.append(["Majority class (never bets)", 0, "+0.00", "n/a", "n/a",
                 f"{np.mean(Y == 0):.4f}", f"{np.mean(Y == 0):.4f}",
                 f"{log_loss(Y, np.full(len(Y), PRIOR)):.4f} (prior)"])
    b, w, pr = every_home(Y, ODDS)
    ci = boot["every_home"]
    rows.append(["Bet on every home team", b, f"{pr:+.2f}", roi(pr, b),
                 f"[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]", f"{np.mean(Y == 1):.4f}",
                 f"{np.mean(Y == 1):.4f}", "n/a"])
    return pd.DataFrame(rows, columns=["Model", "Bets", "Test profit (units)", "ROI",
                                       "95% bootstrap interval", "Accuracy @ betting threshold",
                                       "Accuracy @ 0.5", "Log loss"])


NOTE = ("**Read profit with care.** Profit from a few dozen bets is statistically noisy: one season of "
        "bets can swing by several units either way, and the bootstrap interval includes zero. "
        "The threshold was the best of many noisy validation values, so a small positive number here "
        "is not evidence of a betting edge. The bootstrap interval is descriptive; nothing was chosen with it.")


def fig_validation_sweep():
    s = sweep_val
    ok = s[s.bets >= meta["min_bets"]]
    lo = s[s.bets < meta["min_bets"]]
    n_pos = int((ok.profit > 0).sum())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=ok.threshold, y=ok.profit, mode="lines", name="Validation profit (>= 50 bets)",
                             line=dict(color=COL["sklearn"], width=2),
                             hovertemplate="threshold %{x:.3f}<br>profit %{y:+.2f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=lo.threshold, y=lo.profit, mode="lines", name="Excluded (< 50 bets)",
                             line=dict(color="#a8a7a2", width=1.5, dash="dot"),
                             hovertemplate="threshold %{x:.3f}<br>profit %{y:+.2f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=s.threshold, y=s.bets, mode="lines", name="Number of bets",
                             line=dict(color=INK2, width=2),
                             hovertemplate="threshold %{x:.3f}<br>%{y} bets<extra></extra>"), 2, 1)
    fig.add_hline(y=0, line=dict(color=INK2, width=1), row=1, col=1)
    fig.add_vline(x=THR, line=dict(color=MARK, width=2, dash="dash"))
    fig.add_annotation(x=THR, y=0.405, yref="paper", text=f"validation-chosen {THR:.3f}", showarrow=False,
                       xanchor="left", yanchor="bottom", bgcolor=SURFACE, font=dict(color=MARK))
    fig.update_yaxes(title_text="Profit (units)", row=1, col=1)
    fig.update_yaxes(title_text="Bets", row=2, col=1)
    fig.update_xaxes(title_text="Decision threshold", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return style(fig, f"Validation profit by threshold: only {n_pos} of {len(ok)} eligible thresholds are profitable", 520)


def fig_bootstrap():
    fig = go.Figure()
    for k, label in MODELS:
        ci = boot[k]
        fig.add_trace(go.Scatter(
            x=[ci["profit"]], y=[label], mode="markers", name=label,
            error_x=dict(type="data", symmetric=False, array=[ci["ci_high"] - ci["profit"]],
                         arrayminus=[ci["profit"] - ci["ci_low"]], color=COL[k], thickness=2, width=8),
            marker=dict(color=COL[k], size=11, symbol=SYM[k], line=dict(color=SURFACE, width=2)),
            hovertemplate=f"{label}<br>profit %{{x:+.2f}}<br>95% interval [{ci['ci_low']:+.2f}, "
                          f"{ci['ci_high']:+.2f}]<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK2, width=1))
    fig.update_xaxes(title_text="Test profit (units) with 95% bootstrap interval")
    fig.update_yaxes(autorange="reversed")
    ci = boot["sklearn"]
    where = ("includes zero" if ci["ci_low"] < 0 < ci["ci_high"]
             else "is entirely above zero" if ci["ci_low"] > 0 else "is entirely below zero")
    fig = style(fig, f"Test profit at {THR:.3f}: the 95% interval {where}", 340)
    fig.update_layout(legend=dict(y=-0.62), margin=dict(l=64, r=24, t=64, b=40))
    return fig


# --------------------------------------------------------------------------
# Model comparison tab
# --------------------------------------------------------------------------
def calibration_table(bins=10):
    rows = []
    for k, label in MODELS:
        order = np.argsort(P[k])
        for i, idx in enumerate(np.array_split(order, bins), 1):
            rows.append({"model": label, "bin": i, "n": len(idx),
                         "mean_predicted": float(P[k][idx].mean()), "actual_rate": float(Y[idx].mean())})
    return pd.DataFrame(rows)


def fig_calibration():
    cal = calibration_table()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color="#a8a7a2", dash="dot", width=1.5), hoverinfo="skip"))
    for k, label in MODELS:
        c = cal[cal.model == label]
        fig.add_trace(go.Scatter(
            x=c.mean_predicted, y=c.actual_rate, mode="lines+markers", name=label,
            line=dict(color=COL[k], width=2, dash=DASH[k]),
            marker=dict(color=COL[k], size=9, symbol=SYM[k], line=dict(color=SURFACE, width=2)),
            customdata=c.n, hovertemplate=label + "<br>predicted %{x:.3f}<br>actual %{y:.3f}"
                                                   "<br>%{customdata} matches<extra></extra>"))
    fig.update_xaxes(title_text="Mean predicted P(home win), ten equal-size bins", range=[0, 1])
    fig.update_yaxes(title_text="Actual home-win rate", range=[0, 1])
    return style(fig, "Three implementations, one calibration curve", 460)


def fig_hist():
    fig = make_subplots(rows=1, cols=3, subplot_titles=[n for _, n in MODELS], shared_yaxes=True)
    for j, (k, label) in enumerate(MODELS, 1):
        for val, name, color in ((1, "Home win", WIN), (0, "Not a home win", LOSE)):
            fig.add_trace(go.Histogram(
                x=P[k][Y == val], name=name, marker=dict(color=color, line=dict(color=SURFACE, width=1)),
                opacity=0.65, xbins=dict(start=0, end=1, size=0.05), legendgroup=name,
                showlegend=(j == 1), hovertemplate="P(home win) %{x}<br>%{y} matches<extra>" + name + "</extra>"),
                1, j)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Predicted P(home win)", range=[0, 1])
    fig.update_yaxes(title_text="Matches", col=1)
    hi, lo = P["sklearn"][Y == 1].mean(), P["sklearn"][Y == 0].mean()
    title = ("Home wins get higher predicted probabilities, but the two groups overlap heavily"
             if hi > lo else "Predicted probabilities by actual outcome")
    return style(fig, title, 400)


def model_gap_text():
    d = {f"{a} vs {b}": float(np.abs(P[a] - P[b]).max()) for a, b in (("sklearn", "manual"), ("sklearn", "nn"))}
    return ("Largest probability difference between implementations on test: "
            + "; ".join(f"{k} {v:.1e}" for k, v in d.items())
            + ". They are the same model trained three ways, so the curves coincide.")


# --------------------------------------------------------------------------
# Distributions tab
# --------------------------------------------------------------------------
def fig_target():
    f = feats.copy()
    f["league"] = f[[c for c in f.columns if c.startswith("lg_")]].idxmax(axis=1).str[3:]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Home-win rate by split", "Home-win rate by league"))
    sp = f.groupby("split").y_home_win.agg(["mean", "size"]).reindex(["train", "val", "test"])
    fig.add_trace(go.Bar(x=sp.index, y=sp["mean"], marker=dict(color=COL["sklearn"]), name="All leagues",
                         customdata=sp["size"], hovertemplate="%{x}<br>rate %{y:.3f}<br>%{customdata} matches<extra></extra>",
                         text=[f"{v:.3f}" for v in sp["mean"]], textposition="outside", showlegend=False), 1, 1)
    for s, color in zip(["train", "val", "test"], (COL["sklearn"], COL["manual"], COL["nn"])):
        g = f[f.split == s].groupby("league").y_home_win.mean()
        fig.add_trace(go.Bar(x=g.index, y=g.values, name=s, marker=dict(color=color),
                             hovertemplate="%{x} (" + s + ")<br>rate %{y:.3f}<extra></extra>"), 1, 2)
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[0, 0.6], title_text="Share of matches won by the home team", col=1)
    fig.update_yaxes(range=[0, 0.6], col=2)
    return style(fig, f"The home team wins {sp['mean'].min():.0%} to {sp['mean'].max():.0%} of matches in every split", 420)


def fig_feature(feature, view):
    fig = go.Figure()
    if view == "Train vs validation vs test":
        for s, color in zip(["train", "val", "test"], (COL["sklearn"], COL["manual"], COL["nn"])):
            fig.add_trace(go.Histogram(x=feats.loc[feats.split == s, feature], name=s, histnorm="probability",
                                       opacity=0.6, marker=dict(color=color, line=dict(color=SURFACE, width=1))))
    else:
        tr = feats[feats.split == "train"]
        for val, name, color in ((1, "Home win", WIN), (0, "Not a home win", LOSE)):
            fig.add_trace(go.Histogram(x=tr.loc[tr.y_home_win == val, feature], name=name, histnorm="probability",
                                       opacity=0.6, marker=dict(color=color, line=dict(color=SURFACE, width=1))))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text=feature)
    fig.update_yaxes(title_text="Share of matches")
    return style(fig, f"Distribution of {feature}", 420)


# --------------------------------------------------------------------------
# Threshold explorer tab
# --------------------------------------------------------------------------
def fig_confusion(thr):
    fig = make_subplots(rows=1, cols=3, subplot_titles=["scikit-learn", "Manual PyTorch", "nn.Module"],
                        horizontal_spacing=0.08)
    decisions = [P[k] >= thr for k, _ in MODELS]
    for j, (k, label) in enumerate(MODELS, 1):
        pred, act = P[k] >= thr, Y == 1
        z = np.array([[(pred & act).sum(), (~pred & act).sum()], [(pred & ~act).sum(), (~pred & ~act).sum()]])
        fig.add_trace(go.Heatmap(
            z=z, x=["Bet", "No bet"], y=["Home win", "Not a home win"], text=z.astype(str),
            texttemplate="%{text}", textfont=dict(size=15), colorscale=[[0, "#eaf2fc"], [1, "#8db8ee"]],
            showscale=False, zmin=0, zmax=max(int(z.max()), 1), xgap=2, ygap=2,
            hovertemplate="actual: %{y}<br>decision: %{x}<br>%{z} matches<extra></extra>"), 1, j)
    fig.update_yaxes(autorange="reversed")
    fig.update_yaxes(showticklabels=False, col=2)
    fig.update_yaxes(showticklabels=False, col=3)
    same = all(np.array_equal(decisions[0], d) for d in decisions[1:])
    verdict = "bet identically" if same else "bet differently"
    fig.update_annotations(font=dict(size=12, color=INK2))
    fig = style(fig, f"At {thr:.3f} the three models {verdict}", 330)
    fig.update_layout(margin=dict(l=96, r=16, t=84, b=40))
    return fig


def fig_test_curve(thr):
    ts = np.round(np.arange(0.30, 0.9001, 0.005), 3)
    fig = go.Figure()
    for k, label in MODELS:
        pr = [profit(P[k], Y, ODDS, t) for t in ts]
        fig.add_trace(go.Scatter(
            x=ts, y=[p[2] for p in pr], mode="lines", name=label, customdata=[p[0] for p in pr],
            line=dict(color=COL[k], width=2, dash=DASH[k]),
            hovertemplate=label + "<br>threshold %{x:.3f}<br>profit %{y:+.2f}<br>%{customdata} bets<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=INK2, width=1))
    fig.add_vline(x=THR, line=dict(color=MARK, width=2, dash="dash"))
    fig.add_annotation(x=THR, y=0, yref="paper", text=f"validation-chosen {THR:.3f}", showarrow=False,
                       xanchor="left", yanchor="bottom", bgcolor=SURFACE, font=dict(color=MARK))
    if abs(thr - THR) > 1e-9:
        fig.add_vline(x=thr, line=dict(color=INK, width=1.5, dash="dot"))
        fig.add_annotation(x=thr, y=0.5, yref="paper", text=f"slider {thr:.3f}", showarrow=False,
                           xanchor="left", yanchor="middle", bgcolor=SURFACE, font=dict(color=INK))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="Decision threshold")
    fig.update_yaxes(title_text="Test profit (units)")
    return style(fig, f"Test profit by threshold: the fixed {THR:.3f} is one point on a noisy curve", 420)


def explore(thr):
    thr = float(thr)
    if abs(thr - THR) < 1e-9:
        status = (f"**Slider is at the validation-chosen threshold ({THR:.3f}): the profit below is the "
                  f"headline result.**")
    else:
        status = (f"Slider at {thr:.3f}. The validation-chosen threshold is {THR:.3f}; its profit is the "
                  f"headline result. Values at other thresholds are exploratory test-set numbers.")
    rows = []
    for k, label in MODELS:
        b, w, pr = profit(P[k], Y, ODDS, thr)
        rows.append([label, b, w, f"{pr:+.2f}", roi(pr, b)])
    rows.append(["Majority class (never bets)", 0, 0, "+0.00", "n/a"])
    b, w, pr = every_home(Y, ODDS)
    rows.append(["Bet on every home team", b, w, f"{pr:+.2f}", roi(pr, b)])
    table = pd.DataFrame(rows, columns=["Model", "Bets", "Won", "Profit (units)", "ROI"])
    return status, table, fig_confusion(thr), fig_test_curve(thr)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
cards_main, cards_base = headline_cards()
hi_p, lo_p = P["sklearn"][Y == 1].mean(), P["sklearn"][Y == 0].mean()

with gr.Blocks(title="Football home-win model dashboard") as demo:
    gr.Markdown("# Football home-win model: three implementations, one logistic regression")
    caption(
        "Four leagues (E0, SP1, D1, I1). Target: home win. Train 2019/20-2023/24, validation 2024/25, "
        "test 2025/26. A bettor stakes 1 unit on the home team at Bet365 odds when P(home win) clears the "
        "threshold; a win pays odds - 1, a loss costs 1. Odds are never model inputs. "
        "Everything here comes from the saved training artifacts."
    )
    with gr.Tabs():
        with gr.Tab("Headline"):
            gr.Markdown(f"## Test 2025/26 at the fixed validation-chosen threshold ({THR:.3f})")
            caption("The threshold was picked on 2024/25 validation data with the scikit-learn model and applied "
                    "unchanged to all three models on the test season.")
            gr.HTML(cards_main)
            gr.Markdown("## Baselines for comparison")
            gr.HTML(cards_base)
            gr.Markdown(NOTE, elem_classes="note")
            gr.Markdown("## All models side by side")
            gr.Dataframe(headline_table(), interactive=False, wrap=True, show_row_numbers=False)
            caption("Each row is one way of playing the 2025/26 season with 1 unit per bet; accuracy is shown at "
                    "the betting threshold and at the usual 0.5 cut-off.")
            gr.Markdown("## Can the profit be told apart from luck?")
            gr.Plot(fig_bootstrap(), show_label=False)
            caption(f"Each marker is the actual test profit; the bar is the range of test profit across "
                    f"{BOOT_DRAWS:,} resamples of the {len(Y):,} test matches (fixed seed). A bar that crosses "
                    f"zero means we cannot tell the result apart from luck.")
            gr.Markdown("## How the threshold was chosen")
            gr.Plot(fig_validation_sweep(), show_label=False)
            caption("The top panel is validation profit if the bettor backs every match above each threshold, and "
                    "the bottom panel is how many bets that means; the jagged profit line is what noise looks like, "
                    "and the dark dashed line is the threshold that was picked.")
            with gr.Accordion("Table view: validation threshold sweep", open=False):
                gr.Dataframe(sweep_val.round(3), interactive=False, show_row_numbers=False)

        with gr.Tab("Model comparison"):
            gr.Markdown("## Do the three implementations agree?")
            caption(model_gap_text())
            gr.Plot(fig_calibration(), show_label=False)
            caption("Points on the diagonal mean that when the model says 60%, home teams win about 60% of the "
                    "time; each point is a tenth of the test matches.")
            gr.Markdown("## What does the model predict for wins and non-wins?")
            gr.Plot(fig_hist(), show_label=False)
            caption(f"Each bar counts test matches by predicted chance of a home win, split by what really happened; "
                    f"the average prediction was {hi_p:.2f} for home wins and {lo_p:.2f} for everything else.")
            with gr.Accordion("Table view: calibration bins", open=False):
                gr.Dataframe(calibration_table().round(4), interactive=False, show_row_numbers=False)

        with gr.Tab("Threshold explorer"):
            gr.Markdown("## Move the threshold and watch the test-set result change")
            caption(f"The validation-chosen threshold is {THR:.3f}; the button returns to it. Do not use this "
                    f"slider to pick a threshold, because the test set was meant to be touched once.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=340):
                    slider = gr.Slider(0.30, 0.90, value=THR, step=0.005,
                                       label=f"Decision threshold (validation-chosen: {THR:.3f})")
                    reset = gr.Button(f"Reset to validation-chosen threshold ({THR:.3f})", variant="primary")
                    status = gr.Markdown()
                    table = gr.Dataframe(interactive=False, wrap=True, show_row_numbers=False)
                with gr.Column(scale=3, min_width=420):
                    cm = gr.Plot(show_label=False)
                    caption("Each box counts test matches by what happened (rows) and what the bettor was told to "
                            "do (columns); a home win with a bet is a winning bet.")
            curve = gr.Plot(show_label=False)
            caption("Each point is the test profit if the bettor had used that threshold; it is for exploring "
                    "only, since the real threshold was fixed in advance on validation.")
            gr.Markdown(NOTE, elem_classes="note")
            outs = [status, table, cm, curve]
            slider.change(explore, slider, outs)
            reset.click(lambda: THR, None, slider)
            demo.load(explore, slider, outs)

        with gr.Tab("Distributions"):
            gr.Markdown("## Target: how often the home team wins")
            gr.Plot(fig_target(), show_label=False)
            caption("Bars show the share of matches the home team won, overall by split on the left and by "
                    "league on the right.")
            gr.Markdown("## Features: what the model sees")
            gr.Markdown(
                "**Reading the feature names.** `h_` and `a_` mean the home and away team. A trailing number is "
                "the rolling window in matches, and every value is a per-match average over that team's "
                "previous matches (earlier seasons included), so `h_gf10` is the home team's goals scored per "
                "game over its last 10 matches (`ga` is goals conceded). `ppg10` is points per game over the "
                "last 10; `home_ppg5` and `away_ppg5` are points per game over the last 5 matches at that "
                "venue only (the home team's home matches, the away team's away matches). `sotdiff10` is "
                "shots on target for minus against and `shotdiff10` is total shots for minus against, both per "
                "game over the last 10; `gd38` is goal difference per game over the last 38 matches, roughly "
                "one season; `cards10` is the team's own cards per game over the last 10 (yellow = 1, red = 2). "
                "`rest` is days since the team's last match (capped at 14); `elo` is the team's Elo rating "
                "before the match; `cold` is 1 when any of that team's features fell back to the cold-start "
                "default because it had too little match history, for example a newly promoted team. "
                "`lg_E0`, `lg_SP1`, `lg_D1` and `lg_I1` are one-hot league indicators (Premier League, La Liga, "
                "Bundesliga, Serie A). Every feature uses only matches played before the one being predicted.",
                elem_classes="note",
            )
            with gr.Row():
                feat = gr.Dropdown(FEATURES, value=FEATURES[0], label="Feature")
                view = gr.Radio(["Train vs validation vs test", "Home win vs not (train)"],
                                value="Train vs validation vs test", label="Compare")
            fplot = gr.Plot(show_label=False)
            caption("Bars show how common each value is; when the train, validation and test shapes line up, "
                    "the model meets familiar inputs at test time.")
            feat.change(fig_feature, [feat, view], fplot)
            view.change(fig_feature, [feat, view], fplot)
            demo.load(fig_feature, [feat, view], fplot)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=THEME, css=CSS, js=JS,
                footer_links=[])
