import math
from datetime import date

import gradio as gr
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.model.predict import MODELS, cost, good, margin

INK = "#3f3f3c"
MUTED = "#898781"
LIGHT = "#c3c2b7"
GRID = "#e8e7e2"
BLUE = "#2a78d6"
GOOD = "#0ca30c"
BAD = "#d03b3b"
WARN = "#eda100"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
COLORS = {"sklearn": BLUE, "manual_torch": "#eb6834", "standard_torch": "#1baf7a", "naive": MUTED}
LABELS = {
    "sklearn": "scikit-learn",
    "manual_torch": "manual PyTorch",
    "standard_torch": "standard PyTorch",
    "naive": "naive (yesterday)",
}

THEME = gr.themes.Soft(primary_hue="slate", neutral_hue="stone", font=[FONT])
# same look whatever the OS theme, so the gray plots always read
for key in list(vars(THEME)):
    if key.endswith("_dark"):
        setattr(THEME, key, getattr(THEME, key[: -len("_dark")]))

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
.tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;}
.tile {background: #f4f4f2; border-radius: 10px; padding: 12px 16px;}
.tile .label {color: #898781; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;}
.tile .value {color: #0b0b0b; font-size: 26px; font-weight: 600; margin-top: 2px;}
.tile .sub {color: #52514e; font-size: 13px;}
.zwrap {overflow-x: auto;}
.zgrid {border-collapse: separate; border-spacing: 2px; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums;}
.zgrid, .zgrid tr, .zgrid th, .zgrid td {border: none !important;}
.zgrid th {color: #52514e; font-weight: 500; text-align: right; padding: 6px 10px; white-space: nowrap; background: none;}
.zgrid td {text-align: right; padding: 6px 10px; border-radius: 4px; white-space: nowrap; color: #0b0b0b; background: #f4f4f2;}
.zgrid td span {color: inherit; opacity: .7;}
.score {border-collapse: separate; border-spacing: 0 2px; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums;}
.score, .score tr, .score th, .score td {border: none !important;}
.score th {color: #898781; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; text-align: right; padding: 8px 12px; background: none;}
.score td {text-align: right; padding: 10px 12px; color: #0b0b0b; background: #f4f4f2;}
.score th:first-child, .score td:first-child {text-align: left;}
.score tr.selected td {background: #e4e9f0; font-weight: 600;}
.znote {color: #898781; font-size: 12px; margin-top: 6px;}
.edgewrap {max-height: 380px; overflow: auto;}
.edgewrap td, .edgewrap th {white-space: nowrap; padding: 8px 10px !important;}
.score td.pos {color: #0ca30c;} .score td.neg {color: #d03b3b;}
"""


def _layout(
    fig: go.Figure,
    title: str,
    height: int = 380,
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 15, "color": INK}},
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": MUTED, "size": 12},
        margin={"l": 55, "r": 20, "t": 50, "b": 70},
        legend={"orientation": "h", "x": 0, "y": -0.12, "yanchor": "top", "font": {"color": INK}},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=LIGHT, zeroline=False)
    return fig.update_yaxes(gridcolor=GRID, linecolor=LIGHT, zeroline=False)


def _mark_test_start(
    fig: go.Figure,
    df: pl.DataFrame,
    test_start: date,
) -> None:
    if df["day"].min() < test_start <= df["day"].max():
        fig.add_vline(x=test_start, line={"color": LIGHT, "width": 1})
        fig.add_annotation(
            x=test_start,
            y=1,
            yref="paper",
            text="test start",
            showarrow=False,
            xanchor="left",
            font={"color": MUTED},
        )


def stat_tiles(tiles: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="tile"><div class="label">{a}</div><div class="value">{b}</div><div class="sub">{c}</div></div>'
        for a, b, c in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}"


def _zcolor(z: float | None) -> str:
    # red above normal, blue below, darker = more extreme
    z = z if z is not None and math.isfinite(z) else 0.0
    rgb = "208,59,59" if z > 0 else "42,120,214"
    return f"rgba({rgb},{min(abs(z), 3) / 3 * 0.85:.2f})"


def _cell(z: float | None, v: float | None) -> str:
    if z is None or v is None or not math.isfinite(z) or not math.isfinite(v):
        return "<td>n/a</td>"
    ink = "color: #fff;" if abs(z) > 1.75 else ""
    return f'<td style="background: {_zcolor(z)}; {ink}"><b>{z:+.1f}</b> <span>({_fmt(v)})</span></td>'


def zscore_grid(
    z: pl.DataFrame,
    labels: dict[str, str],
    window: int,
) -> str:
    # rows = variables, columns = days, cell = z (value)
    head = "".join(f"<th>{d:%a %d %b}</th>" for d in z["day"])
    rows = "".join(
        f"<tr><th>{label}</th>{''.join(_cell(a, b) for a, b in zip(z[f'z|{c}'], z[c]))}</tr>"
        for c, label in labels.items()
    )
    note = f"z-score vs the previous {window} days (value). Red above normal, blue below."
    return f'<div class="zwrap"><table class="zgrid"><tr><th></th>{head}</tr>{rows}</table></div><div class="znote">{note}</div>'


def spread_chart(
    df: pl.DataFrame,
    model: str,
    threshold: float,
    test_start: date,
) -> go.Figure:
    cycle = pl.col(f"p_{model}") >= threshold
    lost = df.filter(cycle & ~good)
    missed = df.filter(~cycle & good)
    marker = {"size": 9, "line": {"color": "#fff", "width": 2}}
    fig = go.Figure(
        [
            go.Scatter(
                x=df["day"],
                y=df["spread_30d"],
                name="30 day average",
                line={"color": MUTED, "width": 2, "dash": "dot"},
            ),
            go.Scatter(
                x=df["day"],
                y=df["y_spread"],
                name="realised spread",
                line={"color": INK, "width": 2},
            ),
            go.Scatter(
                x=lost["day"],
                y=lost["y_spread"],
                name="cycled, lost",
                mode="markers",
                marker={**marker, "color": BAD},
            ),
            go.Scatter(
                x=missed["day"],
                y=missed["y_spread"],
                name="missed",
                mode="markers",
                marker={**marker, "color": WARN},
            ),
        ],
    )
    fig.add_hline(
        y=cost,
        line={"color": BAD, "width": 1, "dash": "dash"},
        annotation_text=f"cycle cost {cost:.0f}",
        annotation_font_color=BAD,
    )
    _mark_test_start(fig, df, test_start)
    return _layout(fig, f"Daily spread and errors of {LABELS[model]} (EUR/MWh)")


def probability_chart(
    df: pl.DataFrame,
    model: str,
    threshold: float,
    test_start: date,
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=df["day"],
            y=df.select(good.cast(pl.Int8)).to_series(),
            name="spread beat cost",
            marker={"color": "#f0efec", "line": {"width": 0}},
            hoverinfo="skip",
        ),
    )
    for m in MODELS:
        selected = m == model
        fig.add_scatter(
            x=df["day"],
            y=df[f"p_{m}"],
            name=LABELS[m],
            line={"color": COLORS[m], "width": 2.5 if selected else 1},
            opacity=1 if selected else 0.6,
        )
    fig.add_hline(
        y=threshold, line={"color": INK, "width": 1, "dash": "dash"}, annotation_text=f"threshold {threshold:.0%}"
    )
    _mark_test_start(fig, df, test_start)
    fig.update_layout(bargap=0)
    return _layout(fig, "Probability vs outcome", 340).update_yaxes(
        range=[0, 1.05], tickformat=".0%"
    )


def pnl_chart(df: pl.DataFrame, threshold: float) -> go.Figure:
    # cumulative P&L minus naive
    cum = lambda cycle: pl.when(cycle).then(margin).otherwise(0).cum_sum()  # noqa: E731
    pnl = df.select(
        "day",
        perfect=cum(good) - cum(pl.col("p_naive") >= threshold),
        **{m: cum(pl.col(f"p_{m}") >= threshold) - cum(pl.col("p_naive") >= threshold) for m in MODELS},
    )
    fig = go.Figure(
        go.Scatter(
            x=pnl["day"],
            y=pnl["perfect"],
            name="perfect foresight",
            line={"color": LIGHT, "width": 2, "dash": "dot"},
        )
    )
    for m in MODELS:
        fig.add_scatter(
            x=pnl["day"],
            y=pnl[m],
            name=LABELS[m],
            line={"color": COLORS[m], "width": 2},
        )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1, "dash": "dash"}, annotation_text="naive")
    return _layout(fig, "Cumulative P&L vs naive (EUR/MW)").update_yaxes(tickformat=",.0f")


def distribution_chart(
    df: pl.DataFrame,
    reference: pl.DataFrame,
    column: str,
    title: str,
    line: tuple[float, str] | None = None,
) -> go.Figure:
    fig = go.Figure(
        [
            go.Histogram(
                x=reference[column],
                name="training period",
                histnorm="percent",
                bingroup=1,
                marker_color=LIGHT,
                opacity=0.8,
            ),
            go.Histogram(
                x=df[column],
                name="selected range",
                histnorm="percent",
                bingroup=1,
                marker_color=BLUE,
                opacity=0.6,
            ),
        ],
    )
    if line is not None and math.isfinite(line[0]):
        fig.add_vline(x=line[0], line={"color": INK, "width": 1, "dash": "dash"}, annotation_text=line[1])
    fig.update_layout(barmode="overlay", hovermode="closest")
    return _layout(fig, title, 340).update_yaxes(ticksuffix="%")


def scorecard_table(scores: pl.DataFrame, model: str) -> str:
    heads = [
        "Model",
        "Cycles",
        "Hit rate",
        "Accuracy",
        "ROC AUC",
        "Brier",
        "P&L EUR/MW",
        "Missed EUR/MW",
        "Capture of perfect",
    ]
    head = "".join(f"<th>{h}</th>" for h in heads)
    rows = "".join(
        f'<tr class="{"selected" if r["model"] == model else ""}"><td>{LABELS[r["model"]]}</td><td>{r["cycles"]}</td>'
        f"<td>{r['hit_rate'] or 0:.1%}</td><td>{r['accuracy']:.1%}</td>"
        f"<td>{'n/a' if r['roc_auc'] is None else f'{r["roc_auc"]:.3f}'}</td><td>{r['brier']:.3f}</td>"
        f"<td>{r['profit_eur_per_mw']:,.0f}</td><td>{r['missed_eur_per_mw']:,.0f}</td><td>{r['capture'] or 0:.1%}</td></tr>"
        for r in scores.iter_rows(named=True)
    )
    return f'<table class="score"><tr>{head}</tr>{rows}</table>'


def threshold_chart(sweep: pl.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    for m in ["naive", *MODELS]:
        rows = sweep.filter(model=m)
        fig.add_scatter(
            x=rows["threshold"],
            y=rows["profit_eur_per_mw"],
            name=LABELS[m],
            line={"color": COLORS[m], "width": 2, "dash": "dash" if m == "naive" else "solid"},
        )
    fig.add_vline(x=threshold, line={"color": INK, "width": 1, "dash": "dot"}, annotation_text="current")
    fig = _layout(fig, "P&L by threshold (EUR/MW)")
    fig.update_xaxes(tickformat=".0%")
    return fig.update_yaxes(tickformat=",.0f")


def calibration_chart(cal: pl.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="perfectly calibrated",
            line={"color": LIGHT, "width": 1, "dash": "dash"},
        )
    )
    for m in MODELS:
        rows = cal.filter(model=m)
        fig.add_scatter(
            x=rows["p_mean"],
            y=rows["observed"],
            name=LABELS[m],
            mode="lines+markers",
            line={"color": COLORS[m], "width": 2},
            marker={"size": [4 + 14 * math.sqrt(n / cal["n"].max()) for n in rows["n"]]},
            customdata=rows["n"],
            hovertemplate="predicted %{x:.0%}<br>observed %{y:.0%}<br>%{customdata} days",
        )
    fig = _layout(fig, "Calibration")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return fig.update_yaxes(range=[0, 1.02], tickformat=".0%")


def contribution_chart(
    terms: pl.DataFrame,
    bias: float,
    title: str,
    top: int = 12,
) -> go.Figure:
    # log-odds waterfall: bias, top features, rest
    head, rest = terms.head(top), terms.slice(top)
    labels = ["baseline", *head["feature"], f"{rest.height} other features", "total"]
    values = [bias, *head["contribution"], rest["contribution"].sum(), 0]
    hover = [
        "model intercept",
        *[f"value {_fmt(v)}" if v is not None and math.isfinite(v) else "missing" for v in head["value"]],
        "",
        "",
    ]
    logit = bias + terms["contribution"].sum()
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=labels,
            x=values,
            measure=["absolute", *["relative"] * (len(labels) - 2), "total"],
            customdata=hover,
            hovertemplate="%{y}: %{x:+.2f}<br>%{customdata}<extra></extra>",
            increasing={"marker": {"color": GOOD}},
            decreasing={"marker": {"color": BAD}},
            totals={"marker": {"color": BLUE}},
            connector={"line": {"color": LIGHT, "width": 1}},
        ),
    )
    fig.add_vline(x=0, line={"color": MUTED, "width": 1})
    fig.add_annotation(
        x=logit,
        y="total",
        text=f"p = {1 / (1 + math.exp(-logit)):.0%}",
        showarrow=False,
        xanchor="left" if logit >= 0 else "right",
        xshift=8 if logit >= 0 else -8,
        font={"color": INK},
    )
    fig = _layout(fig, title, 480)
    fig.update_layout(hovermode="closest", showlegend=False, margin={"l": 190})
    fig.update_xaxes(title="log-odds (green pushes to cycle)")
    return fig.update_yaxes(autorange="reversed")


def weights_chart(weights: pl.DataFrame, top: int = 15) -> go.Figure:
    rows = weights.sort(pl.mean_horizontal([pl.col(m).abs() for m in MODELS]), descending=True).head(top)
    fig = go.Figure(
        [
            go.Bar(
                y=rows["feature"],
                x=rows[m],
                name=LABELS[m],
                orientation="h",
                marker_color=COLORS[m],
            )
            for m in MODELS
        ],
    )
    fig = _layout(fig, "Model weights (per std)", 480)
    fig.update_layout(barmode="group", hovermode="y unified", margin={"l": 190})
    return fig.update_yaxes(autorange="reversed")


def edge_table(edge: pl.DataFrame, model: str) -> str:
    # days where the model and naive disagree
    days = edge.filter(pl.col("cycle") != pl.col("naive")).sort("day", descending=True)
    call = lambda c: "cycle" if c else "rest"  # noqa: E731
    head = "".join(f"<th>{h}</th>" for h in ["Day", "Spread", "Prob.", "Model", "Naive", "Edge EUR/MW"])
    rows = "".join(
        f"<tr><td>{r['day']:%d %b %y}</td><td>{r['y_spread']:.1f}</td><td>{r['p']:.0%}</td><td>{call(r['cycle'])}</td>"
        f'<td>{call(r["naive"])}</td><td class="{"pos" if r["edge"] > 0 else "neg"}">{r["edge"]:+,.0f}</td></tr>'
        for r in days.iter_rows(named=True)
    )
    if not rows:
        return '<div class="znote">Same call as naive on every day.</div>'
    note = f"{days.height} days where {LABELS[model]} and naive disagree."
    return (
        f'<div class="znote">{note}</div><div class="edgewrap"><table class="score"><tr>{head}</tr>{rows}</table></div>'
    )


def day_price_chart(
    prices: pl.DataFrame,
    day: date,
    window: int,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
    )
    band = prices.drop_nulls("std")
    fig.add_scatter(
        x=band["hour"],
        y=band["mean"] + band["std"],
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=band["hour"],
        y=band["mean"] - band["std"],
        name=f"normal range ({window}d mean ± 1 sd)",
        fill="tonexty",
        fillcolor="rgba(195,194,183,0.3)",
        line={"width": 0},
        hoverinfo="skip",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=band["hour"],
        y=band["mean"],
        name="same-hour mean",
        line={"color": MUTED, "width": 1, "dash": "dot"},
        hoverinfo="skip",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=prices["hour"],
        y=prices["price"],
        name="price",
        line={"color": INK, "width": 2},
        customdata=prices["z"],
        hovertemplate="%{x}h  %{y:.1f} EUR/MWh, z %{customdata:+.1f}<extra></extra>",
        row=1,
        col=1,
    )
    for role, color in [("charge", BLUE), ("discharge", BAD)]:
        rows = prices.filter(role=role)
        fig.add_scatter(
            x=rows["hour"],
            y=rows["price"],
            name=f"{role}: {round(rows['price'].mean()) + 0} avg" if rows.height else role,
            mode="markers",
            marker={"color": color, "size": 11, "line": {"color": "#fff", "width": 2}},
            hoverinfo="skip",
            row=1,
            col=1,
        )
    fig.add_bar(
        x=prices["hour"],
        y=prices["z"],
        marker_color=[_zcolor(z) for z in prices["z"]],
        showlegend=False,
        hovertemplate="%{x}h  z %{y:+.1f}<extra></extra>",
        row=2,
        col=1,
    )
    fig = _layout(fig, f"Hourly price, {day:%d %b %Y} (EUR/MWh)", 480)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(dtick=2, ticksuffix="h")
    return fig.update_yaxes(title_text="z", row=2, col=1)


def hourly_z_heatmap(prices: pl.DataFrame, window: int) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            x=prices["day"],
            y=prices["hour"],
            z=prices["z"],
            zmid=0,
            zmin=-3,
            zmax=3,
            colorscale=[[0, "rgba(42,120,214,0.85)"], [0.5, "#f4f4f2"], [1, "rgba(208,59,59,0.85)"]],
            colorbar={"title": {"text": "z"}, "thickness": 10},
            hovertemplate="%{x|%a %d %b %Y} %{y}h<br>z %{z:+.1f}<extra></extra>",
        ),
    )
    fig = _layout(fig, f"Hourly price z-score ({window}d)", 420)
    fig.update_layout(hovermode="closest")
    return fig.update_yaxes(dtick=3, ticksuffix="h", autorange="reversed")


OUTCOMES = {
    "earned": ("earned (cycled, profitable)", GOOD, 1, 1.0),
    "saved": ("saved (rested, would have lost)", BLUE, 1, 0.5),
    "lost": ("lost (cycled, unprofitable)", BAD, -1, 1.0),
    "missed": ("missed (rested, would have earned)", WARN, -1, 0.5),
}


def earnings_chart(
    out: pl.DataFrame,
    model: str,
    test_start: date,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for cell, (name, color, sign, opacity) in OUTCOMES.items():
        rows = out.filter(cell=cell)
        fig.add_bar(
            x=rows["day"],
            y=sign * rows["eur"],
            name=name,
            marker_color=color,
            opacity=opacity,
            secondary_y=False,
        )
    cum = lambda cells, sign=1: pl.when(pl.col("cell").is_in(cells)).then(sign * pl.col("eur")).otherwise(0.0)  # noqa: E731
    pnl = out.select(
        "day", model=(cum(["earned"]) + cum(["lost"], -1)).cum_sum(), perfect=cum(["earned", "missed"]).cum_sum()
    )
    fig.add_scatter(
        x=pnl["day"],
        y=pnl["perfect"],
        name="cumulative, perfect foresight",
        line={"color": LIGHT, "width": 2, "dash": "dot"},
        secondary_y=True,
    )
    fig.add_scatter(
        x=pnl["day"],
        y=pnl["model"],
        name=f"cumulative P&L, {LABELS[model]}",
        line={"color": INK, "width": 2.5},
        secondary_y=True,
    )
    _mark_test_start(fig, out, test_start)
    fig = _layout(fig, "Earnings (EUR/MW)", 420)
    fig.update_layout(barmode="relative", bargap=0.1)
    fig.update_yaxes(tickformat=",.0f", secondary_y=False)
    return fig.update_yaxes(tickformat=",.0f", showgrid=False, secondary_y=True)


def confusion_chart(out: pl.DataFrame, threshold: float) -> go.Figure:
    agg = {r["cell"]: r for r in out.group_by("cell").agg(days=pl.len(), eur=pl.col("eur").sum()).iter_rows(named=True)}
    days = lambda c: agg.get(c, {}).get("days", 0)  # noqa: E731
    eur = lambda c: agg.get(c, {}).get("eur", 0.0)  # noqa: E731
    grid = [["earned", "lost"], ["missed", "saved"]]
    text = [[f"<b>{days(c)} days</b><br>{c} {eur(c):,.0f} EUR/MW" for c in row] for row in grid]
    fig = go.Figure(
        go.Heatmap(
            x=["profitable", "unprofitable"],
            y=["cycle", "rest"],
            z=[[1, 0], [0, 1]],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": INK},
            colorscale=[[0, "#f6dcdc"], [1, "#dcefdc"]],
            showscale=False,
            hoverinfo="skip",
            xgap=4,
            ygap=4,
        ),
    )
    title = f"Confusion at {threshold:.0%}, errors cost {eur('lost') + eur('missed'):,.0f} EUR/MW"
    fig = _layout(fig, title, 380)
    fig.update_xaxes(title="realised day", side="bottom", showgrid=False)
    return fig.update_yaxes(title="model call", autorange="reversed", showgrid=False)


def hour_roles_chart(prices: pl.DataFrame) -> go.Figure:
    days = prices["day"].n_unique()
    share = (
        prices.group_by("hour")
        .agg(
            charge=(pl.col("role") == "charge").sum() / days,
            discharge=(pl.col("role") == "discharge").sum() / days,
        )
        .sort("hour")
    )
    fig = go.Figure(
        [
            go.Bar(
                x=share["hour"],
                y=share["charge"],
                name="cheapest 4h (charge)",
                marker_color=BLUE,
            ),
            go.Bar(
                x=share["hour"],
                y=share["discharge"],
                name="dearest 4h (discharge)",
                marker_color=BAD,
            ),
        ],
    )
    fig = _layout(fig, "Charge and discharge hours", 340)
    fig.update_layout(barmode="group", bargap=0.15)
    fig.update_xaxes(dtick=2, ticksuffix="h")
    return fig.update_yaxes(tickformat=".0%")


def drift_chart(table: pl.DataFrame, top: int = 12) -> go.Figure:
    rows = table.head(top)
    colors = [BAD if v >= 0.25 else WARN if v >= 0.1 else LIGHT for v in rows["psi"]]
    fig = go.Figure(
        go.Bar(
            y=rows["column"],
            x=rows["psi"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: PSI %{x:.2f}<extra></extra>",
        )
    )
    for x, text in [(0.1, "moderate"), (0.25, "large")]:
        fig.add_vline(x=x, line={"color": MUTED, "width": 1, "dash": "dot"}, annotation_text=text)
    fig = _layout(fig, "Input drift vs training (PSI)", 420)
    fig.update_layout(hovermode="closest", showlegend=False, margin={"l": 190})
    return fig.update_yaxes(autorange="reversed")


RANGE_HTML = """
<div class="drange">
  <div class="drange-head">
    <span>${first}</span>
    <b class="drange-value">${value[0]} to ${value[1]}</b>
    <span>${last}</span>
  </div>
  <div class="drange-track">
    <div class="drange-fill" style="left: ${100 * (new Date(value[0]) - new Date(first)) / (new Date(last) - new Date(first))}%;
      right: ${100 - 100 * (new Date(value[1]) - new Date(first)) / (new Date(last) - new Date(first))}%;"></div>
    <input type="range" class="lo" min="0" max="${(new Date(last) - new Date(first)) / 864e5}" step="1"
      value="${(new Date(value[0]) - new Date(first)) / 864e5}">
    <input type="range" class="hi" min="0" max="${(new Date(last) - new Date(first)) / 864e5}" step="1"
      value="${(new Date(value[1]) - new Date(first)) / 864e5}">
  </div>
</div>
"""

RANGE_CSS = """
.drange {padding: 4px 8px 10px;}
.drange-head {display: flex; justify-content: space-between; font-size: 12px; color: #898781; margin-bottom: 10px;}
.drange-value {color: #3f3f3c; font-size: 14px;}
.drange-track {position: relative; height: 6px; background: #e8e7e2; border-radius: 3px; margin: 0 8px;}
.drange-fill {position: absolute; top: 0; bottom: 0; background: #2a78d6; border-radius: 3px;}
.drange input[type=range] {position: absolute; left: -8px; width: calc(100% + 16px); top: -6px; height: 18px; margin: 0;
  background: none; pointer-events: none; -webkit-appearance: none; appearance: none;}
.drange input[type=range]::-webkit-slider-runnable-track {background: none; border: none;}
.drange input[type=range]::-webkit-slider-thumb {pointer-events: auto; -webkit-appearance: none; width: 16px; height: 16px;
  border-radius: 50%; background: #fff; border: 2px solid #2a78d6; cursor: grab;}
.drange input[type=range]::-moz-range-thumb {pointer-events: auto; width: 14px; height: 14px; border-radius: 50%;
  background: #fff; border: 2px solid #2a78d6; cursor: grab;}
"""

# moves the handles live, commits [start, end] on release
RANGE_JS = """
const span = (new Date(props.last) - new Date(props.first)) / 864e5;
const iso = (i) => new Date(new Date(props.first).getTime() + i * 864e5).toISOString().slice(0, 10);
const read = (e) => {
  const lo = element.querySelector('.lo'), hi = element.querySelector('.hi');
  if (+lo.value > +hi.value) (e.target === lo ? hi : lo).value = e.target.value;
  return [lo, hi];
};
element.addEventListener('input', (e) => {
  const [lo, hi] = read(e);
  const fill = element.querySelector('.drange-fill');
  fill.style.left = `${100 * lo.value / span}%`;
  fill.style.right = `${100 - 100 * hi.value / span}%`;
  element.querySelector('.drange-value').textContent = `${iso(+lo.value)} to ${iso(+hi.value)}`;
});
element.addEventListener('change', (e) => {
  const [lo, hi] = read(e);
  props.value = [iso(+lo.value), iso(+hi.value)];
  trigger('change');
});
"""


# a function, not a gr.HTML subclass: gradio writes a .pyi stub for every subclass
def DateRange(  # noqa: N802
    first: date,
    last: date,
    value: list[str],
) -> gr.HTML:
    return gr.HTML(
        value=value,
        html_template=RANGE_HTML,
        css_template=RANGE_CSS,
        js_on_load=RANGE_JS,
        first=str(first),
        last=str(last),
    )
