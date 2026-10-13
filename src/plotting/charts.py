import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from src.config import CONFIG, PlotSize

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# offer_type, legend name, color, sort descending
SIDES = [
    ("V", "supply (V)", "#d62728", False),
    ("C", "demand (C)", "#2ca02c", True),
]


def line_chart(
    df: pl.DataFrame,
    x: str,
    y: str | list[str],
    *,
    every: str | None = None,
    size: PlotSize = "medium",
    title: str | None = None,
) -> go.Figure:
    # every (e.g. "1d") resamples intraday data by mean
    ys = [y] if isinstance(y, str) else y
    df = df.sort(x)
    if every:
        df = df.group_by_dynamic(x, every=every).agg(pl.col(ys).mean())
    colors = CONFIG.plot.line_colors
    fig = go.Figure(
        [
            go.Scattergl(
                x=df[x],
                y=df[col],
                name=col,
                mode="lines",
                line={"color": colors[i % len(colors)], "width": 1},
            )
            for i, col in enumerate(ys)
        ],
    )
    fig.update_layout(hovermode="x unified")
    return _layout(fig, size, title)


def histogram(
    df: pl.DataFrame,
    x: str | list[str],
    *,
    nbins: int | None = None,
    size: PlotSize = "medium",
    title: str | None = None,
) -> go.Figure:
    xs = [x] if isinstance(x, str) else x
    colors = CONFIG.plot.line_colors
    fig = go.Figure(
        [
            go.Histogram(
                x=df[col],
                name=col,
                nbinsx=nbins,
                marker_color=colors[i % len(colors)],
                opacity=0.6 if len(xs) > 1 else 1.0,
            )
            for i, col in enumerate(xs)
        ],
    )
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title="count")
    return _layout(fig, size, title)


def carpet_plot(
    df: pl.DataFrame,
    time: str,
    value: str,
    *,
    every: str = "1h",
    size: PlotSize = "medium",
    title: str | None = None,
) -> go.Figure:
    # day on x, time of day on y
    t = pl.col(time).dt.truncate(every)
    cells = (
        df.group_by(
            t.dt.date().alias("day"),
            (t.dt.hour() + t.dt.minute() / 60).alias("hour"),
        )
        .agg(pl.col(value).mean())
        .sort("day", "hour")
    )
    fig = go.Figure(
        go.Heatmap(
            x=cells["day"],
            y=cells["hour"],
            z=cells[value],
            colorscale=CONFIG.plot.carpet_colorscale,
            colorbar={"title": value},
        ),
    )
    fig.update_yaxes(title="hour of day", dtick=3)
    return _layout(fig, size, title)


def calendar_plot(
    df: pl.DataFrame,
    time: str,
    value: str,
    *,
    size: PlotSize = "medium",
    title: str | None = None,
) -> go.Figure:
    # one row per year: week on x, weekday on y
    jan1_weekday = pl.date(pl.col("day").dt.year(), 1, 1).dt.weekday() - 1
    daily = (
        df.group_by(pl.col(time).cast(pl.Date).alias("day"))
        .agg(pl.col(value).mean())
        .with_columns(
            pl.col("day").dt.year().alias("year"),
            (pl.col("day").dt.weekday() - 1).alias("weekday"),
            # week of year anchored on Jan 1's weekday, no ISO year wrap
            ((pl.col("day").dt.ordinal_day() - 1 + jan1_weekday) // 7).alias("week"),
        )
    )
    years = daily["year"].unique().sort().to_list()
    fig = make_subplots(
        rows=len(years),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.3 / len(years),
        subplot_titles=[str(yr) for yr in years],
    )
    for row, yr in enumerate(years, start=1):
        d = daily.filter(pl.col("year") == yr)
        fig.add_trace(
            go.Heatmap(
                x=d["week"],
                y=d["weekday"],
                z=d[value],
                customdata=d["day"].cast(pl.String),
                hovertemplate="%{customdata}<br>%{z:.2f}<extra></extra>",
                coloraxis="coloraxis",
                xgap=1,
                ygap=1,
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(
        tickvals=list(range(7)),
        ticktext=WEEKDAYS,
        autorange="reversed",
    )
    fig.update_layout(
        coloraxis={
            "colorscale": CONFIG.plot.calendar_colorscale,
            "colorbar": {"title": value},
        },
    )
    return _layout(fig, size, title)


def acf_chart(
    acf: pl.DataFrame,
    *,
    size: PlotSize = "medium",
    title: str | None = None,
) -> go.Figure:
    ci = acf["ci"][0]
    fig = go.Figure(
        go.Bar(
            x=acf["lag"],
            y=acf["acf"],
            marker_color=CONFIG.plot.line_colors[0],
        ),
    )
    for bound in (ci, -ci):
        fig.add_hline(y=bound, line_dash="dot", line_color="grey")
    fig.update_xaxes(title="lag")
    return _layout(fig, size, title)


def heatmap(
    df: pl.DataFrame,
    x: str,
    y: str,
    z: str,
    *,
    zmid: float | None = None,
    size: PlotSize = "medium",
    title: str | None = None,
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            x=df[x],
            y=df[y],
            z=df[z],
            zmid=zmid,
            colorscale=CONFIG.plot.heatmap_colorscale,
            colorbar={"title": z},
        ),
    )
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return _layout(fig, size, title)


def bid_curves(
    curves: pl.DataFrame,
    *,
    size: PlotSize = "medium",
    title: str | None = None,
) -> go.Figure:
    fig = go.Figure(
        [
            go.Scatter(
                x=d["quantity_mw"].cum_sum(),
                y=d["price_eur_mwh"],
                name=name,
                line_shape="hv",
                line_color=color,
            )
            for name, color, d in _sides(curves)
        ],
    )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title="cumulative MW")
    fig.update_yaxes(title="EUR/MWh")
    return _layout(fig, size, title)


def grid(
    figs: list[go.Figure],
    rows: int,
    cols: int,
    *,
    size: PlotSize = "large",
    title: str | None = None,
) -> go.Figure:
    # single panel figures into one grid, row major, keeps axis titles
    out = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f.layout.title.text or "" for f in figs],
    )
    for i, fig in enumerate(figs):
        if {t.xaxis for t in fig.data} - {None, "x"}:
            raise ValueError("can't nest a multi-panel figure (e.g. calendar_plot)")
        row, col = divmod(i, cols)
        sub = out.get_subplot(row + 1, col + 1)
        for axis, update in (("xaxis", out.update_xaxes), ("yaxis", out.update_yaxes)):
            props = fig.layout[axis].to_plotly_json()
            for key in ("domain", "anchor", "rangeslider"):
                props.pop(key, None)
            update(props, row=row + 1, col=col + 1)
        for trace in fig.data:
            if isinstance(trace, go.Heatmap):
                # colorbar beside its own panel, not stacked at the far right
                y0, y1 = sub.yaxis.domain
                trace.colorbar.update(x=sub.xaxis.domain[1] + 0.005, y=(y0 + y1) / 2, len=y1 - y0)
            out.add_trace(trace, row=row + 1, col=col + 1)
    return _layout(out, size, title)


def metrics_table(
    df: pl.DataFrame,
    *,
    lower_better: tuple[str, ...] = (),
    precision: int = 2,
    footer: str | None = None,
    title: str | None = None,
) -> go.Figure:
    # numeric cells shaded red (worst) to green (best)
    numeric = df.select(cs.numeric()).columns

    def fill(col: str) -> str | list[str]:
        if col not in numeric:
            return "#f7f7f7"
        v = df[col]
        t = (v - v.min()) / ((v.max() - v.min()) or 1)
        if col in lower_better:
            t = 1 - t
        # pale end of the scale so the text stays readable
        return sample_colorscale("RdYlGn", (0.25 + 0.5 * t).to_list())

    # each column as wide as its longest text
    text = [[f"{v:,.{precision}f}" if c in numeric else str(v) for v in df[c]] for c in df.columns]
    widths = [max(len(c), *map(len, col)) for c, col in zip(df.columns, text)]
    # text columns bold
    values = [col if c in numeric else [f"<b>{v}</b>" for v in col] for c, col in zip(df.columns, text)]
    fig = go.Figure(
        go.Table(
            columnwidth=widths,
            header={
                "values": [f"<b>{c}</b>" for c in df.columns],
                "fill_color": "#2f3e4e",
                "font": {"color": "white", "size": 13},
                "align": ["right" if c in numeric else "left" for c in df.columns],
                "line_color": "white",
                "height": 32,
            },
            cells={
                "values": values,
                "fill_color": [fill(c) for c in df.columns],
                "font": {"color": "#222", "size": 13},
                "align": ["right" if c in numeric else "left" for c in df.columns],
                "line_color": "white",
                "height": 30,
            },
        ),
    )
    if footer:
        fig.add_annotation(
            text=footer,
            x=0,
            y=0,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font={"size": 11, "color": "#666"},
        )
    # height fits the rows, plus one 18px line per footer line
    footer_px = 18 * ((footer or "").count("<br>") + 1) + 16
    fig = _layout(fig, "medium", title)
    return fig.update_layout(
        height=55 + 32 + 30 * len(df) + footer_px,
        margin={"l": 20, "r": 20, "t": 50, "b": footer_px},
        title={"font": {"size": 16}},
    )


def _sides(curves: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    return [
        (name, color, curves.filter(offer_type=side).sort("price_eur_mwh", descending=desc))
        for side, name, color, desc in SIDES
    ]


def _layout(
    fig: go.Figure,
    size: PlotSize,
    title: str | None,
) -> go.Figure:
    width, height = CONFIG.plot.sizes[size]
    return fig.update_layout(
        width=width,
        height=height,
        title=title,
        template="plotly_white",
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
    )
