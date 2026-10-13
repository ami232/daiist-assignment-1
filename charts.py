"""
Plotly figures for the dashboard. Pure functions: data in, figure out.

Nothing here loads files or touches a model, so app.py stays about layout
and interaction, and every chart can be read on its own.

Colour rules used throughout:
- The three implementations always wear the same three colours.
- A failed inspection is red and a passed one is grey, wherever they appear.
- Text is always ink-coloured; only marks carry colour.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Palette -----------------------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

MODEL_COLORS = {"sklearn": "#2a78d6", "manual": "#eb6834", "module": "#1baf7a"}
MODEL_NAMES = {
    "sklearn": "scikit-learn",
    "manual": "PyTorch manual loop",
    "module": "PyTorch nn.Module + optim",
}
BLUE = "#2a78d6"
FAILED = "#d03b3b"
PASSED = "#b4b2a9"
RAISES_RISK = "#e34948"
LOWERS_RISK = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SPLIT_FILL = {"train": "#f0efec", "validation": "#e6e5df", "test": "#dbd9d1"}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def styled(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    """Shared chrome: quiet axes, hairline grid, ink-coloured text."""
    fig.update_layout(
        height=height,
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=64, r=32, t=44 if legend else 20, b=52),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_SECONDARY)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK, size=13)),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS,
        tickfont=dict(color=MUTED), title_font=dict(color=INK_SECONDARY, size=12.5),
    )
    fig.update_yaxes(
        gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
        tickfont=dict(color=MUTED), title_font=dict(color=INK_SECONDARY, size=12.5),
    )
    return fig


def _split_bands(fig: go.Figure, splits: dict, label_y: float = 1.0) -> None:
    """Shade the train / validation / test periods behind a time series."""
    for name, info in splits.items():
        start = pd.Timestamp(info["start"]).to_period("M").start_time
        end = pd.Timestamp(info["end"]).to_period("M").end_time
        fig.add_vrect(x0=start, x1=end, fillcolor=SPLIT_FILL[name], opacity=0.7, line_width=0, layer="below")
        fig.add_annotation(
            x=start + (end - start) / 2, y=label_y, yref="paper", yanchor="bottom", showarrow=False,
            text=name.capitalize(), font=dict(color=INK_SECONDARY, size=12),
        )


# --- Overview ------------------------------------------------------------------------
def timeline(table: pd.DataFrame, splits: dict) -> go.Figure:
    """Monthly actual fail rate against the average predicted probability."""
    monthly = (
        table.groupby(table["inspection_date"].dt.to_period("M"))
        .agg(actual=("failed", "mean"), predicted=("p_sklearn", "mean"), inspections=("failed", "size"))
        .reset_index()
    )
    months = monthly["inspection_date"].dt.to_timestamp() + pd.Timedelta(days=14)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=monthly["actual"], name="Actual fail rate", mode="lines+markers",
        line=dict(color=INK_SECONDARY, width=2), marker=dict(size=6, color=INK_SECONDARY),
        customdata=monthly["inspections"],
        hovertemplate="<b>%{y:.1%}</b> actual  (%{customdata:,} inspections)<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=monthly["predicted"], name="Average predicted probability", mode="lines",
        line=dict(color=BLUE, width=2),
        hovertemplate="<b>%{y:.1%}</b> predicted<extra></extra>",
    ))
    styled(fig, height=360)
    _split_bands(fig, splits, label_y=0.0)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", range=[0, 0.45], title_text="Share of inspections failed")
    fig.update_xaxes(hoverformat="%B %Y")
    return fig


def stability(quarters: list[dict], splits: dict) -> go.Figure:
    """AUC quarter by quarter: does the model decay, or just vary?"""
    frame = pd.DataFrame(quarters)
    midpoints = pd.PeriodIndex(frame["quarter"], freq="Q").to_timestamp() + pd.Timedelta(days=45)
    fig = go.Figure(go.Scatter(
        x=midpoints, y=frame["roc_auc"], mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=9, color=BLUE, line=dict(color=SURFACE, width=2)),
        customdata=np.stack([frame["quarter"], frame["rows"]], axis=1),
        hovertemplate="%{customdata[0]}<br><b>AUC %{y:.3f}</b>  (%{customdata[1]:,} inspections)<extra></extra>",
    ))
    styled(fig, height=320, legend=False)
    _split_bands(fig, splits, label_y=0.0)
    fig.add_hline(y=0.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=0.5, text="0.50 = coin flip", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=MUTED, size=12))
    fig.update_yaxes(range=[0.45, 0.78], title_text="ROC AUC in the quarter", tickformat=".2f")
    return fig


# --- Three models ----------------------------------------------------------------------
def calibration(rows: pd.DataFrame, kinds: list[str], n_bins: int = 10) -> go.Figure:
    """Prediction vs actual: average predicted probability against the observed fail rate.

    Inspections are sorted by predicted probability and cut into ten equal
    groups. A model whose probabilities can be trusted sits on the diagonal.
    The three markers have different sizes, so when the implementations
    agree they stack into a bullseye.
    """
    sizes = {"sklearn": 20, "manual": 14, "module": 8}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip",
    ))
    upper = 0.0
    for kind in kinds:
        p = rows[f"p_{kind}"]
        groups = rows.groupby(pd.qcut(p, n_bins, labels=False, duplicates="drop"))
        predicted, actual, count = groups[f"p_{kind}"].mean(), groups["failed"].mean(), groups.size()
        upper = max(upper, predicted.max(), actual.max())
        fig.add_trace(go.Scatter(
            x=predicted, y=actual, name=MODEL_NAMES[kind], mode="markers",
            marker=dict(size=sizes[kind], color=MODEL_COLORS[kind], line=dict(color=SURFACE, width=2)),
            customdata=count,
            hovertemplate=f"{MODEL_NAMES[kind]}<br>predicted <b>%{{x:.1%}}</b>, actual <b>%{{y:.1%}}</b>"
                          "  (%{customdata:,} inspections)<extra></extra>",
        ))
    styled(fig, height=440)
    limit = min(1.0, upper + 0.06)
    fig.update_xaxes(range=[0, limit], tickformat=".0%", title_text="Average predicted probability of failing", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(range=[0, limit], tickformat=".0%", title_text="Share that actually failed")
    return fig


def prediction_gap(rows: pd.DataFrame) -> go.Figure:
    """How far each PyTorch prediction is from scikit-learn's, inspection by inspection."""
    fig = go.Figure()
    for kind in ["manual", "module"]:
        gap = 100 * (rows[f"p_{kind}"] - rows["p_sklearn"])
        fig.add_trace(go.Scattergl(
            x=rows["p_sklearn"], y=gap, name=MODEL_NAMES[kind], mode="markers",
            marker=dict(size=5, color=MODEL_COLORS[kind], opacity=0.55),
            hovertemplate=f"{MODEL_NAMES[kind]}<br>scikit-learn says %{{x:.1%}}<br>"
                          "difference <b>%{y:.3f}</b> percentage points<extra></extra>",
        ))
    styled(fig, height=440)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(tickformat=".0%", title_text="scikit-learn's predicted probability")
    fig.update_yaxes(range=[-0.5, 0.5], title_text="Difference from scikit-learn (percentage points)", tickformat=".1f")
    return fig


def coefficients(models: dict, labels: dict) -> go.Figure:
    """scikit-learn's coefficients as bars, the two PyTorch versions as markers on top."""
    reference = pd.Series(models["sklearn"]["coefficients"]).sort_values()
    names = [labels[name] for name in reference.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=reference.values, orientation="h", name=MODEL_NAMES["sklearn"], width=0.5,
        marker=dict(color=MODEL_COLORS["sklearn"]),
        hovertemplate="scikit-learn <b>%{x:.4f}</b><extra></extra>",
    ))
    for kind, symbol, size in [("manual", "diamond", 13), ("module", "circle", 8)]:
        values = pd.Series(models[kind]["coefficients"]).reindex(reference.index)
        fig.add_trace(go.Scatter(
            y=names, x=values.values, mode="markers", name=MODEL_NAMES[kind],
            marker=dict(symbol=symbol, size=size, color=MODEL_COLORS[kind], line=dict(color=SURFACE, width=2)),
            hovertemplate=f"{MODEL_NAMES[kind]} <b>%{{x:.4f}}</b><extra></extra>",
        ))
    styled(fig, height=560)
    fig.update_layout(barcornerradius=4, hovermode="y unified", margin=dict(l=250))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title_text="Coefficient (change in log-odds per standard deviation)", showgrid=True, gridcolor=GRID, tickformat=".2f")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SECONDARY))
    return fig


def convergence(search: list[dict], sklearn_objective: float, chosen_lr: float, color: str) -> go.Figure:
    """Distance from scikit-learn's minimum after each epoch, one line per learning rate."""
    floor = 1e-12  # below this the difference is floating-point noise
    greys = ["#c3c2b7", "#a3a199", "#898781", "#6f6d68"]
    fig = go.Figure()
    ends = []
    for i, run in enumerate(search):
        history = [np.nan if value is None else max(value - sklearn_objective, floor) for value in run["history"]]
        is_chosen = run["lr"] == chosen_lr
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(history) + 1), y=history, mode="lines", name=f"lr {run['lr']:g}",
            line=dict(color=color if is_chosen else greys[i % len(greys)], width=2.5 if is_chosen else 1.5),
            hovertemplate=f"lr {run['lr']:g}: <b>%{{y:.1e}}</b><extra></extra>",
        ))
        ends.append([np.log10(history[-1]), len(history), f"lr {run['lr']:g}" + (" (chosen)" if is_chosen else ""), is_chosen])

    # Label each line at its right-hand end. Where two lines finish close
    # together, push the upper label up just enough to keep both readable.
    ends.sort(key=lambda end: end[0])
    span = max(ends[-1][0] - ends[0][0], 1.0)
    for lower, upper in zip(ends, ends[1:]):
        upper[0] = max(upper[0], lower[0] + 0.075 * span)
    for position, epochs, text, is_chosen in ends:
        fig.add_annotation(
            x=epochs, y=position, text=text, showarrow=False, xanchor="left", xshift=6,
            font=dict(color=INK if is_chosen else INK_SECONDARY, size=12),
        )
    styled(fig, height=380, legend=False)
    fig.update_layout(hovermode="x unified", margin=dict(r=110))
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(type="log", dtick=3, exponentformat="power", title_text="Training objective above scikit-learn's minimum")
    return fig


# --- Features --------------------------------------------------------------------------
def ablation_ladder(steps: list[dict]) -> go.Figure:
    """Validation AUC as feature groups are added one after another."""
    labels = [("" if i == 0 else "+ ") + step["step"] for i, step in enumerate(steps)]
    aucs = [step["roc_auc"] for step in steps]
    fig = go.Figure(go.Scatter(
        x=labels, y=aucs, mode="lines+markers+text", line=dict(color=BLUE, width=2),
        marker=dict(size=11, color=BLUE, line=dict(color=SURFACE, width=2)),
        text=[f"{auc:.3f}" for auc in aucs], textposition="top center", textfont=dict(color=INK, size=13),
        customdata=[step["n_features"] for step in steps],
        hovertemplate="%{x}<br><b>AUC %{y:.3f}</b> with %{customdata} features<extra></extra>",
    ))
    styled(fig, height=340, legend=False)
    fig.add_hline(y=0.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=0.5, text="0.50 = coin flip", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=MUTED, size=12))
    fig.update_yaxes(range=[0.47, 0.76], tickformat=".2f", title_text="Validation ROC AUC")
    fig.update_xaxes(tickfont=dict(color=INK_SECONDARY))
    return fig


def ablation_drop(rows: list[dict]) -> go.Figure:
    """AUC lost when one group is removed from the full model."""
    frame = pd.DataFrame(rows).sort_values("auc_lost")
    frame["auc_lost"] = frame["auc_lost"].clip(lower=0)  # noise can put "adds nothing" a hair below zero
    fig = go.Figure(go.Bar(
        y=frame["removed"], x=frame["auc_lost"], orientation="h", width=0.45, marker=dict(color=BLUE),
        text=[f"{value:.3f}" for value in frame["auc_lost"]], textposition="outside", textfont=dict(color=INK, size=13),
        cliponaxis=False, hovertemplate="Without %{y}: AUC falls by <b>%{x:.3f}</b><extra></extra>",
    ))
    styled(fig, height=340, legend=False)
    fig.update_layout(barcornerradius=4, margin=dict(l=150, r=60))
    fig.update_xaxes(title_text="Validation AUC lost when the group is removed", rangemode="tozero", showgrid=True, gridcolor=GRID, tickformat=".3f")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SECONDARY))
    return fig


def result_mix(table: pd.DataFrame) -> go.Figure:
    """Inspection outcomes in each period. Only Fail is the positive class."""
    order = ["Pass", "Pass w/ Conditions", "Fail"]
    colors = {"Pass": "#dbd9d1", "Pass w/ Conditions": "#b4b2a9", "Fail": FAILED}
    shares = (
        table.groupby("split")["results"].value_counts(normalize=True).unstack().reindex(["test", "validation", "train"])[order]
    )
    fig = go.Figure()
    for result in order:
        fig.add_trace(go.Bar(
            y=[name.capitalize() for name in shares.index], x=shares[result], orientation="h", name=result, width=0.5,
            marker=dict(color=colors[result], line=dict(color=SURFACE, width=2)),
            text=[f"{value:.0%}" for value in shares[result]], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white" if result == "Fail" else INK, size=13),
            hovertemplate=f"{result}: <b>%{{x:.1%}}</b><extra></extra>",
        ))
    styled(fig, height=250)
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(tickformat=".0%", range=[0, 1], title_text="Share of routine inspections")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SECONDARY))
    return fig


def _is_discrete(values: pd.Series) -> bool:
    return values.dtype == object or pd.api.types.is_string_dtype(values) or values.nunique() <= 13


def feature_distribution(rows: pd.DataFrame, column: str, label: str, percent: bool) -> go.Figure:
    """Where passed and failed inspections sit along one feature."""
    values = rows[column]
    fig = go.Figure()
    outcomes = [("Passed", 0, PASSED), ("Failed", 1, FAILED)]
    if _is_discrete(values):
        categories = sorted(values.dropna().unique())
        for name, flag, color in outcomes:
            share = values[rows["failed"] == flag].value_counts(normalize=True).reindex(categories).fillna(0)
            fig.add_trace(go.Bar(
                x=[str(c) for c in categories], y=share.values, name=name, marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                hovertemplate=f"{name}: <b>%{{y:.1%}}</b> of that group<extra></extra>",
            ))
        fig.update_layout(barmode="group", bargap=0.45, barcornerradius=4)
        fig.update_xaxes(type="category")
    else:
        low, high = values.quantile([0.01, 0.99])
        edges = np.linspace(low, high, 31)
        centres = (edges[:-1] + edges[1:]) / 2
        for name, flag, color in outcomes:
            counts, _ = np.histogram(values[rows["failed"] == flag].dropna().clip(low, high), bins=edges)
            fig.add_trace(go.Scatter(
                x=centres, y=counts / counts.sum(), name=name, mode="lines", line=dict(color=color, width=2, shape="hvh"),
                fill="tozeroy", fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.10)",
                hovertemplate=f"{name}: <b>%{{y:.1%}}</b><extra></extra>",
            ))
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(tickformat=".0%" if percent else None)
    styled(fig, height=360)
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(tickformat=".0%", title_text="Share of inspections with that outcome", rangemode="tozero")
    return fig


def feature_fail_rate(rows: pd.DataFrame, column: str, label: str, percent: bool) -> go.Figure:
    """Fail rate across the range of one feature."""
    values = rows[column]
    overall = rows["failed"].mean()
    if _is_discrete(values):
        groups = rows.groupby(values)["failed"].agg(["mean", "size"])
        x, mode = [str(c) for c in groups.index], "markers"
    else:
        groups = rows.groupby(pd.qcut(values, 10, duplicates="drop"), observed=True)["failed"].agg(["mean", "size"])
        x, mode = rows.groupby(pd.qcut(values, 10, duplicates="drop"), observed=True)[column].median().values, "lines+markers"

    fig = go.Figure(go.Scatter(
        x=x, y=groups["mean"], mode=mode, line=dict(color=BLUE, width=2),
        marker=dict(size=11, color=BLUE, line=dict(color=SURFACE, width=2)),
        customdata=groups["size"],
        hovertemplate="<b>%{y:.1%}</b> failed  (%{customdata:,} inspections)<extra></extra>",
    ))
    styled(fig, height=360, legend=False)
    fig.add_hline(y=overall, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_annotation(x=1, xref="paper", y=overall, text=f"overall {overall:.1%}", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=MUTED, size=12))
    fig.update_xaxes(title_text=label, tickformat=".0%" if percent and not _is_discrete(values) else None,
                     type="category" if _is_discrete(values) else None)
    fig.update_yaxes(tickformat=".0%", title_text="Share that failed", rangemode="tozero")
    return fig


# --- Threshold and cost ------------------------------------------------------------------
def confusion_matrix(tp: int, fp: int, fn: int, tn: int) -> go.Figure:
    counts = np.array([[tp, fn], [fp, tn]])
    meaning = np.array([["Caught early", "Missed"], ["Wasted priority visit", "Correctly left in queue"]])
    share = counts / counts.sum()
    fig = go.Figure(go.Heatmap(
        z=share, x=["Prioritised", "Normal queue"], y=["Actually failed", "Actually passed"],
        colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)], zmin=0, zmax=0.8,
        showscale=False, xgap=2, ygap=2,
        customdata=np.dstack([counts, meaning]),
        hovertemplate="%{y}, %{x}<br><b>%{customdata[0]:,}</b> inspections (%{z:.1%})<extra></extra>",
    ))
    for row in range(2):
        for col in range(2):
            dark = share[row, col] > 0.4
            fig.add_annotation(
                x=col, y=row, showarrow=False, align="center",
                text=f"<b style='font-size:24px'>{counts[row, col]:,}</b><br>{meaning[row, col]}",
                font=dict(color="white" if dark else INK, size=13),
            )
    styled(fig, height=380, legend=False)
    fig.update_layout(margin=dict(l=120, t=44))
    fig.update_xaxes(side="top", ticks="", showline=False, tickfont=dict(color=INK_SECONDARY, size=13), title_text="Model's decision")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_SECONDARY, size=13))
    return fig


def cost_curve(thresholds, costs, current, current_cost, chosen, cost_all, cost_none) -> go.Figure:
    """Total cost of mistakes at every threshold, with the two no-model policies."""
    fig = go.Figure(go.Scatter(
        x=thresholds, y=costs, mode="lines", line=dict(color=BLUE, width=2),
        hovertemplate="threshold %{x:.2f}<br><b>$%{y:,.0f}</b><extra></extra>",
    ))
    for value, text in [(cost_all, "Prioritise everyone"), (cost_none, "Prioritise no one")]:
        fig.add_hline(y=value, line=dict(color=MUTED, width=1, dash="dash"))
        fig.add_annotation(x=0.5, y=value, text=text, showarrow=False, yanchor="bottom", font=dict(color=MUTED, size=12))
    fig.add_vline(x=chosen, line=dict(color=AXIS, width=1))
    fig.add_annotation(x=chosen, y=1, yref="paper", text=f"chosen on validation ({chosen:.2f})", showarrow=False,
                       yanchor="bottom", font=dict(color=INK_SECONDARY, size=12))
    fig.add_trace(go.Scatter(
        x=[current], y=[current_cost], mode="markers",
        marker=dict(size=14, color=INK, line=dict(color=SURFACE, width=2)),
        hovertemplate="your threshold %{x:.2f}<br><b>$%{y:,.0f}</b><extra></extra>",
    ))
    styled(fig, height=380, legend=False)
    fig.update_xaxes(range=[0, 1], title_text="Decision threshold", tickformat=".1f")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", title_text="Cost of mistakes over the period", rangemode="tozero")
    return fig


# --- Score a restaurant ----------------------------------------------------------------------
def contributions(values: pd.Series, labels: dict) -> go.Figure:
    """How much each feature moves this restaurant's log-odds away from an average one."""
    ordered = values.reindex(values.abs().sort_values().index)
    fig = go.Figure(go.Bar(
        y=[labels[name] for name in ordered.index], x=ordered.values, orientation="h", width=0.5,
        marker=dict(color=[RAISES_RISK if value > 0 else LOWERS_RISK for value in ordered.values]),
        hovertemplate="%{y}<br><b>%{x:.3f}</b> log-odds<extra></extra>",
    ))
    styled(fig, height=520, legend=False)
    fig.update_layout(barcornerradius=4, margin=dict(l=250))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title_text="Lowers risk  <  contribution to log-odds vs an average restaurant  >  raises risk",
                     showgrid=True, gridcolor=GRID, tickformat=".2f")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SECONDARY))
    return fig
