"""
Assignment 1 - Gradio dashboard.

Loads the models and data saved by train.py (the "artifacts" folder) and shows:
  1. Discount strategy: threshold slider, confusion matrix and business cost.
  2. Customer insights: target and feature distributions, and what drives a purchase.
  3. Model comparison: the three trained models, prediction vs. actual.

Nothing is trained here: the app only loads saved models and applies them.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

# ---------------------------------------------------------------------------
# 1. Load everything train.py saved
# ---------------------------------------------------------------------------
with open(ART / "settings.json") as f:
    settings = json.load(f)

X_test = np.load(ART / "X_test.npy")      # prepared inputs of the test sessions
y_test = np.load(ART / "y_test.npy")      # 1 = bought, 0 = did not buy
results_df = pd.read_csv(ART / "results.csv")
raw_df = pd.read_csv(ROOT / "data" / "online_shoppers_intention.csv")

X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Model 1: scikit-learn
sk_model = joblib.load(ART / "sk_model.joblib")
sk_prob = sk_model.predict_proba(X_test)[:, 1]

# Model 2: manual PyTorch loop (just the saved weights w and bias b)
manual = torch.load(ART / "manual_model.pt")
with torch.no_grad():
    manual_prob = torch.sigmoid(X_test_t @ manual["w"] + manual["b"]).numpy()


# Model 3: nn.Module (same class as in train.py, filled with the saved weights)
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


nn_model = LogisticRegressionModel(X_test.shape[1])
nn_model.load_state_dict(torch.load(ART / "nn_model.pt"))
nn_model.eval()
with torch.no_grad():
    nn_prob = torch.sigmoid(nn_model(X_test_t)).numpy()

# Predicted probability of buying, per model, for every test session
PROBS = {
    "scikit-learn": sk_prob,
    "Manual PyTorch loop": manual_prob,
    "nn.Module + torch.optim": nn_prob,
}
MODEL_NAMES = list(PROBS)

# ---------------------------------------------------------------------------
# 2. Look and feel shared by every chart
# ---------------------------------------------------------------------------
BLUE, AMBER, TEAL = "#2b5fa6", "#c9822b", "#2a9d8f"   # the three series colors
CONTEXT = "#c3cad5"                                    # quiet gray for background groups
INK, MUTED, GRID, SURFACE = "#1b2430", "#5b6675", "#eceff3", "#ffffff"
FONT = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MODEL_COLORS = dict(zip(MODEL_NAMES, [BLUE, AMBER, TEAL]))


def style(fig, title, subtitle="", x_title="", y_title="", height=400):
    """Apply the same clean layout to every chart."""
    heading = f"<b>{title}</b>"
    if subtitle:
        heading += f"<br><span style='font-size:13px;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=heading, font=dict(size=16, color=INK, family=FONT),
                   x=0, xanchor="left", xref="paper", y=0.95, yanchor="top"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=MUTED, size=13, family=FONT),
        height=height,
        margin=dict(l=70, r=30, t=90, b=60),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right",
                    font=dict(color=INK, size=13)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, family=FONT)),
        bargap=0.35,
    )
    fig.update_xaxes(title_text=x_title, title_font=dict(size=13), showgrid=False,
                     zeroline=False, linecolor="#d5dae1", ticks="outside", tickcolor="#d5dae1")
    fig.update_yaxes(title_text=y_title, title_font=dict(size=13), gridcolor=GRID,
                     zeroline=False, linecolor=SURFACE)
    return fig


# Readable names for the original columns
FEATURE_LABELS = {
    "ProductRelated_Duration": "Time on product pages (seconds)",
    "ProductRelated": "Product pages viewed",
    "Administrative_Duration": "Time on account pages (seconds)",
    "Administrative": "Account pages viewed",
    "Informational_Duration": "Time on information pages (seconds)",
    "Informational": "Information pages viewed",
    "ExitRates": "Exit rate of pages viewed",
    "BounceRates": "Bounce rate of pages viewed",
    "PageValues": "Page value (not used by the model: leaks the outcome)",
    "SpecialDay": "Closeness to a special day",
    "Month": "Month",
    "VisitorType": "Visitor type",
    "Weekend": "Weekend visit",
    "TrafficType": "Traffic source (code)",
    "OperatingSystems": "Operating system (code)",
    "Browser": "Browser (code)",
    "Region": "Region (code)",
}
CATEGORY_COLUMNS = ["Month", "VisitorType", "Weekend", "TrafficType",
                    "OperatingSystems", "Browser", "Region", "SpecialDay"]
MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
OVERALL_RATE = raw_df["Revenue"].mean()


# ---------------------------------------------------------------------------
# 3. Tab "Discount strategy" - threshold, confusion matrix and business cost
# ---------------------------------------------------------------------------
def count_outcomes(prob, threshold):
    """Compare predictions with what really happened.
    A session at or above the threshold is predicted to buy, so it gets no discount."""
    pred_buy = prob >= threshold
    return {
        "missed_sales": int(np.sum(pred_buy & (y_test == 0))),         # predicted buy, left
        "wasted_discounts": int(np.sum(~pred_buy & (y_test == 1))),    # predicted no buy, bought
        "correct_no_discount": int(np.sum(pred_buy & (y_test == 1))),  # predicted buy, bought
        "correct_discount": int(np.sum(~pred_buy & (y_test == 0))),    # predicted no buy, left
    }


def cost_of(outcomes, price, discount_pct, conversion_pct):
    cost_missed_sale = price * conversion_pct / 100     # sale lost, if the discount would have worked
    cost_wasted_discount = price * discount_pct / 100   # discount given to someone who buys anyway
    return (outcomes["missed_sales"] * cost_missed_sale
            + outcomes["wasted_discounts"] * cost_wasted_discount)


def confusion_plot(outcomes, cost_missed_sale, cost_wasted_discount):
    """Four tiles. Correct decisions are quiet gray-blue, the two costly mistakes are amber."""
    counts = [[outcomes["correct_discount"], outcomes["missed_sales"]],
              [outcomes["wasted_discounts"], outcomes["correct_no_discount"]]]
    names = [["Discount well targeted", "Missed sale"],
             ["Wasted discount", "Full-price sale"]]
    costs = [["no cost", f"{counts[0][1]:,} × ${cost_missed_sale:,.2f} = ${counts[0][1] * cost_missed_sale:,.0f}"],
             [f"{counts[1][0]:,} × ${cost_wasted_discount:,.2f} = ${counts[1][0] * cost_wasted_discount:,.0f}", "no cost"]]
    is_mistake = [[0, 1], [1, 0]]

    fig = go.Figure(go.Heatmap(
        z=is_mistake,
        x=["Predicted: won't buy<br>→ discount shown", "Predicted: will buy<br>→ no discount"],
        y=["Did not buy", "Bought"],
        colorscale=[[0, "#eef2f7"], [1, "#f6e3c8"]], zmin=0, zmax=1,
        showscale=False, xgap=10, ygap=10, hoverinfo="skip",
    ))
    for r in range(2):
        for c in range(2):
            fig.add_annotation(x=c, y=r, yshift=26, showarrow=False,
                               text=f"<b>{counts[r][c]:,}</b>",
                               font=dict(size=30, color=INK, family=FONT))
            fig.add_annotation(x=c, y=r, yshift=-8, showarrow=False,
                               text=f"<b>{names[r][c]}</b>",
                               font=dict(size=14, color=INK, family=FONT))
            fig.add_annotation(x=c, y=r, yshift=-30, showarrow=False, text=costs[r][c],
                               font=dict(size=12, color=MUTED, family=FONT))
    style(fig, "What happens to the test sessions at this threshold",
          "Amber tiles are the two mistakes that cost money", height=470)
    fig.update_xaxes(side="bottom", showline=False, ticks="", tickfont=dict(size=13, color=INK))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=13, color=INK),
                     title_text="What really happened")
    fig.update_layout(margin=dict(l=90, r=20, t=90, b=70))
    return fig


def cost_curve_plot(prob, threshold, price, discount_pct, conversion_pct):
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = [cost_of(count_outcomes(prob, t), price, discount_pct, conversion_pct) for t in thresholds]
    current = cost_of(count_outcomes(prob, threshold), price, discount_pct, conversion_pct)
    cost_everyone = int(y_test.sum()) * price * discount_pct / 100

    fig = go.Figure()
    fig.add_hline(y=cost_everyone, line=dict(color="#9aa4b2", width=1))
    fig.add_annotation(x=0.99, y=cost_everyone, xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"Discount for everyone, no model: ${cost_everyone:,.0f}",
                       font=dict(size=12, color=MUTED, family=FONT))
    fig.add_trace(go.Scatter(
        x=thresholds, y=costs, mode="lines", line=dict(color=BLUE, width=2.5), showlegend=False,
        hovertemplate="Threshold %{x:.2f}<br>Total cost $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[threshold], y=[current], mode="markers", showlegend=False,
        marker=dict(color=AMBER, size=14, line=dict(color=SURFACE, width=2)),
        hovertemplate="Your threshold %{x:.2f}<br>Total cost $%{y:,.0f}<extra></extra>",
    ))
    fig.add_annotation(
        x=threshold, y=current, ax=40, ay=-70, showarrow=True, arrowcolor="#9aa4b2", arrowwidth=1,
        arrowhead=0, standoff=9, xanchor="left", bgcolor=SURFACE,
        text=f"<b>Your threshold {threshold:.2f}</b><br>Total cost ${current:,.0f}", align="left",
        font=dict(color=INK, size=13, family=FONT))
    style(fig, "Total cost at every possible threshold",
          "Low thresholds give almost nobody a discount; high thresholds give it to almost everybody",
          "Decision threshold", "Total cost on the test group", height=400)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(range=[0, 1], dtick=0.1)
    return fig


def kpi_card(label, value, note="", highlight=False):
    extra = " kpi-highlight" if highlight else ""
    return (f"<div class='kpi{extra}'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>")


def update_strategy_tab(model_name, threshold, price, discount_pct, conversion_pct):
    prob = PROBS[model_name]
    outcomes = count_outcomes(prob, threshold)
    cost_missed_sale = price * conversion_pct / 100
    cost_wasted_discount = price * discount_pct / 100
    cost = cost_of(outcomes, price, discount_pct, conversion_pct)

    # The two strategies that need no model
    n_buyers, n_leavers = int(y_test.sum()), int((y_test == 0).sum())
    cost_everyone = n_buyers * cost_wasted_discount     # every buyer gets a wasted discount
    cost_no_one = n_leavers * cost_missed_sale          # every leaver is a missed sale
    best_naive = min(cost_everyone, cost_no_one)
    best_naive_name = "discount for everyone" if cost_everyone <= cost_no_one else "discount for no one"
    saving = best_naive - cost
    share_discounted = (outcomes["correct_discount"] + outcomes["wasted_discounts"]) / len(y_test)

    if saving >= 0:
        saving_card = kpi_card("Saving vs. no model", f"${saving:,.0f}",
                               f"{saving / best_naive:.1%} cheaper than {best_naive_name}", highlight=True)
    else:
        saving_card = kpi_card("Extra cost vs. no model", f"${-saving:,.0f}",
                               f"{best_naive_name} would be cheaper", highlight=True)
    cards = (
        "<div class='kpi-row'>"
        + kpi_card("Total business cost", f"${cost:,.0f}", f"on {len(y_test):,} test sessions")
        + saving_card
        + kpi_card("Visitors shown a discount", f"{share_discounted:.0%}",
                   f"{outcomes['correct_discount'] + outcomes['wasted_discounts']:,} sessions")
        + "</div>"
        + "<div class='kpi-row'>"
        + kpi_card("Missed sales", f"{outcomes['missed_sales']:,}",
                   f"${cost_missed_sale:,.2f} each: no discount, visitor left")
        + kpi_card("Wasted discounts", f"{outcomes['wasted_discounts']:,}",
                   f"${cost_wasted_discount:,.2f} each: would have bought anyway")
        + "</div>"
        + f"<p class='footnote'>Without a model: discount for everyone costs ${cost_everyone:,.0f}; "
          f"discount for no one costs ${cost_no_one:,.0f}.</p>"
    )
    return (cards, confusion_plot(outcomes, cost_missed_sale, cost_wasted_discount),
            cost_curve_plot(prob, threshold, price, discount_pct, conversion_pct))


# ---------------------------------------------------------------------------
# 4. Tab "Customer insights" - target, features and what drives a purchase
# ---------------------------------------------------------------------------
def target_plot():
    counts = raw_df["Revenue"].value_counts()
    values = [int(counts[False]), int(counts[True])]
    shares = [v / sum(values) for v in values]
    fig = go.Figure(go.Bar(
        y=["Did not buy", "Bought"], x=values, orientation="h",
        marker=dict(color=[CONTEXT, BLUE], cornerradius=4), width=0.5,
        text=[f"  {v:,} ({s:.1%})" for v, s in zip(values, shares)],
        textposition="outside", textfont=dict(color=INK, size=13, family=FONT), cliponaxis=False,
        hovertemplate="%{y}: %{x:,} sessions<extra></extra>",
    ))
    style(fig, "Only about 1 in 6 sessions ends in a purchase",
          "Target distribution across all sessions",
          height=260)
    fig.update_xaxes(visible=False, range=[0, max(values) * 1.6])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=13, color=INK), autorange="reversed")
    fig.update_layout(margin=dict(l=100, r=20, t=90, b=20))
    return fig


def month_plot():
    table = raw_df.groupby("Month")["Revenue"].agg(rate="mean", sessions="size").loc[MONTH_ORDER]
    best = table["rate"].idxmax()
    colors = [BLUE if m == best else CONTEXT for m in table.index]
    fig = go.Figure(go.Bar(
        x=table.index, y=table["rate"], marker=dict(color=colors, cornerradius=4),
        customdata=table["sessions"],
        text=[f"{r:.0%}" for r in table["rate"]], textposition="outside",
        textfont=dict(color=INK, size=12, family=FONT), cliponaxis=False,
        hovertemplate="%{x}<br>Purchase rate: %{y:.1%}<br>Sessions: %{customdata:,}<extra></extra>",
    ))
    fig.add_hline(y=OVERALL_RATE, line=dict(color="#9aa4b2", width=1))
    fig.add_annotation(x=0, xref="paper", y=OVERALL_RATE, xanchor="left", yanchor="bottom",
                       showarrow=False, text=f"Average {OVERALL_RATE:.1%}",
                       font=dict(size=12, color=MUTED, family=FONT))
    style(fig, f"Purchase rate peaks in {best}",
          "Share of sessions ending in a purchase, by month",
          height=380)
    fig.update_yaxes(tickformat=".0%", range=[0, table["rate"].max() * 1.2])
    return fig


def drivers_plot():
    """The model's largest weights. Inputs are standardized, so the weights can be compared."""
    names = settings["feature_names"]
    weights = pd.Series(sk_model.coef_[0], index=names)
    top = weights.reindex(weights.abs().sort_values(ascending=False).index)[:10][::-1]

    def readable(name):
        if name.startswith("Month_"):
            return f"Month: {name[6:]}"
        if name.startswith("TrafficType_"):
            return f"Traffic source {name[12:]}"
        if name.startswith("VisitorType_"):
            return name[12:].replace("_", " ")
        if name == "NoBounce":
            return "Bounce rate exactly zero"
        return FEATURE_LABELS.get(name, name).replace(" (seconds)", "")

    fig = go.Figure(go.Bar(
        y=[readable(n) for n in top.index], x=top.values, orientation="h",
        marker=dict(color=[TEAL if v > 0 else AMBER for v in top.values], cornerradius=4), width=0.6,
        hovertemplate="%{y}<br>Model weight: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#9aa4b2", width=1))
    style(fig, "What the model relies on most",
          "Ten largest weights. Teal raises the predicted chance of buying, amber lowers it",
          "Model weight (inputs are standardized, so bars are comparable)", height=440)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=13, color=INK))
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_layout(margin=dict(l=230, r=30, t=90, b=60))
    return fig


def fmt(value):
    """Short, readable number for chart labels."""
    if float(value).is_integer() or abs(value) >= 10:
        return f"{value:,.0f}"
    return f"{value:.3f}"


def feature_plots(feature):
    """Two views of one feature: how it is distributed, and how the purchase rate changes with it."""
    label = FEATURE_LABELS[feature]
    bought = raw_df["Revenue"]

    if feature in CATEGORY_COLUMNS:
        table = raw_df.groupby(feature)["Revenue"].agg(rate="mean", sessions="size")
        if feature == "Month":
            table = table.loc[MONTH_ORDER]
        groups = table.index.astype(str)
        dist = go.Figure(go.Bar(
            x=groups, y=table["sessions"], marker=dict(color=CONTEXT, cornerradius=4),
            hovertemplate="%{x}: %{y:,} sessions<extra></extra>"))
        style(dist, "How sessions are distributed", label, "", "Sessions")
        dist.update_xaxes(type="category")
    else:
        # Numbers: the top 1% is grouped into the last bar so extreme sessions do not flatten the chart
        values = raw_df[feature]
        upper = values.quantile(0.99)
        edges = np.linspace(0, upper if upper > 0 else 1, 16)
        centers = (edges[:-1] + edges[1:]) / 2
        dist = go.Figure()
        for name, mask, color in [("Did not buy", ~bought, CONTEXT), ("Bought", bought, BLUE)]:
            counts, _ = np.histogram(values[mask].clip(upper=upper), bins=edges)
            dist.add_trace(go.Bar(
                x=centers, y=counts / counts.sum(), name=name, marker=dict(color=color, cornerradius=3),
                hovertemplate="Around %{x:.3g}<br>%{y:.1%} of this group<extra>" + name + "</extra>"))
        style(dist, "Buyers vs. non-buyers", label, "", "Share of each group")
        dist.update_layout(barmode="group", bargap=0.2)
        dist.update_yaxes(tickformat=".0%")

        # Five equal-sized bands, from the lowest values to the highest
        bands = pd.qcut(values.rank(method="first"), 5, labels=False)
        table = raw_df.groupby(bands)["Revenue"].agg(rate="mean", sessions="size")
        limits = values.groupby(bands).agg(["min", "max"])
        groups = [f"{fmt(lo)} to {fmt(hi)}" for lo, hi in zip(limits["min"], limits["max"])]

    best = int(np.argmax(table["rate"].values))
    rate = go.Figure(go.Bar(
        x=[g.replace(" to ", "<br>to ") for g in groups], y=table["rate"],
        marker=dict(color=[BLUE if i == best else CONTEXT for i in range(len(table))], cornerradius=4),
        customdata=table["sessions"],
        hovertemplate="%{x}<br>Purchase rate: %{y:.1%}<br>Sessions: %{customdata:,}<extra></extra>"))
    rate.add_hline(y=OVERALL_RATE, line=dict(color="#9aa4b2", width=1))
    rate.add_annotation(x=0, xref="paper", y=OVERALL_RATE, xanchor="left", yanchor="bottom",
                        showarrow=False, text=f"Average {OVERALL_RATE:.1%}",
                        font=dict(size=12, color=MUTED, family=FONT))
    style(rate, "Purchase rate by group", label, "", "Share that bought")
    rate.update_xaxes(type="category", tickangle=0)
    rate.update_yaxes(tickformat=".0%")

    low, high = table["rate"].min(), table["rate"].max()
    insight = (f"**Insight:** the purchase rate ranges from **{low:.1%}** to **{high:.1%}** across the groups "
               f"of *{label.lower()}*, against an average of {OVERALL_RATE:.1%}. "
               f"The highest group is **{groups[best]}**.")
    return dist, rate, insight


# ---------------------------------------------------------------------------
# 5. Tab "Model comparison" - the three implementations
# ---------------------------------------------------------------------------
def results_table_html():
    header = ["Model", "Log loss (lower is better)", "AUC (higher is better)",
              f"Total cost at threshold {settings['best_threshold']:.2f}"]
    cells = "".join(f"<div class='rt-head'>{h}</div>" for h in header)
    for _, row in results_df.iterrows():
        quiet = " rt-quiet" if row["Model"] == "Naive baseline" else ""
        values = [row["Model"], f"{row['Log loss']:.4f}", f"{row['AUC']:.4f}",
                  f"${row['Total cost ($)']:,.0f}"]
        cells += "".join(f"<div class='rt-cell{quiet}'>{v}</div>" for v in values)
    return f"<div class='results-grid'>{cells}</div>"


def prediction_vs_actual_plot():
    """Sort the test sessions by predicted probability, cut them into 10 equal groups, and
    compare each group's average prediction with the share that really bought."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 0.5], y=[0, 0.5], mode="lines", name="Perfect prediction",
        line=dict(color="#9aa4b2", width=1), hoverinfo="skip"))
    # Decreasing marker sizes, because the three lines sit on top of each other
    for name, size in zip(MODEL_NAMES, [17, 11, 5]):
        prob = PROBS[name]
        groups = pd.qcut(prob, 10, labels=False, duplicates="drop")
        table = pd.DataFrame({"prob": prob, "actual": y_test, "group": groups})
        by_group = table.groupby("group").agg(pred=("prob", "mean"), actual=("actual", "mean"))
        fig.add_trace(go.Scatter(
            x=by_group["pred"], y=by_group["actual"], mode="lines+markers", name=name,
            line=dict(color=MODEL_COLORS[name], width=2),
            marker=dict(size=size, color=MODEL_COLORS[name], line=dict(color=SURFACE, width=2)),
            hovertemplate="Predicted: %{x:.1%}<br>Actually bought: %{y:.1%}<extra>" + name + "</extra>"))
    style(fig, "Predicted probability vs. actual purchase rate",
          "Test sessions in 10 equal groups. The three models overlap, and all follow the line of perfect prediction",
          "Average predicted probability of buying", "Share that actually bought", height=460)
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(tickformat=".0%")
    return fig


def agreement_text():
    diffs = [np.abs(PROBS[a] - PROBS[b]).max()
             for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i + 1:]]
    return (f"**Do the three implementations agree?** Yes. For any single test session, the largest difference "
            f"in predicted probability between two of the models is **{max(diffs):.4f}** on a scale from 0 to 1.")


# ---------------------------------------------------------------------------
# 6. Page layout
# ---------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
.page-title h1 {font-size: 28px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 4px;}
.page-title p {color: var(--body-text-color-subdued); font-size: 15px;}
.kpi-row {display: flex; gap: 14px; margin-bottom: 14px; flex-wrap: wrap;}
.kpi {flex: 1; min-width: 150px; padding: 16px 18px; border-radius: 12px;
      border: 1px solid var(--border-color-primary); background: var(--background-fill-primary);}
.kpi-highlight {border-color: #2b5fa6; box-shadow: inset 4px 0 0 #2b5fa6;}
.kpi-label {font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--body-text-color-subdued);}
.kpi-value {font-size: 30px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.25;
            color: var(--body-text-color);}
.kpi-note {font-size: 13px; color: var(--body-text-color-subdued);}
.footnote {font-size: 13px; color: var(--body-text-color-subdued); margin: 4px 2px;}
.results-grid {display: grid; grid-template-columns: 1.4fr 1fr 1fr 1fr; font-size: 14px;
               border: 1px solid var(--border-color-primary); border-radius: 12px; overflow: hidden;
               background: var(--background-fill-primary); margin-bottom: 14px;}
.rt-head {font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
          color: var(--body-text-color-subdued); padding: 12px 16px;
          border-bottom: 1px solid var(--border-color-primary);}
.rt-cell {padding: 13px 16px; border-top: 1px solid var(--border-color-primary);
          font-variant-numeric: tabular-nums; color: var(--body-text-color);}
.rt-quiet {color: var(--body-text-color-subdued);}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
)

# Headline numbers for the top of the page
model_row = results_df[results_df["Model"] == "scikit-learn"].iloc[0]
baseline_row = results_df[results_df["Model"] == "Naive baseline"].iloc[0]
headline_saving = baseline_row["Total cost ($)"] - model_row["Total cost ($)"]
HEADLINE = (
    "<div class='kpi-row'>"
    + kpi_card("Sessions in the data", f"{len(raw_df):,}", "one row per visitor session")
    + kpi_card("Sessions that end in a purchase", f"{OVERALL_RATE:.1%}", "the outcome the model predicts")
    + kpi_card("Model quality (AUC)", f"{model_row['AUC']:.2f}", "0.50 is guessing, 1.00 is perfect")
    + kpi_card("Saving on the test group", f"${headline_saving:,.0f}",
               f"{headline_saving / baseline_row['Total cost ($)']:.1%} less than discounting everyone",
               highlight=True)
    + "</div>"
)

with gr.Blocks(title="Who should get a discount?") as demo:
    gr.Markdown(
        "# Who should get a discount?\n"
        "An online shop shows a discount to visitors who are predicted to leave without buying. "
        "A logistic regression makes that prediction from each visitor's browsing behavior.",
        elem_classes="page-title",
    )
    gr.HTML(HEADLINE)

    with gr.Tab("Discount strategy"):
        gr.Markdown(
            "A visitor whose predicted chance of buying is **below the threshold** is shown the discount. "
            "Move the threshold to trade one mistake against the other and see what it costs."
        )
        with gr.Row():
            threshold_slider = gr.Slider(
                0.01, 0.99, value=settings["best_threshold"], step=0.01, scale=3,
                label=f"Decision threshold ({settings['best_threshold']:.2f} was chosen on the validation group)")
            model_choice = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model", scale=2)
        with gr.Accordion("Business assumptions", open=False):
            with gr.Row():
                price_input = gr.Number(value=settings["price"], label="Product price ($)", minimum=1)
                discount_input = gr.Slider(1, 50, value=settings["discount"] * 100, step=1,
                                           label="Discount size (%)")
                conversion_input = gr.Slider(1, 50, value=settings["conversion"] * 100, step=1,
                                             label="Leaving visitors the discount convinces (%)")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                confusion_chart = gr.Plot(show_label=False)
            with gr.Column(scale=4):
                cost_cards = gr.HTML()
        cost_chart = gr.Plot(show_label=False)

        controls = [model_choice, threshold_slider, price_input, discount_input, conversion_input]
        outputs = [cost_cards, confusion_chart, cost_chart]
        for control in controls:
            control.change(update_strategy_tab, inputs=controls, outputs=outputs)
        demo.load(update_strategy_tab, inputs=controls, outputs=outputs)

    with gr.Tab("Customer insights"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                gr.Plot(value=target_plot(), show_label=False)
            with gr.Column(scale=5):
                gr.Plot(value=month_plot(), show_label=False)
        gr.Plot(value=drivers_plot(), show_label=False)
        gr.Markdown(
            "<p class='footnote'>How to read it: months are compared with August, traffic sources with the "
            "group of rare sources, and returning visitors with new ones. Exit rate and bounce rate move "
            "together, so their two weights should be read as a pair.</p>"
        )
        gr.Markdown("### Explore one feature")
        feature_choice = gr.Dropdown(
            [(label, column) for column, label in FEATURE_LABELS.items()],
            value="ProductRelated_Duration", label="Feature (original values, before any transformation)")
        with gr.Row():
            feature_dist_chart = gr.Plot(show_label=False)
            feature_rate_chart = gr.Plot(show_label=False)
        feature_insight = gr.Markdown()
        feature_outputs = [feature_dist_chart, feature_rate_chart, feature_insight]
        feature_choice.change(feature_plots, inputs=feature_choice, outputs=feature_outputs)
        demo.load(feature_plots, inputs=feature_choice, outputs=feature_outputs)

    with gr.Tab("Model comparison"):
        gr.Markdown(
            "The same logistic regression was trained three ways. Results are for the test group, "
            "which was never used for training or for choosing the threshold."
        )
        gr.HTML(results_table_html())
        gr.Plot(value=prediction_vs_actual_plot(), show_label=False)
        gr.Markdown(agreement_text())

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
