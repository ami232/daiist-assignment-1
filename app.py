"""
Assignment 1 - Gradio dashboard. Loads the artifacts written by train.py
(preprocessor + three trained models + metadata) and never retrains.

Run with: uv run python main.py app
"""

import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, log_loss, precision_recall_curve, roc_auc_score,
                             roc_curve)

from pipeline import (ARTIFACTS, TARGET, LogisticRegressionNet, booking_revenue, business_cost, clean,
                      cost_curve, load_data, manual_predict_proba)

# ---------------------------------------------------------------- palette (dataviz reference palette)

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MODEL_STYLE = {
    "sklearn": {"color": "#2a78d6", "dash": "solid", "label": "scikit-learn"},
    "manual_torch": {"color": "#eb6834", "dash": "dash", "label": "Manual PyTorch"},
    "standard_torch": {"color": "#1baf7a", "dash": "dot", "label": "nn.Module + optim"},
}
CLASS_COLOR = {0: "#2a78d6", 1: "#eb6834"}
CLASS_LABEL = {0: "Not canceled", 1: "Canceled"}
SPLIT_COLOR = {"train": "#e1e0d9", "val": "#cde2fb", "test": "#fde0d3"}


def style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)), height=height,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK_2, size=12),
        margin=dict(l=55, r=20, t=55, b=45), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False)
    return fig


# ---------------------------------------------------------------- load artifacts (no training here)

META = json.loads((ARTIFACTS / "metadata.json").read_text())
PRE = joblib.load(ARTIFACTS / "preprocessor.joblib")
SK = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
MANUAL = torch.load(ARTIFACTS / "manual_torch.pt", weights_only=True)
STD = LogisticRegressionNet(len(META["feature_names"]))
STD.load_state_dict(torch.load(ARTIFACTS / "standard_torch.pt", weights_only=True))
STD.eval()
COEFS = pd.read_csv(ARTIFACTS / "coefficients.csv", index_col="feature")

FN_DEFAULT = META["costs"]["fn_mult"]
FP_DEFAULT = META["costs"]["fp_mult"]
OFFICIAL_T = META["theoretical_threshold"]

DF = clean(load_data())
val_start, test_start = pd.Timestamp(META["split"]["val_start"]), pd.Timestamp(META["split"]["test_start"])
DF["split"] = np.where(DF.arrival_date < val_start, "train", np.where(DF.arrival_date < test_start, "val", "test"))

SPLITS = {}
for name in ("val", "test"):
    d = DF[DF.split == name]
    X = PRE.transform(d).astype(np.float64)
    SPLITS[name] = {
        "df": d, "y": d[TARGET].to_numpy(), "revenue": booking_revenue(d, META["adr_cap"]),
        "p": {"sklearn": SK.predict_proba(X)[:, 1],
              "manual_torch": manual_predict_proba(MANUAL, X),
              "standard_torch": STD.predict_proba(X)},
        "rule": ((d.deposit_type == "Non Refund") | (d.previous_cancellations > 0)).to_numpy().astype(float),
    }


# ---------------------------------------------------------------- tab 1: model comparison

def metrics_table(split: str) -> pd.DataFrame:
    s = SPLITS[split]
    y, rev = s["y"], s["revenue"]
    rows = []
    for name, p in [("Accept all (no model)", np.zeros(len(y))), ("Rule: non-refund or prev. cancel", s["rule"])] \
            + [(MODEL_STYLE[m]["label"], p) for m, p in s["p"].items()]:
        is_model = name in [v["label"] for v in MODEL_STYLE.values()]
        c = business_cost(y, p, rev, OFFICIAL_T if is_model else 0.5, FN_DEFAULT, FP_DEFAULT)
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        rows.append({
            "Model": name,
            "Log-loss": round(log_loss(y, pc), 4) if is_model else None,
            "ROC-AUC": round(roc_auc_score(y, p), 4) if is_model else None,
            "PR-AUC": round(average_precision_score(y, p), 4) if is_model else None,
            "Precision": round(c["tp"] / max(c["tp"] + c["fp"], 1), 3),
            "Recall": round(c["tp"] / max(c["tp"] + c["fn"], 1), 3),
            "Business cost (EUR)": round(c["total_cost"]),
        })
    return pd.DataFrame(rows)


def prob_by_outcome_plot(split: str) -> go.Figure:
    s = SPLITS[split]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[v["label"] for v in MODEL_STYLE.values()],
                        shared_yaxes=True)
    for i, (m, p) in enumerate(s["p"].items(), 1):
        for cls in (0, 1):
            fig.add_trace(go.Histogram(x=p[s["y"] == cls], xbins=dict(start=0, end=1, size=0.05),
                                       name=CLASS_LABEL[cls], marker_color=CLASS_COLOR[cls], opacity=0.75,
                                       showlegend=(i == 1), legendgroup=str(cls)), row=1, col=i)
        fig.add_vline(x=OFFICIAL_T, line_dash="dash", line_color=INK_2, row=1, col=i)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Predicted P(cancel)", range=[0, 1])
    fig.update_yaxes(title_text="Bookings", col=1)
    return style(fig, f"Predicted probability vs. actual outcome ({split}); dashed = threshold {OFFICIAL_T:.3f}")


def roc_pr_plot(split: str) -> go.Figure:
    s = SPLITS[split]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC curve", "Precision-recall curve"])
    for m, p in s["p"].items():
        st = MODEL_STYLE[m]
        fpr, tpr, _ = roc_curve(s["y"], p)
        prec, rec, _ = precision_recall_curve(s["y"], p)
        line = dict(color=st["color"], dash=st["dash"], width=2)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=line, legendgroup=m,
                                 name=f"{st['label']} (AUC {roc_auc_score(s['y'], p):.3f})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=line, legendgroup=m, showlegend=False,
                                 name=st["label"]), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             name="Random"), row=1, col=1)
    fig.add_hline(y=s["y"].mean(), line_dash="dot", line_color=MUTED, row=1, col=2)
    fig.update_xaxes(title_text="False positive rate", row=1, col=1)
    fig.update_yaxes(title_text="True positive rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    return style(fig, f"Ranking quality ({split})", height=420)


def calibration_plot(split: str) -> go.Figure:
    s = SPLITS[split]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             name="Perfect calibration"))
    for m, p in s["p"].items():
        st = MODEL_STYLE[m]
        frac, mean_p = calibration_curve(s["y"], p, n_bins=10, strategy="quantile")
        fig.add_trace(go.Scatter(x=mean_p, y=frac, mode="lines+markers", name=st["label"],
                                 line=dict(color=st["color"], dash=st["dash"], width=2), marker=dict(size=8)))
    fig.update_xaxes(title_text="Mean predicted P(cancel)", range=[0, 1])
    fig.update_yaxes(title_text="Actual cancellation rate", range=[0, 1])
    return style(fig, f"Calibration: predicted vs. actual rate ({split}, deciles)")


def over_time_plot(split: str) -> go.Figure:
    s = SPLITS[split]
    d = pd.DataFrame({"week": s["df"].arrival_date.dt.to_period("W").dt.start_time.to_numpy(), "y": s["y"],
                      **{m: p for m, p in s["p"].items()}})
    g = d.groupby("week").mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g.y, mode="lines+markers", name="Actual",
                             line=dict(color=INK, width=2), marker=dict(size=8)))
    for m in s["p"]:
        st = MODEL_STYLE[m]
        fig.add_trace(go.Scatter(x=g.index, y=g[m], mode="lines", name=f"Predicted - {st['label']}",
                                 line=dict(color=st["color"], dash=st["dash"], width=2)))
    fig.update_yaxes(title_text="Cancellation rate", tickformat=".0%")
    fig.update_xaxes(title_text="Arrival week")
    return style(fig, f"Weekly cancellation rate: predicted vs. actual ({split})")


def agreement_plot(split: str) -> go.Figure:
    s = SPLITS[split]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             name="y = x"))
    for m in ("manual_torch", "standard_torch"):
        st = MODEL_STYLE[m]
        diff = np.abs(s["p"][m] - s["p"]["sklearn"]).max()
        fig.add_trace(go.Scattergl(x=s["p"]["sklearn"], y=s["p"][m], mode="markers", opacity=0.6,
                                   name=f"{st['label']} (max |diff| {diff:.1e})",
                                   marker=dict(color=st["color"], size=8, line=dict(color=SURFACE, width=1))))
    fig.update_xaxes(title_text="scikit-learn P(cancel)")
    fig.update_yaxes(title_text="PyTorch P(cancel)")
    return style(fig, f"Do the three implementations agree? ({split})")


def coef_plot() -> go.Figure:
    c = COEFS.drop(index="(intercept)")
    top = c.reindex(c.sklearn.abs().sort_values(ascending=False).index[:20])[::-1]
    names = [n.replace("num__", "").replace("cat__", "") for n in top.index]
    fig = go.Figure()
    for m in MODEL_STYLE:
        fig.add_trace(go.Bar(y=names, x=top[m], orientation="h", name=MODEL_STYLE[m]["label"],
                             marker_color=MODEL_STYLE[m]["color"]))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_xaxes(title_text="Coefficient (log-odds per 1 std / vs. reference category)")
    return style(fig, "Top 20 coefficients by |scikit-learn coefficient|", height=640)


def loss_curve_plot() -> go.Figure:
    lc = META["loss_curves"]
    fig = go.Figure()
    for m in ("manual_torch", "standard_torch"):
        st = MODEL_STYLE[m]
        epochs = np.arange(len(lc[m])) * lc["every"]
        fig.add_trace(go.Scatter(x=epochs, y=lc[m], mode="lines", name=st["label"],
                                 line=dict(color=st["color"], dash=st["dash"], width=2)))
    fig.add_hline(y=META["train_objective"]["sklearn"], line_dash="dot", line_color=MODEL_STYLE["sklearn"]["color"],
                  annotation_text="scikit-learn optimum", annotation_font_color=INK_2)
    fig.update_xaxes(title_text="Epoch (full batch)")
    fig.update_yaxes(title_text="Train objective: BCE + lambda/2 ||w||^2", type="log")
    return style(fig, f"PyTorch training curves (lr = {META['chosen']['lr']}, {META['chosen']['epochs']:,} epochs)")


def agreement_table() -> pd.DataFrame:
    rows = []
    for pair, v in META["agreement"].items():
        row = {"Pair": pair, "max |prob diff|": f"{v['max_abs_prob_diff']:.2e}",
               "mean |prob diff|": f"{v['mean_abs_prob_diff']:.2e}",
               "max |coef diff|": f"{v['max_abs_coef_diff']:.2e}", "|intercept diff|": f"{v['intercept_diff']:.2e}"}
        row.update({f"same decision @ {t}": f"{s:.2%}" for t, s in v["same_decision_by_threshold"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_table() -> pd.DataFrame:
    df = pd.DataFrame(META["convergence"])
    for col in ("max_abs_coef_diff", "intercept_diff", "max_abs_prob_diff", "mean_abs_prob_diff", "objective_gap"):
        df[col] = df[col].map(lambda v: f"{v:.2e}")
    return df


def update_comparison(split):
    return (metrics_table(split), prob_by_outcome_plot(split), over_time_plot(split), roc_pr_plot(split),
            calibration_plot(split), agreement_plot(split))


# ---------------------------------------------------------------- tab 2: distributions

NUMERIC_FEATURES = ["lead_time", "adr", "total_nights", "total_guests", "stays_in_weekend_nights",
                    "previous_cancellations", "previous_bookings_not_canceled", "days_in_waiting_list",
                    "total_of_special_requests"]
CATEGORICAL_FEATURES = ["hotel", "deposit_type", "market_segment", "distribution_channel", "customer_type",
                        "meal", "reserved_room_type", "arrival_date_month", "is_repeated_guest",
                        "required_car_parking_spaces"]


def feature_plot(feature: str, split: str) -> go.Figure:
    d = DF if split == "all" else DF[DF.split == split]
    note = " (post-booking, excluded from model)" if feature in META["excluded_columns"] else ""
    if feature in NUMERIC_FEATURES:
        fig = go.Figure()
        cap = d[feature].quantile(0.99)
        for cls in (0, 1):
            fig.add_trace(go.Histogram(x=d.loc[d[TARGET] == cls, feature].clip(upper=cap), nbinsx=40,
                                       name=CLASS_LABEL[cls], marker_color=CLASS_COLOR[cls], opacity=0.75))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text=f"{feature} (clipped at p99 = {cap:.0f})")
        fig.update_yaxes(title_text="Bookings")
        return style(fig, f"{feature} by outcome{note}", height=420)
    g = d.groupby(feature)[TARGET].agg(["size", "mean"]).sort_values("size", ascending=False).head(15)
    labels = g.index.astype(str)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Bookings", "Cancellation rate"])
    fig.add_trace(go.Bar(x=labels, y=g["size"], marker_color="#2a78d6", name="Bookings", showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=g["mean"], marker_color="#eb6834", name="Cancel rate", showlegend=False,
                         text=[f"{v:.0%}" for v in g["mean"]], textposition="outside"), row=1, col=2)
    fig.add_hline(y=d[TARGET].mean(), line_dash="dot", line_color=MUTED, row=1, col=2,
                  annotation_text="overall", annotation_font_color=INK_2)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1], row=1, col=2)
    return style(fig, f"{feature}{note}", height=420)


def target_over_time_plot() -> go.Figure:
    g = DF.groupby([DF.arrival_date.dt.to_period("M").dt.start_time, "hotel"])[TARGET].mean().unstack()
    fig = go.Figure()
    for split, (lo, hi) in {"train": (DF.arrival_date.min(), val_start), "val": (val_start, test_start),
                            "test": (test_start, DF.arrival_date.max())}.items():
        fig.add_vrect(x0=lo, x1=hi, fillcolor=SPLIT_COLOR[split], opacity=0.5, line_width=0,
                      annotation_text=split, annotation_position="top left", annotation_font_color=INK_2)
    for hotel, color, dash in (("City Hotel", "#2a78d6", "solid"), ("Resort Hotel", "#eb6834", "dash")):
        fig.add_trace(go.Scatter(x=g.index, y=g[hotel], mode="lines+markers", name=hotel,
                                 line=dict(color=color, dash=dash, width=2), marker=dict(size=8)))
    fig.update_yaxes(title_text="Cancellation rate", tickformat=".0%")
    fig.update_xaxes(title_text="Arrival month")
    return style(fig, "Target over time and the temporal split (shaded)", height=400)


def split_table() -> pd.DataFrame:
    rows = []
    for name, v in META["split"]["info"].items():
        rows.append({"Split": name, "Rows": v["rows"], "Arrivals": f"{v['from']} -> {v['to']}",
                     "Cancel rate": f"{v['cancel_rate']:.1%}", "City share": f"{v['city_share']:.1%}",
                     "Non-refund share": f"{v['non_refund_share']:.1%}",
                     "Non-refund cancel rate": f"{v['non_refund_cancel_rate']:.1%}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tab 3: threshold & business cost

def threshold_view(model: str, split: str, threshold: float, fn_mult: float, fp_mult: float):
    s = SPLITS[split]
    y, rev, p = s["y"], s["revenue"], s["p"][model]
    fn_mult, fp_mult = max(float(fn_mult or 0), 0.0), max(float(fp_mult or 0), 0.0)
    c = business_cost(y, p, rev, threshold, fn_mult, fp_mult)

    z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    text = [[f"TN<br>{c['tn']:,}", f"FP<br>{c['fp']:,}<br>cost {c['fp_cost']:,.0f}"],
            [f"FN<br>{c['fn']:,}<br>cost {c['fn_cost']:,.0f}", f"TP<br>{c['tp']:,}"]]
    cm = go.Figure(go.Heatmap(z=z, x=["Predicted: shows up", "Predicted: cancels"],
                              y=["Actual: shows up", "Actual: cancels"], text=text, texttemplate="%{text}",
                              colorscale=[[0, "#cde2fb"], [1, "#1c5cab"]], showscale=False,
                              textfont=dict(size=14), hovertemplate="%{text}<extra></extra>"))
    cm.update_yaxes(autorange="reversed")
    style(cm, f"Confusion matrix - {MODEL_STYLE[model]['label']}, {split}, threshold {threshold:.4f}", height=380)

    grid, costs = cost_curve(y, p, rev, fn_mult, fp_mult)
    accept_all = fn_mult * rev[y == 1].sum()
    rule = business_cost(y, s["rule"], rev, 0.5, fn_mult, fp_mult)["total_cost"]
    theory = fp_mult / (fp_mult + fn_mult) if fp_mult + fn_mult > 0 else 0.5
    best = grid[int(np.argmin(costs))]
    cc = go.Figure()
    cc.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name="Model cost",
                            line=dict(color=MODEL_STYLE[model]["color"], width=2)))
    cc.add_hline(y=accept_all, line_dash="dot", line_color=MUTED, annotation_text="accept all",
                 annotation_font_color=INK_2)
    cc.add_hline(y=rule, line_dash="dash", line_color=MUTED, annotation_text="simple rule",
                 annotation_position="bottom right", annotation_font_color=INK_2)
    cc.add_vline(x=theory, line_dash="dash", line_color=INK_2, annotation_text=f"theory {theory:.2f}",
                 annotation_position="top left", annotation_font_color=INK_2)
    cc.add_trace(go.Scatter(x=[best], y=[costs.min()], mode="markers", name=f"Min on this split ({best:.2f})",
                            marker=dict(color=INK, size=10, symbol="diamond")))
    cc.add_trace(go.Scatter(x=[threshold], y=[c["total_cost"]], mode="markers", name="Current threshold",
                            marker=dict(color=MODEL_STYLE[model]["color"], size=12,
                                        line=dict(color=SURFACE, width=2))))
    cc.update_xaxes(title_text="Threshold", range=[0, 1])
    cc.update_yaxes(title_text="Business cost (EUR)")
    style(cc, "Business cost vs. threshold", height=380)

    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    saving = accept_all - c["total_cost"]
    summary = (
        f"### Business cost: **{c['total_cost']:,.0f} EUR**\n"
        f"- Missed cancellations (FN): {c['fn']:,} bookings -> {c['fn_cost']:,.0f} EUR of empty rooms\n"
        f"- False alarms (FP): {c['fp']:,} bookings -> {c['fp_cost']:,.0f} EUR of relocations\n"
        f"- vs. accepting every booking ({accept_all:,.0f} EUR): **{'saves' if saving >= 0 else 'loses'} "
        f"{abs(saving):,.0f} EUR** ({saving / accept_all:+.1%})\n"
        f"- vs. simple rule ({rule:,.0f} EUR): {rule - c['total_cost']:+,.0f} EUR\n"
        f"- Precision {precision:.1%} - Recall {recall:.1%} - Bookings flagged {c['tp'] + c['fp']:,}\n"
        f"- Cost-optimal threshold: theory FP/(FP+FN) = **{theory:.3f}**; empirical minimum on this split = {best:.2f}"
    )
    return cm, cc, summary


def theory_threshold(fn_mult, fp_mult):
    fn_mult, fp_mult = max(float(fn_mult or 0), 0.0), max(float(fp_mult or 0), 0.0)
    return round(fp_mult / (fp_mult + fn_mult), 4) if fp_mult + fn_mult > 0 else 0.5


# ---------------------------------------------------------------- tab 4: pipeline decisions

def table(records, cols) -> pd.DataFrame:
    return pd.DataFrame(records)[cols]


variants = table(META["feature_variants"], ["axis", "label", "n_features", "log_loss", "roc_auc", "pr_auc"])
c_search = table(META["c_search"], ["C", "log_loss", "roc_auc", "pr_auc"])
lr_search = table(META["lr_search"], ["method", "lr", "val_log_loss", "final_train_objective"])
ablation = table(META["leakage_ablation"], ["variant", "val_roc_auc", "test_roc_auc", "test_pr_auc",
                                             "test_log_loss", "test_cost"])
excluded = pd.DataFrame([{"Column": k, "Why excluded": v} for k, v in META["excluded_columns"].items()])

# ---------------------------------------------------------------- layout

with gr.Blocks(title="Hotel cancellation - three logistic regressions") as demo:
    gr.Markdown(
        "# Hotel booking cancellation: one logistic regression, three implementations\n"
        f"Predicting at **booking time** whether a booking will be canceled, to drive overbooking and deposit "
        f"decisions. Models are loaded from `artifacts/` (trained by `train.py`), never retrained here. "
        f"Temporal split by arrival date: train < {META['split']['val_start']} <= val < "
        f"{META['split']['test_start']} <= test. Official threshold = FP/(FP+FN) = **{OFFICIAL_T:.3f}**."
    )
    with gr.Tab("Model comparison"):
        split_cmp = gr.Radio(["test", "val"], value="test", label="Evaluation split")
        metrics_out = gr.Dataframe(label=f"Metrics (models at threshold {OFFICIAL_T:.3f}; default costs)")
        prob_out = gr.Plot()
        time_out = gr.Plot()
        with gr.Row():
            roc_out = gr.Plot()
        with gr.Row():
            cal_out = gr.Plot()
            agree_out = gr.Plot()
        gr.Markdown("### Agreement between implementations (test set, from training run)")
        gr.Dataframe(agreement_table())
        gr.Markdown("### Gap to scikit-learn after 1k / 3k / 10k epochs (shrinks with training = convergence, "
                    "not a different model)")
        gr.Dataframe(convergence_table())
        with gr.Row():
            gr.Plot(coef_plot())
            gr.Plot(loss_curve_plot())
        cmp_outputs = [metrics_out, prob_out, time_out, roc_out, cal_out, agree_out]
        split_cmp.change(update_comparison, split_cmp, cmp_outputs)
        demo.load(update_comparison, split_cmp, cmp_outputs)

    with gr.Tab("Distributions"):
        gr.Plot(target_over_time_plot())
        gr.Dataframe(split_table(), label="Split summary")
        with gr.Row():
            feat = gr.Dropdown(NUMERIC_FEATURES + CATEGORICAL_FEATURES, value="lead_time", label="Feature")
            feat_split = gr.Radio(["all", "train", "val", "test"], value="all", label="Rows")
        feat_out = gr.Plot()
        feat.change(feature_plot, [feat, feat_split], feat_out)
        feat_split.change(feature_plot, [feat, feat_split], feat_out)
        demo.load(feature_plot, [feat, feat_split], feat_out)

    with gr.Tab("Threshold & business cost"):
        gr.Markdown(
            "A **false negative** (missed cancellation) costs `FN x ADR x nights` (empty room). A **false "
            "positive** (flagged booking that shows up, after overbooking) costs `FP x ADR x nights` "
            "(relocation, compensation, reputation). Edit the multipliers to change the business assumptions."
        )
        with gr.Row():
            t_model = gr.Dropdown([(v["label"], k) for k, v in MODEL_STYLE.items()], value="sklearn", label="Model")
            t_split = gr.Radio(["test", "val"], value="test", label="Split")
            fn_in = gr.Number(value=FN_DEFAULT, label="FN cost (x booking revenue)", minimum=0, step=0.5)
            fp_in = gr.Number(value=FP_DEFAULT, label="FP cost (x booking revenue)", minimum=0, step=0.5)
        with gr.Row():
            t_slider = gr.Slider(0.01, 0.99, value=OFFICIAL_T, step=0.0001, label="Decision threshold")
            t_button = gr.Button("Set to cost-optimal threshold FP/(FP+FN)")
        summary_out = gr.Markdown()
        with gr.Row():
            cm_out = gr.Plot()
            cc_out = gr.Plot()
        t_inputs = [t_model, t_split, t_slider, fn_in, fp_in]
        t_outputs = [cm_out, cc_out, summary_out]
        for comp in (t_model, t_split, fn_in, fp_in):
            comp.change(threshold_view, t_inputs, t_outputs)
        t_slider.release(threshold_view, t_inputs, t_outputs)
        t_button.click(theory_threshold, [fn_in, fp_in], t_slider).then(threshold_view, t_inputs, t_outputs)
        demo.load(threshold_view, t_inputs, t_outputs)

    with gr.Tab("Pipeline decisions"):
        gr.Markdown("### Columns excluded because they are not known at booking time")
        gr.Dataframe(excluded)
        gr.Markdown("### Leakage ablation: what post-booking variables would add (scikit-learn, same C)")
        gr.Dataframe(ablation)
        gr.Markdown("### Dependence on deposit_type (scikit-learn, same C)")
        gr.Dataframe(pd.DataFrame(META["without_deposit"]))
        gr.Markdown("### Drift or seasonality? Evaluation windows vs the same dates one year earlier")
        gr.Dataframe(pd.DataFrame(META["seasonality"]))
        gr.Markdown("### Composition: ROC-AUC with and without Non Refund bookings (scikit-learn)")
        gr.Dataframe(pd.DataFrame(META["composition"]))
        gr.Markdown(f"### Test cost differences, block bootstrap over {META['bootstrap']['weeks']} arrival weeks "
                    f"({META['bootstrap']['resamples']:,} resamples, EUR)")
        gr.Dataframe(pd.DataFrame([{"comparison": k, **v} for k, v in META["bootstrap"].items()
                                   if isinstance(v, dict)]))
        gr.Markdown("### Feature variants (validation, one axis at a time, C = 1)")
        gr.Dataframe(variants)
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### Regularisation search (chosen C = {META['chosen']['C']})")
                gr.Dataframe(c_search)
            with gr.Column():
                gr.Markdown(f"### Learning-rate search (chosen {META['chosen']['lr']})")
                gr.Dataframe(lr_search)


if __name__ == "__main__":
    # No fixed port: Gradio starts at 7860 (what CI checks) and moves to the next free port if taken.
    demo.launch(server_name="127.0.0.1", inbrowser=not os.environ.get("CI"))
