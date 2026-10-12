"""
Assignment 1 — Gradio dashboard (run with: uv run python main.py app).

Loads ONLY what train.py saved in artifacts/ (preprocessor, three trained
models, test split, metrics). Nothing is fit or trained here: the test-set
predictions are recomputed at startup by calling the saved models.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from data_utils import ARTIFACTS_DIR, load_wine
from features import RAW_FEATURES, engineer, make_target
from models import LogisticRegressionNet, predict_manual, predict_standard

# ---------------- load artifacts (no training) ----------------
if not (ARTIFACTS_DIR / "metrics.json").exists():
    raise SystemExit("artifacts/ is empty: run `uv run python main.py train` first.")

summary = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
profile = json.loads((ARTIFACTS_DIR / "data_profile.json").read_text())
pre = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS_DIR / "sklearn_logreg.joblib")
alc_model = joblib.load(ARTIFACTS_DIR / "baseline_alcohol_only.joblib")
manual_params = torch.load(ARTIFACTS_DIR / "manual_pytorch_params.pt")
net = LogisticRegressionNet(summary["n_features"])
net.load_state_dict(torch.load(ARTIFACTS_DIR / "nn_module_pytorch.pt"))
coefs = pd.read_csv(ARTIFACTS_DIR / "coefficients.csv", index_col=0)
history = pd.read_csv(ARTIFACTS_DIR / "training_history.csv")
ablation = pd.read_csv(ARTIFACTS_DIR / "ablation.csv")
test = pd.read_csv(ARTIFACTS_DIR / "test_set.csv")

MODELS = {
    "sklearn": lambda X: sk_model.predict_proba(X)[:, 1],
    "manual_pytorch": lambda X: predict_manual(manual_params, X),
    "nn_module_pytorch": lambda X: predict_standard(net, X),
}
LABELS = {"sklearn": "scikit-learn", "manual_pytorch": "Manual PyTorch loop",
          "nn_module_pytorch": "nn.Module + optim", "baseline_alcohol_only": "Baseline: alcohol only",
          "baseline_prior": "Baseline: never flag", "baseline_flag_all": "Baseline: flag all"}
COLORS = {"sklearn": "#2a78d6", "manual_pytorch": "#eb6834", "nn_module_pytorch": "#1baf7a",
          "baseline_alcohol_only": "#8a8984"}
WINE = {"red": "#8c1c3a", "white": "#d4b84a"}
TEMPLATE = "plotly_white"

X_test = pre.transform(engineer(test))
y_test = make_target(test).to_numpy()
alcohol_idx = summary["feature_names"].index("alcohol")
probs = {name: f(X_test) for name, f in MODELS.items()}
probs["baseline_alcohol_only"] = alc_model.predict_proba(X_test[:, [alcohol_idx]])[:, 1]

full = load_wine()
full["color"] = full.is_red.map({1: "red", 0: "white"})
full["premium"] = make_target(full).map({1: "premium (≥7)", 0: "not premium"})
FEATURES = [c for c in RAW_FEATURES if c != "is_red"]


# ---------------- threshold / cost ----------------
def confusion(y, p, t):
    pred = p >= t
    return {"tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()),
            "fn": int((~pred & (y == 1)).sum()), "tn": int((~pred & (y == 0)).sum())}


def threshold_view(model, t, cost_fp, cost_fn):
    p = probs[model]
    c = confusion(y_test, p, t)
    cost = c["fp"] * cost_fp + c["fn"] * cost_fn
    never = int(y_test.sum()) * cost_fn
    flag_all = int((y_test == 0).sum()) * cost_fp
    best_trivial = min(never, flag_all)

    z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    cm = go.Figure(go.Heatmap(
        z=z, x=["Predicted: not premium", "Predicted: premium"],
        y=["Actual: not premium", "Actual: premium"],
        text=[[f"TN<br>{c['tn']}", f"FP<br>{c['fp']}<br>€{c['fp'] * cost_fp:,.0f}"],
              [f"FN<br>{c['fn']}<br>€{c['fn'] * cost_fn:,.0f}", f"TP<br>{c['tp']}"]],
        texttemplate="%{text}", textfont={"size": 16}, colorscale="Blues", showscale=False,
        hovertemplate="%{y}<br>%{x}<br>%{z} batches<extra></extra>"))
    cm.update_layout(template=TEMPLATE, title=f"Confusion matrix on test set — {LABELS[model]} @ {t:.2f}",
                     yaxis={"autorange": "reversed"}, height=420)

    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    curve = pd.DataFrame({"threshold": grid})
    for m in MODELS:
        curve[LABELS[m]] = [confusion(y_test, probs[m], g)["fp"] * cost_fp +
                            confusion(y_test, probs[m], g)["fn"] * cost_fn for g in grid]
    fig = go.Figure()
    for m in MODELS:
        fig.add_scatter(x=grid, y=curve[LABELS[m]], name=LABELS[m], mode="lines",
                        line={"color": COLORS[m], "width": 2, "dash": "dot" if m != "sklearn" else "solid"})
    fig.add_hline(y=never, line_dash="dash", line_color="#8a8984",
                  annotation_text=f"never flag: €{never:,}", annotation_position="bottom right")
    fig.add_hline(y=flag_all, line_dash="dash", line_color="#8a8984",
                  annotation_text=f"flag all: €{flag_all:,}", annotation_position="top right")
    fig.add_vline(x=t, line_color="#0b0b0b", line_width=1)
    fig.add_vline(x=cost_fp / (cost_fp + cost_fn), line_dash="dot", line_color="#8a8984",
                  annotation_text="FP/(FP+FN)", annotation_position="bottom left")
    fig.update_layout(template=TEMPLATE, title="Test-set cost vs threshold",
                      xaxis_title="decision threshold", yaxis_title="total cost (€)", height=420,
                      legend={"orientation": "h", "y": -0.2})

    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0
    saved = best_trivial - cost
    md = (f"### Business cost at threshold {t:.2f}: **€{cost:,.0f}** "
          f"(€{cost / len(y_test):.2f} per batch, {len(y_test)} test batches)\n\n"
          f"- Panel sends: **{c['tp'] + c['fp']}** batches, of which {c['tp']} truly premium "
          f"(precision {prec:.1%}); premium batches caught: {c['tp']}/{c['tp'] + c['fn']} (recall {rec:.1%})\n"
          f"- Best trivial policy (never flag / flag all): €{best_trivial:,.0f} → model "
          f"{'saves' if saved >= 0 else 'loses'} **€{abs(saved):,.0f}**\n"
          f"- Theory: with calibrated probabilities the cost-minimising threshold is "
          f"FP/(FP+FN) = {cost_fp / (cost_fp + cost_fn):.2f}. Threshold chosen on *validation* in "
          f"train.py: **{summary['results'][model]['val']['threshold']:.2f}**")
    return cm, fig, md


# ---------------- predicted vs actual ----------------
def calibration_plot(n_bins=10):
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                    line={"color": "#c3c2b7", "dash": "dash", "width": 1})
    for m in list(MODELS) + ["baseline_alcohol_only"]:
        df = pd.DataFrame({"p": probs[m], "y": y_test})
        df["bin"] = pd.qcut(df.p, n_bins, duplicates="drop")
        g = df.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        fig.add_scatter(x=g.p, y=g.y, mode="lines+markers", name=LABELS[m],
                        marker={"size": 9, "color": COLORS[m]},
                        line={"width": 2, "color": COLORS[m], "dash": "dot" if m.startswith("baseline") else "solid"},
                        customdata=g.n, hovertemplate="mean predicted %{x:.2f}<br>actual premium rate %{y:.2f}"
                                                       "<br>%{customdata} batches<extra>" + LABELS[m] + "</extra>")
    fig.update_layout(template=TEMPLATE, height=480, xaxis_title="mean predicted P(premium) in decile",
                      yaxis_title="actual share premium in decile",
                      title="Predicted vs actual: calibration by predicted-probability decile (test set)")
    return fig


def distribution_by_class():
    rows = [pd.DataFrame({"model": LABELS[m], "P(premium)": probs[m],
                          "actual": np.where(y_test == 1, "premium", "not premium")}) for m in MODELS]
    fig = px.box(pd.concat(rows), x="model", y="P(premium)", color="actual", points="outliers",
                 color_discrete_map={"premium": "#2a78d6", "not premium": "#c3c2b7"},
                 title="P(premium) by actual class")
    fig.update_layout(template=TEMPLATE, height=440)
    return fig


def agreement_scatter():
    fig = go.Figure()
    for m in ("manual_pytorch", "nn_module_pytorch"):
        fig.add_scatter(x=probs["sklearn"], y=probs[m], mode="markers", name=LABELS[m],
                        marker={"size": 8, "color": COLORS[m], "opacity": 0.6,
                                "line": {"width": 1, "color": "white"}})
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x", line={"color": "#8a8984", "dash": "dash"})
    fig.update_layout(template=TEMPLATE, height=440, xaxis_title="scikit-learn P(premium)",
                      yaxis_title="PyTorch P(premium)", title="Per-batch agreement on the test set")
    return fig


def results_table():
    keys = ["sklearn", "manual_pytorch", "nn_module_pytorch", "baseline_alcohol_only",
            "baseline_prior", "baseline_flag_all"]
    rows = []
    for k in keys:
        r = summary["results"][k]["test"]
        rows.append({"model": LABELS[k], "ROC AUC": round(r["roc_auc"], 4), "PR AUC": round(r["pr_auc"], 4),
                     "log loss": round(r["log_loss"], 4), "threshold (from val)": r["threshold"],
                     "precision": round(r["precision"], 3), "recall": round(r["recall"], 3),
                     "FP": r["fp"], "FN": r["fn"], "test cost €": r["cost_eur"]})
    return pd.DataFrame(rows)


def agreement_table():
    a = summary["agreement_vs_sklearn"]
    return pd.DataFrame([{"model": LABELS[k],
                          "train objective gap": f"{a[k]['objective_gap_vs_sklearn']:.2e}",
                          "max |Δ coef|": f"{a[k]['max_abs_coef_diff_vs_sklearn']:.2e}",
                          "Δ intercept": f"{a[k]['intercept_diff_vs_sklearn']:.2e}",
                          "max |Δ prob| (test)": f"{a[k]['max_abs_prob_diff_vs_sklearn']:.2e}",
                          "label disagreements (test)": a[k]["test_label_disagreements_at_sklearn_threshold"]}
                         for k in ("manual_pytorch", "nn_module_pytorch")])


def coef_plot():
    c = coefs.drop(index="(intercept)").copy()
    c = c.loc[c["sklearn"].abs().sort_values().index]
    fig = go.Figure()
    for m in MODELS:
        fig.add_bar(y=c.index, x=c[m], name=LABELS[m], orientation="h", marker_color=COLORS[m])
    fig.update_layout(template=TEMPLATE, barmode="group", height=900, bargap=0.25,
                      title="Coefficients on standardized features (all three should be identical)",
                      xaxis_title="coefficient (log-odds per 1 std)", legend={"orientation": "h", "y": 1.03})
    return fig


def convergence_plot():
    fig = go.Figure()
    for m in ("manual_pytorch", "nn_module_pytorch"):
        h = history[history.model == m]
        fig.add_scatter(x=h.epoch, y=h.max_coef_gap, mode="lines", name=LABELS[m],
                        line={"color": COLORS[m], "width": 2, "dash": "dot" if m == "nn_module_pytorch" else "solid"})
    fig.update_layout(template=TEMPLATE, height=420, yaxis_type="log", xaxis_title="epoch (full-batch GD step)",
                      yaxis_title="max |coef − sklearn coef| (log)",
                      title=f"Convergence to the sklearn solution (lr={summary['best_lr']}, C={summary['best_C']})")
    return fig


# ---------------- data ----------------
def target_plot():
    counts = full.groupby(["quality", "color"]).size().reset_index(name="n")
    fig = px.bar(counts, x="quality", y="n", color="color", barmode="group", color_discrete_map=WINE,
                 title="Quality score distribution (all 6,497 rows). Premium = quality ≥ 7")
    fig.add_vline(x=6.5, line_dash="dash", line_color="#0b0b0b", annotation_text="premium cut-off")
    fig.update_layout(template=TEMPLATE)
    return fig


def feature_plot(feature, by):
    if by == "premium vs not":
        fig = px.histogram(full, x=feature, color="premium", barmode="overlay", nbins=60, histnorm="probability density",
                           color_discrete_map={"premium (≥7)": "#2a78d6", "not premium": "#c3c2b7"},
                           facet_col="color", title=f"{feature}: premium vs not (density, by colour)")
    elif by == "quality score":
        fig = px.box(full, x="quality", y=feature, color="color", color_discrete_map=WINE,
                     title=f"{feature} by quality score")
    else:
        fig = px.histogram(full, x=feature, color="color", barmode="overlay", nbins=60, color_discrete_map=WINE,
                           title=f"Distribution of {feature}")
    fig.update_layout(template=TEMPLATE, height=460)
    return fig


# ---------------- score a new batch ----------------
medians = full[RAW_FEATURES].median()


def score_batch(color, *vals):
    row = pd.DataFrame([dict(zip(FEATURES, vals), is_red=int(color == "red"))])
    X = pre.transform(engineer(row))
    out = {LABELS[m]: float(f(X)[0]) for m, f in MODELS.items()}
    t = summary["results"]["sklearn"]["val"]["threshold"]
    p = out["scikit-learn"]
    decision = "✅ Send to tasting panel (premium candidate)" if p >= t else "❌ Regular line"
    return (pd.DataFrame({"model": out.keys(), "P(premium)": [f"{v:.4f}" for v in out.values()]}),
            f"### {decision}\nP(premium) = {p:.3f} vs threshold {t:.2f}")


# ---------------- layout ----------------
with gr.Blocks(title="Wine Premium Screening") as demo:
    gr.Markdown(
        "# 🍷 Wine premium screening — logistic regression, three ways\n"
        f"**Decision:** which batches should go to the expert tasting panel as premium candidates (quality ≥ 7)? "
        f"Missing a premium batch costs **€{summary['costs_eur']['fn']}** (lost margin); sending an ordinary one "
        f"costs **€{summary['costs_eur']['fp']}** (panel time). Data: {profile['n_rows']:,} rows → "
        f"{profile['n_rows_after_dedup']:,} after dropping {profile['n_exact_duplicates']:,} exact duplicates; "
        f"split {summary['split_sizes']}. Everything below is loaded from `artifacts/`; nothing is retrained.")

    with gr.Tab("Threshold & business cost"):
        with gr.Row():
            model_dd = gr.Dropdown([(LABELS[m], m) for m in MODELS], value="sklearn", label="Model")
            thr = gr.Slider(0.01, 0.99, value=summary["results"]["sklearn"]["val"]["threshold"], step=0.01,
                            label="Decision threshold")
            cfp = gr.Number(value=summary["costs_eur"]["fp"], label="Cost of a false positive (€)")
            cfn = gr.Number(value=summary["costs_eur"]["fn"], label="Cost of a false negative (€)")
        cost_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        inputs = [model_dd, thr, cfp, cfn]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, cost_plot, cost_md])
        demo.load(threshold_view, inputs, [cm_plot, cost_plot, cost_md])

    with gr.Tab("Predicted vs actual"):
        gr.Markdown("Test-set results (threshold chosen on validation, never on test):")
        gr.Dataframe(results_table(), interactive=False)
        gr.Plot(calibration_plot())
        with gr.Row():
            gr.Plot(distribution_by_class())
            gr.Plot(agreement_scatter())

    with gr.Tab("Three-way agreement"):
        gr.Markdown("All three minimise the same objective (mean log loss + L2 with λ = 1/(2·C·n), bias unpenalised), "
                    "so they should reach the same coefficients. They do, but only after enough gradient steps:")
        gr.Dataframe(agreement_table(), interactive=False)
        gr.Plot(convergence_plot())
        gr.Plot(coef_plot())

    with gr.Tab("Feature engineering"):
        gr.Markdown("Ablation, 5-fold stratified CV on train+val (test untouched), sklearn C=1. "
                    "Condition number of XᵀX shows how hard each design is for plain gradient descent.")
        gr.Dataframe(ablation.round(4), interactive=False)

    with gr.Tab("Data distributions"):
        gr.Plot(target_plot())
        with gr.Row():
            feat = gr.Dropdown(FEATURES, value="alcohol", label="Feature")
            by = gr.Radio(["overall", "premium vs not", "quality score"], value="premium vs not", label="Split by")
        fplot = gr.Plot(feature_plot("alcohol", "premium vs not"))
        feat.change(feature_plot, [feat, by], fplot)
        by.change(feature_plot, [feat, by], fplot)

    with gr.Tab("Score a batch"):
        gr.Markdown("Enter a lab analysis; defaults are dataset medians.")
        color = gr.Radio(["white", "red"], value="white", label="Wine colour")
        nums = []
        with gr.Row():
            for f in FEATURES[:6]:
                nums.append(gr.Number(value=round(float(medians[f]), 4), label=f))
        with gr.Row():
            for f in FEATURES[6:]:
                nums.append(gr.Number(value=round(float(medians[f]), 4), label=f))
        btn = gr.Button("Score", variant="primary")
        score_df = gr.Dataframe(interactive=False)
        score_md = gr.Markdown()
        btn.click(score_batch, [color] + nums, [score_df, score_md])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
