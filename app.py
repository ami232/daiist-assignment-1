"""
Assignment 1 — Gradio dashboard.

Loads what train.ipynb saved in artifacts/ (preprocessors, the three trained
models of each version, costs and threshold) and never trains anything. At
startup it scores the test set with the saved models; every view below is
computed from those predictions.
"""

import json
import os
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline import (CATEGORIES, COUNT_COLS, LogisticRegressionNet, NUMERIC_COLS, TARGET,
                      outcome_values)

ARTIFACTS = Path("artifacts")
DATA_PATH = Path("data/Car_Insurance_Claim.csv")

VERSIONS = ["full", "legal", "strict"]
VERSION_LABELS = {
    "full": "Full (illegal: uses gender and race)",
    "legal": "Legal (no gender, no race)",
    "strict": "Strict (also no postal code, no credit score)",
}
IMPLEMENTATIONS = ["scikit-learn", "PyTorch manual", "PyTorch nn.Module"]

# Chart colours: fixed categorical order, the same implementation always gets the same colour
IMPL_COLORS = dict(zip(IMPLEMENTATIONS, ["#2a78d6", "#eb6834", "#1baf7a"]))
# The three implementations agree, so their curves land on the same points. Drawing them
# from thickest to thinnest keeps all three visible: nested rings and stacked lines.
IMPL_LINE_WIDTH = dict(zip(IMPLEMENTATIONS, [7, 4, 1.5]))
IMPL_MARKER_SIZE = dict(zip(IMPLEMENTATIONS, [18, 11, 5]))
OVERLAP_NOTE = "<br><sup>The lines overlap because the three models agree; click a legend entry to hide it.</sup>"
GROUP_COLORS = ["#2a78d6", "#eb6834"]
INK = "#898781"
GRID = "rgba(137, 135, 129, 0.25)"


# ---------------------------------------------------------------- loading

def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in ["VEHICLE_OWNERSHIP", "MARRIED", "CHILDREN", TARGET]:
        df[col] = df[col].astype(int)
    split = pd.read_csv(ARTIFACTS / "split.csv")
    df = df.merge(split, on="ID", validate="one_to_one")
    return df[df["split"] == "train"], df[df["split"] == "test"]


def load_version(name):
    folder = ARTIFACTS / name
    prep = joblib.load(folder / "preprocessor.joblib")
    manual = torch.load(folder / "manual.pt")
    net = LogisticRegressionNet(len(prep.feature_names_))
    net.load_state_dict(torch.load(folder / "standard.pt"))
    net.eval()
    return {"prep": prep, "sklearn": joblib.load(folder / "sklearn.joblib"),
            "w": manual["w"], "b": manual["b"], "net": net}


def predict(model, df):
    """Claim probability from each of the three implementations."""
    X = model["prep"].transform(df)
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        manual = torch.sigmoid(X_t @ model["w"] + model["b"]).numpy()
        standard = model["net"](X_t).numpy()
    return {"scikit-learn": model["sklearn"].predict_proba(X)[:, 1],
            "PyTorch manual": manual, "PyTorch nn.Module": standard}


def coefficients(model):
    """Intercept and weights of the three implementations, side by side."""
    sk, net = model["sklearn"], model["net"]
    return pd.DataFrame(
        {"scikit-learn": np.r_[sk.intercept_, sk.coef_[0]],
         "PyTorch manual": np.r_[model["b"].numpy(), model["w"].numpy()],
         "PyTorch nn.Module": np.r_[net.linear.bias.detach().numpy(),
                                    net.linear.weight.detach().numpy()[0]]},
        index=["(intercept)"] + model["prep"].feature_names_,
    )


META = json.loads((ARTIFACTS / "meta.json").read_text())
VALUES = META["costs"]["values"]
T_STAR = META["threshold"]
TRAIN, TEST = load_data()
Y_TEST = TEST[TARGET].to_numpy()
MODELS = {name: load_version(name) for name in VERSIONS}
TEST_PROBS = {name: predict(model, TEST) for name, model in MODELS.items()}
THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)


# ---------------------------------------------------------------- helpers

def style(fig, title, x_title, y_title, height=420):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        height=height, margin=dict(l=60, r=20, t=85, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK),
        legend=dict(orientation="h", y=-0.2, x=0),
        hoverlabel=dict(font_family="system-ui, -apple-system, Segoe UI, sans-serif"),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def profit_curve(p):
    """Profit per applicant at every threshold in THRESHOLDS."""
    return np.array([outcome_values(Y_TEST, p, t, VALUES).mean() for t in THRESHOLDS])


def euro(x):
    return f"€{x:,.0f}" if x >= 0 else f"−€{-x:,.0f}"


# ---------------------------------------------------------------- tab: three models

def metrics_table(version):
    rows = [{"model": "naive baseline (training claim rate)",
             **_metrics(np.full(len(Y_TEST), META["train_claim_rate"]))}]
    rows += [{"model": impl, **_metrics(TEST_PROBS[version][impl])} for impl in IMPLEMENTATIONS]
    return pd.DataFrame(rows).round(4)


def _metrics(p):
    return {"log loss": log_loss(Y_TEST, p), "Brier": brier_score_loss(Y_TEST, p),
            "ROC-AUC": roc_auc_score(Y_TEST, p), "expected claims": p.sum(),
            "actual claims": int(Y_TEST.sum())}


def reliability_plot(version):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(color=INK, width=1), hoverinfo="skip"))
    bins = np.linspace(0, 1, 11)
    for impl in IMPLEMENTATIONS:
        p = TEST_PROBS[version][impl]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        table = pd.DataFrame({"p": p, "y": Y_TEST, "bin": idx}).groupby("bin").agg(
            predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(
            x=table["predicted"], y=table["observed"], mode="lines+markers", name=impl,
            line=dict(color=IMPL_COLORS[impl], width=IMPL_LINE_WIDTH[impl]),
            marker=dict(size=IMPL_MARKER_SIZE[impl], line=dict(width=0)),
            customdata=table["n"],
            hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>%{customdata} applicants"))
    return style(fig, "Predicted vs. actual claim rate (test set, 10 bins)" + OVERLAP_NOTE,
                 "mean predicted probability", "observed claim rate")


def agreement_plot(version):
    base = TEST_PROBS[version]["scikit-learn"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical",
                             line=dict(color=INK, width=1), hoverinfo="skip"))
    for impl in IMPLEMENTATIONS[1:]:
        fig.add_trace(go.Scattergl(
            x=base, y=TEST_PROBS[version][impl], mode="markers", name=impl,
            marker=dict(color=IMPL_COLORS[impl], size=IMPL_MARKER_SIZE[impl] - 2),
            hovertemplate="scikit-learn %{x:.4f}<br>" + impl + " %{y:.4f}"))
    gap = max(np.abs(TEST_PROBS[version][i] - base).max() for i in IMPLEMENTATIONS[1:])
    return style(fig, f"Each test applicant: PyTorch vs. scikit-learn (largest gap {gap:.5f})"
                 "<br><sup>Every point lies on the diagonal: both PyTorch versions give the same "
                 "probabilities as scikit-learn.</sup>",
                 "scikit-learn probability", "PyTorch probability")


def prediction_histogram(version, implementation):
    p = TEST_PROBS[version][implementation]
    fig = go.Figure()
    for outcome, label, color in [(0, "did not claim", GROUP_COLORS[0]), (1, "claimed", GROUP_COLORS[1])]:
        fig.add_trace(go.Histogram(x=p[Y_TEST == outcome], name=label, marker_color=color,
                                   xbins=dict(start=0, end=1, size=0.05), opacity=0.75))
    fig.add_vline(x=T_STAR, line=dict(color=INK, width=1),
                  annotation_text=f"t* = {T_STAR:.3f}", annotation_position="top")
    fig.update_layout(barmode="overlay", bargap=0.05)
    return style(fig, f"Predicted probability by actual outcome ({implementation}, test set)",
                 "predicted claim probability", "applicants")


def coefficient_plot(version):
    coef = coefficients(MODELS[version]).drop(index="(intercept)")
    fig = go.Figure()
    for impl in IMPLEMENTATIONS:
        fig.add_trace(go.Bar(y=coef.index, x=coef[impl], name=impl, orientation="h",
                             marker_color=IMPL_COLORS[impl],
                             hovertemplate="%{y}<br>%{x:.4f}<extra>" + impl + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.1)
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Coefficients on standardised features", "coefficient", "",
                 height=max(420, 26 * len(coef) + 120))


def loss_plot(version):
    fig = go.Figure()
    for impl, history in META["loss_histories"][version].items():
        fig.add_trace(go.Scatter(x=np.arange(1, len(history) + 1), y=history, mode="lines",
                                 name=f"{impl}, lr = {META['learning_rates'][impl]}",
                                 line=dict(color=IMPL_COLORS[impl], width=2)))
    fig.update_xaxes(type="log")
    return style(fig, "Training loss (BCE + L2) of the two PyTorch versions", "epoch", "loss")


def update_models_tab(version, implementation):
    return (metrics_table(version), reliability_plot(version), agreement_plot(version),
            prediction_histogram(version, implementation), coefficient_plot(version),
            loss_plot(version))


# ---------------------------------------------------------------- tab: data

FEATURES = list(CATEGORIES) + ["VEHICLE_OWNERSHIP", "MARRIED", "CHILDREN"] + COUNT_COLS + NUMERIC_COLS


def target_summary():
    rows = []
    for name, part in [("train", TRAIN), ("test", TEST)]:
        rows.append({"split": name, "applicants": len(part), "claims": int(part[TARGET].sum()),
                     "claim rate": round(part[TARGET].mean(), 3)})
    return pd.DataFrame(rows)


def feature_plots(feature):
    data = TRAIN
    labels = {0: "did not claim", 1: "claimed"}
    dist = go.Figure()
    if feature in CATEGORIES or data[feature].nunique() <= 2:
        levels = CATEGORIES.get(feature, sorted(data[feature].unique()))
        for outcome in [0, 1]:
            counts = data[data[TARGET] == outcome][feature].value_counts().reindex(levels, fill_value=0)
            dist.add_trace(go.Bar(x=[str(l) for l in levels], y=counts.values, name=labels[outcome],
                                  marker_color=GROUP_COLORS[outcome]))
        dist.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
        rate = data.groupby(feature)[TARGET].agg(["mean", "size"]).reindex(levels)
        x_values = [str(l) for l in levels]
    else:
        for outcome in [0, 1]:
            dist.add_trace(go.Histogram(x=data[data[TARGET] == outcome][feature].dropna(),
                                        name=labels[outcome], marker_color=GROUP_COLORS[outcome],
                                        opacity=0.75, nbinsx=25))
        dist.update_layout(barmode="overlay", bargap=0.05)
        binned = pd.cut(data[feature], bins=10)
        rate = data.groupby(binned, observed=True)[TARGET].agg(["mean", "size"])
        x_values = [f"{iv.left:,.2f}–{iv.right:,.2f}" for iv in rate.index]
    missing = data[feature].isna().mean()
    note = f" ({missing:.0%} missing, not shown)" if missing else ""
    style(dist, f"{feature}: distribution in the training set{note}", feature, "applicants")

    rate_fig = go.Figure(go.Bar(x=x_values, y=rate["mean"], marker_color=GROUP_COLORS[0],
                                customdata=rate["size"], name="claim rate",
                                hovertemplate="%{x}<br>claim rate %{y:.1%}<br>%{customdata} applicants<extra></extra>"))
    rate_fig.add_hline(y=data[TARGET].mean(), line=dict(color=INK, width=1),
                       annotation_text="overall", annotation_position="top right")
    rate_fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    rate_fig.update_layout(bargap=0.3)
    style(rate_fig, f"Claim rate by {feature} (training set)", feature, "claim rate")
    return dist, rate_fig


# ---------------------------------------------------------------- tab: threshold

def threshold_view(version, implementation, threshold):
    p = TEST_PROBS[version][implementation]
    declined = p >= threshold
    claimed = Y_TEST == 1
    cells = {"TP": (declined & claimed).sum(), "FP": (declined & ~claimed).sum(),
             "FN": (~declined & claimed).sum(), "TN": (~declined & ~claimed).sum()}
    total = outcome_values(Y_TEST, p, threshold, VALUES).sum()
    accept_all = outcome_values(Y_TEST, np.zeros_like(p), threshold, VALUES).sum()
    n = len(Y_TEST)

    summary = (
        f"### {euro(total / n)} profit per applicant\n"
        f"**{euro(total)}** on the {n:,} test applicants, against **{euro(accept_all)}** "
        f"if everyone were accepted at the standard premium.\n\n"
        f"- Declined: **{declined.mean():.1%}** of applicants\n"
        f"- Claimers accepted anyway (false negatives): **{cells['FN']}**, "
        f"each a net loss of {euro(-VALUES['FN'])}\n"
        f"- Non-claimers turned away (false positives): **{cells['FP']}**, "
        f"each a lost profit of {euro(VALUES['TN'])}\n\n"
        f"Cost-based threshold t* = {T_STAR:.3f}. Standard premium "
        f"{euro(META['costs']['standard_premium'])}, average claim {euro(META['costs']['avg_claim_cost'])}."
    )

    z = [[cells["TN"], cells["FP"]], [cells["FN"], cells["TP"]]]
    money = [[cells["TN"] * VALUES["TN"], cells["FP"] * VALUES["FP"]],
             [cells["FN"] * VALUES["FN"], cells["TP"] * VALUES["TP"]]]
    text = [[f"{z[r][c]}<br>{euro(money[r][c])}" for c in range(2)] for r in range(2)]
    matrix = go.Figure(go.Heatmap(
        z=z, x=["accepted", "declined"], y=["did not claim", "claimed"], text=text,
        texttemplate="%{text}", textfont=dict(size=16), showscale=False,
        colorscale=[[0, "#cde2fb"], [1, "#2a78d6"]], xgap=2, ygap=2,
        hovertemplate="%{y}, %{x}: %{z} applicants<extra></extra>"))
    matrix.update_yaxes(autorange="reversed")
    style(matrix, f"Confusion matrix at threshold {threshold:.2f} (test set)", "decision", "actual",
          height=380)
    matrix.update_xaxes(showgrid=False)
    matrix.update_yaxes(showgrid=False)

    curve = go.Figure(go.Scatter(x=THRESHOLDS, y=profit_curve(p), mode="lines", name=implementation,
                                 line=dict(color=IMPL_COLORS[implementation], width=2),
                                 hovertemplate="threshold %{x:.2f}<br>€%{y:,.0f} per applicant<extra></extra>"))
    curve.add_vline(x=T_STAR, line=dict(color=INK, width=1), annotation_text="t*",
                    annotation_position="bottom right")
    curve.add_vline(x=threshold, line=dict(color="#52514e", width=2), annotation_text="selected",
                    annotation_position="top left")
    style(curve, "Profit per applicant vs. threshold (test set)", "threshold (decline when p ≥ threshold)",
          "EUR per applicant", height=380)
    return summary, matrix, curve


# ---------------------------------------------------------------- tab: legal vs. illegal

def versions_table(threshold):
    rows = []
    legal_profit = outcome_values(Y_TEST, TEST_PROBS["legal"]["scikit-learn"], threshold, VALUES).mean()
    for name in VERSIONS:
        p = TEST_PROBS[name]["scikit-learn"]
        profit = outcome_values(Y_TEST, p, threshold, VALUES).mean()
        rows.append({"version": VERSION_LABELS[name], "features": len(MODELS[name]["prep"].feature_names_),
                     "ROC-AUC": round(roc_auc_score(Y_TEST, p), 4),
                     "log loss": round(log_loss(Y_TEST, p), 4),
                     "declined": f"{(p >= threshold).mean():.1%}",
                     "profit per applicant": euro(profit),
                     "vs. legal, per 10,000 applicants": euro((profit - legal_profit) * 10_000)})
    return pd.DataFrame(rows)


def group_plot(attribute, threshold):
    groups = CATEGORIES[attribute]
    x = ["actual claim rate"] + [f"declined: {name}" for name in VERSIONS]
    fig = go.Figure()
    for i, group in enumerate(groups):
        mask = (TEST[attribute] == group).to_numpy()
        values = [Y_TEST[mask].mean()] + [(TEST_PROBS[name]["scikit-learn"][mask] >= threshold).mean()
                                          for name in VERSIONS]
        fig.add_trace(go.Bar(x=x, y=values, name=f"{group} ({mask.sum()} applicants)",
                             marker_color=GROUP_COLORS[i],
                             hovertemplate="%{x}<br>%{y:.1%}<extra>" + group + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.7])
    return style(fig, f"{attribute.lower().capitalize()}: claim rate and share declined (test set, scikit-learn)",
                 "", "share of the group")


def gains_table():
    rows = [{"comparison": g["comparison"], "gain from": g["gain from"],
             "per applicant": euro(g["per applicant (EUR)"]),
             "95% interval": f"{euro(g['95% low'])} to {euro(g['95% high'])}"} for g in META["gains"]]
    return pd.DataFrame(rows)


def proxy_table():
    table = pd.DataFrame(META["proxy_auc"]).round(3)
    table.index.name = "banned attribute"
    return table.rename(columns={"legal": "from Legal columns", "strict": "from Strict columns"}).reset_index()


def update_versions_tab(threshold):
    return versions_table(threshold), group_plot("GENDER", threshold), group_plot("RACE", threshold)


# ---------------------------------------------------------------- tab: score an applicant

def score_applicant(age, gender, race, experience, education, income, credit, ownership, vehicle_year,
                    married, children, postal, mileage, vehicle_type, speeding, duis, accidents):
    applicant = pd.DataFrame([{
        "AGE": age, "GENDER": gender, "RACE": race, "DRIVING_EXPERIENCE": experience,
        "EDUCATION": education, "INCOME": income, "CREDIT_SCORE": credit,
        "VEHICLE_OWNERSHIP": int(ownership), "VEHICLE_YEAR": vehicle_year, "MARRIED": int(married),
        "CHILDREN": int(children), "POSTAL_CODE": int(postal),
        # An empty mileage box is treated as missing and filled like in training
        "ANNUAL_MILEAGE": float(mileage) if mileage is not None else np.nan,
        "VEHICLE_TYPE": vehicle_type, "SPEEDING_VIOLATIONS": int(speeding or 0),
        "DUIS": int(duis or 0), "PAST_ACCIDENTS": int(accidents or 0),
    }])
    rows = []
    for name in VERSIONS:
        probs = predict(MODELS[name], applicant)
        row = {"version": VERSION_LABELS[name]}
        row.update({impl: f"{probs[impl][0]:.1%}" for impl in IMPLEMENTATIONS})
        row["decision at t*"] = "decline" if probs["scikit-learn"][0] >= T_STAR else "accept"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- layout

with gr.Blocks(title="Car insurance claim risk") as demo:
    gr.Markdown(
        "# Car insurance claim risk\n"
        "A car insurer accepts each applicant at a standard premium or declines them. "
        "Logistic regression estimates the probability of a claim, trained three ways "
        "(scikit-learn, PyTorch from scratch, PyTorch `nn.Module` + `torch.optim`), in three "
        "versions that differ in which columns they may use. Everything shown is computed "
        "from the models saved by `train.ipynb`; nothing is retrained here."
    )

    with gr.Tab("Three implementations"):
        with gr.Row():
            version_in = gr.Radio([(VERSION_LABELS[v], v) for v in VERSIONS], value="legal", label="Version")
            impl_in = gr.Dropdown(IMPLEMENTATIONS, value="scikit-learn",
                                  label="Implementation (prediction histogram)")
        metrics_out = gr.Dataframe(label="Test-set metrics", interactive=False)
        with gr.Row():
            reliability_out = gr.Plot()
            agreement_out = gr.Plot()
        histogram_out = gr.Plot()
        with gr.Row():
            coef_out = gr.Plot()
            loss_out = gr.Plot()
        models_outputs = [metrics_out, reliability_out, agreement_out, histogram_out, coef_out, loss_out]
        for control in (version_in, impl_in):
            control.change(update_models_tab, [version_in, impl_in], models_outputs)

    with gr.Tab("Data"):
        gr.Dataframe(target_summary(), label="Target: claims per split", interactive=False)
        feature_in = gr.Dropdown(FEATURES, value="DRIVING_EXPERIENCE", label="Feature")
        with gr.Row():
            dist_out = gr.Plot()
            rate_out = gr.Plot()
        feature_in.change(feature_plots, feature_in, [dist_out, rate_out])

    with gr.Tab("Threshold and business cost"):
        with gr.Row():
            t_version_in = gr.Radio([(VERSION_LABELS[v], v) for v in VERSIONS], value="legal", label="Version")
            t_impl_in = gr.Dropdown(IMPLEMENTATIONS, value="scikit-learn", label="Implementation")
            threshold_in = gr.Slider(0.01, 0.99, value=round(T_STAR, 3), step=0.001,
                                     label="Decline when the claim probability is at least")
        with gr.Row():
            with gr.Column(scale=2):
                summary_out = gr.Markdown()
            with gr.Column(scale=3):
                matrix_out = gr.Plot()
        curve_out = gr.Plot()
        for control in (t_version_in, t_impl_in, threshold_in):
            control.change(threshold_view, [t_version_in, t_impl_in, threshold_in],
                           [summary_out, matrix_out, curve_out])

    with gr.Tab("Legal vs. illegal"):
        gr.Markdown(
            "**What would the insurer gain from the banned columns, and does the ban change who is "
            "declined?** Full uses gender and race, illegal in the EU (race everywhere; gender since "
            "the Test-Achats ruling, applied from 21 December 2012). Legal drops them. Strict also "
            "drops postal code and credit score."
        )
        v_threshold_in = gr.Slider(0.01, 0.99, value=round(T_STAR, 3), step=0.001,
                                   label="Decline when the claim probability is at least")
        versions_out = gr.Dataframe(label="Test set, scikit-learn implementation", interactive=False)
        with gr.Row():
            gender_out = gr.Plot()
            race_out = gr.Plot()
        with gr.Row():
            gr.Dataframe(gains_table(), interactive=False,
                         label="Profit gain at t*, with 95% bootstrap interval (from train.ipynb)")
            gr.Dataframe(proxy_table(), interactive=False,
                         label="How well the remaining columns predict the banned attribute (ROC-AUC, 0.5 = not at all)")
        v_threshold_in.change(update_versions_tab, v_threshold_in, [versions_out, gender_out, race_out])

    with gr.Tab("Score an applicant"):
        gr.Markdown("Claim probability for one applicant, from all nine saved models.")
        with gr.Row():
            age = gr.Dropdown(CATEGORIES["AGE"], value="26-39", label="Age")
            gender = gr.Radio(CATEGORIES["GENDER"], value="male", label="Gender")
            race = gr.Radio(CATEGORIES["RACE"], value="majority", label="Race")
            experience = gr.Dropdown(CATEGORIES["DRIVING_EXPERIENCE"], value="0-9y", label="Driving experience")
        with gr.Row():
            education = gr.Dropdown(CATEGORIES["EDUCATION"], value="high school", label="Education")
            income = gr.Dropdown(CATEGORIES["INCOME"], value="middle class", label="Income")
            credit = gr.Slider(0, 1, value=0.5, step=0.01, label="Credit score")
            postal = gr.Dropdown([str(c) for c in CATEGORIES["POSTAL_CODE"]], value="10238", label="Postal code")
        with gr.Row():
            ownership = gr.Checkbox(value=True, label="Owns the vehicle")
            married = gr.Checkbox(value=False, label="Married")
            children = gr.Checkbox(value=False, label="Has children")
            vehicle_year = gr.Radio(CATEGORIES["VEHICLE_YEAR"], value="after 2015", label="Vehicle year")
            vehicle_type = gr.Radio(CATEGORIES["VEHICLE_TYPE"], value="sedan", label="Vehicle type")
        with gr.Row():
            mileage = gr.Number(value=12000, label="Annual mileage")
            speeding = gr.Number(value=0, precision=0, minimum=0, label="Speeding violations")
            duis = gr.Number(value=0, precision=0, minimum=0, label="DUIs")
            accidents = gr.Number(value=0, precision=0, minimum=0, label="Past accidents")
        score_btn = gr.Button("Score applicant", variant="primary")
        score_out = gr.Dataframe(interactive=False, label="Claim probability")
        score_btn.click(score_applicant,
                        [age, gender, race, experience, education, income, credit, ownership, vehicle_year,
                         married, children, postal, mileage, vehicle_type, speeding, duis, accidents],
                        score_out)

    # Fill every view once when the page opens
    demo.load(update_models_tab, [version_in, impl_in], models_outputs)
    demo.load(feature_plots, feature_in, [dist_out, rate_out])
    demo.load(threshold_view, [t_version_in, t_impl_in, threshold_in], [summary_out, matrix_out, curve_out])
    demo.load(update_versions_tab, v_threshold_in, [versions_out, gender_out, race_out])


if __name__ == "__main__":
    # Open a browser tab locally; CI (which sets CI=true) only needs the server to respond
    demo.launch(inbrowser=not os.environ.get("CI"))
