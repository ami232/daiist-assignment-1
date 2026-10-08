"""
Madrid bargain finder — Gradio dashboard.

Loads everything produced by `uv run python main.py train` (the train.ipynb pipeline) from
artifacts/ and never retrains anything. Tabs:

0. Home hub           — find bargains (filter + sort), then see why one is a bargain, feature by feature.
1. Model comparison   — prediction vs. actual for the three implementations + baseline, metrics.
2. Distributions      — target and feature distributions across train / val / test.
3. Bargain finder     — margin-of-safety slider: which listings count as bargains, and how many.
4. What drives price  — the model's coefficients as % effects on the asking price.
5. Training           — learning curves and how closely the three methods' weights agree.

Run with:  uv run python main.py app
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

ARTIFACTS = Path("artifacts")
SEED = 42
MAX_POINTS = 5000  # scatter plots sample this many flats so the browser stays fast

MODEL_LABELS = {
    "sklearn": "scikit-learn",
    "manual_pytorch": "Manual PyTorch",
    "standard_pytorch": "Standard PyTorch",
    "baseline": "Naive baseline",
}
COLORS = {"sklearn": "#1f77b4", "manual_pytorch": "#ff7f0e",
          "standard_pytorch": "#2ca02c", "baseline": "#7f7f7f"}


# --------------------------------------------------------------------------- loading artifacts
def load_artifacts():
    """Load the saved pipeline outputs. Fails with a clear message if training hasn't been run."""
    required = ["metrics.json", "predictions.csv", "learning_curves.csv", "pipeline.joblib", "eval_flats.csv",
                "models/sklearn_linear.joblib", "models/manual_pytorch.pt", "models/standard_pytorch.pt"]
    missing = [f for f in required if not (ARTIFACTS / f).exists()]
    if missing:
        raise SystemExit(f"Missing artifacts {missing}. Run `uv run python main.py train` first.")

    pipeline = joblib.load(ARTIFACTS / "pipeline.joblib")
    features = pipeline["features"]

    # The three trained models, loaded from disk (weights only, no training here)
    sk = joblib.load(ARTIFACTS / "models" / "sklearn_linear.joblib")
    manual = torch.load(ARTIFACTS / "models" / "manual_pytorch.pt")
    standard = torch.nn.Linear(len(features), 1)
    state = torch.load(ARTIFACTS / "models" / "standard_pytorch.pt")
    standard.load_state_dict({k.replace("linear.", ""): v for k, v in state.items()})

    coefs = pd.DataFrame({
        "sklearn": np.r_[sk.intercept_, sk.coef_],
        "manual_pytorch": np.r_[manual["b"].numpy(), manual["w"].numpy()],
        "standard_pytorch": np.r_[standard.bias.detach().numpy(),
                                  standard.weight.detach().numpy().ravel()],
    }, index=["(intercept)"] + features)

    return {
        "metrics": json.loads((ARTIFACTS / "metrics.json").read_text()),
        "preds": pd.read_csv(ARTIFACTS / "predictions.csv"),
        "curves": pd.read_csv(ARTIFACTS / "learning_curves.csv"),
        "pipeline": pipeline,
        "coefs": coefs,
        "sk": sk,
        "eval": pd.read_csv(ARTIFACTS / "eval_flats.csv"),
    }


A = load_artifacts()
PREDS, METRICS, PIPE, COEFS = A["preds"], A["metrics"], A["pipeline"], A["coefs"]


def euros(x):
    return f"€{x:,.0f}"


def sample(df, n=MAX_POINTS):
    return df.sample(n, random_state=SEED) if len(df) > n else df


def metrics_table(split):
    rows = METRICS["validation" if split == "val" else "test"]
    return pd.DataFrame([
        {"Model": MODEL_LABELS[m],
         "RMSE (log price)": round(v["RMSE_log"], 4),
         "Typical error": f"{v['median_abs_pct_error']:.1%}",
         "R²": round(v["R2"], 3)}
        for m, v in rows.items()
    ])


# --------------------------------------------------------------------------- tab 0: home hub
# Estimates are computed live with the loaded scikit-learn model on the saved feature table.
# Because the model is linear, each estimate splits exactly into "typical flat" + one push per
# feature: push_i = weight_i × (this flat's value_i − typical value_i), all on the log scale.
FEATURES = PIPE["features"]
EVAL = A["eval"].copy()


def scale_features(X):
    X = X.copy()
    cols = PIPE["scaler"]["columns"]
    X[cols] = (X[cols] - PIPE["scaler"]["mean"]) / PIPE["scaler"]["std"]
    return X[FEATURES].to_numpy(dtype=np.float64)


XS_EVAL = scale_features(EVAL[[f"f_{c}" for c in FEATURES]].set_axis(FEATURES, axis=1))
XS_TYPICAL = scale_features(PIPE["train_feature_means"].to_frame().T)[0]
WEIGHTS = A["sk"].coef_
EVAL["log_estimate"] = A["sk"].predict(XS_EVAL)
EVAL["estimate"] = np.exp(EVAL["log_estimate"])
EVAL["vs_estimate"] = EVAL["PRICE"] / EVAL["estimate"] - 1
EVAL["price_m2"] = EVAL["PRICE"] / EVAL["CONSTRUCTEDAREA"]
EVAL = EVAL.merge(PREDS[["ASSETID", "zone", "pred_sklearn"]], on="ASSETID", how="left")
assert np.allclose(EVAL["log_estimate"], EVAL["pred_sklearn"], atol=1e-6), "Saved and live predictions differ"
EVAL = EVAL.set_index("ASSETID", drop=False)
PUSHES = pd.DataFrame((XS_EVAL - XS_TYPICAL) * WEIGHTS, columns=FEATURES, index=EVAL.index)
TYPICAL_PRICE = float(np.exp(A["sk"].intercept_ + XS_TYPICAL @ WEIGHTS))

train_preds = PREDS[PREDS["split"] == "train"]
ZONE_PRICE_M2 = (train_preds["PRICE"] / train_preds["CONSTRUCTEDAREA"]).groupby(train_preds["zone"]).median()

GROUPS = {
    "Location": [f for f in FEATURES if f.startswith(("ZONE_", "DISTANCE_"))],
    "Size and layout": [f for f in FEATURES if f in ("LOG_AREA", "ROOMNUMBER", "BATHNUMBER", "ISDUPLEX", "ISSTUDIO")],
    "Building and age": [f for f in FEATURES if f.startswith(("AGE_", "QUALITY_"))
                         or f in ("CADMAXBUILDINGFLOOR", "LOG_DWELLINGS", "BUILTTYPEID_1", "BUILTTYPEID_2")],
    "Floor and position": [f for f in FEATURES if f in ("FLOOR", "FLOOR_MISSING", "FLOOR_X_LIFT", "IS_EXTERIOR",
                                                        "EXTERIOR_MISSING", "ISINTOPFLOOR")],
}
_grouped = {f for fs in GROUPS.values() for f in fs}
GROUPS["Amenities and extras"] = [f for f in FEATURES if f not in _grouped]

SORTS = {
    "Biggest discount": ("vs_estimate", True),
    "Lowest price": ("PRICE", True),
    "Highest price": ("PRICE", False),
    "Lowest €/m²": ("price_m2", True),
    "Largest size": ("CONSTRUCTEDAREA", False),
    "Closest to centre": ("DISTANCE_TO_CITY_CENTER", True),
}
GOOD, BAD = "#2e7d32", "#c62828"
CARD = ("background:var(--block-background-fill); border:1px solid var(--border-color-primary); "
        "border-radius:12px; padding:14px 18px;")
METRIC = "background:var(--background-fill-secondary); border-radius:8px; padding:12px 14px;"
MUTED = "color:var(--body-text-color-subdued);"


def find_bargains(split, sort_by, max_price, min_beds, max_km, need_lift, exterior_only, no_restore, margin_pct):
    margin, cap = margin_pct / 100, PIPE["max_discount"]
    d = EVAL[EVAL["split"] == split]
    d = d[(d["vs_estimate"] <= -margin) & (d["vs_estimate"] > -cap)]
    d = d[(d["PRICE"] <= max_price * 1000) & (d["ROOMNUMBER"] >= min_beds) & (d["DISTANCE_TO_CITY_CENTER"] <= max_km)]
    if need_lift:
        d = d[d["HASLIFT"] == 1]
    if exterior_only:
        d = d[d["FLATLOCATIONID"] == 1]
    if no_restore:
        d = d[d["BUILTTYPEID_2"] == 0]
    col, ascending = SORTS[sort_by]
    d = d.sort_values(col, ascending=ascending)

    ids = d["ASSETID"].tolist()
    count = (f"**{len(d):,} bargains** match your filters "
             f"(at least {margin:.0%} below the estimate, not more than {cap:.0%} below).")
    table = pd.DataFrame({
        "#": range(1, len(d) + 1),
        "Listing price": d["PRICE"].map(euros),
        "Estimate": d["estimate"].map(euros),
        "Below": (-d["vs_estimate"]).map(lambda v: f"{v:.0%}"),
        "m²": d["CONSTRUCTEDAREA"].astype(int),
        "Beds": d["ROOMNUMBER"].astype(int),
        "€/m²": d["price_m2"].map(lambda v: f"€{v:,.0f}"),
        "Km to centre": d["DISTANCE_TO_CITY_CENTER"].round(1),
    }).head(200).reset_index(drop=True)
    choices = [(f"#{i} · {int(r.CONSTRUCTEDAREA)} m², {int(r.ROOMNUMBER)} bed · {euros(r.PRICE)} · "
                f"{-r.vs_estimate:.0%} below", r.ASSETID)
               for i, r in enumerate(d.head(200).itertuples(), 1)]
    first = ids[0] if ids else None
    return count, table, gr.Dropdown(choices=choices, value=first), ids[:200]


def floor_text(r):
    if pd.isna(r["FLOORCLEAN"]):
        floor = "Floor unknown"
    else:
        f = int(r["FLOORCLEAN"])
        floor = {-1: "Basement", 0: "Ground floor"}.get(f, f"Floor {f}")
    side = {1: "exterior", 2: "interior"}.get(r["FLATLOCATIONID"], "side unknown")
    parts = [floor, side, "lift" if r["HASLIFT"] == 1 else "no lift"]
    if r["ISINTOPFLOOR"] == 1:
        parts.append("top floor")
    return ", ".join(parts)


def flat_html(asset_id, margin_pct):
    if not asset_id or asset_id not in EVAL.index:
        return "<p>No bargains match these filters. Try widening them.</p>"
    r = EVAL.loc[asset_id]
    margin = margin_pct / 100
    below = -r["vs_estimate"]
    zone_m2 = ZONE_PRICE_M2.get(r["zone"], np.nan)

    extras = [name for col, name in [("HASTERRACE", "terrace"), ("HASAIRCONDITIONING", "AC"),
              ("HASPARKINGSPACE", "parking"), ("HASDOORMAN", "doorman"), ("HASSWIMMINGPOOL", "pool"),
              ("HASGARDEN", "garden"), ("HASBOXROOM", "storage room"), ("HASWARDROBE", "built-in wardrobes")]
              if r[col] == 1]
    condition = ("New build" if r["BUILTTYPEID_1"] == 1 else
                 "Needs restoration" if r["BUILTTYPEID_2"] == 1 else "Second-hand, good condition")
    rows = [
        ("Size", f"{int(r['CONSTRUCTEDAREA'])} m² · {int(r['ROOMNUMBER'])} bed · {int(r['BATHNUMBER'])} bath"),
        ("Floor", floor_text(r)),
        ("Extras", (", ".join(extras)[:1].upper() + ", ".join(extras)[1:]) if extras else "None listed"),
        ("Building", f"Built {int(r['CADCONSTRUCTIONYEAR'])}, cadastral quality "
                     f"{'unknown' if pd.isna(r['CADASTRALQUALITYID']) else int(r['CADASTRALQUALITYID'])} (0 = best)"),
        ("Condition", condition),
        ("Location", f"Zone {int(r['zone'])} · {r['DISTANCE_TO_CITY_CENTER']:.1f} km to centre"),
        ("Metro", f"{r['DISTANCE_TO_METRO'] * 1000:,.0f} m"),
        ("€/m²", f"€{r['price_m2']:,.0f} vs zone €{zone_m2:,.0f}"),
    ]
    flat_rows = "".join(f"<tr><td style='{MUTED} padding:3px 0;'>{k}</td>"
                        f"<td style='text-align:right; padding:3px 0;'>{v}</td></tr>" for k, v in rows)

    group_pct = {g: float(np.exp(PUSHES.loc[asset_id, fs].sum()) - 1) for g, fs in GROUPS.items()}
    biggest = max(abs(v) for v in group_pct.values()) or 1
    bars = ""
    for g, v in group_pct.items():
        width = abs(v) / biggest * 48
        side = f"left:50%;" if v >= 0 else f"right:50%;"
        color = GOOD if v >= 0 else BAD
        bars += (f"<span>{g}</span><div style='position:relative; height:12px;'>"
                 f"<div style='position:absolute; {side} width:{width:.1f}%; height:12px; background:{color}; "
                 f"border-radius:2px;'></div><div style='position:absolute; left:50%; top:-3px; width:1px; "
                 f"height:18px; background:var(--border-color-primary);'></div></div>"
                 f"<span style='color:{color}; text-align:right;'>{v:+.0%}</span>")

    verdict_ok = below >= margin
    vs_zone = 1 - r["price_m2"] / zone_m2 if zone_m2 == zone_m2 else np.nan
    zone_sentence = (f" and {vs_zone:.0%} under its zone's typical €/m²" if vs_zone > 0 else
                     f", though it's {-vs_zone:.0%} above its zone's typical €/m²") if vs_zone == vs_zone else ""
    checks = ["asking price, not sale price",
              f"the model is typically off by {METRICS['test']['sklearn']['median_abs_pct_error']:.1%}",
              "ask why it's cheap (condition, legal status, occupancy)"]
    if r["BUILTTYPEID_2"] == 1:
        checks.insert(0, "marked as needing restoration, so budget for works")
    if r["HASLIFT"] == 0 and (r["FLOORCLEAN"] or 0) >= 3:
        checks.insert(0, "high floor without a lift")

    return f"""
<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:12px;">
  <div style="{METRIC}"><div style="font-size:13px; {MUTED}">Listing price</div>
    <div style="font-size:24px; font-weight:600;">{euros(r['PRICE'])}</div></div>
  <div style="{METRIC}"><div style="font-size:13px; {MUTED}">Estimated market price</div>
    <div style="font-size:24px; font-weight:600;">{euros(r['estimate'])}</div></div>
  <div style="{METRIC} background:rgba(46,125,50,0.12);"><div style="font-size:13px; color:{GOOD};">Below estimate</div>
    <div style="font-size:24px; font-weight:600; color:{GOOD};">{below:.0%}</div></div>
  <div style="{METRIC}"><div style="font-size:13px; {MUTED}">Verdict</div>
    <div style="font-size:16px; font-weight:600; margin-top:6px;">{'✓ Bargain' if verdict_ok else 'Not a bargain'}</div>
    <div style="font-size:12px; {MUTED}">{'beats' if verdict_ok else 'misses'} {margin:.0%} margin</div></div>
</div>
<div style="display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.3fr); gap:12px;">
  <div style="{CARD}"><div style="font-weight:600; font-size:15px; margin-bottom:8px;">The flat</div>
    <table style="width:100%; font-size:13px; border:none;">{flat_rows}</table>
    <div style="font-size:11px; {MUTED} margin-top:6px;">ID {asset_id}</div></div>
  <div style="{CARD}"><div style="font-weight:600; font-size:15px;">Why the model values it at {euros(r['estimate'])}</div>
    <div style="font-size:12px; {MUTED} margin-bottom:10px;">Starts from a typical Madrid flat ({euros(TYPICAL_PRICE)}),
      then each group of features pushes the estimate up or down</div>
    <div style="font-size:13px; display:grid; grid-template-columns:130px 1fr 48px; gap:8px; align-items:center;">{bars}</div>
    <div style="font-size:13px; {MUTED} margin-top:12px; line-height:1.6;">Comparable flats are listed around
      {euros(r['estimate'])}. This one asks {euros(r['PRICE'])}, {below:.0%} less{zone_sentence}.</div></div>
</div>
<div style="background:rgba(237,108,2,0.12); border-radius:8px; padding:10px 14px; margin-top:12px; font-size:13px;">
  ⚠ Check before buying: {' · '.join(checks)}</div>
"""


def step_flat(current, ids, delta):
    if not ids:
        return None
    i = ids.index(current) if current in ids else 0
    return ids[(i + delta) % len(ids)]


def pick_from_table(ids, evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    return ids[row] if ids and row < len(ids) else None


# --------------------------------------------------------------------------- tab 1: comparison
def prediction_plot(split, models, scale):
    df = sample(PREDS[PREDS["split"] == split])
    fig = go.Figure()
    to_axis = (lambda v: np.exp(v)) if scale == "Price (€)" else (lambda v: v)
    actual = to_axis(df["log_price"])
    for m in models:
        fig.add_trace(go.Scattergl(
            x=actual, y=to_axis(df[f"pred_{m}"]), mode="markers", name=MODEL_LABELS[m],
            marker=dict(size=4, opacity=0.35, color=COLORS[m])))
    lo, hi = actual.min(), actual.max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color="black", dash="dash")))
    log_axes = scale == "Price (€)"
    fig.update_layout(
        title=f"Predicted vs. actual asking price — {split} set ({len(df):,} flats shown)",
        xaxis=dict(title="Actual asking price", type="log" if log_axes else "linear"),
        yaxis=dict(title="Predicted (estimated market price)", type="log" if log_axes else "linear"),
        height=520, legend=dict(orientation="h", y=-0.15))
    return fig


def comparison_update(split, models, scale):
    return prediction_plot(split, models, scale), metrics_table(split)


# --------------------------------------------------------------------------- tab 2: distributions
DIST_VARS = {
    "Asking price (€)": "PRICE",
    "log(asking price)": "log_price",
    "Constructed area (m²)": "CONSTRUCTEDAREA",
    "Bedrooms": "ROOMNUMBER",
    "Bathrooms": "BATHNUMBER",
    "Construction year (cadastre)": "CADCONSTRUCTIONYEAR",
    "Distance to centre (km)": "DISTANCE_TO_CITY_CENTER",
    "Prediction error, scikit-learn (log)": "residual",
}


def distribution_plot(label, splits, clip):
    col = DIST_VARS[label]
    fig = go.Figure()
    for s in splits:
        d = PREDS[PREDS["split"] == s]
        values = d["pred_sklearn"] - d["log_price"] if col == "residual" else d[col]
        if clip:  # hide the most extreme 1% on each side so the bulk is readable
            values = values[values.between(values.quantile(0.01), values.quantile(0.99))]
        fig.add_trace(go.Histogram(x=values, name=f"{s} ({len(d):,})", opacity=0.55,
                                   histnorm="percent", nbinsx=60))
    fig.update_layout(barmode="overlay", title=f"Distribution: {label}",
                      xaxis_title=label, yaxis_title="% of flats in split", height=460)
    return fig


def distribution_summary(label):
    col = DIST_VARS[label]
    out = []
    for s in ["train", "val", "test"]:
        d = PREDS[PREDS["split"] == s]
        v = d["pred_sklearn"] - d["log_price"] if col == "residual" else d[col]
        out.append({"Split": s, "Flats": len(d), "Median": round(v.median(), 3),
                    "Mean": round(v.mean(), 3), "Std": round(v.std(), 3)})
    return pd.DataFrame(out)


def distribution_update(label, splits, clip):
    return distribution_plot(label, splits, clip), distribution_summary(label)


# --------------------------------------------------------------------------- tab 3: bargains
def bargain_update(split, model, margin_pct, cap_pct):
    margin, cap = margin_pct / 100, cap_pct / 100
    d = PREDS[PREDS["split"] == split].copy()
    d["estimate"] = np.exp(d[f"pred_{model}"])
    d["vs_estimate"] = d["PRICE"] / d["estimate"] - 1          # -0.30 = listed 30% below estimate
    d["bargain"] = (d["vs_estimate"] <= -margin) & (d["vs_estimate"] > -cap)
    suspicious = (d["vs_estimate"] <= -cap).sum()
    bargains = d[d["bargain"]]
    gap = (bargains["estimate"] - bargains["PRICE"])

    summary = (
        f"### {len(bargains):,} bargains out of {len(d):,} listings ({len(bargains) / len(d):.1%})\n"
        f"Listed **at least {margin:.0%} below** the estimated market price and **not more than "
        f"{cap:.0%} below**. {suspicious:,} listings fall beyond the cap and are excluded as suspicious.\n\n"
        f"Median discount vs. estimate: **{(-bargains['vs_estimate']).median():.1%}** "
        f"(≈ {euros(gap.median()) if len(bargains) else '€0'} per flat). "
        f"Model's typical error: **{METRICS['test']['sklearn']['median_abs_pct_error']:.1%}**, "
        f"saved margin: **{PIPE['margin']:.1%}** ({METRICS['margin']['percentile']}th percentile of validation error)."
    )

    plot_df = sample(d)
    fig = go.Figure()
    for flag, name, color in [(False, "Other listings", "#bbbbbb"), (True, "Bargains", "#d62728")]:
        part = plot_df[plot_df["bargain"] == flag]
        fig.add_trace(go.Scattergl(
            x=part["estimate"], y=part["PRICE"], mode="markers", name=name,
            marker=dict(size=4 if not flag else 6, opacity=0.4 if not flag else 0.8, color=color)))
    lo, hi = plot_df["estimate"].min(), plot_df["estimate"].max()
    for factor, label, dash in [(1, "Listing = estimate", "dash"), (1 - margin, f"{margin:.0%} below", "dot"),
                                (1 - cap, f"{cap:.0%} below (cap)", "dot")]:
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo * factor, hi * factor], mode="lines", name=label,
                                 line=dict(color="black", dash=dash, width=1)))
    fig.update_layout(title="Listing price vs. estimated market price",
                      xaxis=dict(title="Estimated market price (€)", type="log"),
                      yaxis=dict(title="Listing price (€)", type="log"),
                      height=520, legend=dict(orientation="h", y=-0.15))

    top = bargains.sort_values("vs_estimate").head(25)
    table = pd.DataFrame({
        "Flat ID": top["ASSETID"],
        "Listing price": top["PRICE"].map(euros),
        "Estimate": top["estimate"].map(euros),
        "Below estimate": (-top["vs_estimate"]).map(lambda v: f"{v:.0%}"),
        "m²": top["CONSTRUCTEDAREA"].astype(int),
        "Bedrooms": top["ROOMNUMBER"].astype(int),
        "Km to centre": top["DISTANCE_TO_CITY_CENTER"].round(1),
        "Built": top["CADCONSTRUCTIONYEAR"].astype(int),
    })
    return summary, fig, table


# --------------------------------------------------------------------------- tab 4: drivers
def drivers_plot(top_n, group):
    c = COEFS["sklearn"].drop("(intercept)")
    numeric = set(PIPE["numeric"])
    if group == "Yes/no features only":
        c = c[[f for f in c.index if f not in numeric]]
    elif group == "Numeric features only":
        c = c[[f for f in c.index if f in numeric]]
    c = c.reindex(c.abs().sort_values(ascending=False).index).head(int(top_n))[::-1]
    pct = (np.exp(c) - 1) * 100
    labels = [f"{f} (per +1 std)" if f in numeric else f for f in c.index]
    fig = go.Figure(go.Bar(x=pct, y=labels, orientation="h",
                           marker_color=["#2ca02c" if v > 0 else "#d62728" for v in pct]))
    fig.update_layout(title="Effect on asking price, holding everything else equal",
                      xaxis_title="% change in asking price", height=max(400, 22 * len(c) + 120),
                      margin=dict(l=260))
    return fig


# --------------------------------------------------------------------------- tab 5: training
def curves_plot():
    curves = A["curves"]
    fig = go.Figure()
    for m in ["manual_pytorch", "standard_pytorch"]:
        d = curves[curves["model"] == m]
        fig.add_trace(go.Scatter(x=d["epoch"], y=d["train_mse"], name=f"{MODEL_LABELS[m]} — train",
                                 line=dict(color=COLORS[m])))
        fig.add_trace(go.Scatter(x=d["epoch"], y=d["val_mse"], name=f"{MODEL_LABELS[m]} — validation",
                                 line=dict(color=COLORS[m], dash="dot")))
    sk_val = METRICS["validation"]["sklearn"]["RMSE_log"] ** 2
    fig.add_hline(y=sk_val, line_dash="dash", annotation_text="scikit-learn (validation)")
    fig.update_layout(title="Learning curves (full-batch gradient descent)", xaxis_title="Epoch",
                      yaxis=dict(title="MSE on log price", type="log"), height=440)
    return fig


def coef_agreement_plot():
    c = COEFS
    fig = go.Figure()
    for m in ["manual_pytorch", "standard_pytorch"]:
        fig.add_trace(go.Scatter(x=c["sklearn"], y=c[m], mode="markers", name=MODEL_LABELS[m],
                                 text=c.index, marker=dict(size=7, opacity=0.7, color=COLORS[m])))
    lo, hi = c.min().min(), c.max().max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Identical",
                             line=dict(color="black", dash="dash")))
    fig.update_layout(title="Weights of the loaded models vs. scikit-learn (one dot per feature)",
                      xaxis_title="scikit-learn weight", yaxis_title="PyTorch weight", height=440)
    return fig


def agreement_table():
    c = COEFS
    return pd.DataFrame([
        {"Model": MODEL_LABELS[m],
         "Max |weight diff| vs scikit-learn": f"{(c[m] - c['sklearn']).abs().max():.5f}",
         "Max |prediction diff| (val, log)": f"{METRICS['agreement_vs_sklearn'][m]['max |pred diff| vs sklearn (log)']:.5f}"}
        for m in ["manual_pytorch", "standard_pytorch"]
    ])


# --------------------------------------------------------------------------- layout
test_sk = METRICS["test"]["sklearn"]
HEADER = f"""
# 🏠 Madrid bargain finder
Linear regression on **log(asking price)** of {len(PREDS):,} Madrid flats (idealista18, 2018),
trained three ways. Train = Q1–Q3, validation and test = Q4 halves.
**Test set:** RMSE {test_sk['RMSE_log']:.3f} (log) · typical error {test_sk['median_abs_pct_error']:.1%} ·
R² {test_sk['R2']:.3f} — vs. naive baseline typical error
{METRICS['test']['baseline']['median_abs_pct_error']:.1%}. Nothing here is retrained: all results load
from `artifacts/`. Start in the **Home hub** to find bargains and see why each one is cheap.
"""

with gr.Blocks(title="Madrid bargain finder") as demo:
    gr.Markdown(HEADER)

    with gr.Tab("Home hub"):
        gr.Markdown("### Find a bargain")
        with gr.Row():
            hub_split = gr.Radio(["test", "val"], value="test", label="Listings")
            hub_sort = gr.Dropdown(list(SORTS), value="Biggest discount", label="Sort by")
            hub_margin = gr.Slider(0, 50, value=round(PIPE["margin"] * 100), step=1,
                                   label="Margin of safety: at least X% below estimate")
        with gr.Row():
            hub_price = gr.Slider(50, 3000, value=3000, step=25, label="Max listing price (€ thousands)")
            hub_beds = gr.Slider(0, 6, value=0, step=1, label="Min bedrooms")
            hub_km = gr.Slider(0.5, 20, value=20, step=0.5, label="Max km to centre")
        with gr.Row():
            hub_lift = gr.Checkbox(value=False, label="Must have a lift")
            hub_ext = gr.Checkbox(value=False, label="Exterior only")
            hub_restore = gr.Checkbox(value=True, label="Hide flats needing restoration")
        hub_count = gr.Markdown()
        hub_ids = gr.State([])
        with gr.Accordion("Matching bargains (click a row to open it)", open=False):
            hub_table = gr.Dataframe(interactive=False, max_height=360)
        gr.Markdown("### Bargain spotlight")
        with gr.Row():
            hub_pick = gr.Dropdown(choices=[], label="Bargain", scale=4)
            hub_prev = gr.Button("Previous", scale=1)
            hub_next = gr.Button("Next", scale=1)
        hub_detail = gr.HTML()

        finder_inputs = [hub_split, hub_sort, hub_price, hub_beds, hub_km, hub_lift, hub_ext, hub_restore, hub_margin]
        finder_outputs = [hub_count, hub_table, hub_pick, hub_ids]
        for ctl in finder_inputs:
            ctl.change(find_bargains, finder_inputs, finder_outputs)
        hub_pick.change(flat_html, [hub_pick, hub_margin], hub_detail)
        hub_prev.click(lambda cur, ids: step_flat(cur, ids, -1), [hub_pick, hub_ids], hub_pick)
        hub_next.click(lambda cur, ids: step_flat(cur, ids, 1), [hub_pick, hub_ids], hub_pick)
        hub_table.select(pick_from_table, hub_ids, hub_pick)

    with gr.Tab("1 · Model comparison"):
        with gr.Row():
            cmp_split = gr.Radio(["val", "test"], value="test", label="Data split")
            cmp_models = gr.CheckboxGroup(list(MODEL_LABELS), value=["sklearn", "manual_pytorch",
                                          "standard_pytorch"], label="Models (internal names)")
            cmp_scale = gr.Radio(["Price (€)", "log(price)"], value="Price (€)", label="Axis")
        cmp_plot = gr.Plot()
        cmp_table = gr.Dataframe(label="Metrics")
        gr.Markdown("The three implementations sit on top of each other: they learned the same model. "
                    "Dots on the dashed line are perfect predictions; the baseline is a flat line.")
        for ctl in (cmp_split, cmp_models, cmp_scale):
            ctl.change(comparison_update, [cmp_split, cmp_models, cmp_scale], [cmp_plot, cmp_table])

    with gr.Tab("2 · Distributions"):
        with gr.Row():
            dist_var = gr.Dropdown(list(DIST_VARS), value="Asking price (€)", label="Variable")
            dist_splits = gr.CheckboxGroup(["train", "val", "test"], value=["train", "test"], label="Splits")
            dist_clip = gr.Checkbox(value=True, label="Hide extreme 1% on each side")
        dist_plot = gr.Plot()
        dist_table = gr.Dataframe(label="Summary by split")
        for ctl in (dist_var, dist_splits, dist_clip):
            ctl.change(distribution_update, [dist_var, dist_splits, dist_clip], [dist_plot, dist_table])

    with gr.Tab("3 · Bargain finder"):
        with gr.Row():
            bar_split = gr.Radio(["val", "test"], value="test", label="Listings")
            bar_model = gr.Dropdown(["sklearn", "manual_pytorch", "standard_pytorch"], value="sklearn",
                                    label="Model giving the estimate")
        with gr.Row():
            bar_margin = gr.Slider(0, 50, value=round(PIPE["margin"] * 100), step=1,
                                   label="Margin of safety: at least X% below estimate")
            bar_cap = gr.Slider(10, 90, value=round(PIPE["max_discount"] * 100), step=1,
                                label="Suspicious cap: exclude listings more than X% below estimate")
        bar_summary = gr.Markdown()
        bar_plot = gr.Plot()
        bar_table = gr.Dataframe(label="Biggest bargains (top 25)")
        for ctl in (bar_split, bar_model, bar_margin, bar_cap):
            ctl.change(bargain_update, [bar_split, bar_model, bar_margin, bar_cap],
                       [bar_summary, bar_plot, bar_table])

    with gr.Tab("4 · What drives price"):
        with gr.Row():
            drv_n = gr.Slider(5, 40, value=20, step=1, label="Show top N features")
            drv_group = gr.Radio(["All", "Yes/no features only", "Numeric features only"], value="All",
                                 label="Features")
        drv_plot = gr.Plot()
        gr.Markdown("Yes/no features: % change in asking price when the flat has it, compared to the "
                    "baseline category. Numeric features: % change for a one-standard-deviation increase.")
        for ctl in (drv_n, drv_group):
            ctl.change(drivers_plot, [drv_n, drv_group], drv_plot)

    with gr.Tab("5 · Training"):
        gr.Plot(curves_plot())
        gr.Plot(coef_agreement_plot())
        gr.Dataframe(agreement_table(), label="Agreement with scikit-learn")

    # Fill every tab once when the page opens
    demo.load(find_bargains, finder_inputs, finder_outputs)
    demo.load(comparison_update, [cmp_split, cmp_models, cmp_scale], [cmp_plot, cmp_table])
    demo.load(distribution_update, [dist_var, dist_splits, dist_clip], [dist_plot, dist_table])
    demo.load(bargain_update, [bar_split, bar_model, bar_margin, bar_cap], [bar_summary, bar_plot, bar_table])
    demo.load(drivers_plot, [drv_n, drv_group], drv_plot)


if __name__ == "__main__":
    demo.launch()
