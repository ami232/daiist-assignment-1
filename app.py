"""
Assignment 1 -- Gradio dashboard (plain-language version).

    uv run python main.py app

Loads ONLY artifacts produced by train.py (artifacts/ + the committed data sample).
Nothing is trained here.
"""
import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from common import ARTIFACTS, BID_CAP, engineer, load_raw, simulate_bids, split_name

REQUIRED = ["summary.json", "preprocessor.joblib", "sklearn_ridge.joblib", "manual_torch.pt",
            "nn_linear.pt", "predictions_val.csv", "predictions_test.csv", "coefficients.csv"]
missing = [f for f in REQUIRED if not (ARTIFACTS / f).exists()]
if missing:
    raise SystemExit(f"Missing artifacts {missing}. Run `uv run python main.py train` first.")

# ------------------------------------------------------------------ load artifacts
S = json.loads((ARTIFACTS / "summary.json").read_text())
PRE = joblib.load(ARTIFACTS / "preprocessor.joblib")
SK = joblib.load(ARTIFACTS / "sklearn_ridge.joblib")
_man = torch.load(ARTIFACTS / "manual_torch.pt")
_nn = torch.load(ARTIFACTS / "nn_linear.pt")
NN = torch.nn.Linear(_nn["d_in"], 1)
NN.load_state_dict({k.replace("linear.", ""): v for k, v in _nn["state_dict"].items()})
PREDS = {"val": pd.read_csv(ARTIFACTS / "predictions_val.csv", parse_dates=["sale_date"]),
         "test": pd.read_csv(ARTIFACTS / "predictions_test.csv", parse_dates=["sale_date"])}
COEFS = pd.read_csv(ARTIFACTS / "coefficients.csv")
RAW = load_raw()
RAW["split"] = split_name(RAW["sale_date"])
FEATS = engineer(RAW)

MODELS = {"scikit-learn": "pred_sklearn", "PyTorch manual": "pred_manual_torch",
          "PyTorch nn.Module": "pred_nn_module", "Baseline (trust mmr)": "pred_baseline_mmr"}
COLORS = {"scikit-learn": "#1f77b4", "PyTorch manual": "#ff7f0e",
          "PyTorch nn.Module": "#2ca02c", "Baseline (trust mmr)": "#7f7f7f"}
SPLIT_LABEL = {"val": "May (validation)", "test": "June (test)"}

# Friendly, human-readable view of the variables for the "what drives the gap" tab
DISP = pd.DataFrame({
    "Car age (years)": FEATS["age"],
    "Mileage (odometer)": RAW["odometer"].clip(upper=300_000),
    "Condition score (1-49)": FEATS["condition"],
    "Make": FEATS["make"],
    "Body type": FEATS["body_group"],
    "State where it is sold": FEATS["state"],
    "Seller is a lease company": np.where(FEATS["seller_is_lease"] > 0, "yes (lease return)", "no"),
})

# headline numbers (all read from artifacts, nothing recomputed)
M_T = S["metrics"]["test"]
B_T = S["business"]["test"]
RM_MODEL, RM_BASE = M_T["sklearn"]["rmse"] * 100, M_T["baseline_mmr"]["rmse"] * 100
X_DEF = S["cushion_default"]
IMPROVE = (1 - RM_MODEL / RM_BASE) * 100


def card(title, big, small):
    return (f"<div style='flex:1;min-width:210px;padding:14px 16px;border:1px solid var(--border-color-primary);"
            f"border-radius:10px;background:var(--block-background-fill)'>"
            f"<div style='font-size:13px;opacity:.75'>{title}</div>"
            f"<div style='font-size:28px;font-weight:700;margin:4px 0'>{big}</div>"
            f"<div style='font-size:13px;opacity:.8'>{small}</div></div>")


def timeline():
    def block(w, bg, t1, t2):
        return (f"<div style='flex:{w};background:{bg};color:#fff;padding:10px 12px;border-radius:8px;"
                f"font-size:13px'><b>{t1}</b><br>{t2}</div>")
    return ("<div style='display:flex;gap:6px;margin:8px 0'>"
            + block(4, "#1f77b4", "Dec - Mar: TRAIN", f"{S['n_rows']['train']:,} sales. The model learns from these.")
            + block(1.6, "#ff7f0e", "May: VALIDATE", f"{S['n_rows']['val']:,} sales. We choose settings here, including the cushion X.")
            + block(1.6, "#2ca02c", "June: TEST", f"{S['n_rows']['test']:,} sales. Used once at the end, as if it were the future.")
            + "</div>")


# ------------------------------------------------------------------ tab: model comparison
def comparison_table(split):
    rows = []
    for label, col in MODELS.items():
        m = S["metrics"][split][col.replace("pred_", "")]
        rows.append({"Model": label,
                     "Typical error, RMSE (pp)": round(m["rmse"] * 100, 2),
                     "Average bias: predicted - actual (pp)": round(m["mean_signed_error"] * 100, 2),
                     "% of cars where it predicted too high": round(m["overestimate_share"] * 100, 1)})
    return pd.DataFrame(rows)


def pred_vs_actual(split, selected):
    d = PREDS[split]
    fig = go.Figure()
    for label in selected:
        fig.add_trace(go.Scattergl(x=d["target"] * 100, y=d[MODELS[label]] * 100, mode="markers",
                                   name=label, opacity=0.35, marker=dict(size=4, color=COLORS[label])))
    fig.add_trace(go.Scatter(x=[-100, 100], y=[-100, 100], mode="lines", name="perfect prediction",
                             line=dict(color="black", dash="dash")))
    fig.add_annotation(x=-60, y=45, text="Above the line: predicted too high<br>(the costly mistake)",
                       showarrow=False, font=dict(size=12), align="left")
    fig.add_annotation(x=45, y=-45, text="Below the line: predicted too low<br>(we would lose the car)",
                       showarrow=False, font=dict(size=12), align="right")
    fig.update_layout(title=f"Each dot is one car sold in {SPLIT_LABEL[split]}",
                      xaxis_title="what really happened: final price vs mmr (%)",
                      yaxis_title="what the model predicted (%)", height=520,
                      xaxis=dict(range=[-100, 100]), yaxis=dict(range=[-60, 60]))
    return fig


def compare(split, selected):
    return comparison_table(split), pred_vs_actual(split, selected or list(MODELS))


def loss_figure():
    fig = go.Figure()
    for name, curve in S["loss_curves"].items():
        fig.add_trace(go.Scatter(y=curve, x=np.arange(len(curve)) * 10, name=name))
    fig.update_layout(title="Training loss: the two PyTorch versions follow the same path",
                      xaxis_title="training step", yaxis_title="loss", yaxis_type="log", height=360)
    return fig


def agreement_df():
    a = S["agreement"]
    nice = {"max_abs_pred_diff_manual_vs_sklearn": "Largest prediction gap: manual PyTorch vs scikit-learn",
            "max_abs_pred_diff_nn_vs_sklearn": "Largest prediction gap: nn.Module vs scikit-learn",
            "max_abs_pred_diff_manual_vs_nn": "Largest prediction gap: manual PyTorch vs nn.Module",
            "max_abs_coef_diff_manual_vs_sklearn": "Largest coefficient gap: manual PyTorch vs scikit-learn",
            "max_abs_coef_diff_nn_vs_sklearn": "Largest coefficient gap: nn.Module vs scikit-learn"}
    return pd.DataFrame({"Check": [nice[k] for k in a], "Value": [f"{v:.1e}" for v in a.values()]})


def top_coefs(n):
    c = COEFS.reindex(COEFS["coef_sklearn"].abs().sort_values(ascending=False).index).head(int(n))
    c = c.rename(columns={"feature": "variable", "coef_sklearn": "scikit-learn",
                          "coef_manual": "PyTorch manual", "coef_nn": "PyTorch nn.Module"})
    return c.round(4)


# ------------------------------------------------------------------ tab: what drives the gap
def target_hist():
    fig = go.Figure()
    for sp, c, lab in zip(("train", "val", "test"), ("#1f77b4", "#ff7f0e", "#2ca02c"),
                          ("Dec-Mar (train)", "May (validation)", "June (test)")):
        v = RAW.loc[RAW["split"] == sp, "target"] * 100
        fig.add_trace(go.Histogram(x=v.clip(-100, 150), name=lab, opacity=0.55,
                                   histnorm="probability density", nbinsx=80, marker_color=c))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(barmode="overlay", height=400,
                      xaxis_title="final price vs mmr (%)  [0 = sold exactly at mmr; shown cut at -100 / +150]",
                      yaxis_title="share of cars",
                      title="How far from mmr do cars actually sell?")
    return fig


def feature_hist(feature, split):
    s = DISP.loc[RAW["split"] == split, feature]
    if not pd.api.types.is_numeric_dtype(s):
        vc = s.value_counts().head(20)
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values))
        fig.update_layout(yaxis_title="number of cars")
    else:
        fig = go.Figure(go.Histogram(x=s.dropna(), nbinsx=40))
        fig.update_layout(yaxis_title="number of cars", xaxis_title=feature)
    fig.update_layout(title=f"How the cars are spread: {feature} ({ {'train':'Dec-Mar','val':'May','test':'June'}[split] })", height=380)
    return fig


def target_by_feature(feature):
    s = DISP[feature]
    d = pd.DataFrame({"f": s, "t": RAW["target"] * 100})
    if not pd.api.types.is_numeric_dtype(s):
        top = s.value_counts().head(15).index
        g = d[d["f"].isin(top)].groupby("f")["t"].mean().sort_values()
    else:
        g = d.groupby(pd.qcut(d["f"], 10, duplicates="drop"), observed=True)["t"].mean()
        g.index = g.index.astype(str)
    fig = go.Figure(go.Bar(x=g.index, y=g.values, marker_color=["#d62728" if v < 0 else "#2ca02c" for v in g.values]))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title=f"Does '{feature}' change how far from mmr the car sells?",
                      yaxis_title="average final price vs mmr (%)", height=380)
    return fig


# ------------------------------------------------------------------ tab: bid simulator
def bid_sim(split, cushion_pp):
    cushion = cushion_pp / 100
    d = PREDS[split]
    rows = []
    xs = np.arange(0, 30.01, 1.0) / 100
    f_lost, f_exc = go.Figure(), go.Figure()
    for label, col in MODELS.items():
        r = simulate_bids(d["mmr"], d["sellingprice"], d[col], cushion)
        rows.append({"Model": label,
                     "Bid above final price (%)": round(r["bid_above_price_share"] * 100, 1),
                     "Average $ we were ready to overpay": round(r["mean_excess_usd"]),
                     "Cars lost (%)": round(r["lost_share"] * 100, 1),
                     "Average $ we were short by": round(r["mean_shortfall_usd"])})
        sims = [simulate_bids(d["mmr"], d["sellingprice"], d[col], x) for x in xs]
        f_lost.add_trace(go.Scatter(x=xs * 100, y=[s["lost_share"] * 100 for s in sims], name=label,
                                    line=dict(color=COLORS[label])))
        f_exc.add_trace(go.Scatter(x=xs * 100, y=[s["mean_excess_usd"] for s in sims], name=label,
                                   line=dict(color=COLORS[label])))
    for f, title, yt in ((f_lost, "The price of safety: a bigger cushion loses more cars", "% of cars lost"),
                         (f_exc, "The benefit: a bigger cushion cuts the overpayment", "average $ overpaid when we bid too high")):
        f.add_vline(x=cushion_pp, line_dash="dash")
        f.update_layout(title=title, xaxis_title="cushion X (pp of mmr)  [dashed line = current slider]",
                        yaxis_title=yt, height=360)
    k = simulate_bids(d["mmr"], d["sellingprice"], d["pred_sklearn"], cushion)
    b = simulate_bids(d["mmr"], d["sellingprice"], d["pred_baseline_mmr"], cushion)
    summary = (f"**With X = {cushion_pp:.1f} pp, on the cars sold in {SPLIT_LABEL[split]}:** using our model, our maximum bid "
               f"would have been **above the final price on {k['bid_above_price_share']*100:.1f}% of cars**, by "
               f"${k['mean_excess_usd']:,.0f} on average (this is where overpaying risk lives). "
               f"On the other {k['lost_share']*100:.1f}% the bid was too low and the car would be lost, "
               f"${k['mean_shortfall_usd']:,.0f} short on average. "
               f"Trusting mmr alone with the same X: {b['bid_above_price_share']*100:.1f}% above the final price, "
               f"${b['mean_excess_usd']:,.0f} excess.")
    return summary, pd.DataFrame(rows), f_lost, f_exc


def fixed_x_table():
    rows = []
    for label, col in MODELS.items():
        k = col.replace("pred_", "")
        v, t = S["business"]["val"][k], S["business"]["test"][k]
        rows.append({"Model": label, "Cushion X set on May (pp)": round(v["cushion"] * 100, 2),
                     "May: bid above final price (%)": round(v["bid_above_price_share"] * 100, 1),
                     "June: bid above final price (%)": round(t["bid_above_price_share"] * 100, 1),
                     "June: avg $ overpaid": round(t["mean_excess_usd"]),
                     "June: cars lost (%)": round(t["lost_share"] * 100, 1)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tab: try a car
CATS = {name: list(c) for name, c in zip(
    ["make", "body_group", "transmission", "state", "color", "seller"],
    PRE.named_transformers_["cat"].categories_)}


def predict_car(year, odometer, condition, make, body, transmission, state, color, seller, mmr, cushion_pp):
    row = pd.DataFrame([{"sale_date": pd.Timestamp("2015-06-15"), "year": year, "odometer": odometer,
                         "condition": condition if condition else np.nan, "make": make, "body": body,
                         "transmission": transmission, "state": state, "color": color, "seller": seller}])
    X = PRE.transform(engineer(row))
    Xt = torch.tensor(X, dtype=torch.float32)
    preds = {"scikit-learn": float(SK.predict(X)[0]),
             "PyTorch manual": float((Xt @ _man["w"] + _man["b"]).item()),
             "PyTorch nn.Module": float(NN(Xt).item()), "Baseline (trust mmr)": 0.0}
    out = []
    for k, p in preds.items():
        bid = min(mmr * (1 + p - cushion_pp / 100), BID_CAP * mmr)
        out.append({"Model": k, "Expected final price vs mmr (%)": round(p * 100, 2),
                    "Expected final price ($)": round(mmr * (1 + p)), "Maximum bid we would place ($)": round(bid)})
    p = preds["scikit-learn"]
    bid = min(mmr * (1 + p - cushion_pp / 100), BID_CAP * mmr)
    msg = (f"**Result:** the model expects this car to close about **{p*100:+.1f}%** vs mmr "
           f"(around ${mmr*(1+p):,.0f}). Subtracting the cushion ({cushion_pp:.1f} pp), "
           f"the maximum bid would be **${bid:,.0f}**.")
    return msg, pd.DataFrame(out)


# ------------------------------------------------------------------ layout
INTRO = f"""
# Used-car auctions: how high should we bid?

**The situation.** We are a car dealership. We buy used cars at wholesale auctions (some of them returned from a lease).
For every car we must decide **the maximum we are willing to bid**. Bid too low and we lose the car. Bid too high and
we pay more than the car is worth to the market, and that money leaves our cash.

**The reference price.** Each car comes with an **mmr**: a third-party estimate of what that car is worth on the wholesale market.
It is a good starting point, but cars rarely sell exactly at mmr.

**What this dashboard does.** We built a model that predicts **how far above or below mmr a car will end up selling**,
using only the car's characteristics (age, mileage, condition, make, state, seller...). It never sees mmr itself.
Then we turn that prediction into a bid:

> **maximum bid = mmr × (1 + predicted gap − X)**, and never more than 20% above mmr.

**X is a safety cushion** that we subtract so that, if the model is too optimistic, we do not overpay.

**Worked example.** A car has an mmr of $14,000. The model predicts it will sell 2% above mmr. With a cushion of X = 7.5 points,
the maximum bid is 14,000 × (1 + 0.02 − 0.075) = **$13,230**.

*The data has no costs or resale prices, so we do not compute profit. We only measure how often the bid ends up above or below
the price at which the car really sold.*
"""

GLOSSARY = """
| Term | Meaning |
|---|---|
| **mmr** | Third-party estimate of the car's wholesale market value. Our reference price. |
| **Gap (what we predict)** | final selling price ÷ mmr − 1. A gap of +0.05 (= +5%) means the car sold 5% above mmr. |
| **pp (percentage points)** | Differences between percentages. If the real gap was +3% and we predicted +8%, we were off by 5 pp. |
| **Baseline (trust mmr)** | The naive rival: always predict a gap of 0, i.e. "it will sell exactly at mmr". A useful model must beat it. |
| **RMSE (typical error)** | The size of the model's typical mistake, in pp. It punishes big mistakes more. Lower is better. |
| **Bias (predicted − actual)** | Average of the signed mistakes. Positive means the model tends to predict too high (the costly side). |
| **Cushion X** | What we subtract from the prediction before bidding, to protect against predicting too high. |
| **Train / validation / test** | Dec-Mar: the model learns. May: we tune settings and fix X. June: final exam, used once, like the real future. |
"""

with gr.Blocks(title="Used-car auctions: how high should we bid?") as demo:
    with gr.Tab("1. Start here"):
        gr.Markdown(INTRO)
        gr.Markdown("## The result in four numbers  *(June, cars the model had never seen)*")
        gr.HTML("<div style='display:flex;gap:12px;flex-wrap:wrap'>"
                + card("Typical prediction error", f"{RM_MODEL:.1f} pp",
                       f"vs {RM_BASE:.1f} pp if we just trust mmr: {IMPROVE:.0f}% smaller")
                + card("Cushion X (fixed on May)", f"{X_DEF*100:.2f} pp",
                       "chosen so that only ~25% of May bids ended above the final price")
                + card("June cars where we'd bid above the final price", f"{B_T['sklearn']['bid_above_price_share']*100:.1f}%",
                       f"vs {B_T['baseline_mmr']['bid_above_price_share']*100:.1f}% trusting mmr alone")
                + card("Average overpayment when that happens", f"${B_T['sklearn']['mean_excess_usd']:,.0f}",
                       f"vs ${B_T['baseline_mmr']['mean_excess_usd']:,.0f} trusting mmr alone")
                + "</div>")
        gr.Markdown("**In short:** the model is somewhat better than trusting mmr, but not magic. Much of what makes a car sell above or "
                    "below mmr seems to be bidding noise that the car's characteristics cannot explain. "
                    "The trade-off is real: the safer the cushion, the more cars we lose.")
        gr.Markdown("## How we tested it fairly")
        gr.HTML(timeline())
        gr.Markdown("We never test on the past, because when we use the model for real we only know the past. "
                    "A random split would let the model see cars sold the same week (or the very same car) in both training and testing.")
        with gr.Accordion("Glossary: what do these words mean?", open=False):
            gr.Markdown(GLOSSARY)
        gr.Markdown("**Where to go next:**  tab 2 shows how good the predictions are · tab 3 shows what makes cars sell above or below mmr · "
                    "tab 4 lets you move the cushion and see the trade-off · tab 5 lets you try your own car.")

    with gr.Tab("2. How good are the predictions?"):
        gr.Markdown("### Is the model better than simply trusting mmr?\n"
                    "Each dot below is a car. **The closer the dots are to the dashed line, the better the prediction.** "
                    "Dots above the line are cars where we predicted too high: the mistake that makes us overpay. "
                    "We compare three versions of the same model (they are built differently but should give the same answer) "
                    "and the baseline that just trusts mmr.")
        with gr.Row():
            split_c = gr.Radio([("June (test): the final exam", "test"), ("May (validation)", "val")],
                               value="test", label="Which month?")
            sel = gr.CheckboxGroup(list(MODELS), value=list(MODELS)[:3], label="Show these models")
        plot = gr.Plot(value=pred_vs_actual("test", list(MODELS)[:3]))
        gr.Markdown("**Scorecard** — *typical error* (lower is better) and *bias* (positive = predicts too high).")
        tbl = gr.Dataframe(value=comparison_table("test"))
        for c in (split_c, sel):
            c.change(compare, [split_c, sel], [tbl, plot])
        gr.Markdown("The three model versions score exactly the same: they are the same model trained three ways "
                    "(scikit-learn, a hand-written PyTorch loop, and the standard PyTorch workflow).")
        with gr.Accordion("For the technical reader: do the three implementations really agree?", open=False):
            gr.Markdown("Gaps near 1e-7 are just rounding. If they were large, one of the implementations would be wrong or not fully trained.")
            gr.Dataframe(value=agreement_df())
            gr.Plot(value=loss_figure())
            n = gr.Slider(5, 40, value=15, step=1, label="How many of the strongest variables to show")
            coef_tbl = gr.Dataframe(value=top_coefs(15))
            n.change(top_coefs, n, coef_tbl)

    with gr.Tab("3. What drives the gap?"):
        gr.Markdown("### What does the data look like?\n"
                    "**Chart 1** shows how far cars sell from mmr: half of the cars close within about 7% of it (the dashed line is 0 = exactly mmr), "
                    "but some sell much higher or lower. Those extreme cases are the very wrong bids the model finds hardest.")
        gr.Plot(value=target_hist())
        gr.Markdown("**Charts 2 and 3** let you pick a car characteristic. Chart 2 shows how the cars are distributed; "
                    "chart 3 shows whether cars with that characteristic tend to sell above (green) or below (red) mmr.")
        with gr.Row():
            feat = gr.Dropdown(list(DISP.columns), value="Condition score (1-49)", label="Car characteristic")
            sp = gr.Radio([("Dec-Mar (train)", "train"), ("May (validation)", "val"), ("June (test)", "test")],
                          value="train", label="Period (chart 2)")
        fh = gr.Plot(value=feature_hist("Condition score (1-49)", "train"))
        tf = gr.Plot(value=target_by_feature("Condition score (1-49)"))
        feat.change(feature_hist, [feat, sp], fh)
        sp.change(feature_hist, [feat, sp], fh)
        feat.change(target_by_feature, feat, tf)

    with gr.Tab("4. Bid simulator"):
        gr.Markdown(f"### How much cushion should we keep?\n"
                    f"Our rule: **maximum bid = mmr × (1 + predicted gap − X)**, never above {BID_CAP-1:.0%} over mmr. "
                    "Move the slider to change the cushion **X**:\n\n"
                    "- **Bigger X** → we bid lower → fewer times we pay more than the market, but we **lose more cars** (only an opportunity cost).\n"
                    "- **Smaller X** → we bid higher → we win more cars, but **overpay more often** (the costly mistake).\n\n"
                    f"The default X ({X_DEF*100:.2f} pp) was fixed using May's errors only: it is the value for which about 25% of May bids "
                    "ended above the final price. Then we checked it on June, which it had never seen.")
        with gr.Row():
            bs = gr.Radio([("May (validation)", "val"), ("June (test)", "test")], value="test", label="Which month?")
            bx = gr.Slider(0, 30, value=round(X_DEF * 100, 1), step=0.5, label="Cushion X (percentage points of mmr)")
        s0, bt0, bp0, bq0 = bid_sim("test", X_DEF * 100)
        summ = gr.Markdown(value=s0)
        bt = gr.Dataframe(value=bt0)
        bp, bq = gr.Plot(value=bp0), gr.Plot(value=bq0)
        for c in (bs, bx):
            c.change(bid_sim, [bs, bx], [summ, bt, bp, bq])
        with gr.Accordion("The final check: X fixed on May, applied once to June", open=False):
            gr.Dataframe(value=fixed_x_table())
            gr.Markdown("On May, ~25% of bids ended above the final price by design. On June it was lower (~20%): June cars sold relatively higher "
                        "vs mmr than May cars (average gap -1.2% vs -3.1%), so the cushion was a bit more cautious than needed. With only two months "
                        "after training, we cannot tell if that is a seasonal change or chance.")

    with gr.Tab("5. Try your own car"):
        gr.Markdown("### What would we bid for this car?\nDescribe a car and the dashboard predicts how far from mmr it will sell, "
                    "then applies the bidding rule.")
        with gr.Row():
            y_ = gr.Number(value=2012, label="Model year", precision=0)
            o_ = gr.Number(value=60000, label="Odometer (miles)")
            c_ = gr.Number(value=35, label="Condition score (1-49, 0 = unknown)")
            m_ = gr.Number(value=14000, label="mmr of this car ($)")
        with gr.Row():
            mk = gr.Dropdown(CATS["make"], value="ford", label="Make")
            bd = gr.Dropdown(CATS["body_group"], value="sedan", label="Body type")
            tr_ = gr.Dropdown(CATS["transmission"], value="automatic", label="Transmission")
            st = gr.Dropdown(CATS["state"], value="ca", label="State")
        with gr.Row():
            co = gr.Dropdown(CATS["color"], value="black", label="Color")
            se = gr.Dropdown(CATS["seller"], value=CATS["seller"][0], label="Seller")
            cu = gr.Slider(0, 30, value=round(X_DEF * 100, 1), step=0.5, label="Cushion X (pp)")
        btn = gr.Button("Predict and compute the bid", variant="primary")
        msg_out = gr.Markdown()
        out = gr.Dataframe()
        btn.click(predict_car, [y_, o_, c_, mk, bd, tr_, st, co, se, m_, cu], [msg_out, out])

if __name__ == "__main__":
    demo.launch()