import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_prep import load_data, split

# Everything below is loaded from disk. Nothing is trained here.
REPORT = json.load(open("models/metrics.json"))
PRED = np.load("models/predictions.npz")
MODELS = ["sklearn", "manual", "module"]

df = load_data()
tr, va, te = split(df)
assert len(te) == len(PRED["y_test"]), "test set does not match saved predictions: rerun train"

DIST_COLS = ["target", "acousticness", "danceability", "energy", "loudness", "tempo",
             "valence", "speechiness", "instrumentalness", "liveness", "duration_ms",
             "log_duration_ms"]


def metrics_table(split_name):
    rows = []
    for name in ["baseline"] + MODELS:
        r = REPORT[name][split_name]
        rows.append({"model": name, "MAE": round(r["mae"], 4),
                     "RMSE": round(r["rmse"], 4), "R2": round(r["r2"], 4)})
    return pd.DataFrame(rows)


def coef_table():
    c = REPORT["coefficients"]
    t = pd.DataFrame({"feature": c["feature"], "sklearn": c["sklearn"],
                      "manual": c["manual"], "module": c["module"]})
    t["max |diff|"] = t[["manual", "module"]].sub(t["sklearn"], axis=0).abs().max(axis=1)
    return t.round(6)


def pred_vs_actual(model):
    y, p = PRED["y_test"], PRED[model]
    fig = px.scatter(x=y, y=p, opacity=0.35,
                     labels={"x": "Actual target (popularity vs cohort)",
                             "y": "Predicted target"},
                     title=f"{model}: predicted vs actual (test, 2018 to May 2020)")
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
                             line=dict(dash="dash")))
    return fig


def distribution(col):
    data = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    if col == "log_duration_ms":
        data = data.assign(log_duration_ms=np.log(data["duration_ms"]))
    return px.histogram(data, x=col, color="split", barmode="overlay", opacity=0.55,
                        histnorm="probability density", nbins=50,
                        title=f"Distribution of {col} by split")


def promote(model, k):
    y, p = PRED["y_test"], PRED[model]
    n = max(1, int(len(p) * k / 100))
    top = np.argsort(-p)[:n]
    hit_sel, hit_all = float((y[top] > 0).mean()), float((y > 0).mean())
    table = pd.DataFrame({
        "": ["songs promoted", "hit rate (beat their cohort), promoted",
             "hit rate, all songs (random pick)", "lift vs random",
             "mean actual target, promoted", "mean actual target, all"],
        "value": [n, round(hit_sel, 3), round(hit_all, 3), round(hit_sel / hit_all, 3),
                  round(float(y[top].mean()), 3), round(float(y.mean()), 3)],
    })
    fig = go.Figure(go.Bar(x=["promoted (top k%)", "all songs"], y=[hit_sel, hit_all]))
    fig.update_layout(title="Share of songs that beat their release-year cohort",
                      yaxis_title="hit rate", yaxis_range=[0, 1])
    return table, fig


with gr.Blocks(title="Spotify popularity: three-method comparison") as demo:
    gr.Markdown(
        "# Which new releases to promote?\n"
        "Target = popularity minus the mean popularity of the song's release year. "
        "Trained on 2000-2015, tuned on 2016-2017, tested on 2018 to May 2020. "
        "All models are loaded from `models/`; nothing is retrained here."
    )

    with gr.Tab("Model comparison"):
        split_dd = gr.Dropdown(["test", "val", "train"], value="test", label="Split")
        mt = gr.Dataframe(label="Metrics (baseline = predict the train mean)")
        model_dd = gr.Dropdown(MODELS, value="sklearn", label="Model")
        scatter = gr.Plot()
        gr.Markdown("### Coefficients: the three methods should agree")
        gr.Markdown(f"Max coefficient difference vs sklearn: "
                    f"manual {REPORT['max_coef_diff']['manual']:.2e}, "
                    f"module {REPORT['max_coef_diff']['module']:.2e}")
        gr.Dataframe(value=coef_table())
        split_dd.change(metrics_table, split_dd, mt)
        model_dd.change(pred_vs_actual, model_dd, scatter)

    with gr.Tab("Distributions"):
        col_dd = gr.Dropdown(DIST_COLS, value="target", label="Column")
        dist_plot = gr.Plot()
        col_dd.change(distribution, col_dd, dist_plot)

    with gr.Tab("Promotion simulator"):
        gr.Markdown("Promote the top k% of the test songs ranked by the model. "
                    "A hit is a song that really beat its cohort (target > 0).")
        pm = gr.Dropdown(MODELS, value="sklearn", label="Model")
        pk = gr.Slider(5, 50, value=20, step=1, label="Promote top k% of songs")
        ptable = gr.Dataframe()
        pplot = gr.Plot()
        pm.change(promote, [pm, pk], [ptable, pplot])
        pk.change(promote, [pm, pk], [ptable, pplot])

    demo.load(metrics_table, split_dd, mt)
    demo.load(pred_vs_actual, model_dd, scatter)
    demo.load(distribution, col_dd, dist_plot)
    demo.load(promote, [pm, pk], [ptable, pplot])

if __name__ == "__main__":
    demo.launch()