import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# ---------- Load what train.py saved (nothing is trained here) ----------
with open("artifacts/config.json") as f:
    cfg = json.load(f)
COSTE_FN, COSTE_FP = cfg["coste_fn"], cfg["coste_fp"]
FEATURES = cfg["features"]

X = {s: pd.read_csv(f"artifacts/X_{s}.csv") for s in ("train", "val", "test")}
meta = {s: pd.read_csv(f"artifacts/meta_{s}.csv") for s in ("train", "val", "test")}

scaler = joblib.load("models/scaler.joblib")
sk_model = joblib.load("models/sklearn_logreg.joblib")
manual = torch.load("models/torch_manual.pt")


class LogReg(nn.Module):  # same class as in train.py, needed to load the weights
    def __init__(self, n_in):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


nn_model = LogReg(len(FEATURES))
nn_model.load_state_dict(torch.load("models/torch_nn.pt"))
nn_model.eval()

# ---------- Predictions on the test set (the movies the models never saw) ----------
Xte = scaler.transform(X["test"])
Xte_t = torch.tensor(Xte, dtype=torch.float32)
y_test = meta["test"]["success"].values

with torch.no_grad():
    PROBS = {
        "scikit-learn": sk_model.predict_proba(Xte)[:, 1],
        "PyTorch manual": torch.sigmoid(Xte_t @ manual["w"] + manual["b"]).numpy(),
        "PyTorch nn.Module": torch.sigmoid(nn_model(Xte_t)).numpy(),
    }
AUCS = {name: roc_auc_score(y_test, p) for name, p in PROBS.items()}
DEFAULT_THR = cfg["umbrales"]["sklearn"]


def confusion(y_real, prob, thr):
    pred = prob >= thr
    tp = int((pred & (y_real == 1)).sum())
    fp = int((pred & (y_real == 0)).sum())
    fn = int(((~pred) & (y_real == 1)).sum())
    tn = int(((~pred) & (y_real == 0)).sum())
    return tn, fp, fn, tp


def update(model_name, thr):
    # Table comparing the three models at the chosen threshold
    rows = []
    for name, prob in PROBS.items():
        tn, fp, fn, tp = confusion(y_test, prob, thr)
        rows.append({
            "Model": name,
            "Accuracy": round((tp + tn) / len(y_test), 3),
            "AUC": round(AUCS[name], 3),
            "Missed hits (FN)": fn,
            "Screens to flops (FP)": fp,
            "Business cost": COSTE_FN * fn + COSTE_FP * fp,
        })
    table = pd.DataFrame(rows)

    # Confusion matrix for the selected model
    prob = PROBS[model_name]
    tn, fp, fn, tp = confusion(y_test, prob, thr)
    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted flop", "Predicted hit"],
        y=["Actual flop", "Actual hit"],
        text=[[tn, fp], [fn, tp]], texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    cm.update_layout(title=f"Confusion matrix: {model_name} (threshold {thr:.2f})",
                     yaxis_autorange="reversed", height=380)

    # Predicted probability vs actual revenue
    df = pd.DataFrame({
        "Predicted probability of hit": prob,
        "Actual revenue ($)": meta["test"]["revenue"].clip(lower=1).values,
        "Actual": np.where(y_test == 1, "Hit", "Flop"),
        "Title": meta["test"]["title"].values,
    })
    sc = px.scatter(df, x="Predicted probability of hit", y="Actual revenue ($)",
                    color="Actual", hover_name="Title", log_y=True,
                    color_discrete_map={"Hit": "#1f77b4", "Flop": "#d62728"},
                    title=f"Prediction vs actual: {model_name}")
    sc.add_vline(x=thr, line_dash="dash", annotation_text="threshold")
    sc.add_hline(y=cfg["mediana_revenue"], line_dash="dot",
                 annotation_text="success = above train median revenue")
    sc.update_layout(height=450)
    return table, cm, sc


def plot_target():
    rev = meta["train"]["revenue"].clip(lower=1)
    fig = px.histogram(np.log10(rev), nbins=40,
                       labels={"value": "log10(revenue)"},
                       title="Target: revenue (log scale), training set")
    fig.add_vline(x=np.log10(cfg["mediana_revenue"]), line_dash="dash",
                  annotation_text="median = success cutoff")
    fig.update_layout(showlegend=False, height=380)
    return fig


def plot_feature(feature):
    df = pd.DataFrame({
        feature: X["train"][feature],
        "Outcome": np.where(meta["train"]["success"] == 1, "Hit", "Flop"),
    })
    fig = px.histogram(df, x=feature, color="Outcome", barmode="overlay", opacity=0.65,
                       color_discrete_map={"Hit": "#1f77b4", "Flop": "#d62728"},
                       title=f"Distribution of {feature} by outcome (training set)")
    fig.update_layout(height=380)
    return fig


with gr.Blocks(title="Movie screens: hit or flop?") as demo:
    gr.Markdown(
        "# Which movies deserve more screens?\n"
        "Three versions of the same logistic regression, evaluated on movies from the most "
        "recent part of the dataset (test set). A *hit* is a movie whose revenue is above the "
        f"training median. Missing a hit costs **{COSTE_FN}**, giving screens to a flop costs "
        f"**{COSTE_FP}**."
    )
    with gr.Tab("Models and threshold"):
        with gr.Row():
            model_dd = gr.Dropdown(list(PROBS), value="scikit-learn", label="Model to inspect")
            thr_slider = gr.Slider(0.01, 0.99, value=DEFAULT_THR, step=0.01,
                                   label="Decision threshold")
        table_out = gr.Dataframe(label="Comparison of the three models at this threshold")
        with gr.Row():
            cm_out = gr.Plot()
            sc_out = gr.Plot()
        inputs = [model_dd, thr_slider]
        outputs = [table_out, cm_out, sc_out]
        model_dd.change(update, inputs, outputs)
        thr_slider.change(update, inputs, outputs)
        demo.load(update, inputs, outputs)
    with gr.Tab("Data"):
        gr.Plot(value=plot_target())
        feat_dd = gr.Dropdown(FEATURES, value="log_budget", label="Feature")
        feat_out = gr.Plot()
        feat_dd.change(plot_feature, feat_dd, feat_out)
        demo.load(plot_feature, feat_dd, feat_out)

demo.launch()
