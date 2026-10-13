from pathlib import Path          # for building file paths that work on any machine
import numpy as np                # numerical arrays and math
import pandas as pd               # dataframes (reading the test CSV)
import joblib                     # loading the saved sklearn model and preprocessor
import plotly.graph_objects as go # interactive plots for the dashboard
import gradio as gr              # the web UI framework
import torch                     # loading and running the two PyTorch models

ROOT = Path(__file__).resolve().parent   # folder this script lives in (repo root)
ART = ROOT / "artifacts"                 # where train.py saved the models and data

# Must match train.py's feature lists / target.
# adults / children / babies removed: perfectly collinear with total_guests (VIF = inf).
NUMERIC = [                              # numeric columns, in the same order train.py used
    "lead_time", "total_nights", "total_guests", "adr",
    "booking_changes", "days_in_waiting_list", "total_requests",
    "previous_cancellations", "previous_bookings_not_canceled",
    "required_car_parking_spaces", "arrival_date_week_number",
    "adr_per_person", "cancel_ratio",
]
CATEGORICAL = [                          # categorical columns, matching train.py
    "hotel", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
    "is_repeated_guest", "has_agent", "has_company", "is_family",
    "arrival_date_month",
]
COUNTRY = ["country"]                     # high-cardinality column, matching train.py
FEATURES = NUMERIC + CATEGORICAL + COUNTRY  # full input list the preprocessor expects
TARGET = "is_canceled"                   # the column being predicted

# ---- Business cost assumptions (edit to match REPORT.md) ----
COST_FN = 100.0   # euros lost when a cancellation slips through (empty room); near one ADR, what the room would have earned
COST_FP = 40.0    # euros of goodwill cost when a guest who would have shown up is wrongly flagged


class LogReg(torch.nn.Module):          # must redefine the model class to load the standard model's weights
    def __init__(self, d):
        super().__init__()               # initialize the parent nn.Module
        self.linear = torch.nn.Linear(d, 1)  # one linear layer, d inputs -> 1 output (same as in train.py)

    def forward(self, x):
        return self.linear(x)            # forward pass returns the raw logit


# ---- Load artifacts and run each model once on the test set ----
pre = joblib.load(ART / "preprocessor.joblib")   # load the fitted preprocessor (same transforms as training)
test = pd.read_csv(ART / "test.csv")             # load the saved test set
y_true = test[TARGET].to_numpy()                 # the true cancel/not-cancel labels
X = pre.transform(test[FEATURES])                # transform raw test features the exact way training did

skl = joblib.load(ART / "sklearn.joblib")        # load the trained sklearn model
manual = torch.load(ART / "manual.pt")           # load the manual model's saved {w, b} dict
std_model = LogReg(X.shape[1])                    # build an empty standard model sized to the features
std_model.load_state_dict(torch.load(ART / "standard.pt"))  # load its trained weights into it
std_model.eval()                                 # set to evaluation mode (no training behaviour)

with torch.no_grad():                            # predicting, not training, so skip gradient tracking
    Xt = torch.tensor(X, dtype=torch.float32)    # features as a float tensor
    p_manual = torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy().ravel()  # manual model probabilities
    p_standard = torch.sigmoid(std_model(Xt)).numpy().ravel()                 # standard model probabilities

PROBAS = {                                        # one probability array per model, keyed by name
    "scikit-learn": skl.predict_proba(X)[:, 1],   # sklearn's probability of the "cancel" class
    "manual pytorch": p_manual,                   # from the manual model
    "standard pytorch": p_standard,               # from the standard model
}
results_table = pd.read_csv(ART / "results.csv")  # the metrics table train.py saved, for display
lead_time = test["lead_time"].to_numpy()          # one feature pulled out for the distribution plot


def confusion_counts(y, proba, thr):
    pred = (proba >= thr).astype(int)             # turn probabilities into 0/1 decisions at this threshold
    tp = int(((pred == 1) & (y == 1)).sum())      # true positives: predicted cancel AND really cancelled
    tn = int(((pred == 0) & (y == 0)).sum())      # true negatives: predicted show AND really showed
    fp = int(((pred == 1) & (y == 0)).sum())      # false positives: predicted cancel BUT really showed (false alarm)
    fn = int(((pred == 0) & (y == 1)).sum())      # false negatives: predicted show BUT really cancelled (missed)
    return tp, tn, fp, fn


def calibration_figure():
    fig = go.Figure()                             # start an empty plot
    bins = np.linspace(0, 1, 11)                  # 10 probability buckets from 0 to 1
    centers = (bins[:-1] + bins[1:]) / 2          # the midpoint of each bucket, for the x-axis
    for name, proba in PROBAS.items():            # draw one line per model
        idx = np.clip(np.digitize(proba, bins) - 1, 0, len(centers) - 1)  # which bucket each prediction falls in
        actual = [y_true[idx == b].mean() if (idx == b).any() else np.nan  # actual cancel rate within each bucket
                  for b in range(len(centers))]
        fig.add_trace(go.Scatter(x=centers, y=actual, mode="lines+markers", name=name))  # plot predicted vs actual
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",            # the diagonal = perfect calibration
                             line=dict(dash="dash", color="gray"), name="perfect"))
    fig.update_layout(title="Predicted probability vs actual rate (calibration)",  # labels
                      xaxis_title="Predicted probability of cancelling",
                      yaxis_title="Actual cancellation rate", height=420)
    return fig                                     # hand back the finished figure


def distribution_figure():
    fig = go.Figure()                             # empty plot
    fig.add_trace(go.Histogram(x=lead_time[y_true == 0], name="showed up",     # lead-time distribution for show-ups
                               opacity=0.6, nbinsx=40))
    fig.add_trace(go.Histogram(x=lead_time[y_true == 1], name="canceled",      # lead-time distribution for cancellations
                               opacity=0.6, nbinsx=40))
    fig.update_layout(barmode="overlay", title="Lead time by outcome (test set)",  # overlay the two so they compare
                      xaxis_title="Lead time (days before arrival)",
                      yaxis_title="Number of bookings", height=420)
    return fig


def threshold_view(model_name, thr):
    proba = PROBAS[model_name]                     # the chosen model's probabilities
    tp, tn, fp, fn = confusion_counts(y_true, proba, thr)  # counts at the current threshold
    cm = go.Figure(data=go.Heatmap(               # draw the confusion matrix as a 2x2 heatmap
        z=[[tn, fp], [fn, tp]],                    # the four counts laid out in the grid
        x=["pred: show", "pred: cancel"],          # columns = what the model predicted
        y=["actual: show", "actual: cancel"],      # rows = what really happened
        text=[[tn, fp], [fn, tp]], texttemplate="%{text}",  # print the numbers on the cells
        colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix @ threshold {thr:.2f}", height=380)  # title shows current threshold
    cost = fp * COST_FP + fn * COST_FN             # total business cost = each error type times its cost
    caught = tp / (tp + fn) if (tp + fn) else 0    # recall: fraction of real cancellations the model caught
    summary = (                                    # markdown block summarizing the business impact
        f"### Business impact at threshold {thr:.2f}\n"
        f"- Cancellations caught: **{caught:.0%}** ({tp} of {tp + fn})\n"
        f"- False alarms (guests wrongly flagged): **{fp}**\n"
        f"- Missed cancellations: **{fn}**\n"
        f"- Estimated cost: **{cost:,.0f} EUR**  "
        f"(FN x {COST_FN:.0f} + FP x {COST_FP:.0f})"
    )
    return cm, summary                             # return the matrix and the text for the UI to display


with gr.Blocks(title="Hotel cancellation dashboard") as demo:   # build the app; 'demo' is what main.py launches
    gr.Markdown("# Hotel booking cancellation model")           # page heading
    gr.Markdown("Logistic regression trained three ways. The preprocessor and "  # short description
                "all three models are loaded from `artifacts/` and run here; "
                "nothing is retrained.")

    with gr.Tab("Model comparison"):               # first tab
        gr.Dataframe(results_table, label="Test-set metrics")  # show the metrics table
        gr.Plot(calibration_figure())              # show the calibration plot

    with gr.Tab("Distributions"):                  # second tab
        gr.Plot(distribution_figure())             # show the lead-time distribution

    with gr.Tab("Threshold & business cost"):      # third tab (the interactive one)
        model_pick = gr.Dropdown(list(PROBAS.keys()), value="scikit-learn",  # choose which model to inspect
                                 label="Model")
        thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Decision threshold")  # the threshold slider
        cm_plot = gr.Plot()                        # placeholder for the confusion matrix
        cost_md = gr.Markdown()                     # placeholder for the cost summary
        for ctrl in (model_pick, thr):             # whenever the dropdown OR the slider changes...
            ctrl.change(threshold_view, [model_pick, thr], [cm_plot, cost_md])  # ...recompute and update both outputs
        demo.load(threshold_view, [model_pick, thr], [cm_plot, cost_md])  # also populate them once when the app opens


if __name__ == "__main__":                         # only runs when executed directly (via main.py app)
    demo.launch(server_name="0.0.0.0", server_port=7860)  # start the server on port 7860 (CI checks this port)