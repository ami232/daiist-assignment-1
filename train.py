"""
Assignment 1 — training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file — main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py train`, this file must, start to finish, with no
manual steps in between:

- Load your committed dataset.
- Apply the feature engineering, preprocessing, and train/val/test split
  consistent with the business framing you wrote in REPORT.md.
- Train the same model — linear regression for a regression target,
  logistic regression for a binary one — three ways, on the same split:
  scikit-learn, a from-scratch PyTorch loop (Session 5's manual approach:
  raw tensors, autograd, a manual gradient-step update), and the standard
  torch.nn.Module + torch.optim workflow.
- Evaluate all three against each other and against a naive baseline.
- Save whatever app.py needs to build its dashboard without retraining
  anything.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
FEATURES = ["ret_1", "ret_5", "ret_10", "volatility_5", "volatility_20",
            "sma_5", "sma_20", "intraday", "range", "volume_20",
            "monday", "friday"]
L2, LR, EPOCHS = 0.01, 0.2, 800


#DATA AND CREATE FEATURES

def prepare_data():
    df = pd.read_csv(ROOT / "data/spy_data.csv", skiprows=[1, 2]).rename(columns={"Price": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("Date").drop_duplicates("Date")
    df = df[(df[["Open", "High", "Low", "Close", "Volume"]] > 0).all(axis=1)].copy()

    r = df["Close"].pct_change()
    df["ret_1"], df["ret_5"], df["ret_10"] = r, df["Close"].pct_change(5), df["Close"].pct_change(10)
    df["volatility_5"], df["volatility_20"] = r.rolling(5).std(), r.rolling(20).std()
    df["sma_5"] = df["Close"] / df["Close"].rolling(5).mean() - 1
    df["sma_20"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["intraday"] = df["Close"] / df["Open"] - 1
    df["range"] = (df["High"] - df["Low"]) / df["Open"]
    df["volume_20"] = df["Volume"] / df["Volume"].rolling(20).mean() - 1
    df["monday"] = (df["Date"].dt.weekday == 0).astype(int)
    df["friday"] = (df["Date"].dt.weekday == 4).astype(int)

    # Target being SPY closing above its opening price tomorrow?
    df["next_return"] = df["Close"].shift(-1) / df["Open"].shift(-1) - 1
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["next_return"])
    df["target"] = (df["next_return"] > 0).astype(int)
    return df.reset_index(drop=True)

#COST AND EVALUATION
def cost(y, probability, threshold):
    pred = probability >= threshold
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    return float((3 * fp + fn) * 100 / len(y))

def evaluate(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "roc_auc": round(roc_auc_score(y, probability), 4),
            "cost_per_100": round(cost(y, probability, threshold), 2),
            "false_positives": int(np.sum((pred == 1) & (y == 0))),
            "false_negatives": int(np.sum((pred == 0) & (y == 1)))}

#TRAINING IN THREE WAYS
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    df = prepare_data()
    i, j = int(len(df) * .70), int(len(df) * .85)
    train, val, test = df.iloc[:i-1].copy(), df.iloc[i:j-1].copy(), df.iloc[j:].copy()

    for label, part in [("train", train), ("validation", val), ("test", test)]:
        part["split"] = label

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[FEATURES])
    Xval, Xte = scaler.transform(val[FEATURES]), scaler.transform(test[FEATURES])
    ytr, yval, yte = train.target.to_numpy(), val.target.to_numpy(), test.target.to_numpy()

    #1. SCIKIT-LEARN
    sklearn_model = LogisticRegression(C=1 / (len(ytr) * L2), max_iter=2000)
    sklearn_model.fit(Xtr, ytr)

    x = torch.tensor(Xtr, dtype=torch.float64)
    y = torch.tensor(ytr[:, None], dtype=torch.float64)
    w = torch.zeros((x.shape[1], 1), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    #2. MANUAL PYTORCH
    for _ in range(EPOCHS):
        loss = nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + L2 / 2 * w.square().sum()
        loss.backward()
        with torch.no_grad():
            w -= LR * w.grad
            b -= LR * b.grad
            w.grad.zero_()
            b.grad.zero_()

    #3. STANDARD PYTORCH
    model = nn.Linear(x.shape[1], 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y) + L2 / 2 * model.weight.square().sum()
        loss.backward()
        optimizer.step()

    #COMPARE
    def probabilities(X):
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float64)
            return {"scikit-learn": sklearn_model.predict_proba(X)[:, 1],
                    "manual PyTorch": torch.sigmoid(t @ w + b).numpy().ravel(),
                    "standard PyTorch": torch.sigmoid(model(t)).numpy().ravel()}

    val_prob = probabilities(Xval)["scikit-learn"]
    threshold = min(np.arange(.35, .951, .01), key=lambda t: cost(yval, val_prob, t))
    preds = probabilities(Xte)
    preds["majority baseline"] = np.full(len(yte), ytr.mean())

    scores = {name: evaluate(yte, p, .5 if name == "majority baseline" else threshold)
              for name, p in preds.items()}

    #SAVE RESULTS
    joblib.dump(sklearn_model, OUT / "sklearn_model.joblib")
    joblib.dump(scaler, OUT / "scaler.joblib")
    np.savez(OUT / "manual_model.npz", weights=w.detach().numpy(), bias=b.detach().numpy())
    torch.save(model.state_dict(), OUT / "standard_model.pt")
    pd.concat([train, val, test]).to_csv(OUT / "prepared_data.csv", index=False)
    summary = {"features": FEATURES, "threshold": round(float(threshold), 2),
               "L2": L2, "learning_rate": LR, "epochs": EPOCHS, "test_metrics": scores,
               "date_ranges": {p: [str(d.Date.min().date()), str(d.Date.max().date())]
                               for p, d in [("train", train), ("validation", val), ("test", test)]},
               "max_gap_sklearn_manual": float(np.max(np.abs(preds["scikit-learn"] - preds["manual PyTorch"]))),
               "max_gap_manual_standard": float(np.max(np.abs(preds["manual PyTorch"] - preds["standard PyTorch"])))}

    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"Observations: {len(df)} | Features: {len(FEATURES)} | Threshold: {threshold:.2f}")
    print(pd.DataFrame(scores).T.to_string())
    print("Models and evaluation saved to artifacts/")

if __name__ == "__main__":
    main()
