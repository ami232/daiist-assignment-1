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

"""Stage 1: load, time-split, preprocess, baseline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


import json
from pathlib import Path

import joblib
import torch
from torch import nn
from sklearn.linear_model import Ridge

MODELS = Path("models")

DATA = "data/AmesHousing.csv"
TARGET = "SalePrice"

# ---- Split (our decision: train 2006-08, validate 2009, test 2010) ----
TRAIN_YEARS, VAL_YEAR, TEST_YEAR = [2006, 2007, 2008], 2009, 2010


# Columns an agent would NOT know when pricing a house before it sells.
LEAKY_COLS = ["Sale Type", "Sale Condition", "Mo Sold"]          

# Identifiers carry no information about the house.
ID_COLS = ["Order", "PID"]

#I decided to remove Yr sold because an agent wouldn't know this beforehand
TIME_COLS = ["Yr Sold"]

NEAR_EMPTY_COLS = ["Pool QC", "Misc Feature", "Alley", "Fence"]
# Redundant: Gr Liv Area is exactly 1st Flr SF + 2nd Flr SF + Low Qual Fin SF,
# so we keep the total and drop its parts. Garage Cars and Garage Area measure
# nearly the same thing (keep the area), and Garage Yr Blt overlaps Year Built
# and is empty for every house with no garage.
REDUNDANT_COLS = ["1st Flr SF", "2nd Flr SF", "Low Qual Fin SF",
                  "Garage Cars", "Garage Yr Blt"]
# Experiment switch: does the model still work without assessor ratings?
USE_RATINGS = True
RATING_COLS = ["Overall Qual", "Overall Cond", "Exter Qual", "Exter Cond",
               "Bsmt Qual", "Bsmt Cond", "Heating QC", "Kitchen Qual",
               "Fireplace Qu", "Garage Qual", "Garage Cond"]



def load():
    # "NA" is a real category (feature absent); only empty cells are missing
    return pd.read_csv(DATA, keep_default_na=False, na_values=[""])


QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
TEXT_RATING_COLS = ["Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond",
                    "Heating QC", "Kitchen Qual", "Fireplace Qu",
                    "Garage Qual", "Garage Cond"]


def engineer(df):
    df = df.copy()

    # A. Codes, not quantities: 60 is not "3x" 20, so treat it as a category
    df["MS SubClass"] = df["MS SubClass"].astype(str)

    # B. Age (clip: a few houses sold before they were finished)
    df["House Age"] = (df["Yr Sold"] - df["Year Built"]).clip(lower=0)
    df["Years Since Remodel"] = (df["Yr Sold"] - df["Year Remod/Add"]).clip(lower=0)

    # C. Combine related columns
    df["Total Baths"] = (df["Full Bath"] + 0.5 * df["Half Bath"]
                         + df["Bsmt Full Bath"].fillna(0)
                         + 0.5 * df["Bsmt Half Bath"].fillna(0))
    df["Total SF"] = df["Gr Liv Area"] + df["Total Bsmt SF"].fillna(0)
    df["Porch SF"] = df[["Open Porch SF", "Enclosed Porch",
                         "3Ssn Porch", "Screen Porch"]].sum(axis=1)

    # D. Skew: the target is log price, so log the right-skewed sizes too
    df["Log Lot Area"] = np.log1p(df["Lot Area"])
    df["Log Gr Liv Area"] = np.log1p(df["Gr Liv Area"])

    # E. Rating letters have an order; one-hot throws that away
    for c in TEXT_RATING_COLS:
        df[c] = df[c].map(QUALITY_MAP)

    return df


def split(df):
    in_train = df["Yr Sold"].isin(TRAIN_YEARS)
    # Training only: drop large unfinished (Partial) houses that sold far below
    # what their size implies (3 rows; see REPORT.md)
    outlier = (df["Gr Liv Area"] > 4000) & (df["Sale Condition"] == "Partial")
    train = df[in_train & ~outlier]
    val = df[df["Yr Sold"] == VAL_YEAR]
    test = df[df["Yr Sold"] == TEST_YEAR]
    return train, val, test


def make_xy(df):
    drop = LEAKY_COLS + ID_COLS + TIME_COLS + NEAR_EMPTY_COLS + REDUNDANT_COLS + [TARGET]
    if not USE_RATINGS:
        drop += RATING_COLS
    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = np.log(df[TARGET].to_numpy())  # train on log price (skew ~0 vs 1.74 raw)
    return X, y

def build_preprocessor(X_train):
    num = X_train.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X_train.columns if c not in num]
    return ColumnTransformer([
        # Lot Frontage (17% empty): fill with the training median and add a
        # 0/1 "was missing" column, so the model can see which houses were blank.
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                fill_value="Missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat),
    ])


def metrics(price_true, price_pred):
    """All on the dollar scale, so every model is judged the same way."""
    ape = np.abs(price_pred - price_true) / price_true
    return {
        "median_APE_%": round(100 * float(np.median(ape)), 2),
        "within_10%": round(100 * float(np.mean(ape <= 0.10)), 1),
        "RMSE_$": round(float(np.sqrt(np.mean((price_pred - price_true) ** 2)))),
    }


def get_data():
    """Returns arrays ready for any of the three models, plus the fitted
    preprocessor. Preprocessing is fit on TRAIN ONLY."""
    df = engineer(load())
    tr, va, te = split(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = make_xy(tr), make_xy(va), make_xy(te)
    pre = build_preprocessor(Xtr)
    Xtr = pre.fit_transform(Xtr).astype(np.float32)
    Xva = pre.transform(Xva).astype(np.float32)
    Xte = pre.transform(Xte).astype(np.float32)
    return pre, (Xtr, ytr), (Xva, yva), (Xte, yte)


# ---- add below get_data() ----
def to_price(log_pred):
    return np.exp(log_pred)


def train_sklearn(Xtr, ytr, Xva, yva, alphas=(0.1, 1, 10, 100, 1000)):
    """Pick the regularization strength on the VALIDATION set, never the test."""
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        err = metrics(np.exp(yva), to_price(m.predict(Xva)))["median_APE_%"]
        print(f"  alpha={a:<7} val median APE: {err}%")
        if best is None or err < best[0]:
            best = (err, a, m)
    return best[2], best[1]


def train_manual(Xtr, ytr, alpha, lr=0.05, epochs=20000):
    """Raw tensors, autograd, manual gradient step (Session 5 style)."""
    X = torch.tensor(Xtr)
    y = torch.tensor(ytr, dtype=torch.float32)
    n = len(y)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for epoch in range(epochs):
        pred = X @ w + b
        # MSE + L2 penalty on weights (not the bias). alpha/n makes this the
        # same objective sklearn's Ridge minimizes.
        loss = ((pred - y) ** 2).mean() + (alpha / n) * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if epoch % 1000 == 0:
            print(f"  manual epoch {epoch}: loss {loss.item():.5f}")
    return w.detach(), b.detach()


class LinReg(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)   # same start as the manual loop
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def train_nn(Xtr, ytr, alpha, lr=0.05, epochs=20000):
    """Standard workflow: nn.Module + loss function + torch.optim."""
    X = torch.tensor(Xtr)
    y = torch.tensor(ytr, dtype=torch.float32)
    n = len(y)
    model = LinReg(X.shape[1])
    # SGD's weight_decay adds wd*w to the gradient, which equals the gradient
    # of (alpha/n)*||w||^2 when wd = 2*alpha/n. No decay on the bias.
    opt = torch.optim.SGD([
        {"params": model.linear.weight, "weight_decay": 2 * alpha / n},
        {"params": model.linear.bias, "weight_decay": 0.0},
    ], lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        if epoch % 1000 == 0:
            print(f"  nn epoch {epoch}: loss {loss.item():.5f}")
    return model


def main():
    pre, (Xtr, ytr), (Xva, yva), (Xte, yte) = get_data()
    print("Rows  train/val/test:", len(ytr), len(yva), len(yte))
    print("Features after preprocessing:", Xtr.shape[1])

    print("\nTuning alpha (scikit-learn, on validation):")
    sk_model, alpha = train_sklearn(Xtr, ytr, Xva, yva)
    print("Chosen alpha:", alpha)

    print("\nTraining manual PyTorch loop:")
    w, b = train_manual(Xtr, ytr, alpha)
    print("\nTraining nn.Module + optim:")
    nn_model = train_nn(Xtr, ytr, alpha)

    def preds(X):
        Xt = torch.tensor(X)
        with torch.no_grad():
            return {
                "scikit-learn": sk_model.predict(X),
                "manual PyTorch": (Xt @ w + b).numpy(),
                "nn.Module": nn_model(Xt).numpy(),
            }

    median_price = float(np.exp(np.median(ytr)))
    results, test_preds = {}, preds(Xte)
    for split_name, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
        price = np.exp(y)
        results[split_name] = {"baseline": metrics(price, np.full_like(price, median_price))}
        for name, p in preds(X).items():
            results[split_name][name] = metrics(price, to_price(p))

    for split_name, table in results.items():
        print(f"\n=== {split_name} ===")
        for name, m in table.items():
            print(f"{name:<16}", m)

    # Do the three agree? Max gap in predicted log price on the test set
    p = test_preds
    print("\nMax |log-price gap| on test: sklearn vs manual:",
          float(np.abs(p["scikit-learn"] - p["manual PyTorch"]).max()),
          "| sklearn vs nn:",
          float(np.abs(p["scikit-learn"] - p["nn.Module"]).max()))

    # Save everything app.py needs (it must never retrain)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(pre, MODELS / "preprocessor.joblib")
    joblib.dump(sk_model, MODELS / "sklearn_model.joblib")
    torch.save({"w": w, "b": b}, MODELS / "manual_pytorch.pt")
    torch.save(nn_model.state_dict(), MODELS / "nn_module.pt")
    np.savez(MODELS / "test_predictions.npz", y_true=np.exp(yte),
             **{k: to_price(v) for k, v in test_preds.items()})
    (MODELS / "results.json").write_text(json.dumps(
        {"alpha": alpha, "results": results}, indent=2))
    print("\nSaved models to", MODELS)


if __name__ == "__main__":
    main()