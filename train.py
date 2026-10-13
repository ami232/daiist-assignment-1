
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "hotel_bookings.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12,
}

# Known only at or after check-in -> using them leaks the future.
LEAK_COLS = ["reservation_status", "reservation_status_date", "assigned_room_type"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month"].map(MONTHS),
        day=df["arrival_date_day_of_month"],
    ))
    df["children"] = df["children"].fillna(0) # fill the 4 missing children values with 0 (no children)
    df["country"] = df["country"].fillna("UNK") # fill the 488 missing countries with a placeholder label (UNK)
    df = df[(df["adr"] >= 0) & (df["adr"] <= 1000)].copy() # drop impossible rates: negative, and absurd outliers above 1000
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0].copy() # drop bookings with zero people (data errors)
    df = df.drop(columns=LEAK_COLS) # remove the leakage columns defined above
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["total_requests"] = df["total_of_special_requests"]
    # ADR normalized by party size: separates "pricey because big group" from "pricey per head"
    df["adr_per_person"] = df["adr"] / df["total_guests"]   # total_guests >= 1, guaranteed by cleaning
    # A guest's personal cancellation rate from their history (0 for first-timers)
    prev_total = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["cancel_ratio"] = np.where(prev_total > 0,
                                  df["previous_cancellations"] / prev_total, 0.0)
    return df


# adults / children / babies are deliberately NOT here: total_guests = adults + children
# + babies makes them perfectly collinear (VIF = inf, measured on train). total_guests and
# is_family keep that information without the redundancy.
NUMERIC = [
    "lead_time", "total_nights", "total_guests", "adr",
    "booking_changes", "days_in_waiting_list", "total_requests",
    "previous_cancellations", "previous_bookings_not_canceled",
    "required_car_parking_spaces", "arrival_date_week_number",
    "adr_per_person", "cancel_ratio",
]
CATEGORICAL = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
    "is_repeated_guest", "has_agent", "has_company", "is_family",
    "arrival_date_month",
]
COUNTRY = ["country"]                       # high-cardinality (~178 values), handled on its own branch
FEATURES = NUMERIC + CATEGORICAL + COUNTRY
TARGET = "is_canceled"


def time_split(df: pd.DataFrame):
    df = df.sort_values("arrival_date")
    cut1 = df["arrival_date"].quantile(0.70)
    cut2 = df["arrival_date"].quantile(0.85)
    train = df[df["arrival_date"] <= cut1]
    val = df[(df["arrival_date"] > cut1) & (df["arrival_date"] <= cut2)]
    test = df[df["arrival_date"] > cut2]
    return train, val, test


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")), #fills any missing number with that column’s median.
        ("scale", StandardScaler()), #rescales each column to mean 0 and standard deviation 1. This matters because logistic regression weighs features by size.
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")), #fills any missing category with that column’s mode.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), #turns each category into 0/1 columns.
    ])
    # Country has ~178 values. max_categories keeps the most frequent markets and
    # folds the long tail into one grouped bucket. Because the preprocessor is fit
    # on train only, those top markets are learned from training data = no leakage.
    # 11 = the ~10 biggest markets + 1 grouped "infrequent" bucket.
    country = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 max_categories=11)),
    ])
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", cat, CATEGORICAL),
        ("country", country, COUNTRY),
    ])


def auc(y, proba):
    return roc_auc_score(y, proba)
'''Personal note: AUC is the area under that curve, between 0 and 1.

0.5 is a coin flip. The curve is the diagonal line; the model has no discriminating power. That’s why your baseline’s AUC is 0.5.
1.0 is perfect separation. Every canceller is scored higher than every non-canceller.
Logistic models should land around 0.80 to 0.85, meaning solid but not perfect.'''

# ---- sklearn ----
def c_from_lambda(lam, n):
    # Translate our L2 strength (lam) into sklearn's C so all three models use
    # the same penalty. sklearn's C is the INVERSE of regularization strength,
    # and it sums the loss over samples while we average it, hence the 1/(2*lam*n).
    return 1.0 / (2.0 * lam * n)


def fit_sklearn(Xtr, ytr, lam):
    C = c_from_lambda(lam, len(ytr))        # convert lam to the equivalent C for this sample size
    m = LogisticRegression(max_iter=2000, C=C)  # build the model; max_iter high enough to converge
    m.fit(Xtr, ytr)                          # let sklearn find the best weights internally
    return m                                  # hand back the trained model


# ---- manual PyTorch loop ----
def train_manual(Xtr, ytr, lr, lam, epochs=600):
    X = torch.tensor(Xtr, dtype=torch.float32)          # features as a float tensor
    y = torch.tensor(ytr, dtype=torch.float32).reshape(-1, 1)  # labels as a column vector (n rows, 1 col)
    d = X.shape[1]                                        # number of features (columns)
    w = torch.zeros((d, 1), requires_grad=True)          # weights start at zero; track gradients on them
    b = torch.zeros(1, requires_grad=True)               # bias starts at zero; track gradients
    for _ in range(epochs):                              # repeat the gradient-descent step 600 times
        p = torch.sigmoid(X @ w + b)                     # predicted probabilities: squash the linear score to 0-1
        bce = -(y * torch.log(p + 1e-7) + (1 - y) * torch.log(1 - p + 1e-7)).mean()  # binary cross-entropy (how wrong we are); 1e-7 avoids log(0)
        loss = bce + lam * (w ** 2).sum()                # add the L2 penalty on the weights (not the bias)
        loss.backward()                                  # autograd computes the gradient of loss w.r.t. w and b
        with torch.no_grad():                            # pause gradient tracking while we update the weights
            w -= lr * w.grad                             # step the weights downhill by learning-rate * gradient
            b -= lr * b.grad                             # same step for the bias
            w.grad.zero_()                               # clear w's gradient so it doesn't accumulate next loop
            b.grad.zero_()                               # clear b's gradient
    return w.detach(), b.detach()                        # return final weights, detached from the gradient graph


def proba_manual(X, w, b):
    X = torch.tensor(X, dtype=torch.float32)             # features to tensor
    with torch.no_grad():                                # predicting, not training, so no gradients needed
        return torch.sigmoid(X @ w + b).numpy().ravel()  # probabilities as a flat NumPy array


# ---- standard nn.Module + optim ----
class LogReg(torch.nn.Module):                           # logistic regression as a PyTorch module
    def __init__(self, d):
        super().__init__()                               # required: initialize the parent nn.Module
        self.linear = torch.nn.Linear(d, 1)              # one linear layer: d inputs -> 1 output (holds w and b)

    def forward(self, x):
        return self.linear(x)                            # forward pass returns the raw linear score (logit)


def train_standard(Xtr, ytr, lr, lam, epochs=600):
    X = torch.tensor(Xtr, dtype=torch.float32)           # features to tensor
    y = torch.tensor(ytr, dtype=torch.float32).reshape(-1, 1)  # labels as a column vector
    model = LogReg(X.shape[1])                            # create the model sized to the feature count
    opt = torch.optim.Adam(model.parameters(), lr=lr)    # Adam optimizer does the weight updates for us
    bce = torch.nn.BCEWithLogitsLoss()                   # loss that combines sigmoid + cross-entropy, numerically stable
    for _ in range(epochs):                              # training loop
        opt.zero_grad()                                  # clear gradients from the previous step
        out = model(X)                                   # forward pass: raw logits
        loss = bce(out, y) + lam * (model.linear.weight ** 2).sum()  # loss + same hand-added L2 penalty as the manual loop
        loss.backward()                                  # compute gradients
        opt.step()                                        # let Adam update the weights
    return model                                          # return the trained model


def proba_standard(X, model):
    X = torch.tensor(X, dtype=torch.float32)             # features to tensor
    with torch.no_grad():                                # no gradients for prediction
        return torch.sigmoid(model(X)).numpy().ravel()   # apply sigmoid to logits -> probabilities, as flat NumPy


def evaluate(name, y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)                    # turn probabilities into 0/1 decisions at the threshold (default 0.5)
    return {"model": name,                               # label for the results table
            "accuracy": round(accuracy_score(y, pred), 4),   # fraction predicted correctly
            "auc": round(roc_auc_score(y, proba), 4)}        # ranking quality, threshold-free (uses proba, not pred)


def main():
    print("Loading and cleaning...")
    df = engineer(load_and_clean())                      # load + clean, then add engineered features
    train, val, test = time_split(df)                    # split by arrival date: past -> future
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")  # show the split sizes

    pre = build_preprocessor()                           # build the impute/scale/encode pipeline
    Xtr = pre.fit_transform(train[FEATURES])             # FIT on train only (learn medians, scales, categories), then transform
    Xval = pre.transform(val[FEATURES])                  # transform val with train's learned settings (no re-fitting = no leakage)
    Xte = pre.transform(test[FEATURES])                  # transform test the same way
    ytr, yval, yte = (train[TARGET].to_numpy(), val[TARGET].to_numpy(),  # pull the labels out as arrays
                      test[TARGET].to_numpy())

    # ---- Tune L2 strength on validation (using the cheap sklearn fit) ----
    print("Tuning L2 strength on the validation set...")
    LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2]                   # small grid of penalty strengths to try
    best_lam, best_auc = None, -1                        # track the best so far
    for lam in LAMBDAS:                                  # try each penalty strength
        a = auc(yval, fit_sklearn(Xtr, ytr, lam).predict_proba(Xval)[:, 1])  # fit on train, score AUC on validation
        print(f"  lambda={lam:<7} val AUC={a:.4f}")      # report this value's validation AUC
        if a > best_auc:                                 # if it beat the current best...
            best_lam, best_auc = lam, a                  # ...remember it
    print(f"  chosen lambda={best_lam}")                 # the winning penalty strength

    # ---- Tune learning rate on validation for each torch method ----
    print("Tuning learning rates on the validation set...")
    LRS = [0.05, 0.1, 0.5]                               # small grid of learning rates to try
    best_lr_man = max(LRS, key=lambda lr: auc(           # pick the lr that gives the best validation AUC...
        yval, proba_manual(Xval, *train_manual(Xtr, ytr, lr, best_lam))))  # ...for the manual loop (trained at best_lam)
    best_lr_std = max(LRS, key=lambda lr: auc(           # same selection...
        yval, proba_standard(Xval, train_standard(Xtr, ytr, lr, best_lam))))  # ...for the standard model
    print(f"  manual lr={best_lr_man}   standard lr={best_lr_std}")  # the chosen learning rates

    # ---- Final fits on train, same lambda, evaluated on test ----
    results = []                                         # collect one row per model
    results.append({"model": "baseline (majority)",     # naive baseline: always predict "not canceled"
                    "accuracy": round(accuracy_score(yte, np.zeros(len(yte), int)), 4),  # its accuracy on test
                    "auc": 0.5})                         # a no-skill model has AUC 0.5 by definition

    print("Training final models...")
    skl = fit_sklearn(Xtr, ytr, best_lam)               # final sklearn fit on train with the chosen lambda
    results.append(evaluate("sklearn", yte, skl.predict_proba(Xte)[:, 1]))  # score it ONCE on test

    w, b = train_manual(Xtr, ytr, best_lr_man, best_lam)  # final manual fit with chosen lr + lambda
    results.append(evaluate("manual pytorch", yte, proba_manual(Xte, w, b)))  # score on test

    model = train_standard(Xtr, ytr, best_lr_std, best_lam)  # final standard fit with chosen lr + lambda
    results.append(evaluate("standard pytorch", yte, proba_standard(Xte, model)))  # score on test

    res_df = pd.DataFrame(results)                       # assemble the results table
    print("\nTest-set results:")
    print(res_df.to_string(index=False))                # print it without the row index

    # ---- Save everything the app needs (app loads the models, not cached preds) ----
    joblib.dump(pre, ART / "preprocessor.joblib")       # save the fitted preprocessor (app must transform the same way)
    joblib.dump(skl, ART / "sklearn.joblib")            # save the sklearn model
    torch.save({"w": w, "b": b}, ART / "manual.pt")     # save the manual model's weights and bias
    torch.save(model.state_dict(), ART / "standard.pt") # save the standard model's learned parameters
    test[FEATURES + [TARGET]].to_csv(ART / "test.csv", index=False)  # save the test set so the app can run inference on it
    res_df.to_csv(ART / "results.csv", index=False)     # save the results table for the app and the report
    print(f"\nSaved artifacts to {ART}")


if __name__ == "__main__":                              # only run main() when this file is executed directly
    main()