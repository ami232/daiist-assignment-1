"""
Assignment 1 - training pipeline: predict whether a Resort Hotel booking
will be cancelled, with logistic regression trained three ways.

Run with:  uv run python main.py train

What this file does, in order:
  1. Load the bookings (data/resort_hotel_bookings.csv).
  2. Remove columns that would only be known AFTER the guest arrives (leakage).
  3. Build simple, explainable features.
  4. Split by arrival date: past -> train, next months -> validation, latest -> test.
  5. Train the SAME logistic regression three ways:
       a) scikit-learn
       b) PyTorch "by hand" (raw tensors + autograd + our own update step)
       c) PyTorch "standard" (nn.Module + optim.SGD)
  6. Compare them with each other and with a naive baseline.
  7. Pick a decision threshold that minimises the hotel's cost.
  8. Save everything the dashboard needs into artifacts/ (app.py never trains).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "resort_hotel_bookings.csv"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Business settings (see REPORT.md)
# ---------------------------------------------------------------------------
# Missed cancellation (we predicted "will come", guest cancelled):
#   the room stays empty -> lose about one night of revenue.
COST_MISSED_CANCEL = 100.0  # false negative, in euros
# False alarm (we predicted "will cancel", guest came):
#   wasted retention call / small risk from overbooking.
COST_FALSE_ALARM = 30.0  # false positive, in euros

# Time-based split boundaries (by arrival date)
TRAIN_END = "2017-01-01"  # train: Jul 2015 - Dec 2016
VAL_END = "2017-05-01"    # validation: Jan - Apr 2017, test: May - Aug 2017

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)

    # One real date column, built from year / month name / day.
    month_num = df["arrival_date_month"].map(lambda m: MONTHS.index(m) + 1)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"], month=month_num, day=df["arrival_date_day_of_month"]))

    # Small data-quality fixes
    df["children"] = df["children"].fillna(0)              # missing -> no children
    df["country"] = df["country"].fillna("Unknown")
    df["adr"] = df["adr"].clip(lower=0)                     # a negative price is an error
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]  # bookings with 0 guests
    df = df[(df["stays_in_weekend_nights"] + df["stays_in_week_nights"]) > 0]  # 0-night stays
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2 + 3. Remove leakage, build features
# ---------------------------------------------------------------------------
# Columns we must NOT use, because the hotel only learns them later:
LEAKAGE = [
    "reservation_status",           # literally says "Canceled"
    "reservation_status_date",      # date the status was last updated
    "assigned_room_type",           # decided at check-in
    "required_car_parking_spaces",  # recorded on arrival (0% cancels if > 0)
]

NUMERIC = [
    "log_lead_time", "total_nights", "weekend_share", "total_guests", "has_kids",
    "adr", "price_per_person_night", "is_repeated_guest", "has_cancelled_before",
    "previous_bookings_not_canceled", "booking_changes", "log_days_in_waiting_list",
    "total_of_special_requests", "has_agent", "has_company",
]
CATEGORICAL = [
    "arrival_month", "deposit_type", "market_segment", "customer_type", "meal",
    "distribution_channel", "country_group", "reserved_room_type",
]


def build_features(df: pd.DataFrame, top_countries: list[str]) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    # Lead time is very skewed (0 to 700+ days) -> log squashes the long tail.
    f["log_lead_time"] = np.log1p(df["lead_time"])
    # Shape of the stay
    f["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    f["weekend_share"] = df["stays_in_weekend_nights"] / f["total_nights"]
    # Shape of the group
    f["total_guests"] = df["adults"] + df["children"] + df["babies"]
    f["has_kids"] = ((df["children"] + df["babies"]) > 0).astype(int)
    # Price
    f["adr"] = df["adr"]
    f["price_per_person_night"] = df["adr"] / f["total_guests"]
    # Customer history
    f["is_repeated_guest"] = df["is_repeated_guest"]
    f["has_cancelled_before"] = (df["previous_cancellations"] > 0).astype(int)
    f["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].clip(upper=10)
    f["booking_changes"] = df["booking_changes"].clip(upper=5)
    f["log_days_in_waiting_list"] = np.log1p(df["days_in_waiting_list"])
    f["total_of_special_requests"] = df["total_of_special_requests"]
    # 300+ agent / company IDs -> just "was there one?"
    f["has_agent"] = df["agent"].notna().astype(int)
    f["has_company"] = df["company"].notna().astype(int)
    # Categories
    f["arrival_month"] = df["arrival_date_month"]
    for c in ["deposit_type", "market_segment", "customer_type", "meal",
              "distribution_channel", "reserved_room_type"]:
        f[c] = df[c]
    # 120+ countries -> the 8 most common (in TRAIN data) + "Other"
    f["country_group"] = df["country"].where(df["country"].isin(top_countries), "Other")
    return f


# ---------------------------------------------------------------------------
# 5. The three implementations
# ---------------------------------------------------------------------------
def train_sklearn(X, y, C):
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X, y)
    return model


def l2_strength(C, n):
    """sklearn minimises  sum(loss) + 1/(2C)*||w||^2.
    Divide by n to get the 'average loss' form PyTorch uses:
        mean(loss) + (1/(2*C*n)) * ||w||^2
    so the same C gives the same model in all three versions."""
    return 1.0 / (C * n)


def train_torch_manual(X, y, C, lr, epochs, snapshot_at=()):
    """Logistic regression with raw tensors: no nn.Module, no optimizer.
    snapshot_at: epochs at which to keep a copy of (w, b), so we can show
    how the model slowly converges towards the sklearn solution."""
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    lam = l2_strength(C, len(y))

    w = torch.zeros(X.shape[1], requires_grad=True)  # one weight per feature
    b = torch.zeros(1, requires_grad=True)           # intercept
    history, snapshots = [], {}
    for epoch in range(epochs):
        z = X @ w + b                       # linear score
        p = torch.sigmoid(z)                # -> probability of cancel
        eps = 1e-7
        bce = -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()
        loss = bce + 0.5 * lam * (w ** 2).sum()   # + L2 penalty (not on b)
        loss.backward()                     # autograd computes dloss/dw, dloss/db
        with torch.no_grad():               # our own gradient-descent step
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        history.append(loss.item())
        if epoch + 1 in snapshot_at:
            snapshots[epoch + 1] = (w.detach().clone(), b.detach().clone())
    return w.detach(), b.detach(), history, snapshots


class LogReg(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)
        torch.nn.init.zeros_(self.linear.weight)  # same start as manual version
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(1)  # raw score (logit)


def train_torch_standard(X, y, C, lr, epochs):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    lam = l2_strength(C, len(y))

    model = LogReg(X.shape[1])
    loss_fn = torch.nn.BCEWithLogitsLoss()  # sigmoid + log-loss in one stable step
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y) + 0.5 * lam * (model.linear.weight ** 2).sum()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return model, history


def torch_manual_proba(w, b, X):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X, dtype=torch.float32) @ w + b).numpy()


def torch_standard_proba(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# ---------------------------------------------------------------------------
# Helpers for evaluation
# ---------------------------------------------------------------------------
def business_cost(y, p, threshold):
    pred = (p >= threshold).astype(int)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return fn * COST_MISSED_CANCEL + fp * COST_FALSE_ALARM


def best_threshold(y, p):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [business_cost(y, p, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def scores(y, p, threshold):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), 4),
        "accuracy@0.5": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
        "threshold": threshold,
        "cost_eur": business_cost(y, p, threshold),
        "cost_per_booking_eur": round(business_cost(y, p, threshold) / len(y), 2),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    df = df.drop(columns=LEAKAGE)
    print(f"Loaded {len(df):,} bookings, cancel rate {df.is_canceled.mean():.1%}")

    # 4. Time-based split (decided before looking at any model results)
    df["split"] = np.where(df["arrival_date"] < TRAIN_END, "train",
                           np.where(df["arrival_date"] < VAL_END, "val", "test"))
    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {name:5s}: {len(part):6,} rows, {part.arrival_date.min():%Y-%m-%d} to "
              f"{part.arrival_date.max():%Y-%m-%d}, cancel rate {part.is_canceled.mean():.1%}")

    # Top countries are chosen using the training data only (no peeking).
    top_countries = tr["country"].value_counts().head(8).index.tolist()
    F = build_features(df, top_countries)

    # Scale numbers + one-hot categories. Fit on TRAIN only, then apply to all.
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False), CATEGORICAL),
    ])
    pre.fit(F.loc[tr.index])
    feature_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    parts = [("train", tr), ("val", va), ("test", te)]
    X = {s: pre.transform(F.loc[part.index]).astype(np.float32) for s, part in parts}
    y = {s: part["is_canceled"].to_numpy() for s, part in parts}
    print(f"{len(feature_names)} model inputs after one-hot encoding")

    # 5a. Tune regularisation C on validation (sklearn: fast and exact)
    print("\nTuning C (regularisation) on validation log loss:")
    best_C, best_val = None, np.inf
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        m = train_sklearn(X["train"], y["train"], C)
        v = log_loss(y["val"], m.predict_proba(X["val"])[:, 1])
        print(f"  C={C:<6} val log loss={v:.4f}")
        if v < best_val:
            best_C, best_val = C, v
    print(f"  -> chosen C = {best_C}")

    # 5b. Tune the learning rate for PyTorch (short runs, same C)
    # Full-batch gradient descent converges slowly here (see REPORT.md), so we
    # need many epochs to reach the same optimum sklearn's solver finds.
    EPOCHS = 30000
    CHECKPOINTS = (1000, 3000, 10000, 30000)
    print("\nTuning PyTorch learning rate (manual loop, 500 epochs each):")
    best_lr, best_lr_val = None, np.inf
    for lr in [0.05, 0.2, 0.5, 1.0]:
        w, b, _, _ = train_torch_manual(X["train"], y["train"], best_C, lr, 500)
        v = log_loss(y["val"], np.clip(torch_manual_proba(w, b, X["val"]), 1e-7, 1 - 1e-7))
        print(f"  lr={lr:<5} val log loss={v:.4f}")
        if v < best_lr_val:
            best_lr, best_lr_val = lr, v
    print(f"  -> chosen lr = {best_lr}, training for {EPOCHS} epochs")

    # Final training of the three versions on the SAME train set
    sk = train_sklearn(X["train"], y["train"], best_C)
    w, b, hist_manual, snaps = train_torch_manual(
        X["train"], y["train"], best_C, best_lr, EPOCHS, snapshot_at=CHECKPOINTS)
    std, hist_std = train_torch_standard(X["train"], y["train"], best_C, best_lr, EPOCHS)

    # Naive baseline: give every booking the training cancel rate.
    base_rate = float(y["train"].mean())
    proba = {}
    for s in ["val", "test"]:
        proba[s] = {
            "baseline": np.full(len(y[s]), base_rate),
            "sklearn": sk.predict_proba(X[s])[:, 1],
            "torch_manual": torch_manual_proba(w, b, X[s]),
            "torch_standard": torch_standard_proba(std, X[s]),
        }

    # 7. Pick each model's threshold on VALIDATION (cost-minimising),
    #    then report its cost on TEST.
    print("\nTest-set results (threshold chosen on validation to minimise cost):")
    results = {}
    for name in ["baseline", "sklearn", "torch_manual", "torch_standard"]:
        if name == "baseline":
            # Baseline can only "flag everybody" (t=0) or "flag nobody" (t=1):
            # keep whichever is cheaper on validation.
            t = 0.0 if business_cost(y["val"], proba["val"][name], 0.0) < \
                business_cost(y["val"], proba["val"][name], 1.0) else 1.0
            r = scores(y["test"], proba["test"][name], t)
            r["roc_auc"] = 0.5  # constant prediction cannot rank bookings
        else:
            t = best_threshold(y["val"], proba["val"][name])
            r = scores(y["test"], proba["test"][name], t)
        results[name] = r
    print(pd.DataFrame(results).T.to_string())

    # How closely do the three implementations agree?
    coef = pd.DataFrame({
        "feature": feature_names,
        "sklearn": sk.coef_[0],
        "torch_manual": w.numpy(),
        "torch_standard": std.linear.weight.detach().numpy()[0],
    })
    p = proba["test"]
    agreement = {
        "max_abs_prob_diff_sklearn_vs_manual": float(np.abs(p["sklearn"] - p["torch_manual"]).max()),
        "max_abs_prob_diff_sklearn_vs_standard": float(np.abs(p["sklearn"] - p["torch_standard"]).max()),
        "max_abs_prob_diff_manual_vs_standard": float(np.abs(p["torch_manual"] - p["torch_standard"]).max()),
        "mean_abs_prob_diff_sklearn_vs_manual": float(np.abs(p["sklearn"] - p["torch_manual"]).mean()),
        "max_abs_coef_diff_sklearn_vs_manual": float((coef.sklearn - coef.torch_manual).abs().max()),
        "intercepts": {"sklearn": float(sk.intercept_[0]), "torch_manual": float(b.item()),
                       "torch_standard": float(std.linear.bias.item())},
    }
    # Investigation: how far is the manual model from sklearn after N epochs?
    sk_obj = log_loss(y["train"], sk.predict_proba(X["train"])[:, 1]) + \
        0.5 * l2_strength(best_C, len(y["train"])) * float((sk.coef_ ** 2).sum())
    convergence = []
    for ep, (ws, bs) in snaps.items():
        ps = torch_manual_proba(ws, bs, X["test"])
        convergence.append({
            "epochs": ep, "train_objective": round(hist_manual[ep - 1], 6),
            "max_abs_prob_diff_vs_sklearn": round(float(np.abs(ps - p["sklearn"]).max()), 4),
            "intercept": round(float(bs.item()), 3),
        })
    agreement["sklearn_train_objective"] = round(float(sk_obj), 6)
    agreement["sklearn_intercept"] = round(float(sk.intercept_[0]), 3)
    agreement["convergence"] = convergence
    print("\nAgreement between the three implementations:")
    for k, v in agreement.items():
        print(f"  {k}: {v}")

    # 8. Save everything for the dashboard
    joblib.dump(pre, OUT / "preprocessor.joblib")
    joblib.dump(sk, OUT / "model_sklearn.joblib")
    torch.save({"w": w, "b": b}, OUT / "model_torch_manual.pt")
    torch.save(std.state_dict(), OUT / "model_torch_standard.pt")

    test_out = te[["arrival_date", "is_canceled"]].copy()
    for name, vals in proba["test"].items():
        test_out[f"p_{name}"] = vals
    test_out.to_csv(OUT / "test_predictions.csv", index=False)

    eng = F.copy()
    eng["is_canceled"] = df["is_canceled"]
    eng["split"] = df["split"]
    eng["arrival_date"] = df["arrival_date"]
    eng.to_csv(OUT / "engineered_data.csv", index=False)

    coef.to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame({"torch_manual": hist_manual, "torch_standard": hist_std}).to_csv(
        OUT / "training_loss.csv", index_label="epoch")
    with open(OUT / "metrics.json", "w") as fh:
        json.dump({
            "chosen_C": best_C, "chosen_lr": best_lr, "epochs": EPOCHS,
            "cost_missed_cancel": COST_MISSED_CANCEL, "cost_false_alarm": COST_FALSE_ALARM,
            "split_sizes": {s: int(len(y[s])) for s in y},
            "results_test": results, "agreement": agreement,
        }, fh, indent=2)
    print(f"\nSaved models and results to {OUT}/")


if __name__ == "__main__":
    main()
