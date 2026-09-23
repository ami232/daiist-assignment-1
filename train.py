"""
Assignment 1: training pipeline for Resort Hotel cancellation prediction.

Business framing (see REPORT.md): when a booking comes in, score its
probability of cancelling. Bookings above the threshold are asked to
confirm or pay a partial deposit. The model therefore only sees what is
known at booking time, and the data is split by booking date.

Stages so far:
1. load + filter to Resort Hotel
2. derive booking date (arrival date - lead_time)
3. exclude Non Refund bookings, drop columns that leak the outcome, are
   the business action itself, or are recorded unreliably
4. time-based train / val / test split on booking date
5. feature engineering + one-hot encoding (one reference category dropped per
   feature), fitted on train only
6. naive baseline + the same L2-regularised logistic regression three ways:
   scikit-learn, a manual PyTorch loop, and nn.Module + torch.optim
7. tune C (sklearn, on val log loss) and the learning rate (PyTorch, on val);
   evaluate everything on test and save artifacts for app.py

The decision threshold is not chosen here: it depends on the business costs,
which the dashboard exposes. Metrics here are threshold-free (log loss, AUC).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hotels.csv"
ARTIFACTS = ROOT / "artifacts"
TARGET = "is_canceled"

# Split boundaries on booking date. The data only contains arrivals up to
# 2017-08-31, so bookings made close to that date are censored toward short
# lead times. Test ends at 2017-03-31 so its bookings had ~5 months to arrive.
TRAIN_END = pd.Timestamp("2016-04-01")  # train: booked before this
VAL_END = pd.Timestamp("2016-10-01")    # val:   booked in [TRAIN_END, VAL_END)
TEST_END = pd.Timestamp("2017-04-01")   # test:  booked in [VAL_END, TEST_END)

# Columns never allowed as model inputs.
LEAKY_COLUMNS = [
    "reservation_status",       # the target itself, renamed
    "reservation_status_date",  # date of the outcome
    "assigned_room_type",       # assigned at check-in, reveals who showed up
]
ACTION_COLUMNS = [
    "deposit_type",  # asking for a deposit is the business action we're deciding
]
UNRELIABLE_COLUMNS = [
    "country",          # cancelled bookings recorded as PRT far more often early on
    "booking_changes",  # 0 at booking time, grows mostly for guests who show up
]

# Feature engineering settings.
LEAD_BINS = [-1, 7, 30, 90, 180, np.inf]  # days before arrival
LEAD_LABELS = ["0-7d", "8-30d", "31-90d", "91-180d", "181d+"]
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
AGENT_MIN_SHARE = 0.01  # agents with >= 1% of train bookings keep their own group
GUEST_BINS = [-1, 1, 2, 3, np.inf]  # cancel rate is not monotonic in guests
GUEST_LABELS = ["0-1", "2", "3", "4+"]
WEEKEND_BINS = [-0.01, 0, 0.5, 1]  # share of nights that fall on a weekend
WEEKEND_LABELS = ["none", "up to half", "over half"]
# Reference (dropped) category per feature; alphabetical first unless set here.
# Online TA is the largest segment. The alphabetical default, Complementary, has
# only 65 train bookings, which made gradient descent slow to converge.
REFERENCE = {"market_segment": "Online TA"}
CATEGORICAL_FEATURES = ["lead_bin", "season", "agent_group", "guests_bin", "weekend_bin",
                        "market_segment", "customer_type"]

# Model settings. Only regularisation strength and learning rate are tuned.
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LR_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]  # 10 is too large: oscillates
EPOCHS = 15000  # full-batch GD steps; 3000 left coefs ~0.015 short of the optimum
SEED = 0

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def load_resort(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NULL"])
    df = df[df["hotel"] == "Resort Hotel"].drop(columns="hotel")
    return df.reset_index(drop=True)


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month"].map(MONTHS),
        day=df["arrival_date_day_of_month"],
    ))
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
    return df


def exclude_non_refund(df: pd.DataFrame) -> pd.DataFrame:
    # ~95% of these "cancel", a known data quirk. They already paid, so they
    # are outside the decision the model supports.
    return df[df["deposit_type"] != "Non Refund"].reset_index(drop=True)


def drop_forbidden(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=LEAKY_COLUMNS + ACTION_COLUMNS + UNRELIABLE_COLUMNS)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    booked = df["booking_date"]
    train = df[booked < TRAIN_END]
    val = df[(booked >= TRAIN_END) & (booked < VAL_END)]
    test = df[(booked >= VAL_END) & (booked < TEST_END)]
    return train, val, test


def fit_agent_groups(train: pd.DataFrame) -> list[str]:
    share = agent_ids(train).value_counts(normalize=True)
    return sorted(share[share >= AGENT_MIN_SHARE].index)


def agent_ids(df: pd.DataFrame) -> pd.Series:
    return df["agent"].map(lambda a: "none" if pd.isna(a) else str(int(a)))


def build_features(df: pd.DataFrame, agent_groups: list[str]) -> pd.DataFrame:
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    total_guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    # 0-night bookings (day use) have no weekend share; count them as 0
    weekend_share = (df["stays_in_weekend_nights"] / nights.replace(0, np.nan)).fillna(0)
    agents = agent_ids(df)
    return pd.DataFrame({
        "lead_bin": pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS).astype(str),
        "season": df["arrival_date"].dt.month.map(SEASONS),
        "agent_group": agents.where(agents.isin(agent_groups), "other"),
        "guests_bin": pd.cut(total_guests, GUEST_BINS, labels=GUEST_LABELS).astype(str),
        "weekend_bin": pd.cut(weekend_share, WEEKEND_BINS, labels=WEEKEND_LABELS).astype(str),
        "market_segment": df["market_segment"],
        "customer_type": df["customer_type"],
    }, index=df.index)


def make_preprocessor(train_feats: pd.DataFrame) -> ColumnTransformer:
    # All features are categorical, so one-hot encoding is the only step; no scaling
    # needed. One reference category per feature is dropped so the dummy columns plus
    # intercept are not redundant; coefficients read as "vs. reference".
    # Categories unseen in train (e.g. a new segment) encode as the reference.
    drop = [REFERENCE.get(f, sorted(train_feats[f].unique())[0]) for f in CATEGORICAL_FEATURES]
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop=drop, handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


# ---------------------------------------------------------------- models
#
# All three minimise the same objective, the one scikit-learn uses:
#     C * sum(log loss) + 0.5 * ||w||^2        (intercept not penalised)
# Dividing by C * n gives the mean-loss form used in PyTorch:
#     mean(log loss) + (lam / 2) * ||w||^2     with lam = 1 / (C * n)
# Same objective, convex, one optimum -> the three should agree.

def fit_sklearn(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=5000, tol=1e-8).fit(X, y)


def fit_manual_torch(X: np.ndarray, y: np.ndarray, lam: float, lr: float,
                     epochs: int = EPOCHS) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Session 5 style: raw tensors, autograd, manual gradient step."""
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    history = []
    for _ in range(epochs):
        z = Xt @ w + b
        p = torch.sigmoid(z)
        eps = 1e-12
        data_loss = -(yt * torch.log(p + eps) + (1 - yt) * torch.log(1 - p + eps)).mean()
        loss = data_loss + lam / 2 * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())
    return w.detach(), b.detach(), history


class LogisticModel(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # logits; sigmoid lives in the loss


def fit_torch_module(X: np.ndarray, y: np.ndarray, lam: float, lr: float,
                     epochs: int = EPOCHS) -> tuple[LogisticModel, list[float]]:
    """Standard workflow: nn.Module + BCEWithLogitsLoss + torch.optim."""
    torch.manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    model = LogisticModel(X.shape[1])
    criterion = torch.nn.BCEWithLogitsLoss()
    # Penalty added by hand rather than via weight_decay, which would also
    # shrink the bias and so differ from scikit-learn.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(Xt), yt) + lam / 2 * (model.linear.weight ** 2).sum()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return model, history


def predict_manual(w: torch.Tensor, b: torch.Tensor, X: np.ndarray) -> np.ndarray:
    return torch.sigmoid(torch.tensor(X, dtype=torch.float64) @ w + b).numpy()


def predict_module(model: LogisticModel, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float64))).numpy()


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"log_loss": log_loss(y, p, labels=[0, 1]),
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p)}


# ---------------------------------------------------------------- pipeline

def describe_split(name: str, part: pd.DataFrame) -> None:
    print(f"{name:<5} {len(part):>6} rows | cancel rate {part[TARGET].mean():.3f} | "
          f"booked {part['booking_date'].min().date()} -> {part['booking_date'].max().date()} | "
          f"median lead {part['lead_time'].median():.0f}d")


def main() -> None:
    torch.manual_seed(SEED)
    df = drop_forbidden(exclude_non_refund(add_dates(load_resort())))
    train, val, test = time_split(df)

    print(f"Resort Hotel (excl. Non Refund): {len(df)} bookings, "
          f"{len(df) - len(train) - len(val) - len(test)} booked after "
          f"{TEST_END.date()} excluded (censored)")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        describe_split(name, part)

    # Everything learned from data (agent list, categories) uses train only.
    agent_groups = fit_agent_groups(train)
    feats = {name: build_features(part, agent_groups)
             for name, part in [("train", train), ("val", val), ("test", test)]}
    preprocessor = make_preprocessor(feats["train"]).fit(feats["train"])
    X = {name: preprocessor.transform(f) for name, f in feats.items()}
    y = {"train": train[TARGET].to_numpy(), "val": val[TARGET].to_numpy(),
         "test": test[TARGET].to_numpy()}
    feature_names = list(preprocessor.get_feature_names_out())
    n = len(y["train"])
    print(f"\n{len(agent_groups)} agent groups kept (+ 'other'); "
          f"{X['train'].shape[1]} model inputs after encoding")

    # 1. Naive baseline: every booking gets the train cancellation rate.
    base_rate = y["train"].mean()

    # 2. Tune C on validation log loss (threshold-free: threshold comes from costs).
    print("\nTuning C (scikit-learn, val log loss):")
    val_loss = {}
    for C in C_GRID:
        val_loss[C] = log_loss(y["val"], fit_sklearn(X["train"], y["train"], C).predict_proba(X["val"])[:, 1])
        print(f"  C={C:<5} val log loss {val_loss[C]:.4f}")
    best_C = min(val_loss, key=val_loss.get)
    lam = 1 / (best_C * n)
    sk = fit_sklearn(X["train"], y["train"], best_C)
    print(f"  -> C={best_C} (lam={lam:.2e} in PyTorch)")

    # 3. Tune the learning rate for each PyTorch version on validation log loss,
    #    at the same regularisation. A too-small lr just hasn't converged yet.
    print("\nTuning learning rate (PyTorch, val log loss):")
    manual_runs, module_runs = {}, {}
    for lr in LR_GRID:
        w, b, hist = fit_manual_torch(X["train"], y["train"], lam, lr)
        model, mhist = fit_torch_module(X["train"], y["train"], lam, lr)
        manual_runs[lr] = (w, b, hist, log_loss(y["val"], predict_manual(w, b, X["val"])))
        module_runs[lr] = (model, mhist, log_loss(y["val"], predict_module(model, X["val"])))
        print(f"  lr={lr:<4} manual {manual_runs[lr][3]:.4f} | module {module_runs[lr][2]:.4f}")
    lr_manual = min(manual_runs, key=lambda k: manual_runs[k][3])
    lr_module = min(module_runs, key=lambda k: module_runs[k][2])
    w, b, manual_hist, _ = manual_runs[lr_manual]
    module, module_hist, _ = module_runs[lr_module]
    print(f"  -> manual lr={lr_manual}, module lr={lr_module}")

    # 4. Evaluate on test.
    probs = {
        "baseline": np.full(len(y["test"]), base_rate),
        "sklearn": sk.predict_proba(X["test"])[:, 1],
        "manual_torch": predict_manual(w, b, X["test"]),
        "torch_module": predict_module(module, X["test"]),
    }
    results = {name: scores(y["test"], p) for name, p in probs.items()}
    results["baseline"]["roc_auc"] = 0.5  # constant prediction ranks nothing
    print("\nTest set (booked Oct 2016 - Mar 2017):")
    print(pd.DataFrame(results).T.round(4).to_string())

    # 5. Do the three implementations agree?
    coefs = pd.DataFrame({
        "sklearn": np.r_[sk.intercept_, sk.coef_.ravel()],
        "manual_torch": np.r_[b.numpy(), w.numpy()],
        "torch_module": np.r_[module.linear.bias.detach().numpy(),
                              module.linear.weight.detach().numpy().ravel()],
    }, index=["intercept"] + feature_names)
    agreement = {
        "max_coef_diff_manual_vs_sklearn": float((coefs.manual_torch - coefs.sklearn).abs().max()),
        "max_coef_diff_module_vs_sklearn": float((coefs.torch_module - coefs.sklearn).abs().max()),
        "max_prob_diff_manual_vs_sklearn": float(np.abs(probs["manual_torch"] - probs["sklearn"]).max()),
        "max_prob_diff_module_vs_sklearn": float(np.abs(probs["torch_module"] - probs["sklearn"]).max()),
    }
    print("\nAgreement with scikit-learn (max absolute difference):")
    for k, v in agreement.items():
        print(f"  {k}: {v:.2e}")

    # 6. Save artifacts for app.py (which must never retrain).
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "agent_groups": agent_groups},
                ARTIFACTS / "preprocessing.joblib")
    joblib.dump(sk, ARTIFACTS / "sklearn_logreg.joblib")
    torch.save({"w": w, "b": b}, ARTIFACTS / "manual_torch.pt")
    torch.save(module.state_dict(), ARTIFACTS / "torch_module.pt")
    keep = ["booking_date", "arrival_date", "lead_time", "adr", "stays_in_weekend_nights",
            "stays_in_week_nights", TARGET]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        pd.concat([part[keep], feats[name]], axis=1).to_csv(ARTIFACTS / f"{name}_set.csv", index=False)
    coefs.to_csv(ARTIFACTS / "coefficients.csv")
    pd.DataFrame({"manual_torch": manual_hist, "torch_module": module_hist}).to_csv(
        ARTIFACTS / "loss_history.csv", index_label="epoch")
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({"test": results, "agreement": agreement, "base_rate": base_rate,
                   "best_C": best_C, "lam": lam, "lr_manual": lr_manual, "lr_module": lr_module,
                   "val_log_loss_by_C": {str(k): v for k, v in val_loss.items()}}, f, indent=2)
    print(f"\nSaved artifacts to {ARTIFACTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
