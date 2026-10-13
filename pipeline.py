"""
Shared pieces used by both train.py and app.py, so the app scores data with
exactly the same feature engineering and model definitions the training
stage used.

Business framing (see REPORT.md): before each outbound call of a term-deposit
campaign, decide whether this client is worth calling. Only information known
*before* dialing may be used as a feature.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
ARTIFACTS = ROOT / "artifacts"

# Campaign economics (euros). Calling is worth it when
# P(subscribe) * VALUE_PER_SUBSCRIPTION > COST_PER_CALL.
COST_PER_CALL = 5.0
VALUE_PER_SUBSCRIPTION = 50.0

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Calendar split boundaries (inclusive "YYYY-MM" periods).
TRAIN_END, VAL_END = "2009-12", "2010-06"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    # 12 exact duplicates (identical down to call duration): almost surely
    # double-logged calls, not distinct clients.
    df = df.drop_duplicates().reset_index(drop=True)
    df["y"] = (df["y"] == "yes").astype(int)
    df["period"] = _reconstruct_period(df["month"])
    return df


def _reconstruct_period(month: pd.Series) -> pd.Series:
    """The file has no year column but is ordered by date (May 2008 - Nov
    2010), so every time the month number goes backwards a new year started.
    Used only for the split and the dashboard's timeline, never as a feature."""
    idx = month.map({m: i for i, m in enumerate(MONTHS)}).to_numpy()
    year = 2008 + np.concatenate([[0], np.cumsum(np.diff(idx) < 0)])
    return pd.Series([f"{y}-{m + 1:02d}" for y, m in zip(year, idx)],
                     index=month.index)


def chronological_split(df: pd.DataFrame):
    """Train on May 2008 - Dec 2009, validate on Jan - Jun 2010, test on
    Jul - Nov 2010: fit on past campaigns, deploy on the next one. A random
    split would leak future economic conditions into training.

    Boundaries are calendar-based rather than row-fraction-based because the
    bank changed its calling policy around mid-2009 (call volume fell ~10x,
    success rate rose from ~5% to ~50%). A 70/15/15 row split puts *all* of
    the training data in the old regime and tests on the new one; the model
    then extrapolates macro indicators outside their training range (see
    REPORT.md and `row_fraction_split`, kept for that comparison)."""
    p = df["period"]
    return df[p <= TRAIN_END], df[(p > TRAIN_END) & (p <= VAL_END)], df[p > VAL_END]


def row_fraction_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """The first-attempt split: oldest 70% / next 15% / newest 15% of rows."""
    n = len(df)
    i_val, i_test = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

AGE_BINS = [0, 25, 30, 40, 50, 60, 200]
AGE_LABELS = ["<25", "25-29", "30-39", "40-49", "50-59", "60+"]

# Feature groups, so train.py can ablate them one at a time.
FEATURE_GROUPS = {
    "client": dict(
        num=[],
        cat=["job", "marital", "education", "age_band"],
    ),
    "credit": dict(
        num=["default_unknown", "housing_yes", "loan_yes", "loan_info_unknown"],
        cat=[],
    ),
    "contact": dict(
        num=["cellular", "log_prior_contacts"],
        cat=["day_of_week"],
    ),
    "history": dict(
        num=["recent_contact", "days_since_contact_capped", "previous_capped"],
        cat=["poutcome"],
    ),
    "month": dict(num=[], cat=["month"]),
    "macro": dict(num=["euribor3m", "cons.conf.idx", "cons.price.idx"], cat=[]),
    "macro_collinear": dict(num=["emp.var.rate", "nr.employed"], cat=[]),
}

# Final feature set, chosen by the validation ablation in train.py.
DEFAULT_GROUPS = ["client", "credit", "contact", "history", "macro"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into model-ready ones. Pure row-wise transforms (no
    statistics learned from data), so it's safe to apply before splitting;
    anything fitted (scaling, category lists) happens in the sklearn
    preprocessor, on the training split only."""
    out = pd.DataFrame(index=df.index)

    # `duration` is deliberately absent: it's only known after the call ends.

    # Age has a U-shaped relationship with subscribing (students and retirees
    # say yes far more often), which a single linear term can't express.
    out["age_band"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS,
                             right=False).astype(str)
    out["job"] = df["job"]
    out["marital"] = df["marital"]
    # 18 "illiterate" clients are too few to estimate their own coefficient;
    # closest education level is basic.4y.
    out["education"] = df["education"].replace({"illiterate": "basic.4y"})

    # `default` is "yes" for only 3 clients, so it carries no signal; whether
    # the client *declined to disclose* it ("unknown", 21%) does.
    out["default_unknown"] = (df["default"] == "unknown").astype(int)
    out["housing_yes"] = (df["housing"] == "yes").astype(int)
    out["loan_yes"] = (df["loan"] == "yes").astype(int)
    # housing and loan are "unknown" on exactly the same 990 rows: one flag.
    out["loan_info_unknown"] = (df["loan"] == "unknown").astype(int)

    out["cellular"] = (df["contact"] == "cellular").astype(int)
    out["day_of_week"] = df["day_of_week"]
    out["month"] = df["month"]
    # `campaign` counts contacts *including* the one being decided on, so the
    # pre-call quantity is campaign - 1. Long right tail (max 56): log1p.
    out["log_prior_contacts"] = np.log1p(df["campaign"] - 1)

    # pdays = 999 is a sentinel for "never contacted", not 999 days; treating
    # it as a number would make it the largest, most influential value.
    # ("Ever contacted" needs no flag of its own: it's exactly
    # poutcome != "nonexistent", already one-hot encoded below.)
    contacted = df["pdays"] != 999
    out["recent_contact"] = contacted.astype(int)
    out["days_since_contact_capped"] = np.where(contacted, df["pdays"].clip(upper=30), 0)
    out["previous_capped"] = df["previous"].clip(upper=3)
    out["poutcome"] = df["poutcome"]

    for col in ["euribor3m", "cons.conf.idx", "cons.price.idx",
                "emp.var.rate", "nr.employed"]:
        out[col] = df[col]
    return out


def columns_for(groups):
    num = [c for g in groups for c in FEATURE_GROUPS[g]["num"]]
    cat = [c for g in groups for c in FEATURE_GROUPS[g]["cat"]]
    return num, cat


def make_preprocessor(groups):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num, cat = columns_for(groups)
    encode = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                              sparse_output=False), cat),
    ])
    # Every column, dummies included, is standardized. L2 penalizes all
    # coefficients equally, so it should see them on one scale; and rare
    # dummies (e.g. poutcome_success, 2% of training rows) left as 0/1 have
    # tiny variance, which made the problem badly conditioned (eigenvalue
    # ratio ~2300 vs ~180 scaled) and plain gradient descent crawl.
    return make_pipeline(encode, StandardScaler())


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def l2_penalty(weight: torch.Tensor, C: float, n: int) -> torch.Tensor:
    """sklearn's LogisticRegression minimizes C * sum(log loss) + ||w||^2 / 2.
    Dividing by C * n gives mean(log loss) + ||w||^2 / (2 C n): the same
    optimum, but in the "mean loss" form PyTorch training uses. The
    intercept is not penalized, matching sklearn."""
    return (weight ** 2).sum() / (2 * C * n)


class TorchLogReg(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits


def predict_manual(params: dict, X: np.ndarray) -> np.ndarray:
    w, b = params["w"], params["b"]
    logits = torch.as_tensor(X, dtype=torch.float32) @ w + b
    return torch.sigmoid(logits).numpy()


def predict_module(model: TorchLogReg, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()


def call_mask(proba, threshold, capacity=1.0):
    """Call a client if their score clears the threshold AND they rank within
    the call centre's capacity (a fraction of the list, best scores first)."""
    call = proba >= threshold
    max_calls = int(np.floor(capacity * len(proba)))
    if call.sum() > max_calls:
        top = np.argsort(-proba, kind="stable")[:max_calls]
        capped = np.zeros_like(call)
        capped[top] = True
        call &= capped
    return call


def expected_profit(y_true, proba, threshold, capacity=1.0,
                    cost=COST_PER_CALL, value=VALUE_PER_SUBSCRIPTION):
    call = call_mask(proba, threshold, capacity)
    tp = int((call & (y_true == 1)).sum())
    return tp * value - int(call.sum()) * cost
