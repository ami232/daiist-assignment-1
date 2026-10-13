"""
Shared pipeline code, imported by both train.py and app.py.

Everything that must behave *identically* at training time and at dashboard
time lives here: loading the data, the row-wise feature engineering, the
time-ordered split, the preprocessing definition, the PyTorch model class and
the business-profit calculation. Keeping it in one place means the app can
never drift out of sync with how the models were trained.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
ARTIFACTS = ROOT / "artifacts"
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

SEED = 42

# ---------------------------------------------------------------------------
# Business assumptions (hypothetical, stated in REPORT.md)
# ---------------------------------------------------------------------------
CALL_COST = 10.0   # EUR: fully-loaded cost of one outbound call (agent time + overhead)
SUB_VALUE = 80.0   # EUR: expected net margin from one term-deposit subscription
# For a calibrated model, calling is worth it when p * SUB_VALUE > CALL_COST,
# i.e. p > CALL_COST / SUB_VALUE = 0.125.

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def _find_member(zf: zipfile.ZipFile, target: str) -> bytes:
    """Find `target` inside a zip, looking inside nested zips too (UCI nests them)."""
    names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
    for name in names:
        if name.endswith("/" + target) or name == target:
            return zf.read(name)
    for name in names:
        if name.endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(zf.read(name)))
            try:
                return _find_member(inner, target)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(target)


def _download() -> None:
    print(f"Dataset not found at {DATA_PATH}; downloading it once from UCI ...")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0 (assignment download)"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    DATA_PATH.write_bytes(_find_member(outer, DATA_PATH.name))
    print(f"Saved {DATA_PATH} -- commit it so the repo is self-contained.")


def load_raw() -> pd.DataFrame:
    """Load the committed CSV (download it only if it is missing)."""
    if not DATA_PATH.exists():
        _download()
    df = pd.read_csv(DATA_PATH, sep=";")
    return add_time_columns(df)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The file is ordered by date (May 2008 - Nov 2010) but has no year column.
    The month only goes *down* when a new year starts, so we can rebuild it.
    These columns are for splitting/plotting only, never model inputs."""
    df = df.copy()
    month_num = df["month"].map(MONTHS).to_numpy()
    year = 2008 + np.concatenate([[0], np.cumsum(np.diff(month_num) < 0)])
    df["year"] = year
    df["period"] = [f"{y}-{m:02d}" for y, m in zip(year, month_num)]
    return df


# ---------------------------------------------------------------------------
# 2. Row-wise feature engineering (no statistics learned from data, so it is
#    safe to apply before splitting -- nothing can leak across the split)
# ---------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # `duration` is only known AFTER the call ends, and duration == 0 implies
    # y == "no". Using it would be target leakage for a "whom should we call?"
    # decision, so it is dropped (the UCI documentation says the same).
    out = out.drop(columns=["duration"])

    out = out.rename(columns={
        "emp.var.rate": "emp_var_rate",
        "cons.price.idx": "cons_price_idx",
        "cons.conf.idx": "cons_conf_idx",
        "nr.employed": "nr_employed",
    })
    out["y"] = (out["y"] == "yes").astype(int)

    # pdays == 999 is a sentinel meaning "never contacted before". Left as a
    # number, a linear model would read it as "contacted a very long time
    # ago". Split it into a flag + a clean numeric part.
    out["never_contacted"] = (out["pdays"] == 999).astype(int)
    out["days_since_prev"] = np.where(out["never_contacted"] == 1, 0, out["pdays"])

    # campaign includes the current call; campaign - 1 = calls already made in
    # this campaign, which IS known before deciding to call again. Long right
    # tail (up to 56) with diminishing effect -> log1p.
    out["prior_calls_log"] = np.log1p(out["campaign"] - 1)
    out["previous_log"] = np.log1p(out["previous"])

    # Subscription rate is U-shaped in age (students and retirees respond
    # most). A single linear age term cannot bend, so add two band flags.
    out["age_lt25"] = (out["age"] < 25).astype(int)
    out["age_ge60"] = (out["age"] >= 60).astype(int)

    # default: "no" / "unknown" / "yes" (only 3 rows are "yes"). Collapse to
    # "anything other than a clean no" to avoid a near-empty category.
    out["default_not_no"] = (out["default"] != "no").astype(int)
    return out


# Column groups, so the feature-set ablation in train.py can add/remove them.
NUMERIC_GROUPS = {
    "client": ["age", "age_lt25", "age_ge60", "default_not_no"],
    "history": ["prior_calls_log", "never_contacted", "days_since_prev", "previous_log"],
    "macro": ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"],
}
CATEGORICAL_GROUPS = {
    "client": ["job", "marital", "education", "housing", "loan"],
    "history": ["poutcome"],
    "contact": ["contact", "day_of_week"],
    "month": ["month"],
}
FEATURE_SETS = {
    "client+history+contact": ["client", "history", "contact"],
    "+macro": ["client", "history", "contact", "macro"],
    "+macro+month": ["client", "history", "contact", "macro", "month"],
}


def columns_for(feature_set: str) -> tuple[list[str], list[str]]:
    groups = FEATURE_SETS[feature_set]
    numeric = [c for g in groups for c in NUMERIC_GROUPS.get(g, [])]
    categorical = [c for g in groups for c in CATEGORICAL_GROUPS.get(g, [])]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Fit on TRAIN ONLY. Scaling matters for the PyTorch versions (gradient
    descent converges badly on unscaled features) and for L2 regularisation
    (the penalty should treat every coefficient on the same scale)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 3. Time-ordered split
# ---------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Rows are in date order. The bank would train on past campaigns and
    score future ones, so a random split would leak future economic
    conditions into training. Train = oldest 70%, val = next 15%, test = newest 15%."""
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------
class LogisticRegressionNet(nn.Module):
    """Standard-workflow model: one linear layer producing a logit. The
    sigmoid lives in the loss (BCEWithLogitsLoss) for numerical stability."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def predict_manual(params: dict, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = torch.as_tensor(X, dtype=torch.float32) @ params["w"] + params["b"]
        return torch.sigmoid(z).numpy()


def predict_standard(model: LogisticRegressionNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()


def regularised_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam: float) -> float:
    """mean log-loss + 0.5 * lam * ||w||^2 (intercept not penalised).
    With lam = 1 / (C * n) this is exactly scikit-learn's objective divided by
    C * n, so all three implementations are minimising the same function."""
    w = np.asarray(w, dtype=np.float64)
    z = X.astype(np.float64) @ w + float(b)
    data = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
    return float(data + 0.5 * lam * np.dot(w, w))


# ---------------------------------------------------------------------------
# 5. Business outcome
# ---------------------------------------------------------------------------
def business_outcome(y, p, threshold: float, value: float = SUB_VALUE, cost: float = CALL_COST) -> dict:
    """Call everyone whose predicted probability >= threshold.
    profit = (subscriptions won) * value - (calls made) * cost."""
    y = np.asarray(y).astype(int)
    call = np.asarray(p) >= threshold
    tp = int((call & (y == 1)).sum())
    fp = int((call & (y == 0)).sum())
    fn = int((~call & (y == 1)).sum())
    tn = int((~call & (y == 0)).sum())
    calls = tp + fp
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "calls": calls,
        "precision": tp / calls if calls else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "profit": tp * value - calls * cost,
    }


THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def profit_curve(y, p, value: float = SUB_VALUE, cost: float = CALL_COST) -> np.ndarray:
    return np.array([business_outcome(y, p, t, value, cost)["profit"] for t in THRESHOLD_GRID])


def best_threshold(y, p, value: float = SUB_VALUE, cost: float = CALL_COST) -> float:
    return float(THRESHOLD_GRID[int(np.argmax(profit_curve(y, p, value, cost)))])
