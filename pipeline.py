"""
Shared pieces used by both train.py and app.py: data loading/cleaning, the
feature builder, the preprocessor, the PyTorch model class and the business
cost function. Kept in one importable module so the objects pickled by
train.py can be unpickled by app.py.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hotels_sample.csv"
ARTIFACTS = ROOT / "artifacts"

TARGET = "is_canceled"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Columns never used as features, and the reason (see REPORT.md).
EXCLUDED = {
    "reservation_status": "reveals the target",
    "reservation_status_date": "date of the final outcome",
    "assigned_room_type": "assigned at check-in, after booking",
    "booking_changes": "accumulates after booking",
    "days_in_waiting_list": "waiting happens after the booking request",
    "total_of_special_requests": "often added after booking",
    "required_car_parking_spaces": "often added after booking",
}
LEAKY_FEATURES = ["log_special_requests", "has_parking", "log_waiting_list",
                  "log_booking_changes", "room_changed"]

LEAD_BUCKETS = [-1, 7, 30, 90, 180, np.inf]
LEAD_LABELS = ["0-7", "8-30", "31-90", "91-180", "180+"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["children"] = df["children"].fillna(0)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df.arrival_date_year,
        month=df.arrival_date_month.map({m: i for i, m in enumerate(MONTHS, 1)}),
        day=df.arrival_date_day_of_month,
    ))
    df["total_nights"] = df.stays_in_weekend_nights + df.stays_in_week_nights
    df["total_guests"] = df.adults + df.children + df.babies
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    keep = (df.total_nights > 0) & (df.total_guests > 0)
    return df.loc[keep].sort_values("arrival_date", kind="stable").reset_index(drop=True)


@dataclass
class FeatureConfig:
    lead: str = "log"            # "log" | "bucket" | "both"
    month: str = "cyclic"        # "cyclic" | "onehot" | "onehot_week"
    interactions: bool = True
    meal_room: bool = True
    deposit: bool = True         # False drops deposit_type and its interaction (dependence check)
    leaky: tuple = ()            # subset of LEAKY_FEATURES; only for the leakage ablation

    def as_dict(self) -> dict:
        return asdict(self)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw booking rows -> engineered feature frame. Learns top-k categories
    and the ADR cap from the training split only."""

    def __init__(self, config: FeatureConfig | None = None, top_k: int = 10, adr_quantile: float = 0.99):
        self.config = config
        self.top_k = top_k
        self.adr_quantile = adr_quantile

    def fit(self, X: pd.DataFrame, y=None):
        self.top_countries_ = X.country.value_counts().head(self.top_k).index.tolist()
        self.top_agents_ = X.agent.dropna().astype(int).value_counts().head(self.top_k).index.tolist()
        self.adr_cap_ = float(X.adr.quantile(self.adr_quantile))
        return self

    def adr_clipped(self, X: pd.DataFrame) -> pd.Series:
        return X.adr.clip(lower=0, upper=self.adr_cap_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config or FeatureConfig()
        out = pd.DataFrame(index=X.index)

        if cfg.lead in ("log", "both"):
            out["log_lead_time"] = np.log1p(X.lead_time)
        if cfg.lead in ("bucket", "both"):
            out["lead_bucket"] = pd.cut(X.lead_time, LEAD_BUCKETS, labels=LEAD_LABELS).astype(str)

        out["total_nights"] = np.log1p(X.total_nights)
        out["weekend_ratio"] = X.stays_in_weekend_nights / X.total_nights
        out["total_guests"] = X.total_guests
        out["has_kids"] = ((X.children + X.babies) > 0).astype(int)
        out["arrival_weekday"] = X.arrival_date.dt.day_name()

        month = X.arrival_date.dt.month
        if cfg.month == "cyclic":
            out["month_sin"] = np.sin(2 * np.pi * month / 12)
            out["month_cos"] = np.cos(2 * np.pi * month / 12)
        else:
            out["arrival_month"] = X.arrival_date_month
        if cfg.month == "onehot_week":
            week = X.arrival_date_week_number
            out["week_sin"] = np.sin(2 * np.pi * week / 53)
            out["week_cos"] = np.cos(2 * np.pi * week / 53)

        out["log_prev_cancellations"] = np.log1p(X.previous_cancellations)
        out["log_prev_not_canceled"] = np.log1p(X.previous_bookings_not_canceled)
        out["ever_canceled"] = (X.previous_cancellations > 0).astype(int)
        out["is_repeated_guest"] = X.is_repeated_guest

        for col in ["hotel", "market_segment", "distribution_channel", "customer_type"]:
            out[col] = X[col]
        if cfg.deposit:
            out["deposit_type"] = X.deposit_type
        if cfg.meal_room:
            out["meal"] = X.meal.replace("Undefined", "SC")
            out["reserved_room_type"] = X.reserved_room_type

        # Portugal and "no agent" are captured by the country_grp / agent_grp one-hots; separate
        # is_portugal / has_agent flags would be exact duplicates (singular design matrix).
        out["country_grp"] = X.country.where(X.country.isin(self.top_countries_), "Other").fillna("Unknown")
        out["has_company"] = X.company.notna().astype(int)
        agent = X.agent.fillna(-1).astype(int)
        out["agent_grp"] = np.where(agent.isin(self.top_agents_), agent.astype(str),
                                    np.where(agent == -1, "none", "other"))

        adr = self.adr_clipped(X)
        out["adr"] = adr
        out["adr_per_person"] = adr / X.total_guests
        out["adr_zero"] = (X.adr <= 0).astype(int)

        if cfg.interactions:
            log_lead = np.log1p(X.lead_time)
            out["lead_x_city"] = log_lead * (X.hotel == "City Hotel")
            if cfg.deposit:
                out["lead_x_nonrefund"] = log_lead * (X.deposit_type == "Non Refund")

        leaky = {
            "log_special_requests": lambda: np.log1p(X.total_of_special_requests),
            "has_parking": lambda: (X.required_car_parking_spaces > 0).astype(int),
            "log_waiting_list": lambda: np.log1p(X.days_in_waiting_list),
            "log_booking_changes": lambda: np.log1p(X.booking_changes),
            "room_changed": lambda: (X.reserved_room_type != X.assigned_room_type).astype(int),
        }
        for name in cfg.leaky:
            out[name] = leaky[name]()
        return out


def make_preprocessor(config: FeatureConfig) -> Pipeline:
    """FeatureBuilder -> StandardScaler (numeric) + OneHotEncoder (categorical).
    Every statistic (top-k, ADR cap, means/stds, category lists) is fit on train only."""
    encoder = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
         make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=True)
    return Pipeline([("features", FeatureBuilder(config)), ("encode", encoder)])


class LogisticRegressionNet(torch.nn.Module):
    """Logistic regression as a torch module: one linear layer. The sigmoid is
    applied inside BCEWithLogitsLoss during training and in predict_proba."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self(torch.as_tensor(x, dtype=torch.float64))).numpy()


def manual_predict_proba(params: dict, x: np.ndarray) -> np.ndarray:
    z = torch.as_tensor(x, dtype=torch.float64) @ params["w"] + params["b"]
    return torch.sigmoid(z).numpy()


def booking_revenue(df: pd.DataFrame, adr_cap: float) -> np.ndarray:
    """Room revenue at stake for each booking: clipped ADR x nights."""
    return (df.adr.clip(lower=0, upper=adr_cap) * df.total_nights).to_numpy()


def business_cost(y: np.ndarray, proba: np.ndarray, revenue: np.ndarray, threshold: float,
                  fn_mult: float = 1.0, fp_mult: float = 2.0) -> dict:
    """FN (missed cancellation) costs fn_mult x the booking's revenue (empty room).
    FP (flagged booking that shows up) costs fp_mult x revenue (relocating an
    overbooked guest: compensation + reputation)."""
    pred = proba >= threshold
    fn = (~pred) & (y == 1)
    fp = pred & (y == 0)
    return {
        "tp": int((pred & (y == 1)).sum()), "fp": int(fp.sum()),
        "fn": int(fn.sum()), "tn": int(((~pred) & (y == 0)).sum()),
        "fn_cost": float(fn_mult * revenue[fn].sum()),
        "fp_cost": float(fp_mult * revenue[fp].sum()),
        "total_cost": float(fn_mult * revenue[fn].sum() + fp_mult * revenue[fp].sum()),
    }


def cost_curve(y, proba, revenue, fn_mult=1.0, fp_mult=2.0, grid=None):
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2) if grid is None else grid
    costs = np.array([business_cost(y, proba, revenue, t, fn_mult, fp_mult)["total_cost"] for t in grid])
    return grid, costs
