"""Train one linear regression three ways for hourly bikeshare demand.

The decision this model supports is set in REPORT.md: one hour ahead, how
many bikes should Capital Bikeshare have available system-wide. That
decision is why the split is chronological, why casual/registered are
excluded, and why lag features are allowed.

Run via `uv run python main.py train`. This script trains. app.py only loads
what is written under artifacts/.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hour.csv"
ARTIFACTS = ROOT / "artifacts"

# Chronological blocks. Validation is only for learning rate, ridge alpha,
# and which feature block to keep. The test quarter is not used for that.
TRAIN_END = pd.Timestamp("2012-07-01")
VAL_END = pd.Timestamp("2012-10-01")

# Planning assumptions for the staffing tab, not quantities estimated from
# the fit. Stocking out of a bike wastes a trip; moving a bike that nobody
# rents wastes van time. Underage is more expensive, so the deployed number
# should sit above the mean forecast. See REPORT.md.
COST_UNDER = 5.0
COST_OVER = 1.0

RIDGE_ALPHA = 1.0
LEARNING_RATES = (0.05, 0.1, 0.2)
N_EPOCHS = 3000
RANDOM_SEED = 0

# Feature blocks. Later blocks extend earlier ones. The shipped model uses
# CHOSEN_BLOCK, which is the last block that still reduced validation error
# by about a bike or more. roll_mean_24 was tried and dropped: with lag_1
# and lag_24 already in the model it did not move validation MAE.
RAW_FEATURES = [
    "hr",
    "mnth",
    "weekday",
    "season",
    "weathersit",
    "holiday",
    "workingday",
    "yr",
    "temp_c",
    "hum_pct",
    "wind",
]
CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "holiday",
    "workingday",
    "mist",
    "precip",
    "temp_c",
    "hum_pct",
    "wind",
    "days_since_start",
]
SHAPE_FEATURES = CYCLICAL_FEATURES + [
    "wd_hour_sin",
    "wd_hour_cos",
    "morning_rush",
    "evening_rush",
    "leisure_midday",
]
INTERACTION_FEATURES = SHAPE_FEATURES + [
    "temp_hour_sin",
    "temp_hour_cos",
    "hum_x_temp",
    "precip_x_rush",
    "wind_is_zero",
]
LAG_FEATURES = ["lag_1", "lag_24", "lag_168"]
CHOSEN_FEATURES = INTERACTION_FEATURES + LAG_FEATURES + [
    "delta_temp_168",
    "delta_hum_168",
    "delta_precip_168",
]

FEATURE_BLOCKS = [
    ("raw columns, no engineering", RAW_FEATURES),
    ("cyclical calendar and weather", CYCLICAL_FEATURES),
    ("plus commute shape", SHAPE_FEATURES),
    ("plus weather interactions", INTERACTION_FEATURES),
    ("lags only", LAG_FEATURES),
    ("interactions plus lags", INTERACTION_FEATURES + LAG_FEATURES),
    ("chosen: lags plus change vs last week", CHOSEN_FEATURES),
    ("chosen plus 24h rolling mean (not shipped)", CHOSEN_FEATURES + ["roll_mean_24"]),
]


def load_hourly() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame["ts"] = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"], unit="h")
    return frame.sort_values("ts")


def build_modeling_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clock-align the series, then build features from the past only.

    165 hours are missing from the two-year span. Shifting the raw file by
    one row would treat a multi-hour gap as one hour, so lags are taken on a
    complete hourly index. Hours that were never observed are dropped after
    that, because they have no target.
    """
    observed = raw.set_index("ts").sort_index()
    full_index = pd.date_range(observed.index.min(), observed.index.max(), freq="h")
    base = observed.reindex(full_index)
    n_missing_hours = int(len(full_index) - len(observed))

    base["temp_c"] = base["temp"] * 41.0
    base["hum_pct"] = base["hum"] * 100.0
    base["wind"] = base["windspeed"] * 67.0

    # 0% humidity is not a real reading. All 22 of these hours fall on
    # 2011-03-10, which is inside the training window. The fill value is the
    # training-period median, computed after the zeros are removed.
    humidity_defect = base["hum_pct"] == 0
    n_humidity_defect = int(humidity_defect.fillna(False).sum())
    base.loc[humidity_defect, "hum_pct"] = np.nan
    train_humidity = base.loc[base.index < TRAIN_END, "hum_pct"]
    humidity_median = float(train_humidity.median())
    base["hum_pct"] = base["hum_pct"].fillna(humidity_median)

    hour = base.index.hour
    month = base.index.month
    # pandas dayofweek is Monday=0. The file's own weekday column is Sunday=0.
    # Cyclical encoding only needs a consistent 7-day circle, so Monday=0 is fine.
    dow = base.index.dayofweek
    base["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    base["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    base["month_sin"] = np.sin(2 * np.pi * month / 12)
    base["month_cos"] = np.cos(2 * np.pi * month / 12)
    base["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    base["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    base["days_since_start"] = (base.index - pd.Timestamp("2011-01-01")) / pd.Timedelta(days=1)
    base["hr"] = hour
    base["mnth"] = month

    working = base["workingday"]
    base["wd_hour_sin"] = working * base["hour_sin"]
    base["wd_hour_cos"] = working * base["hour_cos"]
    base["morning_rush"] = ((working == 1) & np.isin(hour, [7, 8, 9])).astype(float)
    base["evening_rush"] = ((working == 1) & np.isin(hour, [16, 17, 18, 19])).astype(float)
    base["leisure_midday"] = ((working == 0) & np.isin(hour, list(range(10, 17)))).astype(float)
    base["temp_hour_sin"] = base["temp_c"] * base["hour_sin"]
    base["temp_hour_cos"] = base["temp_c"] * base["hour_cos"]
    base["hum_x_temp"] = base["hum_pct"] * base["temp_c"]

    # weathersit 4 (heavy storm) occurs 3 times. A linear model cannot learn a
    # separate coefficient from three hours, so those hours join light precip.
    weather = base["weathersit"].clip(upper=3)
    base["mist"] = (weather == 2).astype(float)
    base["precip"] = (weather >= 3).astype(float)
    rush = ((base["morning_rush"] == 1) | (base["evening_rush"] == 1)).astype(float)
    base["precip_x_rush"] = base["precip"] * rush
    base["wind_is_zero"] = (base["wind"] == 0).astype(float)

    # shift() looks backward on the clock. It does not use the current hour
    # or any future hour. At decision time the previous hour has already
    # been counted, and last week's weather is history. This hour's weather
    # stands in for a forecast; REPORT.md states that limitation.
    counts = base["cnt"]
    base["lag_1"] = counts.shift(1)
    base["lag_24"] = counts.shift(24)
    base["lag_168"] = counts.shift(168)
    # Tried and not shipped. Once lag_1 and lag_24 exist, this average
    # repeats them. It is only here so the validation table can show that.
    base["roll_mean_24"] = counts.shift(1).rolling(24, min_periods=12).mean()
    base["delta_temp_168"] = base["temp_c"] - base["temp_c"].shift(168)
    base["delta_hum_168"] = base["hum_pct"] - base["hum_pct"].shift(168)
    base["delta_precip_168"] = base["precip"] - base["precip"].shift(168)

    model_rows = base.loc[base["cnt"].notna()].copy()
    before_lag_drop = len(model_rows)
    model_rows = model_rows.dropna(subset=LAG_FEATURES + [
        "delta_temp_168",
        "delta_hum_168",
        "delta_precip_168",
    ])
    model_rows["split"] = np.where(
        model_rows.index < TRAIN_END,
        "train",
        np.where(model_rows.index < VAL_END, "val", "test"),
    )

    notes = {
        "rows_raw": int(len(raw)),
        "missing_hours_in_span": n_missing_hours,
        "humidity_zero_hours_imputed": n_humidity_defect,
        "humidity_train_median": humidity_median,
        "rows_before_lag_warmup": int(before_lag_drop),
        "rows_modeled": int(len(model_rows)),
        "rows_dropped_for_lags": int(before_lag_drop - len(model_rows)),
        "temp_atemp_correlation": float(raw["temp"].corr(raw["atemp"])),
        "casual_plus_registered_equals_cnt": bool(
            ((raw["casual"] + raw["registered"]) == raw["cnt"]).all()
        ),
        "wind_zero_share": float((raw["windspeed"] == 0).mean()),
        "heavy_storm_hours": int((raw["weathersit"] == 4).sum()),
    }
    return model_rows, notes


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["split"] == "train"]
    val = frame.loc[frame["split"] == "val"]
    test = frame.loc[frame["split"] == "test"]
    return train, val, test


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def scale_target(y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(y_train.mean())
    sigma = float(y_train.std())
    return (y_train - mu) / sigma, mu, sigma


def invert_target(pred_scaled: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Back to bike counts. A negative count cannot be staged, so clip at 0."""
    return np.clip(pred_scaled * sigma + mu, 0, None)


def fit_ridge(
    x_train: np.ndarray,
    y_train_scaled: np.ndarray,
    alpha: float,
) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train_scaled)
    return model


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].to_numpy(dtype=np.float64)


def evaluate_block(
    train: pd.DataFrame,
    val: pd.DataFrame,
    columns: list[str],
    alpha: float,
    target: str = "identity",
) -> dict[str, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(matrix(train, columns))
    x_val = scaler.transform(matrix(val, columns))
    y_train = train["cnt"].to_numpy(dtype=np.float64)
    y_val = val["cnt"].to_numpy(dtype=np.float64)
    if target == "log1p":
        y_train = np.log1p(y_train)
    y_scaled, mu, sigma = scale_target(y_train)
    model = fit_ridge(x_train, y_scaled, alpha)
    pred_scaled = model.predict(x_val)
    pred = pred_scaled * sigma + mu
    if target == "log1p":
        pred = np.expm1(pred)
    pred = np.clip(pred, 0, None)
    scores = regression_scores(y_val, pred)
    scores["alpha"] = alpha
    scores["target"] = target
    return scores


def pytorch_l2(alpha: float, n_train: int) -> float:
    """Match sklearn Ridge on a mean-squared-error objective.

    Ridge minimizes sum of squares + alpha * ||w||^2, and does not penalize
    the intercept. The loops below minimize mean squared error + (l2/2) * ||w||^2.
    Setting l2 = 2 * alpha / n makes those two objectives the same.
    """
    return 2.0 * alpha / n_train


def train_manual(
    x_train: np.ndarray,
    y_train: np.ndarray,
    l2: float,
    lr: float,
    n_epochs: int,
) -> tuple[np.ndarray, float, list[float]]:
    """Session 5 loop: raw tensors, autograd, hand-written gradient step."""
    x = torch.tensor(x_train, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.float64)
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    history: list[float] = []

    def forward(features: torch.Tensor) -> torch.Tensor:
        return features @ weight + bias

    def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean()

    for _ in range(n_epochs):
        pred = forward(x)
        loss = mse_loss(pred, y) + (l2 / 2.0) * (weight ** 2).sum()
        loss.backward()
        with torch.no_grad():
            weight -= lr * weight.grad
            bias -= lr * bias.grad
        weight.grad.zero_()
        bias.grad.zero_()
        history.append(float(loss.detach()))
        if not np.isfinite(history[-1]):
            break
    return weight.detach().numpy().copy(), float(bias.detach()[0]), history


class LinearRegressor(torch.nn.Module):
    """Same linear model as the manual loop, as an nn.Module."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, dtype=torch.float64)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def train_standard(
    x_train: np.ndarray,
    y_train: np.ndarray,
    l2: float,
    lr: float,
    n_epochs: int,
) -> tuple[LinearRegressor, list[float]]:
    """Standard workflow: nn.Module, loss, torch.optim.SGD."""
    x = torch.tensor(x_train, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.float64)
    model = LinearRegressor(x.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        pred = model(x)
        # Penalty is on the weights only, matching Ridge and the manual loop.
        # optimizer weight_decay would also shrink the bias, so it is not used.
        loss = ((pred - y) ** 2).mean() + (l2 / 2.0) * model.linear.weight.pow(2).sum()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if not np.isfinite(history[-1]):
            break
    return model, history


def predict_scaled_manual(x: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    return x @ weight + bias


def predict_scaled_standard(model: LinearRegressor, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float64))
    return pred.numpy()


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def random_split_diagnostic(
    frame: pd.DataFrame,
    columns: list[str],
    alpha: float,
) -> dict[str, float]:
    """Show why a random split flatters this problem.

    Neighboring hours share weather and, through lag_1, almost share a
    target. A random holdout usually keeps an hour's predecessor in train.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(frame))
    cut = int(0.8 * len(order))
    train = frame.iloc[order[:cut]]
    test = frame.iloc[order[cut:]]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(matrix(train, columns))
    x_test = scaler.transform(matrix(test, columns))
    y_scaled, mu, sigma = scale_target(train["cnt"].to_numpy(dtype=np.float64))
    model = fit_ridge(x_train, y_scaled, alpha)
    pred = invert_target(model.predict(x_test), mu, sigma)
    scores = regression_scores(test["cnt"].to_numpy(dtype=np.float64), pred)
    train_stamps = set(train.index)
    previous = test.index - pd.Timedelta(hours=1)
    scores["share_prev_hour_in_train"] = float(np.mean([stamp in train_stamps for stamp in previous]))
    return scores


def coefficient_table(
    names: list[str],
    scaler: StandardScaler,
    y_sigma: float,
    sklearn_coef: np.ndarray,
    manual_coef: np.ndarray,
    standard_coef: np.ndarray,
) -> list[dict[str, float | str]]:
    """Bikes per original unit, so a coefficient can be read in the report.

    All three fits saw standardized inputs and a standardized target.
    Dividing by the input scale and multiplying by the target scale puts the
    weight back in bikes per unit of that column. Correlated columns still
    share credit, so these are not causal effects.
    """
    rows = []
    for i, name in enumerate(names):
        scale = float(scaler.scale_[i])
        rows.append(
            {
                "feature": name,
                "sklearn_bikes_per_unit": float(sklearn_coef[i] * y_sigma / scale),
                "manual_bikes_per_unit": float(manual_coef[i] * y_sigma / scale),
                "standard_bikes_per_unit": float(standard_coef[i] * y_sigma / scale),
            }
        )
    rows.sort(key=lambda row: abs(row["sklearn_bikes_per_unit"]), reverse=True)
    return rows


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    ARTIFACTS.mkdir(exist_ok=True)

    raw = load_hourly()
    frame, prep_notes = build_modeling_frame(raw)
    train, val, test = split_frame(frame)
    print(
        f"rows train/val/test: {len(train)} / {len(val)} / {len(test)} "
        f"(dropped {prep_notes['rows_dropped_for_lags']} for lag warmup or gaps)"
    )

    print("\nValidation MAE by feature block (Ridge, alpha=1, raw count target):")
    block_results = []
    for name, columns in FEATURE_BLOCKS:
        block_train, block_val = train, val
        if "roll_mean_24" in columns:
            # Gaps in the hourly index leave a few rolling windows empty.
            # Score this extra column on the rows where it exists, and say so.
            block_train = train.dropna(subset=["roll_mean_24"])
            block_val = val.dropna(subset=["roll_mean_24"])
        scores = evaluate_block(block_train, block_val, columns, RIDGE_ALPHA, target="identity")
        scores["val_rows"] = int(len(block_val))
        block_results.append({"block": name, "n_features": len(columns), **scores})
        print(
            f"  {name:52}  MAE {scores['mae']:7.2f}  "
            f"RMSE {scores['rmse']:7.2f}  val rows {len(block_val)}"
        )

    roll_rows_train = train.dropna(subset=["roll_mean_24"])
    roll_rows_val = val.dropna(subset=["roll_mean_24"])
    chosen_on_roll_rows = evaluate_block(
        roll_rows_train, roll_rows_val, CHOSEN_FEATURES, RIDGE_ALPHA, target="identity"
    )
    print(
        f"chosen features on the rolling-mean rows only: "
        f"MAE {chosen_on_roll_rows['mae']:.2f} (val rows {len(roll_rows_val)})"
    )

    log_scores = evaluate_block(train, val, CHOSEN_FEATURES, RIDGE_ALPHA, target="log1p")
    print(f"\nlog1p target on the chosen features, validation MAE {log_scores['mae']:.2f}")
    print("That transform is not used. The shipped target is the raw hourly count.")

    print("\nRidge alpha on the chosen features, validation MAE:")
    alpha_results = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        scores = evaluate_block(train, val, CHOSEN_FEATURES, alpha, target="identity")
        alpha_results.append(scores)
        print(f"  alpha {alpha:8}  MAE {scores['mae']:.2f}")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(matrix(train, CHOSEN_FEATURES))
    x_val = scaler.transform(matrix(val, CHOSEN_FEATURES))
    x_test = scaler.transform(matrix(test, CHOSEN_FEATURES))
    y_train = train["cnt"].to_numpy(dtype=np.float64)
    y_val = val["cnt"].to_numpy(dtype=np.float64)
    y_test = test["cnt"].to_numpy(dtype=np.float64)
    y_train_scaled, y_mu, y_sigma = scale_target(y_train)
    l2 = pytorch_l2(RIDGE_ALPHA, len(train))

    ridge = fit_ridge(x_train, y_train_scaled, RIDGE_ALPHA)

    print("\nLearning-rate check, manual loop, validation MAE:")
    lr_results = []
    manual_runs: dict[float, tuple[np.ndarray, float, list[float]]] = {}
    for lr in LEARNING_RATES:
        weight, bias, history = train_manual(x_train, y_train_scaled, l2, lr, N_EPOCHS)
        pred_val = invert_target(predict_scaled_manual(x_val, weight, bias), y_mu, y_sigma)
        gap = max_abs_diff(weight, ridge.coef_)
        finite = bool(np.isfinite(history[-1]))
        converged = bool(finite and gap < 1e-3)
        mae = float(mean_absolute_error(y_val, pred_val)) if converged else float("inf")
        lr_results.append(
            {
                "lr": lr,
                "val_mae": None if not converged else mae,
                "final_loss": history[-1] if finite else None,
                "max_abs_weight_gap_vs_sklearn": gap if finite else None,
                "converged": converged,
            }
        )
        manual_runs[lr] = (weight, bias, history)
        status = "converged" if converged else "diverged"
        print(f"  lr {lr:4}  val MAE {mae:7.2f}  max |w - ridge| {gap:.3e}  {status}")

    usable = [row for row in lr_results if row["converged"]]
    if not usable:
        raise SystemExit("No learning rate reached the Ridge solution. See the table above.")
    best_mae = min(row["val_mae"] for row in usable)
    tied = [row for row in usable if row["val_mae"] <= best_mae + 0.05]
    # Prefer 0.1 when several rates land on the same solution. It is the
    # middle candidate and is easy to defend.
    chosen_lr = 0.1 if any(row["lr"] == 0.1 for row in tied) else tied[0]["lr"]
    print(f"chosen learning rate: {chosen_lr}")

    manual_w, manual_b, manual_loss = manual_runs[chosen_lr]
    standard_model, standard_loss = train_standard(
        x_train, y_train_scaled, l2, chosen_lr, N_EPOCHS
    )
    standard_w = standard_model.linear.weight.detach().numpy().reshape(-1)
    standard_b = float(standard_model.linear.bias.detach()[0])

    def pack(name: str, pred_test_scaled: np.ndarray, pred_val_scaled: np.ndarray) -> dict:
        pred_test = invert_target(pred_test_scaled, y_mu, y_sigma)
        pred_val = invert_target(pred_val_scaled, y_mu, y_sigma)
        return {
            "name": name,
            "val": regression_scores(y_val, pred_val),
            "test": regression_scores(y_test, pred_test),
            "test_negative_before_clip": int((pred_test_scaled * y_sigma + y_mu < 0).sum()),
            "pred_test": pred_test,
        }

    packed = {
        "sklearn": pack("sklearn", ridge.predict(x_test), ridge.predict(x_val)),
        "manual": pack(
            "manual",
            predict_scaled_manual(x_test, manual_w, manual_b),
            predict_scaled_manual(x_val, manual_w, manual_b),
        ),
        "standard": pack(
            "standard",
            predict_scaled_standard(standard_model, x_test),
            predict_scaled_standard(standard_model, x_val),
        ),
    }

    mean_pred = np.full(len(test), y_train.mean())
    lag168_pred = test["lag_168"].to_numpy(dtype=np.float64)
    baselines = {
        "train_mean": regression_scores(y_test, mean_pred),
        "same_hour_last_week": regression_scores(y_test, lag168_pred),
    }
    baselines["train_mean"]["val_mae"] = float(
        mean_absolute_error(y_val, np.full(len(val), y_train.mean()))
    )
    baselines["same_hour_last_week"]["val_mae"] = float(
        mean_absolute_error(y_val, val["lag_168"].to_numpy(dtype=np.float64))
    )

    agreement = {
        "max_abs_weight_sklearn_vs_manual": max_abs_diff(ridge.coef_, manual_w),
        "max_abs_weight_sklearn_vs_standard": max_abs_diff(ridge.coef_, standard_w),
        "max_abs_weight_manual_vs_standard": max_abs_diff(manual_w, standard_w),
        "abs_bias_sklearn_vs_manual": abs(float(ridge.intercept_) - manual_b),
        "abs_bias_manual_vs_standard": abs(manual_b - standard_b),
        "max_abs_test_pred_sklearn_vs_manual": max_abs_diff(
            packed["sklearn"]["pred_test"], packed["manual"]["pred_test"]
        ),
        "max_abs_test_pred_manual_vs_standard": max_abs_diff(
            packed["manual"]["pred_test"], packed["standard"]["pred_test"]
        ),
    }

    leak = random_split_diagnostic(frame, CHOSEN_FEATURES, RIDGE_ALPHA)
    # Diagnostic only. These two columns add up to the target, so a tiny
    # error here means the column was leaked, not that the model is good.
    # It is not used to choose features or hyperparameters.
    leaky_columns_check = evaluate_block(train, test, ["casual", "registered"], RIDGE_ALPHA)

    coefs = coefficient_table(
        CHOSEN_FEATURES,
        scaler,
        y_sigma,
        ridge.coef_.astype(np.float64),
        manual_w,
        standard_w,
    )

    # Where the test quarter is actually hard: daily MAE of the sklearn forecast.
    daily = test.copy()
    daily["pred"] = packed["sklearn"]["pred_test"]
    daily["abs_err"] = (daily["cnt"] - daily["pred"]).abs()
    by_day = (
        daily.groupby(daily.index.date)
        .agg(mae=("abs_err", "mean"), actual=("cnt", "sum"), pred=("pred", "sum"))
        .sort_values("mae", ascending=False)
    )
    worst_days = [
        {
            "date": str(idx),
            "mae": float(row.mae),
            "actual_total": float(row.actual),
            "pred_total": float(row.pred),
        }
        for idx, row in by_day.head(5).iterrows()
    ]

    residual = y_test - packed["sklearn"]["pred_test"]
    # Newsvendor buffer: stock out costs COST_UNDER, an idle bike costs COST_OVER,
    # so the deployed forecast should cover this quantile of (actual - predicted).
    critical_fractile = COST_UNDER / (COST_UNDER + COST_OVER)
    buffer_star = float(np.quantile(residual, critical_fractile))
    buffer_star = max(0.0, buffer_star)

    print("\nTest scores (bike counts, predictions clipped at 0):")
    for name in ("sklearn", "manual", "standard"):
        test_scores = packed[name]["test"]
        print(
            f"  {name:10} MAE {test_scores['mae']:.2f}  "
            f"RMSE {test_scores['rmse']:.2f}  R2 {test_scores['r2']:.3f}  "
            f"negatives clipped {packed[name]['test_negative_before_clip']}"
        )
    print(
        f"  {'mean':10} MAE {baselines['train_mean']['mae']:.2f}  "
        f"RMSE {baselines['train_mean']['rmse']:.2f}"
    )
    print(
        f"  {'lag 168':10} MAE {baselines['same_hour_last_week']['mae']:.2f}  "
        f"RMSE {baselines['same_hour_last_week']['rmse']:.2f}"
    )
    print("weight gaps:", json.dumps(agreement, indent=2))
    print(f"random-split MAE {leak['mae']:.2f} (prev hour in train: {leak['share_prev_hour_in_train']:.1%})")
    print(f"leaky casual+registered MAE on test {leaky_columns_check['mae']:.4f}")
    print("worst test days:", worst_days)
    print(f"newsvendor buffer at {critical_fractile:.2f} quantile: {buffer_star:.1f} bikes")

    metrics = {
        "models": {
            name: {"val": packed[name]["val"], "test": packed[name]["test"],
                   "test_negative_before_clip": packed[name]["test_negative_before_clip"]}
            for name in packed
        },
        "baselines": baselines,
        "agreement": agreement,
        "feature_blocks_val": block_results,
        "chosen_on_rolling_mean_rows": chosen_on_roll_rows,
        "log1p_val": log_scores,
        "alpha_sweep_val": alpha_results,
        "learning_rates": lr_results,
        "chosen_lr": chosen_lr,
        "ridge_alpha": RIDGE_ALPHA,
        "l2": l2,
        "n_epochs": N_EPOCHS,
        "random_split": leak,
        "leaky_casual_registered_test": leaky_columns_check,
        "worst_test_days": worst_days,
        "cost_under": COST_UNDER,
        "cost_over": COST_OVER,
        "critical_fractile": critical_fractile,
        "suggested_buffer_bikes": buffer_star,
        "prep": prep_notes,
        "split_sizes": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "split_bounds": {
            "train": [str(train.index.min()), str(train.index.max())],
            "val": [str(val.index.min()), str(val.index.max())],
            "test": [str(test.index.min()), str(test.index.max())],
        },
    }

    joblib.dump(ridge, ARTIFACTS / "sklearn_ridge.joblib")
    joblib.dump(
        {
            "scaler": scaler,
            "feature_names": CHOSEN_FEATURES,
            "y_mean": y_mu,
            "y_std": y_sigma,
        },
        ARTIFACTS / "preprocessor.joblib",
    )
    torch.save({"w": torch.tensor(manual_w), "b": torch.tensor([manual_b])}, ARTIFACTS / "torch_manual.pt")
    torch.save(standard_model.state_dict(), ARTIFACTS / "torch_standard.pt")

    dashboard = {
        "feature_names": CHOSEN_FEATURES,
        "x_test": x_test,
        "y_test": y_test,
        "ts_test": np.array([stamp.isoformat() for stamp in test.index]),
        "lag168_test": lag168_pred,
        "train_mean": float(y_train.mean()),
        "manual_loss": manual_loss[::10],
        "standard_loss": standard_loss[::10],
        "loss_stride": 10,
        "coefficients": coefs,
        "metrics": metrics,
    }
    joblib.dump(dashboard, ARTIFACTS / "dashboard.joblib")

    export_columns = [
        "cnt",
        "hr",
        "workingday",
        "holiday",
        "temp_c",
        "hum_pct",
        "wind",
        "weathersit",
        "season",
        "lag_1",
        "lag_24",
        "lag_168",
        "precip",
        "morning_rush",
        "evening_rush",
        "split",
    ]
    export = frame[export_columns].copy()
    export.insert(0, "ts", frame.index.astype(str))
    export.to_csv(ARTIFACTS / "distributions.csv", index=False)

    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models and dashboard inputs under {ARTIFACTS}")


if __name__ == "__main__":
    main()
