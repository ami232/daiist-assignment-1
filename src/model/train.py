import joblib
import numpy as np
import optuna
import polars as pl
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.constants import MODELS_DIR
from src.data.export import CURVES_PATH, SERIES_PATH
from src.data.features import BATTERY_HOURS, build_daily
from src.logger import get_logger
from src.plotting.charts import line_chart, metrics_table

cfg = CONFIG.model
log = get_logger(__name__)
Val = tuple[np.ndarray, np.ndarray] | None
THRESHOLDS = np.round(np.arange(0.05, 1, 0.05), 2)


def tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def bce(
    p: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    # custom loss: each day's BCE scaled by |spread - cost|
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(w * (y * p.log() + (1 - y) * (1 - p).log())).mean()


# raw tensors, hand written gradient step
class ManualLogReg:
    def __init__(self, lr: float, l2: float) -> None:
        self.lr = lr
        self.l2 = l2
        self.history: list[tuple[float, float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.w + self.b)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray | None = None,
        val: Val = None,
    ) -> "ManualLogReg":
        self.history = []
        x, y = tensor(x), tensor(y)
        weight = torch.ones_like(y) if weight is None else tensor(weight)
        self.w = torch.zeros(x.shape[1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        for _ in range(cfg.epochs):
            loss = bce(self.forward(x), y, weight)
            (loss + self.l2 * (self.w**2).sum()).backward()
            with torch.no_grad():
                self.w -= self.lr * self.w.grad
                self.b -= self.lr * self.b.grad
            self.w.grad.zero_()
            self.b.grad.zero_()
            if val:
                self.history.append((loss.item(), log_loss(val[1], self.predict_proba(val[0]), labels=[0, 1])))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            p = self.forward(tensor(x)).numpy()
        return np.column_stack([1 - p, p])


# nn.Linear plus Adam
class TorchLogReg:
    def __init__(self, lr: float, weight_decay: float) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.history: list[tuple[float, float]] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        val: Val = None,
    ) -> "TorchLogReg":
        self.history = []
        torch.manual_seed(cfg.seed)
        x, y = tensor(x), tensor(y)
        self.net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 1), torch.nn.Sigmoid())
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy(self.net(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            if val:
                self.history.append((loss.item(), log_loss(val[1], self.predict_proba(val[0]), labels=[0, 1])))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            p = self.net(tensor(x)).squeeze(-1).numpy()
        return np.column_stack([1 - p, p])  # probability of each class


SEARCH = {
    "sklearn": lambda t: LogisticRegression(
        C=t.suggest_float(
            "C",
            1e-3,
            10,
            log=True,
        ),
        max_iter=1000,
    ),
    "manual_torch": lambda t: ManualLogReg(
        lr=t.suggest_float(
            "lr",
            1e-2,
            1.0,
            log=True,
        ),
        l2=t.suggest_float(
            "l2",
            1e-5,
            1e-1,
            log=True,
        ),
    ),
    "standard_torch": lambda t: TorchLogReg(
        lr=t.suggest_float(
            "lr",
            1e-3,
            1e-1,
            log=True,
        ),
        weight_decay=t.suggest_float(
            "weight_decay",
            1e-5,
            1e-1,
            log=True,
        ),
    ),
}


def load_raw() -> pl.DataFrame:
    # exported series and bid curves on one hourly UTC grid
    series = (
        pl.read_parquet(SERIES_PATH)
        .with_columns(pl.col("datetime_utc").dt.truncate("1h"))
        .group_by("datetime_utc", "series")
        .agg(pl.col("value").mean().cast(pl.Float64))
        .pivot(on="series", index="datetime_utc", values="value")
        .sort("datetime_utc")
        .upsample("datetime_utc", every="1h")
    )
    curves = pl.read_parquet(CURVES_PATH).with_columns(pl.exclude("datetime_utc").cast(pl.Float64))
    return series.join(
        curves,
        on="datetime_utc",
        how="left",
        maintain_order="left",
    )


# clips columns to train percentiles
class Winsorizer(
    BaseEstimator,  # get_params/set_params
    TransformerMixin,  # fit_transform
):
    def __init__(self, q: float = 1.0) -> None:
        self.q = q

    def fit(self, x: np.ndarray, y: None = None) -> "Winsorizer":
        self.low_, self.high_ = np.nanpercentile(x, [self.q, 100 - self.q], axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.low_, self.high_)


def preprocessor() -> Pipeline:
    return make_pipeline(
        Winsorizer(q=1.0),
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )


def cross_validate(
    model: LogisticRegression | ManualLogReg | TorchLogReg,
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, list[list[tuple[float, float]]], np.ndarray]:
    # expanding window, returns mean validation log loss
    # only the manual model trains and is tuned on the weighted loss
    weighted = isinstance(model, ManualLogReg)
    losses, histories = [], []
    # out of fold predictions, NaN on the first block
    oof = np.full(len(y), np.nan)
    for train, val in TimeSeriesSplit(cfg.n_folds).split(x):
        # refit on each fold, no leakage
        prep = preprocessor().fit(x[train])
        x_train, x_val = prep.transform(x[train]), prep.transform(x[val])

        # sklearn has no loss history
        if weighted:
            model.fit(
                x_train,
                y[train],
                weight[train],
                val=(x_val, y[val]),
            )
            histories.append(model.history)
        elif hasattr(model, "history"):
            model.fit(x_train, y[train], val=(x_val, y[val]))
            histories.append(model.history)
        else:
            model.fit(x_train, y[train])

        p = model.predict_proba(x_val)
        oof[val] = p[:, 1]
        losses.append(
            log_loss(
                y[val],
                p,
                sample_weight=weight[val] if weighted else None,
                labels=[0, 1],
            )
        )
    return float(np.mean(losses)), histories, oof


def feature_matrix(df: pl.DataFrame) -> np.ndarray:
    x = df.drop(["day", "y_spread", "split"], strict=False).to_numpy().astype(float)
    x[~np.isfinite(x)] = np.nan
    return x


def profit(cycle: np.ndarray, spread: np.ndarray) -> float:
    # profit per MW of battery capacity, mean over all days
    return BATTERY_HOURS * float(np.sum((spread - cfg.cycle_cost) * cycle))


def ece(
    p: np.ndarray,
    y: np.ndarray,
    bins: int = 10,
) -> float:
    # gap between mean probability and observed rate, weighted by bin size
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in np.unique(idx)))


def logit(p: np.ndarray) -> np.ndarray:
    # log odds, clipped to avoid NaN and inf
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


# platt scaling: a 1 feature logistic regression on the model's log odds
def calibrate(
    calibrator: LogisticRegression,
    p: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(logit(p))[:, 1]


def best_threshold(p: np.ndarray, spread: np.ndarray) -> float:
    # most profitable cut on the grid
    return float(max(THRESHOLDS, key=lambda t: profit(p >= t, spread)))


def evaluate(
    p: np.ndarray,
    spread: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = spread > cfg.cycle_cost
    cycle = p >= threshold
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, cycle),
        "roc_auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-3, 1 - 1e-3), labels=[False, True]),
        "ece": ece(p, y),
        "profit_eur_per_mw": profit(cycle, spread),
    }


def train() -> pl.DataFrame:
    # tunes with CV, fits on 80%, ranks on the last 20%
    df = build_daily(load_raw(), start=cfg.start)
    x = feature_matrix(df)
    spread = df["y_spread"].to_numpy()
    y = (spread > cfg.cycle_cost).astype(np.float32)

    n = int(len(df) * (1 - cfg.test_frac))

    # a day weighs what a wrong call on it costs, mean 1 on train
    weight = np.abs(spread - cfg.cycle_cost)
    weight = weight / weight[:n].mean()

    log.info(f"{len(df)} days, {x.shape[1]} features, test from {df['day'][n]}")

    MODELS_DIR.mkdir(exist_ok=True)
    prep = preprocessor().fit(x[:n])
    joblib.dump(prep, MODELS_DIR / "preprocessor.joblib")
    split = pl.Series(["train"] * n + ["test"] * (len(df) - n))
    df.with_columns(split=split).write_parquet(MODELS_DIR / "dataset.parquet")

    # naive: cycle if yesterday's spread beat the cost
    naive = (df["spread_1d"].to_numpy() > cfg.cycle_cost).astype(float)

    rows = [
        {"model": "naive", "params": "", **evaluate(naive[n:], spread[n:], 0.5)},
        {"model": "always_cycle", "params": "", **evaluate(np.ones(len(df) - n), spread[n:], 0.5)},
    ]
    cv_rows = []

    curves = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for name, make in SEARCH.items():
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=cfg.seed))
        study.optimize(
            lambda t: cross_validate(
                make(t),
                x[:n],
                y[:n],
                weight[:n],
            )[0],
            n_trials=cfg.n_trials,
        )
        best = make(optuna.trial.FixedTrial(study.best_params))

        cv_loss, histories, oof = cross_validate(
            best,
            x[:n],
            y[:n],
            weight[:n],
        )

        log.info(f"{name}: cv log loss {cv_loss:.4f} {study.best_params}")

        # calibration and threshold both fitted on the out of fold days, test stays untouched
        seen = ~np.isnan(oof)
        calibrator = LogisticRegression().fit(logit(oof[seen]), y[:n][seen])
        oof_cal = calibrate(calibrator, oof[seen])
        threshold = best_threshold(oof_cal, spread[:n][seen])

        # CV metrics on every validated train day
        params = ", ".join(f"{k}={v:.3g}" for k, v in study.best_params.items())
        cv_rows.append(
            {
                "model": name,
                "params": params,
                "ece_raw": ece(oof[seen], y[:n][seen]),
                **evaluate(oof_cal, spread[:n][seen], threshold),
            }
        )

        if histories:
            history = np.mean(histories, axis=0)
            curves[f"{name} train"], curves[f"{name} val"] = history[:, 0], history[:, 1]

        # refit on the full train set
        x_train = prep.transform(x[:n])
        model = best.fit(x_train, y[:n], weight[:n]) if isinstance(best, ManualLogReg) else best.fit(x_train, y[:n])

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        p_raw = model.predict_proba(prep.transform(x[n:]))[:, 1]
        p = calibrate(calibrator, p_raw)

        rows.append(
            {
                "model": name,
                "params": str(study.best_params),
                "ece_raw": ece(p_raw, y[n:]),
                **evaluate(p, spread[n:], threshold),
            }
        )

    fig = line_chart(
        pl.DataFrame({"epoch": range(cfg.epochs), **curves}),
        "epoch",
        list(curves),
        title="CV loss, mean over folds",
    )
    fig.write_image(MODELS_DIR / "cv_loss.png")

    # naive on the same validated days
    cv_rows.append(
        {
            "model": "naive",
            "params": "yesterday's spread",
            "ece_raw": ece(naive[:n][seen], y[:n][seen]),
            **evaluate(naive[:n][seen], spread[:n][seen], 0.5),
        }
    )
    cv_rows.append(
        {
            "model": "always_cycle",
            "params": "",
            "ece_raw": ece(np.ones(seen.sum()), y[:n][seen]),
            **evaluate(np.ones(seen.sum()), spread[:n][seen], 0.5),
        }
    )
    footer = "<br>".join(
        [
            f"Cycle when the {BATTERY_HOURS}h charge-then-discharge spread after losses beats {cfg.cycle_cost:.0f} EUR/MWh.",
            f"Train {df['day'][0]} to {df['day'][n - 1]}, {cfg.n_folds}-fold expanding window CV on {seen.sum()} validated days."
            f" Test from {df['day'][n]} held out (ranking.csv).",
            "Probabilities Platt calibrated and threshold set for profit on these same out of fold days.",
        ]
    )
    table = metrics_table(
        pl.DataFrame(cv_rows).sort("profit_eur_per_mw", descending=True),
        lower_better=("log_loss", "ece", "ece_raw"),
        footer=footer,
        title="Cross-validation metrics",
    )
    table.write_image(MODELS_DIR / "cv_metrics.png")

    ranking = pl.DataFrame(rows).sort("profit_eur_per_mw", descending=True)
    ranking.write_csv(MODELS_DIR / "ranking.csv")
    return ranking
