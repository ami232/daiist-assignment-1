# Scores every day from the saved train artifacts

from functools import cache

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import CONFIG
from src.constants import MODELS_DIR
from src.data.export import SERIES_PATH
from src.data.features import BATTERY_HOURS, HOUR
from src.data.partitions import market_day
from src.model.train import SEARCH, ManualLogReg, TorchLogReg, feature_matrix
from src.stats.simulation import block_bootstrap_sums

MODELS = list(SEARCH)
cost = CONFIG.model.cycle_cost
good = pl.col("y_spread") > cost
margin = BATTERY_HOURS * (pl.col("y_spread") - cost)


@cache
def _artifact(name: str) -> Pipeline | LogisticRegression | ManualLogReg | TorchLogReg:
    return joblib.load(MODELS_DIR / f"{name}.joblib")


@cache
def feature_names() -> list[str]:
    return [c for c in pl.read_parquet_schema(MODELS_DIR / "dataset.parquet") if c not in ("day", "y_spread", "split")]


def load_predictions() -> pl.DataFrame:
    df = pl.read_parquet(MODELS_DIR / "dataset.parquet")
    x = _artifact("preprocessor").transform(feature_matrix(df))
    probs = {f"p_{m}": _artifact(m).predict_proba(x)[:, 1].astype(float) for m in MODELS}
    return df.with_columns(
        p_naive=(pl.col("spread_1d") > cost).cast(pl.Float64),
        spread_30d=pl.col("y_spread").rolling_mean(30),
        **probs,
    )


def linear_terms(model: str) -> tuple[np.ndarray, float]:
    # weights and bias on standardised features
    m = _artifact(model)
    if isinstance(m, LogisticRegression):
        return m.coef_[0], float(m.intercept_[0])
    if isinstance(m, ManualLogReg):
        return m.w.detach().numpy(), float(m.b.item())
    layer = m.net[0]
    return layer.weight.detach().numpy()[0], float(layer.bias.item())


def weights() -> pl.DataFrame:
    return pl.DataFrame({"feature": feature_names(), **{m: linear_terms(m)[0] for m in MODELS}})


def contributions(
    row: pl.DataFrame,
    model: str,
) -> tuple[pl.DataFrame, float]:
    # log-odds terms of one day, plus the bias
    w, b = linear_terms(model)
    names = feature_names()
    x = _artifact("preprocessor").transform(feature_matrix(row.select(names)))[0]
    terms = pl.DataFrame({"feature": names, "value": row.select(names).row(0), "contribution": w * x})
    return terms.sort(pl.col("contribution").abs(), descending=True), b


def scorecard(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    best = df.select(margin.filter(good).sum()).item()
    y = df.select(good).to_series().to_numpy()
    rows = []
    for m in ["naive", *MODELS]:
        cycle = pl.col(f"p_{m}") >= threshold
        p = df[f"p_{m}"].to_numpy()
        rows.append(
            df.select(
                model=pl.lit(m),
                cycles=cycle.sum(),
                hit_rate=good.filter(cycle).mean(),
                accuracy=(cycle == good).mean(),
                profit_eur_per_mw=margin.filter(cycle).sum(),
                missed_eur_per_mw=margin.filter(good & ~cycle).sum(),
                roc_auc=pl.lit(roc_auc_score(y, p) if 0 < y.sum() < len(y) else None, pl.Float64),
                brier=pl.lit(float(np.mean((p - y) ** 2))),
            ),
        )
    return (
        pl.concat(rows)
        .with_columns(capture=pl.col("profit_eur_per_mw") / best)
        .sort("profit_eur_per_mw", descending=True)
    )


def threshold_sweep(df: pl.DataFrame, thresholds: list[float]) -> pl.DataFrame:
    return pl.concat(scorecard(df, t).select("model", "profit_eur_per_mw", threshold=pl.lit(t)) for t in thresholds)


def calibration(df: pl.DataFrame, bins: int = 10) -> pl.DataFrame:
    return pl.concat(
        df.group_by(bin=(pl.col(f"p_{m}") * bins).floor().clip(0, bins - 1))
        .agg(p_mean=pl.col(f"p_{m}").mean(), observed=good.mean(), n=pl.len())
        .select(
            model=pl.lit(m),
            p_mean="p_mean",
            observed="observed",
            n="n",
        )
        .sort("p_mean")
        for m in MODELS
    )


def outcomes(
    df: pl.DataFrame,
    model: str,
    threshold: float,
) -> pl.DataFrame:
    # each day: earned, lost, saved or missed, EUR/MW
    cycle = pl.col(f"p_{model}") >= threshold
    cell = (
        pl.when(cycle & good)
        .then(pl.lit("earned"))
        .when(cycle)
        .then(pl.lit("lost"))
        .when(good)
        .then(pl.lit("missed"))
        .otherwise(pl.lit("saved"))
    )
    return df.select("day", cell=cell, eur=margin.abs())


def edge(
    df: pl.DataFrame,
    model: str,
    threshold: float,
) -> pl.DataFrame:
    # EUR/MW gained vs naive each day
    cycle = (pl.col(f"p_{model}") >= threshold).cast(pl.Int8)
    naive = (pl.col("p_naive") >= threshold).cast(pl.Int8)
    return df.select(
        "day",
        "y_spread",
        "p_naive",
        p=f"p_{model}",
        cycle=cycle,
        naive=naive,
        edge=margin * (cycle - naive),
    )


def edge_interval(daily_edge: pl.Series, level: float = 0.9) -> tuple[float, float]:
    sums = block_bootstrap_sums(daily_edge.to_numpy())
    lo, hi = np.quantile(sums, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def zscores(
    df: pl.DataFrame,
    columns: list[str],
    window: int,
) -> pl.DataFrame:
    # vs the trailing window, today excluded
    return df.select(
        "day",
        *[pl.col(c) for c in columns],
        *[
            ((pl.col(c) - pl.col(c).shift(1).rolling_mean(window)) / pl.col(c).shift(1).rolling_std(window)).alias(
                f"z|{c}"
            )
            for c in columns
        ],
    )


def load_hourly_prices() -> pl.DataFrame:
    hourly = (
        pl.scan_parquet(SERIES_PATH)
        .filter(series="price_es")
        .group_by(day=market_day("datetime_utc"), hour=HOUR)
        .agg(price=pl.col("value").mean().cast(pl.Float64))
        .sort("day", "hour")
        .collect()
    )
    rank = pl.col("price").rank("ordinal").over("day")
    n = pl.len().over("day")
    return hourly.with_columns(
        role=pl.when(rank <= BATTERY_HOURS)
        .then(pl.lit("charge"))
        .when(rank > n - BATTERY_HOURS)
        .then(pl.lit("discharge")),
    )


def hourly_zscores(prices: pl.DataFrame, window: int) -> pl.DataFrame:
    # each hour vs the same hour over the trailing window
    past = pl.col("price").shift(1)
    return prices.with_columns(
        mean=past.rolling_mean(window).over("hour"),
        std=past.rolling_std(window).over("hour"),
    ).with_columns(z=(pl.col("price") - pl.col("mean")) / pl.col("std"))
