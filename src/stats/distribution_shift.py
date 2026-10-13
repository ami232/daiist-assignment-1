import numpy as np
import polars as pl

ROLLING: dict[str, str] = {
    "mean": "rolling_mean",
    "std": "rolling_std",
    "median": "rolling_median",
    "skew": "rolling_skew",
    "kurt": "rolling_kurtosis",
}


def rolling_moments(
    col: str,
    window: int,
    moments: tuple[str, ...] = ("mean", "std", "skew", "kurt"),
) -> list[pl.Expr]:
    # named <col>_<moment>_<window>
    return [getattr(pl.col(col), ROLLING[m])(window).alias(f"{col}_{m}_{window}") for m in moments]


def spike_flag(
    col: str,
    window: int,
    k: float = 5.0,
) -> pl.Expr:
    # |x - rolling median| > k robust sigmas (MAD)
    dev = (pl.col(col) - pl.col(col).rolling_median(window)).abs()
    return (dev > k * 1.4826 * dev.rolling_median(window)).alias(f"{col}_spike")


def psi(
    expected: pl.Series,
    actual: pl.Series,
    bins: int = 10,
) -> float:
    # population stability index on expected's quantile bins
    edges = np.unique(np.quantile(expected.drop_nulls(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected.drop_nulls(), edges)[0] / expected.drop_nulls().len()
    a = np.histogram(actual.drop_nulls(), edges)[0] / actual.drop_nulls().len()
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    rows = [
        (c, psi(reference[c], current[c]))
        for c in columns
        if reference[c].drop_nulls().len() and current[c].drop_nulls().len()
    ]
    return pl.DataFrame(rows, schema=["column", "psi"], orient="row").sort("psi", descending=True)
