import numpy as np
import polars as pl
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf


def acf_frame(
    x: pl.Series,
    nlags: int,
    partial: bool = False,
) -> pl.DataFrame:
    # lags 1..nlags with a 95% white-noise band
    values = x.drop_nulls().to_numpy()
    r = pacf(values, nlags=nlags) if partial else acf(values, nlags=nlags, fft=True)
    return pl.DataFrame(
        {
            "lag": np.arange(1, nlags + 1),
            "acf": r[1:],
            "ci": 1.96 / np.sqrt(len(values)),
        },
    )


def acf_by_group(
    df: pl.DataFrame,
    value: str,
    group: str,
    nlags: int,
) -> pl.DataFrame:
    # df must be time sorted
    return pl.concat(
        [
            acf_frame(part[value], nlags).with_columns(pl.lit(key[0]).alias(group))
            for key, part in df.partition_by(group, as_dict=True, maintain_order=True).items()
        ],
    )


def ljung_box(
    x: pl.Series,
    lags: list[int],
) -> pl.DataFrame:
    # H0: no autocorrelation up to each lag
    out = acorr_ljungbox(x.drop_nulls().to_numpy(), lags=lags)
    return pl.from_pandas(out.reset_index(names="lag"))
