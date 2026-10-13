import warnings

import polars as pl
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

MIN_OBS = 20


def adf_kpss(x: pl.Series) -> dict[str, float]:
    # ADF H0: unit root, KPSS H0: stationary
    values = x.drop_nulls().to_numpy()
    # too short or constant: untestable
    if values.size < MIN_OBS or values.min() == values.max():
        return dict.fromkeys(("adf_stat", "adf_p", "kpss_stat", "kpss_p"), float("nan"))
    adf_stat, adf_p, *_ = adfuller(values, autolag="AIC", result_object=False)
    # KPSS p-values are capped to [0.01, 0.1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_p, *_ = kpss(
            values,
            regression="c",
            nlags="auto",
            result_object=False,
        )
    return {
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
    }
