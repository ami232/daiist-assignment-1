"""Feature engineering. Every feature uses only information known at the close of day t."""

import numpy as np
import pandas as pd

from .config import LONG_WINDOW, TRADING_DAYS, VOL_FLOOR

# Every feature that was built and tested. The feature ablation in train.py runs on this list.
CANDIDATE_FEATURES = [
    "log_rv_1d",
    "log_rv_5d",
    "log_rv_21d",
    "log_rv_21d_vs_1y",
    "vix_gap",
    "vix_change_5d",
    "range_vs_close_5d",
    "spike_share_21d",
    "down_share_21d",
    "return_21d",
    "drawdown_1y",
]

# Tested and dropped: adding them lowered validation AUC (0.790 to 0.789) and raised validation
# log loss, and their coefficients were close to zero. The spike and down-day information is
# already carried by the 1-day vs 5-day vs 21-day vol terms and by the 21-day return.
DROPPED_FEATURES = ["spike_share_21d", "down_share_21d"]

# Order matters: it is the column order the three models are trained on.
FEATURES = [c for c in CANDIDATE_FEATURES if c not in DROPPED_FEATURES]

FEATURE_DESCRIPTIONS = {
    "log_rv_1d": "Log realized vol, last day (HAR daily term)",
    "log_rv_5d": "Log realized vol, last 5 days (HAR weekly term)",
    "log_rv_21d": "Log realized vol, last 21 days (HAR monthly term, also what the fund sizes on)",
    "log_rv_21d_vs_1y": "Log of 21-day vol over 1-year vol (distance from long-run level, mean reversion)",
    "vix_gap": "Log VIX minus log 21-day realized vol (risk priced by options but not yet realized)",
    "vix_change_5d": "5-day change in log VIX (implied vol moving ahead of realized)",
    "range_vs_close_5d": "Log of Parkinson high-low vol over close-to-close vol, last 5 days (intraday turbulence)",
    "spike_share_21d": "Share of the last 21 days' variance coming from the single biggest day (window drop-off)",
    "down_share_21d": "Share of the last 21 days' variance coming from down days (leverage effect)",
    "return_21d": "S&P 500 log return over the last 21 days",
    "drawdown_1y": "Log distance of the S&P 500 from its 1-year closing high",
}


def realized_vol(squared_returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility: square root of 252 times the mean squared daily return.

    Mean is taken around zero, not around the sample mean: over a few days the mean
    return is noise, and this is the standard realized-variance convention.
    """
    return np.sqrt(TRADING_DAYS * squared_returns.rolling(window).mean())


def _log_vol(vol: pd.Series) -> pd.Series:
    return np.log(vol.clip(lower=VOL_FLOOR))


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    """All model features plus the realized-vol helpers the target and evaluation need."""
    close, high, low, vix = market["close"], market["high"], market["low"], market["vix"]
    r = np.log(close).diff()
    r2 = r**2

    rv_1d = np.sqrt(TRADING_DAYS) * r.abs()
    rv_5d = realized_vol(r2, 5)
    rv_21d = realized_vol(r2, 21)
    rv_1y = realized_vol(r2, LONG_WINDOW)

    # Parkinson (1980): variance estimated from the daily high-low range.
    # Uses high and low only, never the open, because the open is unreliable before 2006.
    parkinson_var = np.log(high / low) ** 2 / (4 * np.log(2))
    parkinson_5d = np.sqrt(TRADING_DAYS * parkinson_var.rolling(5).mean())

    sum_r2_21d = r2.rolling(21).sum()
    down_r2_21d = (r2 * (r < 0)).rolling(21).sum()

    f = pd.DataFrame(index=market.index)
    f["log_rv_1d"] = _log_vol(rv_1d)
    f["log_rv_5d"] = _log_vol(rv_5d)
    f["log_rv_21d"] = _log_vol(rv_21d)
    f["log_rv_21d_vs_1y"] = _log_vol(rv_21d) - _log_vol(rv_1y)
    f["vix_gap"] = np.log(vix / 100) - _log_vol(rv_21d)
    f["vix_change_5d"] = np.log(vix).diff(5)
    f["range_vs_close_5d"] = _log_vol(parkinson_5d) - _log_vol(rv_5d)
    f["spike_share_21d"] = r2.rolling(21).max() / sum_r2_21d
    f["down_share_21d"] = down_r2_21d / sum_r2_21d
    f["return_21d"] = np.log(close).diff(21)
    f["drawdown_1y"] = np.log(close / close.rolling(LONG_WINDOW).max())

    # Helpers, not features: used to build the target, the baselines and the cost.
    f["daily_return"] = close.pct_change()
    f["rv_21d"] = rv_21d
    f["rv_1y"] = rv_1y
    f["vix"] = vix
    f["close"] = close
    return f
