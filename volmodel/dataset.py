"""Target definition and chronological split: the decisions that follow from the business framing."""

import pandas as pd

from . import config
from .data import load_market_data
from .features import FEATURES, build_features


def add_target(f: pd.DataFrame) -> pd.DataFrame:
    """Label = 1 if next month's realized vol ends up above the trailing 21-day vol the fund sizes on.

    The fund sets leverage = target vol / trailing 21-day vol. When the next 21 days turn out
    more volatile than that, the fund runs above its risk target: that is the event we predict.
    """
    f = f.copy()
    h = config.HORIZON
    fwd_vol = f["rv_21d"].shift(-h)                     # realized vol over days t+1 ... t+21
    f["fwd_vol_21d"] = fwd_vol
    f["vol_rises"] = (fwd_vol > f["rv_21d"]).astype(float).where(fwd_vol.notna())

    # What the vol-target fund earns over the next month with no haircut (used for the business cost).
    leverage = (config.TARGET_VOL / f["rv_21d"]).clip(upper=config.MAX_LEVERAGE)
    f["leverage"] = leverage
    f["fund_fwd_return_21d"] = leverage * (f["close"].shift(-h) / f["close"] - 1)

    # Date of the last return inside the label window. Used to purge the split boundaries.
    dates = pd.Series(f.index, index=f.index)
    f["label_end"] = dates.shift(-h)
    return f


def assign_split(f: pd.DataFrame) -> pd.Series:
    """train / val / test by date, keeping a row only if its whole label window sits inside its block.

    Consecutive labels share 20 of their 21 future returns, so a random split would put
    near-duplicate rows on both sides. Requiring the label window to end inside the block
    removes the last 21 rows before each boundary (the purge).
    Rows whose label is not known yet (the last 21 days) are 'live': predicted, never scored.
    Rows still inside the 1-year feature warm-up are 'warmup' and never used.
    """
    date = f.index.to_series()
    end = f["label_end"]
    ready = f[FEATURES].notna().all(axis=1)

    split = pd.Series("purged", index=f.index)
    split[~ready] = "warmup"
    split[ready & (end <= pd.Timestamp(config.TRAIN_END))] = "train"
    split[ready & (date >= pd.Timestamp(config.VAL_START)) & (end <= pd.Timestamp(config.VAL_END))] = "val"
    split[ready & (date >= pd.Timestamp(config.TEST_START)) & end.notna()] = "test"
    split[ready & end.isna()] = "live"
    return split


def build_dataset() -> tuple[pd.DataFrame, dict]:
    """Raw files to one table: features, target, helpers and split label per trading day."""
    market, quality = load_market_data()
    f = add_target(build_features(market))
    f["split"] = assign_split(f)
    quality["rows_by_split"] = {k: int(v) for k, v in f["split"].value_counts().items()}
    return f, quality
