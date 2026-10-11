"""Load the committed raw files and apply the data-quality decisions."""

import pandas as pd

from . import config


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / name, parse_dates=["date"], index_col="date")
    return df.sort_index()


def load_market_data() -> tuple[pd.DataFrame, dict]:
    """Daily S&P 500 high, low and close joined with the VIX close, plus a data-quality log.

    Data-quality decisions applied here:
    1. The S&P open is dropped. From 1990 to 2005 Yahoo's open equals the previous
       close on almost every day, so any feature built on the open would be fake.
    2. Only the VIX close is kept. Its open, high and low are flat in 1990 and 1991.
    3. Inner join on S&P trading days. This drops VIX prints on US market holidays,
       when the S&P did not trade and there is no return to pair them with.
    """
    spx = _read("spx.csv")
    vix = _read("vix.csv")

    prev_close = spx["close"].shift(1)
    fake_open = (spx["open"] - prev_close).abs() < 1e-6
    fake_open_by_year = fake_open.groupby(spx.index.year).mean()

    df = spx[["high", "low", "close"]].join(vix["close"].rename("vix"), how="inner")

    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert not df.isna().any().any(), "missing values in the raw data"
    assert (df["high"] >= df["low"]).all(), "high below low"
    assert (df[["high", "low", "close", "vix"]] > 0).all().all(), "non-positive prices"

    quality = {
        "spx_rows": int(len(spx)),
        "vix_rows": int(len(vix)),
        "joined_rows": int(len(df)),
        "first_date": df.index[0].strftime("%Y-%m-%d"),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "vix_dates_dropped_no_spx_trading": [d.strftime("%Y-%m-%d") for d in vix.index.difference(spx.index)],
        "share_fake_open_1990_2005": float(fake_open.loc["1990":"2005"].mean()),
        "share_fake_open_2006_2013": float(fake_open.loc["2006":"2013"].mean()),
        "share_fake_open_2014_on": float(fake_open.loc["2014":].mean()),
        "vix_flat_ohlc_days_1990_1991": int((vix.loc["1990":"1991", "high"] == vix.loc["1990":"1991", "low"]).sum()),
        "fake_open_share_by_year": {int(k): round(float(v), 3) for k, v in fake_open_by_year.items()},
    }
    return df, quality
