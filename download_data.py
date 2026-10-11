"""Download daily S&P 500 and VIX history from Yahoo Finance into data/raw/.

Run this ONCE on your own machine, then commit the two CSVs to the repo.
The training pipeline reads the committed files and never touches the network,
so anyone (and the CI check) can retrain from scratch on exactly the same data.

yfinance is only needed for this one-off download, so it is not added to
pyproject.toml. uv pulls it in temporarily:

    uv run --with yfinance python download_data.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

START = "1990-01-01"  # first full year of VIX history
OUT_DIR = Path(__file__).resolve().parent / "data" / "raw"
TICKERS = {"spx": "^GSPC", "vix": "^VIX"}
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch(ticker: str) -> pd.DataFrame:
    """Daily OHLCV for one ticker, indexed by a plain (timezone-free) date."""
    df = yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=False)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "date"
    df = df[COLUMNS]
    df.columns = [c.lower() for c in df.columns]
    return df


def quality_report(name: str, df: pd.DataFrame) -> None:
    """Print the checks worth knowing about before building features."""
    print(f"\n{name}: {len(df):,} rows, {df.index.min().date()} to {df.index.max().date()}")
    print(f"  missing values:           {int(df.isna().sum().sum())}")
    print(f"  duplicated dates:         {int(df.index.duplicated().sum())}")
    print(f"  rows with high == low:    {int((df['high'] == df['low']).sum())}")
    print(f"  rows with open == close:  {int((df['open'] == df['close']).sum())}")
    print(f"  rows with high < low:     {int((df['high'] < df['low']).sum())}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ticker in TICKERS.items():
        df = fetch(ticker)
        path = OUT_DIR / f"{name}.csv"
        df.to_csv(path, float_format="%.4f")
        quality_report(name, df)
        print(f"  saved to {path}")


if __name__ == "__main__":
    main()
