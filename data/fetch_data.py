"""
One-off download of the raw monthly series from the Banco Central do Brasil
SGS API (via python-bcb). NOT part of the training pipeline: train.ipynb only
reads the committed CSV this script writes, so training is reproducible and
never depends on the API being up.

Re-run manually only when you deliberately want fresher data:

    uv run python data/fetch_data.py

Series descriptions can be checked at
https://www3.bcb.gov.br/sgspub/ (search by code).
"""

import time
from datetime import date
from pathlib import Path

import pandas as pd
from bcb import sgs

SERIES = {
    "ipca": 433,       # IPCA, monthly % change (target source)
    "ipca15": 7478,    # IPCA-15, monthly % change (mid-month preview of IPCA)
    "selic": 4390,     # Selic accumulated in the month, %
    "usdbrl": 3698,    # USD/BRL PTAX sell rate, monthly average
    "icbr": 27574,     # IC-Br commodity index (BRL), level
    "ibcbr": 24364,    # IBC-Br economic activity, seasonally adjusted, level
}

START = "1999-01-01"  # inflation targeting regime + floating FX start in 1999
OUT = Path(__file__).resolve().parent / "raw" / "bcb_monthly.csv"


def fetch(name: str, code: int, attempts: int = 5) -> pd.DataFrame:
    # The SGS API intermittently answers 502; retry with backoff.
    for i in range(attempts):
        try:
            return sgs.get({name: code}, start=START)
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"  {name} ({code}) failed: {e}; retrying")
            time.sleep(2 ** i)


def main() -> None:
    df = pd.concat([fetch(name, code) for name, code in SERIES.items()], axis=1)
    df.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)
    print(f"Wrote {OUT} ({len(df)} rows x {df.shape[1]} cols), fetched {date.today()}")


if __name__ == "__main__":
    main()
