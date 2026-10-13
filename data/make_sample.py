"""
Builds data/hotels_sample.csv from the full Hotel Booking Demand dataset.

One-off script, not part of `main.py train`: the sample it produces is the
committed dataset. Uniform random sample (fixed seed) across the full
July 2015 - August 2017 arrival window, then sorted by arrival date so the
temporal split in train.py is a simple slice.
"""

from pathlib import Path
import urllib.request

import pandas as pd

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
DATA_DIR = Path(__file__).resolve().parent
FULL_PATH = DATA_DIR / "hotels_full.csv"
SAMPLE_PATH = DATA_DIR / "hotels_sample.csv"
N_ROWS = 20_000
SEED = 42
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def main() -> None:
    if not FULL_PATH.exists():
        urllib.request.urlretrieve(URL, FULL_PATH)
    df = pd.read_csv(FULL_PATH)

    sample = df.sample(n=N_ROWS, random_state=SEED)
    arrival = pd.to_datetime(dict(
        year=sample.arrival_date_year,
        month=sample.arrival_date_month.map({m: i for i, m in enumerate(MONTHS, 1)}),
        day=sample.arrival_date_day_of_month,
    ))
    sample = sample.assign(_arrival=arrival).sort_values("_arrival", kind="stable").drop(columns="_arrival")
    sample.to_csv(SAMPLE_PATH, index=False)

    print(f"full: {len(df):,} rows, cancel rate {df.is_canceled.mean():.3f}")
    print(f"sample: {len(sample):,} rows, cancel rate {sample.is_canceled.mean():.3f}")
    print(sample.hotel.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
