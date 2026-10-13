"""
One-off download of the raw data. NOT part of `main.py train`: the pipeline
only ever reads the committed CSV next to this file, so training is fully
reproducible offline. This script documents exactly how that CSV was built.

Source: City of Chicago Data Portal, "Food Inspections" (dataset 4ijn-s7e5)
https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5

What it keeps
- Restaurants only, every inspection type, 2019-07-01 to 2026-06-30.
  The upper bound freezes the snapshot (the portal is updated weekly) and
  stops before July-September 2026, whose published inspection counts were
  still about half the usual monthly volume when this was downloaded
  (October 2026).
- One row per inspection.

What it changes
- The free-text `violations` column (inspector comments, ~1 KB per row) is
  reduced to the list of violation code numbers cited, e.g. "3|47|55".
  Nothing else is transformed; all feature engineering happens in train.py.

Run:  uv run python data/download_data.py
"""

import io
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ENDPOINT = "https://data.cityofchicago.org/resource/4ijn-s7e5.csv"
START, END = "2019-07-01", "2026-07-01"  # END is exclusive
PAGE = 20_000
OUT = Path(__file__).resolve().parent / "chicago_restaurant_inspections.csv"

COLUMNS = [
    "inspection_id", "license_", "dba_name", "risk", "zip", "inspection_date",
    "inspection_type", "results", "violations", "latitude", "longitude",
]
# Each cited violation starts with its code number: "55. PHYSICAL FACILITIES ... | 47. ..."
VIOLATION_CODE = re.compile(r"(?:^|\|\s*)(\d{1,2})\.\s")


def fetch_page(offset: int) -> pd.DataFrame:
    query = urllib.parse.urlencode({
        "$select": ",".join(COLUMNS),
        "$where": (
            f"facility_type='Restaurant' AND inspection_date>='{START}' "
            f"AND inspection_date<'{END}'"
        ),
        "$order": "inspection_id",
        "$limit": PAGE,
        "$offset": offset,
    })
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=180) as response:
                return pd.read_csv(io.BytesIO(response.read()), dtype=str)
        except Exception as error:  # network hiccup or throttling: back off and retry
            print(f"  retry {attempt + 1} after error: {error}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch page at offset {offset}")


def violation_codes(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return "|".join(VIOLATION_CODE.findall(text))


def main() -> None:
    pages, offset = [], 0
    while True:
        page = fetch_page(offset)
        print(f"offset {offset:>6}: {len(page)} rows")
        if page.empty:
            break
        pages.append(page)
        if len(page) < PAGE:
            break
        offset += PAGE

    raw = pd.concat(pages, ignore_index=True)
    raw["violation_codes"] = raw["violations"].map(violation_codes)
    raw["inspection_date"] = raw["inspection_date"].str.slice(0, 10)
    raw = raw.rename(columns={"license_": "license"}).drop(columns="violations")
    for coordinate in ("latitude", "longitude"):
        raw[coordinate] = pd.to_numeric(raw[coordinate]).round(5)
    raw = raw.sort_values(["inspection_date", "inspection_id"]).reset_index(drop=True)

    raw.to_csv(OUT, index=False)
    print(f"Wrote {len(raw):,} rows to {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
