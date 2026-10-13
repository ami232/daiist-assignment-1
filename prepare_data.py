"""
One-off script that produces data/online_retail_ii.csv.gz from the original
UCI file. It is NOT part of `main.py train`: the training notebook starts from
the committed .csv.gz, so it needs no download.

Run it only if you want to rebuild the committed file from the source:

    uv run --with openpyxl python prepare_data.py

Source: UCI Machine Learning Repository, "Online Retail II" (id 502),
https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0).

What it does, and nothing more:
- reads both yearly sheets of the Excel file and stacks them,
- drops rows without a Customer ID (22.8% of lines): the target is defined per
  customer, so a line we can't attach to a customer can't be used,
- writes the result as gzip CSV so it fits in the repo (~11.5 MB).

All other cleaning (cancellations, non-product codes, duplicates) happens in
train.ipynb, where it can be seen and justified.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
OUT = Path(__file__).resolve().parent / "data" / "online_retail_ii.csv.gz"


def main() -> None:
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    with archive.open("online_retail_II.xlsx") as f:
        sheets = pd.read_excel(f, sheet_name=None, dtype={"Invoice": str, "StockCode": str})

    df = pd.concat(sheets.values(), ignore_index=True)
    n_raw = len(df)
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)

    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, compression={"method": "gzip", "compresslevel": 9})
    print(f"{n_raw:,} raw lines -> {len(df):,} lines with a customer -> {OUT}")


if __name__ == "__main__":
    main()
