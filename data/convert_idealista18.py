"""
One-off script: download the idealista18 Madrid sale listings (R package data)
and convert them to a CSV that train.py can load.

Source: Rey-Blanco, Arbues, Lopez & Paez (2024), idealista18 R package,
https://github.com/paezha/idealista18 — licensed ODbL v1.0 (attribution required).

Run once from the repo root (rdata is only needed for this conversion):
    uv run --with rdata python data/convert_idealista18.py
"""
import urllib.request
from pathlib import Path

import rdata

URL = "https://github.com/paezha/idealista18/raw/master/data/Madrid_Sale.rda"
HERE = Path(__file__).resolve().parent
RDA = HERE / "Madrid_Sale.rda"
OUT = HERE / "madrid_sale_2018.csv"

if not RDA.exists():
    urllib.request.urlretrieve(URL, RDA)

parsed = rdata.parser.parse_file(RDA)
df = rdata.conversion.convert(parsed, default_encoding="utf8")["Madrid_Sale"]

# 'geometry' is the R spatial point object; LATITUDE/LONGITUDE already hold the same info.
df = df.drop(columns=["geometry"])
df.columns = [str(c) for c in df.columns]

df.to_csv(OUT, index=False)
RDA.unlink()
print(f"Saved {len(df):,} rows x {df.shape[1]} columns to {OUT.name}")
