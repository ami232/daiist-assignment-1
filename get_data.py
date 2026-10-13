"""
One-off script that downloads the raw dataset from the World Bank API
(military expenditure series come from SIPRI) and saves it to data/raw.csv.

Run once:  uv run python get_data.py
Then COMMIT data/raw.csv — train.ipynb reads the committed file, so training
never depends on the internet.
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

INDICATORS = {
    # --- military (SIPRI via World Bank) ---
    "MS.MIL.XPND.CN": "milex_lcu",          # military spending, current local currency
    "MS.MIL.XPND.GD.ZS": "milex_gdp",       # military spending, % of GDP
    "MS.MIL.XPND.ZS": "milex_gov",          # military spending, % of government spending
    "MS.MIL.TOTL.P1": "armed_forces",       # armed forces personnel
    # --- economy / demographics ---
    "NY.GDP.DEFL.KD.ZG": "gdp_deflator",    # inflation, GDP deflator (annual %)
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",      # real GDP growth (annual %)
    "NY.GDP.PCAP.KD": "gdp_pc",             # GDP per capita, constant 2015 USD
    "NY.GDP.MKTP.CD": "gdp_usd",            # GDP, current USD
    "SP.POP.TOTL": "population",
    "NE.TRD.GNFS.ZS": "trade_gdp",          # trade, % of GDP
    "NE.IMP.GNFS.ZS": "imports_gdp",
    "NY.GDP.TOTL.RT.ZS": "resource_rents",  # natural resource rents, % of GDP
}
BASE = "https://api.worldbank.org/v2"


def fetch(url: str):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # retry flaky API
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def fetch_indicator(code: str) -> pd.DataFrame:
    url = f"{BASE}/country/all/indicator/{code}?format=json&per_page=20000&date=1960:2025"
    meta, rows = fetch(url)
    return pd.DataFrame(
        {"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
        for r in rows
        if r["countryiso3code"]
    )


def main():
    # Country metadata: lets us drop aggregates ("World", "Euro area"...) and get region/income
    _, countries = fetch(f"{BASE}/country?format=json&per_page=400")
    meta = pd.DataFrame(
        {
            "iso3": c["id"],
            "country": c["name"],
            "region": c["region"]["value"],
            "income": c["incomeLevel"]["value"],
        }
        for c in countries
        if c["region"]["value"] != "Aggregates"
    )

    df = None
    for code, name in INDICATORS.items():
        print("downloading", code)
        part = fetch_indicator(code).rename(columns={"value": name})
        df = part if df is None else df.merge(part, on=["iso3", "year"], how="outer")

    df = meta.merge(df, on="iso3", how="inner")  # keeps real countries only
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/raw.csv", index=False)
    print(df.shape, "-> data/raw.csv")


if __name__ == "__main__":
    main()
