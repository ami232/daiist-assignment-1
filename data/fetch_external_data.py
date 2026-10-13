"""
One-off download of the external data used as features. The resulting CSVs are
committed to the repo, so `main.py train` never depends on network access.

    uv run python data/fetch_external_data.py

Assumed location: Guérande area (Loire-Atlantique). The dataset does not state
where the bakery is, but its products are regional specialities of southern
Brittany / Nantes (Kouign Amann, Far Breton, Gâche, Nantais, Bottereau,
Guérandais), and the strong July-August peak fits a coastal tourist area.
That region is in French school-holiday Zone B.
"""

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

DATA = Path(__file__).resolve().parent
START, END = "2021-01-01", "2022-09-30"
LAT, LON = 47.33, -2.43  # Guérande


def get_json(url, params=None):
    if params:
        url += "?" + urlencode(params)
    with urlopen(url, timeout=60) as r:
        return json.load(r)


def fetch_weather():
    data = get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": LAT,
            "longitude": LON,
            "start_date": START,
            "end_date": END,
            "daily": "temperature_2m_max,precipitation_sum,sunshine_duration",
            "timezone": "Europe/Paris",
        },
    )
    w = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
    w["sunshine_duration"] = w["sunshine_duration"] / 3600  # seconds -> hours
    w = w.rename(columns={"temperature_2m_max": "temp_max_c", "precipitation_sum": "precip_mm", "sunshine_duration": "sunshine_h"})
    w.to_csv(DATA / "weather_guerande.csv", index=False)
    print("weather:", w.shape)


def fetch_public_holidays():
    rows = []
    for year in (2021, 2022):
        days = get_json(f"https://calendrier.api.gouv.fr/jours-feries/metropole/{year}.json")
        rows += [{"date": d, "name": n} for d, n in days.items()]
    h = pd.DataFrame(rows)
    h.to_csv(DATA / "public_holidays_fr.csv", index=False)
    print("public holidays:", h.shape)


def fetch_school_holidays():
    data = get_json(
        "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records",
        {
            "select": "description,start_date,end_date",
            "where": 'zones="Zone B" and (annee_scolaire="2020-2021" or annee_scolaire="2021-2022" or annee_scolaire="2022-2023")',
            "group_by": "description,start_date,end_date",
            "limit": 100,
        },
    )
    s = pd.DataFrame(data["results"])
    # Timestamps are local midnight expressed in UTC; convert back to Paris dates.
    # A period covers [start_date, end_date): end_date is the day classes resume.
    for col in ("start_date", "end_date"):
        s[col] = pd.to_datetime(s[col]).dt.tz_convert("Europe/Paris").dt.date
    s = s.sort_values("start_date")
    s.to_csv(DATA / "school_holidays_zone_b.csv", index=False)
    print("school holiday periods:", s.shape)


if __name__ == "__main__":
    fetch_weather()
    fetch_public_holidays()
    fetch_school_holidays()
