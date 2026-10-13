"""One-off data preparation: extract the Spanish federation (AEP) subset
from the OpenPowerlifting bulk download.

Run once locally (NOT part of the graded pipeline):
    uv run python prepare_data.py path/to/openpowerlifting-latest.zip

The path can be the downloaded .zip, the folder it unzips to, or the .csv
inside it (Safari on macOS unzips downloads automatically).

Source: OpenPowerlifting (https://www.openpowerlifting.org), public domain.
Bulk file: https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip

Filters applied:
  - Federation == "AEP" (Asociacion Espanola de Powerlifting, IPF affiliate)
  - Event == "SBD" (full power: squat, bench, deadlift)
  - Equipment == "Raw"
Lifter names are replaced by anonymous IDs (data minimisation: the model
only needs to know which rows belong to the same lifter, not who they are).
No other cleaning is done here; all cleaning decisions live in train.ipynb.
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd

OUTPUT = Path("data/aep_raw_sbd.csv")

KEEP = [
    "Name", "Sex", "Event", "Equipment", "Age", "AgeClass", "Division",
    "BodyweightKg", "WeightClassKg",
    "Squat1Kg", "Squat2Kg", "Squat3Kg",
    "Bench1Kg", "Bench2Kg", "Bench3Kg",
    "Deadlift1Kg", "Deadlift2Kg", "Deadlift3Kg",
    "Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg", "TotalKg",
    "Place", "Dots", "Goodlift", "Tested",
    "Federation", "Date", "MeetCountry", "MeetName",
]


def filter_chunks(f) -> list:
    parts = []
    for chunk in pd.read_csv(f, usecols=KEEP, dtype=str, chunksize=300_000):
        mask = (
            (chunk["Federation"] == "AEP")
            & (chunk["Event"] == "SBD")
            & (chunk["Equipment"] == "Raw")
        )
        parts.append(chunk[mask])
    return parts


def main(source: str) -> None:
    path = Path(source).expanduser()
    if not path.exists():
        sys.exit(f"Not found: {path}")

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            print(f"Reading {csv_name} from the zip in chunks...")
            with zf.open(csv_name) as f:
                parts = filter_chunks(f)
    else:
        if path.is_dir():
            csvs = sorted(path.rglob("openpowerlifting-*.csv"))
            if not csvs:
                sys.exit(f"No openpowerlifting-*.csv found inside {path}")
            path = csvs[0]
        print(f"Reading {path.name} in chunks...")
        parts = filter_chunks(path)

    df = pd.concat(parts, ignore_index=True)

    # Anonymise: same name -> same ID, but the ID reveals nothing.
    codes, _ = pd.factorize(df["Name"].sort_values().unique())
    mapping = dict(zip(df["Name"].sort_values().unique(), codes))
    df.insert(0, "LifterID", df["Name"].map(mapping).map(lambda i: f"L{i:05d}"))
    df = df.drop(columns=["Name"])

    df = df.sort_values(["Date", "LifterID"]).reset_index(drop=True)
    OUTPUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df):,} rows, {df['LifterID'].nunique():,} lifters, "
          f"{df['Date'].min()} to {df['Date'].max()} -> {OUTPUT} "
          f"({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run python prepare_data.py path/to/openpowerlifting-latest.zip")
    main(sys.argv[1])
