"""
One-off helper that downloads the Hotel Reservations dataset from Kaggle and
copies it into data/hotel_reservations.csv.

The CSV is committed to the repo, so the training pipeline never needs
network access or Kaggle credentials. Only rerun this if you want to refresh
the raw file:

    uv run python scripts/fetch_data.py
"""

import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "hotel_reservations.csv"


def main() -> None:
    path = Path(kagglehub.dataset_download("ahsan81/hotel-reservations-classification-dataset"))
    print("Path to dataset files:", path)

    DEST.parent.mkdir(exist_ok=True)
    shutil.copyfile(path / "Hotel Reservations.csv", DEST)
    print("Copied to", DEST.relative_to(ROOT))


if __name__ == "__main__":
    main()
