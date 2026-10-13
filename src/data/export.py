# Exports the model inputs to data/raw: series.parquet (long datetime_utc, series, value)
# and curves.parquet (hourly cumulative MW per GRID price). Re-runs only add what changed.

import argparse
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.constants import DATA_DIR
from src.data import loader
from src.data.features import curve_grid
from src.data.partitions import REFETCH_FROM_DAYS_AGO, market_day, parse_day
from src.logger import get_logger

log = get_logger(__name__)

EXPORT_DIR = DATA_DIR / "raw"
SERIES_PATH = EXPORT_DIR / "series.parquet"
CURVES_PATH = EXPORT_DIR / "curves.parquet"
KEYS = ["series", "datetime_utc"]

ESIOS_SERIES = [
    "demand_forecast_d1",
    "wind_forecast_d1",
    "solar_pv_forecast_d1",
    "imbalance_price_up",
    "imbalance_price_down",
]
GENERATION_SERIES = ["B04|in", "B05|in", "B10|out", "B11|in", "B12|in"]


def write(df: pl.DataFrame, path: Path) -> None:
    df.write_parquet(path, compression="zstd", compression_level=22)


def cache_series(start: date) -> pl.DataFrame:
    price = loader.load("omie", "marginalpdbc", lazy=True).select("datetime_utc", "price_es", "price_pt")
    generation = loader.load("entsoe", "generation_actual", lazy=True).with_columns(
        series=pl.concat_str(
            pl.lit("generation_actual"),
            "psr_type",
            "direction",
            separator="|",
        ),
    )
    transfer = loader.load("entsoe", "transfer_capacity_da", lazy=True).filter(
        pl.col("out_zone") == "FR",
        pl.col("in_zone") == "ES",
    )
    parts = [
        price.unpivot(index="datetime_utc", variable_name="series"),
        *[
            loader.load("esios", v, lazy=True).select("datetime_utc", series=pl.lit(v), value="value")
            for v in ESIOS_SERIES
        ],
        generation.filter(pl.col("series").is_in([f"generation_actual|{s}" for s in GENERATION_SERIES])).select(
            "datetime_utc",
            "series",
            "value",
        ),
        transfer.select("datetime_utc", series=pl.lit("transfer_capacity_da|FR|ES"), value="value"),
    ]
    return (
        pl.concat(parts)
        .filter(market_day("datetime_utc") >= start)
        .unique(KEYS, keep="last")
        .with_columns(pl.col("value").cast(pl.Float32))
        .sort(KEYS)
        .collect()
    )


def export_series(start: date) -> None:
    fresh = cache_series(start)
    if SERIES_PATH.exists():
        old = pl.read_parquet(SERIES_PATH)
        new = fresh.join(
            old,
            on=[*KEYS, "value"],
            how="anti",
            nulls_equal=True,
        )
        if new.is_empty():
            log.info("series: up to date")
            return
        log.info(f"series: {new.height} new or changed rows")
        fresh = pl.concat([old.join(fresh, on=KEYS, how="anti"), fresh]).sort(KEYS)
    write(fresh, SERIES_PATH)
    log.info(f"series: {fresh.height} rows, {SERIES_PATH.stat().st_size / 1e6:.1f} MB")


def months_to_build(start: date, old: pl.DataFrame | None) -> list[date]:
    # months with days missing from the export, plus yesterday's month
    root = loader.SOURCE_DIRS["omie"] / "curva_pbc"
    cached = {date.fromisoformat(p.stem.rsplit("_", 1)[1]) for p in root.glob("*.parquet")}
    exported = set() if old is None else set(old.select(market_day("datetime_utc").unique()).to_series())
    refetch_from = date.today() - timedelta(days=REFETCH_FROM_DAYS_AGO)
    days = {d for d in cached if d >= start and (d not in exported or d >= refetch_from)}
    return sorted({d.replace(day=1) for d in days})


def export_curves(start: date) -> None:
    old = pl.read_parquet(CURVES_PATH) if CURVES_PATH.exists() else None
    months = months_to_build(start, old)
    if not months:
        log.info("curves: up to date")
        return
    built = pl.concat(
        [
            curve_grid(m, (m + timedelta(days=32)).replace(day=1) - timedelta(days=1))
            for m in tqdm(months, desc="curves")
        ],
    ).with_columns(pl.exclude("datetime_utc").round(0).cast(pl.Float32))
    if old is not None:
        built = pl.concat([old.join(built, on="datetime_utc", how="anti"), built], how="diagonal_relaxed")
    built = built.filter(market_day("datetime_utc") >= start).sort("datetime_utc")
    write(built, CURVES_PATH)
    log.info(f"curves: {len(months)} months rebuilt, {CURVES_PATH.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exports the model inputs to data/raw.")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD or YYYY-MM, default: full cached history")
    args = parser.parse_args()
    start = parse_day(args.start) if args.start else date.min
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_series(start)
    export_curves(start)


if __name__ == "__main__":
    main()
