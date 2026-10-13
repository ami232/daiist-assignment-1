# Reads the daily parquet back, one frame per variable. OMIE gets renamed and typed here.

from datetime import date
from pathlib import Path

import polars as pl

from src.constants import DATA_DIR, MARKET_TZ
from src.data.partitions import available_days, read_days

# marginalpdbc has no header, order from OMIE's spec
MARGINALPDBC_COLUMNS = ["year", "month", "day", "period", "price_pt", "price_es"]

# old curva_pbc header -> current one
COLUMN_ALIASES: dict[str, str] = {
    "Hora": "Periodo",
    "Energía Compra/Venta": "Potencia Compra/Venta",
}

CURVA_PBC_RENAMES: dict[str, str] = {
    "Periodo": "period",
    "Fecha": "date",
    "Pais": "country",
    "Unidad": "unit",
    "Tipo Oferta": "offer_type",
    "Potencia Compra/Venta": "quantity_mw",
    "Precio Compra/Venta": "price_eur_mwh",
    "Ofertada (O)/Casada (C)": "status",
    "Tipología de Oferta": "offer_typology",
}

SOURCE_DIRS: dict[str, Path] = {
    "omie": DATA_DIR / "omie",
    "esios": DATA_DIR / "esios",
    "entsoe": DATA_DIR / "entsoe" / "ES",
}


def variables(source: str) -> list[str]:
    return sorted(p.name for p in SOURCE_DIRS[source].iterdir() if p.is_dir())


def load(
    source: str,
    variable: str,
    start: date | None = None,
    end: date | None = None,
    *,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    # defaults to the full fetched range
    root = SOURCE_DIRS[source]
    first, last = available_days(root, variable) or (None, None)
    if first is None:
        raise FileNotFoundError(f"nothing fetched for {source}/{variable}")
    lf = read_days(
        root,
        variable,
        start or first,
        end or last,
        lazy=True,
    )
    if source == "omie":
        lf = _tidy(variable, lf)
    return lf if lazy else lf.collect()


def parse_euro_number(column: str) -> pl.Expr:
    # "1.234,5" -> 1234.5
    return (
        pl.col(column)
        .str.replace_all(".", "", literal=True)
        .str.replace(",", ".", literal=True)
        .cast(pl.Float64, strict=False)
        .alias(column)
    )


def _tidy(variable: str, df: pl.LazyFrame) -> pl.LazyFrame:
    if variable == "marginalpdbc":
        return _tidy_marginalpdbc(df)
    if variable == "curva_pbc":
        return _tidy_curva_pbc(df)
    raise ValueError(f"unknown OMIE variable {variable}")


def _tidy_marginalpdbc(df: pl.LazyFrame) -> pl.LazyFrame:
    df = df.rename({f"col_{i}": name for i, name in enumerate(MARGINALPDBC_COLUMNS)})
    df = df.with_columns(
        pl.col(["year", "month", "day", "period"]).cast(pl.Int32),
        pl.col("price_pt", "price_es").cast(pl.Float64),
        pl.date("year", "month", "day").alias("date"),
    )
    # hourly until 2025-09-30, 15-min after; periods count from local midnight so DST works in UTC
    step = pl.when(pl.len().over("date") > 25).then(15).otherwise(60)
    local_midnight = pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone(MARKET_TZ)
    return df.with_columns(
        (local_midnight.dt.convert_time_zone("UTC") + pl.duration(minutes=(pl.col("period") - 1) * step)).alias(
            "datetime_utc"
        ),
    )


def _tidy_curva_pbc(df: pl.LazyFrame) -> pl.LazyFrame:
    df = _coalesce_aliases(df)
    columns = df.collect_schema().names()
    df = df.rename({k: v for k, v in CURVA_PBC_RENAMES.items() if k in columns})
    return df.with_columns(
        parse_euro_number("quantity_mw"),
        parse_euro_number("price_eur_mwh"),
        pl.col("date").str.to_date("%d/%m/%Y", strict=False),
    )


def _coalesce_aliases(df: pl.LazyFrame) -> pl.LazyFrame:
    for old, canonical in COLUMN_ALIASES.items():
        columns = df.collect_schema().names()
        if old in columns and canonical in columns:
            df = df.with_columns(pl.coalesce(canonical, old).alias(canonical)).drop(old)
        elif old in columns:
            df = df.rename({old: canonical})
    return df
