# One parquet per variable and market day: <root>/<variable>/<variable>_YYYY-MM-DD.parquet

import argparse
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.constants import MARKET_TZ, PROJECT_DIR

# days from yesterday on may still be filling in, always refetched
REFETCH_FROM_DAYS_AGO = 1


def day_path(
    root: Path,
    variable: str,
    day: date,
) -> Path:
    return root / variable / f"{variable}_{day:%Y-%m-%d}.parquet"


def day_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def month_chunks(days: list[date]) -> list[list[date]]:
    chunks: dict[tuple[int, int], list[date]] = {}
    for day in days:
        chunks.setdefault((day.year, day.month), []).append(day)
    return list(chunks.values())


def days_to_fetch(
    days: list[date],
    root: Path,
    variable: str,
    force: bool,
) -> list[date]:
    refetch_from = date.today() - timedelta(days=REFETCH_FROM_DAYS_AGO)
    return [d for d in days if force or d >= refetch_from or not day_path(root, variable, d).exists()]


def market_day(column: str) -> pl.Expr:
    return pl.col(column).dt.convert_time_zone(MARKET_TZ).dt.date()


def day_start_utc(day: date) -> datetime:
    return datetime.combine(day, time(), ZoneInfo(MARKET_TZ)).astimezone(ZoneInfo("UTC"))


def write_days(
    df: pl.DataFrame,
    day: pl.Expr,
    root: Path,
    variable: str,
    days: list[date],
) -> list[date]:
    written = []
    for (key,), part in df.with_columns(day.alias("_day")).group_by("_day"):
        if key in days:
            path = day_path(root, variable, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            part.drop("_day").write_parquet(path)
            written.append(key)
    return written


def read_days(
    root: Path,
    variable: str,
    start: date,
    end: date,
    *,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    paths = [p for p in (day_path(root, variable, d) for d in day_range(start, end)) if p.exists()]
    if not paths:
        raise FileNotFoundError(f"no fetched parquet for {variable} in {start}..{end}")
    lf = pl.concat([pl.scan_parquet(p) for p in paths], how="diagonal_relaxed")
    return lf if lazy else lf.collect()


def available_days(root: Path, variable: str) -> tuple[date, date] | None:
    days = [date.fromisoformat(p.stem.rsplit("_", 1)[1]) for p in (root / variable).glob("*.parquet")]
    return (min(days), max(days)) if days else None


def parse_day(value: str, *, end: bool = False) -> date:
    # YYYY-MM means the month's first day, or its last with end=True
    if len(value) == 7:
        first = date.fromisoformat(f"{value}-01")
        if not end:
            return first
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        return next_month - timedelta(days=1)
    return date.fromisoformat(value)


def load_token(name: str) -> str:
    if token := os.environ.get(name):
        return token
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip().strip("\"'")
    raise RuntimeError(f"{name} not set (environment or .env)")


def fetch_parser(
    description: str,
    out: Path,
    variables: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--variables", default=variables, help=f"comma list, default: {variables}")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD or YYYY-MM, default: 5 years ago")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD or YYYY-MM, default: tomorrow")
    parser.add_argument("--force", action="store_true", help="refetch days that already exist")
    parser.add_argument("--out", default=str(out), help="output directory")
    return parser


def fetch_window(args: argparse.Namespace) -> tuple[date, date]:
    # default: 5 years back to tomorrow, D+1 day-ahead data exists today
    today = date.today()
    start = parse_day(args.start) if args.start else today - timedelta(days=5 * 365)
    end = parse_day(args.end, end=True) if args.end else today + timedelta(days=1)
    return start, end
