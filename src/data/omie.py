# OMIE day-ahead files -> data/omie/<variable>/<variable>_YYYY-MM-DD.parquet

import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from src.constants import DATA_DIR
from src.data.partitions import day_path, day_range, days_to_fetch, fetch_parser, fetch_window
from src.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://www.omie.es/en/file-download"
USER_AGENT = "Mozilla/5.0 (compatible; omie-fetcher)"
# curva_pbc is ~20MB a day to download
VARIABLES = ["curva_pbc", "marginalpdbc"]


class OmieFetcher:
    def __init__(
        self,
        out_dir: Path = DATA_DIR / "omie",
        session: requests.Session | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._year_zips: dict[tuple[str, int], zipfile.ZipFile | None] = {}

    def fetch_range(
        self,
        variables: list[str],
        start: date,
        end: date,
        force: bool = False,
    ) -> None:
        days = day_range(start, end)
        tasks = [
            (v, d)
            for v in variables
            for d in days_to_fetch(
                days,
                self.out_dir,
                v,
                force,
            )
        ]
        with tqdm(tasks, desc="omie fetch") as bar:
            for variable, day in bar:
                bar.set_postfix_str(f"{variable} {day}")
                self._fetch_day(variable, day)

    def _fetch_day(self, variable: str, day: date) -> None:
        filename = f"{variable}_{day:%Y%m%d}.1"
        # older years only exist inside a yearly zip
        raw = self._download(variable, filename) or self._from_year_zip(variable, day)
        df = _parse_omie_text(raw, filename) if raw else None
        if df is None or df.is_empty():
            logger.warning(f"no data for {variable} {day}")
            return
        path = day_path(self.out_dir, variable, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    def _from_year_zip(self, variable: str, day: date) -> bytes | None:
        key = (variable, day.year)
        if key not in self._year_zips:
            raw = self._download(variable, f"{variable}_{day.year}.zip")
            self._year_zips[key] = zipfile.ZipFile(BytesIO(raw)) if raw else None
        year_zip = self._year_zips[key]
        member = f"{variable}_{day:%Y%m%d}.1"
        return year_zip.read(member) if year_zip and member in year_zip.namelist() else None

    def _download(self, variable: str, filename: str) -> bytes | None:
        url = f"{BASE_URL}?parents={variable}&filename={filename}"
        for _ in range(3):
            try:
                response = self.session.get(url, timeout=30)
            except requests.RequestException:
                continue
            if response.status_code == 404:
                return None
            if response.ok:
                return response.content
        logger.warning(f"failed to fetch {filename} after 3 attempts")
        return None


def _parse_omie_text(raw: bytes, source: str) -> pl.DataFrame:
    lines = raw.decode("latin-1").splitlines()
    while lines and lines[-1].strip() in ("", "*"):
        lines.pop()
    if len(lines) < 2:
        return pl.DataFrame()
    lines = lines[1:]  # title line
    if lines[0].strip() == "":
        # blank line, then a header row
        header = lines[1].split(";")[:-1]
        rows = lines[2:]
    else:
        # no header: width from the widest row, some files open with a short line
        header = [f"col_{i}" for i in range(max(len(line.split(";")) for line in lines) - 1)]
        rows = lines
    width = len(header)
    data = [(row.split(";")[:-1] + [None] * width)[:width] + [source] for row in rows]
    return pl.DataFrame(data, schema=[*_dedupe(header), "source_file"], orient="row")


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return result


def main() -> None:
    args = fetch_parser("Fetches OMIE files.", DATA_DIR / "omie", ",".join(VARIABLES)).parse_args()
    start, end = fetch_window(args)
    OmieFetcher(out_dir=Path(args.out)).fetch_range(
        args.variables.split(","),
        start,
        end,
        force=args.force,
    )


if __name__ == "__main__":
    main()
