# ENTSO-E Transparency data -> data/entsoe/<zone>/<variable>/<variable>_YYYY-MM-DD.parquet, needs ENTSOE_TOKEN
# series: one row per (psr_type, direction, datetime_utc), direction "in" = generation, "out" = load/pumping
# border: one row per (out_zone, in_zone, datetime_utc)
# outage: one row per (notice, availability segment); a day's file holds every segment overlapping it,
#   so dedupe on (doc_mrid, revision, start_utc) across days. created_utc is the latest revision only.

import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from src.constants import DATA_DIR
from src.data.partitions import (
    day_path,
    day_range,
    day_start_utc,
    days_to_fetch,
    fetch_parser,
    fetch_window,
    load_token,
    market_day,
    month_chunks,
    write_days,
)
from src.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://web-api.tp.entsoe.eu/api"
ZONES: dict[str, str] = {
    "ES": "10YES-REE------0",
    "FR": "10YFR-RTE------C",
    "PT": "10YPT-REN------W",
}
NEIGHBOURS: dict[str, list[str]] = {"ES": ["FR", "PT"]}
OUTAGE_PAGE_SIZE = 200  # API cap per request


@dataclass(frozen=True)
class EntsoeVariable:
    document_type: str
    process_type: str | None
    domain_param: str
    shape: str = "series"  # series, border or outage


VARIABLES: dict[str, EntsoeVariable] = {
    # load forecast is out before 12:00, wind/solar by 18:00 D-1
    "load_forecast_da": EntsoeVariable("A65", "A01", "outBiddingZone_Domain"),
    "wind_solar_forecast_da": EntsoeVariable("A69", "A01", "in_Domain"),
    "load_actual": EntsoeVariable("A65", "A16", "outBiddingZone_Domain"),
    "generation_actual": EntsoeVariable("A75", "A16", "in_Domain"),
    "transfer_capacity_da": EntsoeVariable(
        "A61",
        None,
        "in_Domain",
        shape="border",
    ),
    "outages": EntsoeVariable(
        "A80",
        None,
        "biddingZone_Domain",
        shape="outage",
    ),
    # benchmark only: later vintage, not known at 12:00
    "wind_solar_forecast_intraday": EntsoeVariable("A69", "A40", "in_Domain"),
}
BENCHMARK_VARIABLES = ["wind_solar_forecast_intraday"]
CORE_VARIABLES = [v for v in VARIABLES if v not in BENCHMARK_VARIABLES]


class EntsoeFetcher:
    def __init__(
        self,
        token: str,
        zone: str = "ES",
        out_dir: Path = DATA_DIR / "entsoe",
        session: requests.Session | None = None,
    ) -> None:
        self.token = token
        self.zone = zone
        self.root = out_dir / zone
        self.session = session or requests.Session()

    def fetch_range(
        self,
        variables: list[str],
        start: date,
        end: date,
        force: bool = False,
    ) -> None:
        days = day_range(start, end)
        tasks = [
            (v, chunk)
            for v in variables
            for chunk in month_chunks(
                days_to_fetch(
                    days,
                    self.root,
                    v,
                    force,
                )
            )
        ]
        with tqdm(tasks, desc=f"entsoe fetch {self.zone}") as bar:
            for variable, chunk in bar:
                bar.set_postfix_str(f"{variable} {chunk[0]:%Y-%m}")
                self._fetch_days(variable, chunk)

    def _fetch_days(self, variable: str, days: list[date]) -> None:
        spec = VARIABLES[variable]
        period = {
            "periodStart": f"{day_start_utc(days[0]):%Y%m%d%H%M}",
            "periodEnd": f"{day_start_utc(days[-1] + timedelta(days=1)):%Y%m%d%H%M}",
        }
        if spec.shape == "border":
            df = self._fetch_borders(spec, period)
        elif spec.shape == "outage":
            df = self._fetch_outages(spec, period)
        else:
            params = {"documentType": spec.document_type, spec.domain_param: ZONES[self.zone], **period}
            if spec.process_type:
                params["processType"] = spec.process_type
            raw = self._get(params)
            df = _parse_timeseries_xml(raw) if raw else None
        if df is None or df.is_empty():
            logger.warning(f"no data for {self.zone} {variable} {days[0]}..{days[-1]}")
            return
        df = df.select(pl.lit(self.zone).alias("zone"), *df.columns)
        if spec.shape == "outage":
            _write_outage_days(
                df,
                self.root,
                variable,
                days,
            )
        else:
            write_days(
                df,
                market_day("datetime_utc"),
                self.root,
                variable,
                days,
            )

    def _fetch_borders(self, spec: EntsoeVariable, period: dict[str, str]) -> pl.DataFrame | None:
        frames = []
        for neighbour in NEIGHBOURS.get(self.zone, []):
            for out_zone, in_zone in ((self.zone, neighbour), (neighbour, self.zone)):
                params = {
                    "documentType": spec.document_type,
                    "contract_MarketAgreement.Type": "A01",
                    "out_Domain": ZONES[out_zone],
                    "in_Domain": ZONES[in_zone],
                    **period,
                }
                raw = self._get(params)
                df = _parse_timeseries_xml(raw) if raw else None
                if df is not None:
                    frames.append(
                        df.select(
                            pl.lit(out_zone).alias("out_zone"),
                            pl.lit(in_zone).alias("in_zone"),
                            "datetime_utc",
                            "value",
                        ),
                    )
        return pl.concat(frames) if frames else None

    def _fetch_outages(self, spec: EntsoeVariable, period: dict[str, str]) -> pl.DataFrame | None:
        frames = []
        for offset in range(0, 5000, OUTAGE_PAGE_SIZE):
            raw = self._get(
                {
                    "documentType": spec.document_type,
                    spec.domain_param: ZONES[self.zone],
                    "offset": str(offset),
                    **period,
                },
            )
            if raw is None:
                break
            archive = zipfile.ZipFile(BytesIO(raw))
            frames += [_parse_outage_xml(archive.read(name)) for name in archive.namelist()]
            if len(archive.namelist()) < OUTAGE_PAGE_SIZE:
                break
        frames = [f for f in frames if f is not None]
        return pl.concat(frames) if frames else None

    def _get(self, params: dict[str, str]) -> bytes | None:
        for attempt in range(3):
            try:
                response = self.session.get(BASE_URL, params={"securityToken": self.token, **params}, timeout=120)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if response.status_code == 401:
                raise RuntimeError("ENTSO-E rejected the token - check ENTSOE_TOKEN")
            if b"Acknowledgement_MarketDocument" in response.content:
                reason = re.search(rb"<text>(.*?)</text>", response.content)
                if reason and not reason.group(1).startswith(b"No matching data"):
                    logger.warning(f"ENTSO-E refused {params.get('documentType')}: {reason.group(1).decode()}")
                return None
            if response.ok:
                return response.content
            time.sleep(2 * (attempt + 1))  # 5xx under load
        logger.warning(f"failed to fetch {params.get('documentType')} after 3 attempts")
        return None


def _write_outage_days(
    df: pl.DataFrame,
    root: Path,
    variable: str,
    days: list[date],
) -> None:
    for day in days:
        start, end = day_start_utc(day), day_start_utc(day + timedelta(days=1))
        part = df.filter(pl.col("start_utc") < end, pl.col("end_utc") > start)
        if part.is_empty():
            continue
        path = day_path(root, variable, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(path)


def _parse_timeseries_xml(raw: bytes) -> pl.DataFrame | None:
    rows = []
    for series in _children(ET.fromstring(raw), "TimeSeries"):
        psr_type = next((e.text for e in series.iter() if _local(e.tag) == "psrType"), "")
        direction = _direction(series)
        for period in _children(series, "Period"):
            start, end, step = _interval(period)
            points = _points(period)
            # A03 curves skip points equal to the previous one: forward fill
            value = None
            for position in range(1, int((end - start) / step) + 1):
                value = points.get(position, value)
                if value is not None:
                    rows.append((psr_type, direction, start + step * (position - 1), value))
    if not rows:
        return None
    schema = {
        "psr_type": pl.String,
        "direction": pl.String,
        "datetime_utc": pl.Datetime("us", "UTC"),
        "value": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def _parse_outage_xml(raw: bytes) -> pl.DataFrame | None:
    root = ET.fromstring(raw)
    fields = {_local(e.tag): e.text for e in root}  # document level only
    status = next((_child(e, "value").text for e in root if _local(e.tag) == "docStatus"), "")
    unit = "production_RegisteredResource"
    rows = []
    for series in _children(root, "TimeSeries"):
        tags = {_local(e.tag): e.text for e in series.iter()}
        nominal = tags.get(f"{unit}.pSRType.powerSystemResources.nominalP")
        for period in _children(series, "Available_Period"):
            start, end, step = _interval(period)
            points = sorted(_points(period).items())
            for i, (position, available) in enumerate(points):
                seg_end = start + step * (points[i + 1][0] - 1) if i + 1 < len(points) else end
                rows.append(
                    (
                        fields["mRID"],
                        int(fields["revisionNumber"]),
                        _parse_utc(fields["createdDateTime"]),
                        status,
                        tags.get("businessType", ""),
                        tags.get(f"{unit}.mRID", ""),
                        tags.get(f"{unit}.pSRType.powerSystemResources.mRID", ""),
                        tags.get(f"{unit}.name", ""),
                        tags.get(f"{unit}.pSRType.psrType", ""),
                        float(nominal) if nominal else None,
                        start + step * (position - 1),
                        seg_end,
                        available,
                    ),
                )
    if not rows:
        return None
    schema = {
        "doc_mrid": pl.String,
        "revision": pl.Int64,
        "created_utc": pl.Datetime("us", "UTC"),
        "status": pl.String,  # A09 = cancelled
        "business_type": pl.String,  # A53 planned, A54 forced
        "plant_eic": pl.String,
        "unit_eic": pl.String,
        "plant_name": pl.String,
        "psr_type": pl.String,
        "nominal_mw": pl.Float64,
        "start_utc": pl.Datetime("us", "UTC"),
        "end_utc": pl.Datetime("us", "UTC"),
        "available_mw": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def _direction(series: ET.Element) -> str:
    tags = {_local(e.tag) for e in series}
    has_in = bool(tags & {"inBiddingZone_Domain.mRID", "in_Domain.mRID"})
    has_out = bool(tags & {"outBiddingZone_Domain.mRID", "out_Domain.mRID"})
    if has_in == has_out:
        return ""
    return "in" if has_in else "out"


def _interval(period: ET.Element) -> tuple[datetime, datetime, timedelta]:
    interval = _child(period, "timeInterval")
    resolution = re.fullmatch(r"PT(\d+)M", _child(period, "resolution").text)
    if resolution is None:
        raise ValueError(f"unsupported ENTSO-E resolution {_child(period, 'resolution').text}")
    return (
        _parse_utc(_child(interval, "start").text),
        _parse_utc(_child(interval, "end").text),
        timedelta(minutes=int(resolution.group(1))),
    )


def _points(period: ET.Element) -> dict[int, float]:
    return {int(_child(p, "position").text): float(_child(p, "quantity").text) for p in _children(period, "Point")}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in element if _local(c.tag) == name]


def _child(element: ET.Element, name: str) -> ET.Element:
    return _children(element, name)[0]


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    parser = fetch_parser("Fetches ENTSO-E data ('core', 'benchmark' or a comma list).", DATA_DIR / "entsoe", "core")
    parser.add_argument(
        "--zone",
        default="ES",
        choices=sorted(ZONES),
        help="bidding zone",
    )
    args = parser.parse_args()
    groups = {"core": CORE_VARIABLES, "benchmark": BENCHMARK_VARIABLES}
    variables = groups.get(args.variables) or args.variables.split(",")
    start, end = fetch_window(args)
    fetcher = EntsoeFetcher(
        load_token("ENTSOE_TOKEN"),
        zone=args.zone,
        out_dir=Path(args.out),
    )
    fetcher.fetch_range(
        variables,
        start,
        end,
        force=args.force,
    )


if __name__ == "__main__":
    main()
