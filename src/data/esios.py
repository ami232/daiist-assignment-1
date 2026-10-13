# REE ESIOS indicators -> data/esios/<variable>/<variable>_YYYY-MM-DD.parquet, needs ESIOS_TOKEN

from datetime import date
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from src.constants import DATA_DIR
from src.data.partitions import (
    day_range,
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

BASE_URL = "https://api.esios.ree.es/indicators"
ACCEPT = "application/json; application/vnd.esios-api-v1+json"

# variable -> indicator id
VARIABLES: dict[str, int] = {
    # issued the day before for D, hourly, from ~2019
    "demand_forecast_d1": 1775,
    "wind_forecast_d1": 1777,
    "solar_pv_forecast_d1": 1779,
    # imbalance settlement, 15-min since 2024-12
    "imbalance_price_up": 686,
    "imbalance_price_down": 687,
    "imbalance_net_volume": 1338,
    # benchmark only: revised after gate closure, leaks the future
    "demand_forecast": 460,
    "wind_forecast": 541,
    "solar_pv_forecast": 542,
    "residual_demand_forecast": 10249,
}
BENCHMARK_VARIABLES = ["demand_forecast", "wind_forecast", "solar_pv_forecast", "residual_demand_forecast"]
CORE_VARIABLES = [v for v in VARIABLES if v not in BENCHMARK_VARIABLES]


class EsiosFetcher:
    def __init__(
        self,
        token: str,
        out_dir: Path = DATA_DIR / "esios",
        session: requests.Session | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": ACCEPT, "x-api-key": token})

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
                    self.out_dir,
                    v,
                    force,
                )
            )
        ]
        with tqdm(tasks, desc="esios fetch") as bar:
            for variable, chunk in bar:
                bar.set_postfix_str(f"{variable} {chunk[0]:%Y-%m}")
                self._fetch_days(variable, chunk)

    def _fetch_days(self, variable: str, days: list[date]) -> None:
        values = self._download(VARIABLES[variable], days[0], days[-1])
        if not values:
            logger.warning(f"no data for {variable} {days[0]}..{days[-1]}")
            return
        write_days(
            _values_to_frame(values),
            market_day("datetime_utc"),
            self.out_dir,
            variable,
            days,
        )

    def _download(
        self,
        indicator_id: int,
        first: date,
        last: date,
    ) -> list[dict] | None:
        # dates are read as Madrid local time
        params = {"start_date": f"{first}T00:00", "end_date": f"{last}T23:59"}
        for _ in range(3):
            try:
                response = self.session.get(f"{BASE_URL}/{indicator_id}", params=params, timeout=60)
            except requests.RequestException:
                continue
            if response.status_code in (401, 403):
                raise RuntimeError("ESIOS rejected the token - check ESIOS_TOKEN")
            if response.ok:
                return response.json()["indicator"]["values"]
        logger.warning(f"failed to fetch indicator {indicator_id} {first}..{last} after 3 attempts")
        return None


def _values_to_frame(values: list[dict]) -> pl.DataFrame:
    schema = {
        "datetime_utc": pl.String,
        "value": pl.Float64,
        "geo_id": pl.Int64,
        "geo_name": pl.String,
    }
    return pl.DataFrame(
        {c: [v[c] for v in values] for c in schema},
        schema=schema,
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))


def main() -> None:
    parser = fetch_parser("Fetches ESIOS indicators ('core', 'benchmark' or a comma list).", DATA_DIR / "esios", "core")
    args = parser.parse_args()
    groups = {"core": CORE_VARIABLES, "benchmark": BENCHMARK_VARIABLES}
    variables = groups.get(args.variables) or args.variables.split(",")
    start, end = fetch_window(args)
    EsiosFetcher(load_token("ESIOS_TOKEN"), out_dir=Path(args.out)).fetch_range(
        variables,
        start,
        end,
        force=args.force,
    )


if __name__ == "__main__":
    main()
