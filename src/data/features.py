from datetime import date

import holidays
import numpy as np
import polars as pl
import polars.selectors as cs

from src.constants import MARKET_TZ, PRICE_ES
from src.data import loader

GRID = [-10, *range(0, 301, 10), 500, 1000]
BATTERY_HOURS = 4
EFFICIENCY = 0.85  # round trip, NREL ATB

# days each series is shifted so it is only used once known at gate closure
AVAILABLE_LAG_DAYS = {
    "price_es": 1,
    "price_pt": 1,
    "curve": 1,
    "demand_forecast_d1": 0,
    "wind_forecast_d1": 0,
    "solar_pv_forecast_d1": 0,
    "load_forecast_da": 0,
    "transfer_capacity_da": 0,
    "wind_solar_forecast_da": 1,
    "load_actual": 2,
    "generation_actual": 2,
    "imbalance_price_up": 2,
    "imbalance_price_down": 2,
    "imbalance_net_volume": 2,
    "demand_forecast": 2,
    "wind_forecast": 2,
    "solar_pv_forecast": 2,
    "residual_demand_forecast": 2,
    "wind_solar_forecast_intraday": 2,
}

LOCAL = pl.col("datetime_utc").dt.convert_time_zone(MARKET_TZ)
HOUR = LOCAL.dt.hour()
WEEK_HOUR = LOCAL.dt.weekday() * 24 + HOUR

price = pl.col("price_es")
demand = pl.col("demand_forecast_d1")
wind = pl.col("wind_forecast_d1")
solar = pl.col("solar_pv_forecast_d1")
gas = pl.col("generation_actual|B04|in")
hydro = pl.col("generation_actual|B12|in")
residual = demand - wind - solar


def curve_grid(start: date, end: date) -> pl.DataFrame:
    # cumulative MW at each grid price, one row per hour
    p, mw = pl.col("price_eur_mwh"), pl.col("quantity_mw")
    sell = (pl.col("status") == "O") & (pl.col("offer_type") == "V")
    buy = (pl.col("status") == "O") & (pl.col("offer_type") == "C")
    cleared = (pl.col("status") == "C") & (pl.col("offer_type") == "V")
    midnight = pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone(MARKET_TZ).dt.convert_time_zone("UTC")
    period_start = midnight + pl.duration(hours=pl.col("period").str.extract(r"(\d+)").cast(pl.Int64) - 1)
    return (
        loader.load(
            "omie",
            "curva_pbc",
            start,
            end,
            lazy=True,
        )
        .filter(pl.col("country").is_in(["MI", "ES"]))
        .group_by(datetime_utc=period_start)
        .agg(
            **{f"curve|supply|{g}": mw.filter(sell & (p <= g)).sum() for g in GRID},
            **{f"curve|demand|{g}": mw.filter(buy & (p >= g)).sum() for g in GRID},
            **{"curve|cleared": mw.filter(cleared).sum()},
            periods=pl.col("period").n_unique(),
        )
        .with_columns(cs.starts_with("curve|") / pl.col("periods"))
        .drop("periods")
        .collect()
    )


def best_spread(prices: pl.Series) -> float | None:
    p = prices.drop_nulls().to_numpy()
    if len(p) < 2 * BATTERY_HOURS:
        return None
    # charge in the cheapest hours before k, discharge in the dearest from k on
    return float(
        max(
            EFFICIENCY * np.sort(p[k:])[-BATTERY_HOURS:].mean() - np.sort(p[:k])[:BATTERY_HOURS].mean()
            for k in range(BATTERY_HOURS, len(p) - BATTERY_HOURS + 1)
        ),
    )


def spread(p: pl.Expr) -> pl.Expr:
    return p.map_batches(best_spread, return_dtype=pl.Float64, returns_scalar=True).over("day")


def z(
    x: pl.Expr,
    mean: pl.Expr,
    std: pl.Expr,
) -> pl.Expr:
    return (x - mean) / pl.when(std > 0).then(std)


def z_recent(x: pl.Expr, half_life_h: int) -> pl.Expr:
    past = x.shift(1)
    return z(x, past.ewm_mean(half_life=half_life_h), past.ewm_std(half_life=half_life_h))


def z_season(
    x: pl.Expr,
    half_life: int,
    season: pl.Expr,
) -> pl.Expr:
    past = x.shift(1)
    return z(x, past.ewm_mean(half_life=half_life).over(season), past.ewm_std(half_life=half_life).over(season))


def trend(
    x: pl.Expr,
    fast_h: int,
    slow_h: int,
) -> pl.Expr:
    return z(x.ewm_mean(half_life=fast_h), x.ewm_mean(half_life=slow_h), x.ewm_std(half_life=slow_h))


def price_at(q: pl.Expr) -> pl.Expr:
    # lowest grid price where yesterday's supply reaches q MW
    return pl.min_horizontal(pl.when(pl.col(f"curve|supply|{g}") >= q).then(pl.lit(g)) for g in GRID)


def feature_blocks() -> dict[str, pl.Expr]:
    renewables = wind + solar
    volume = pl.col("curve|cleared") + (demand - demand.shift(24)) - (renewables - renewables.shift(24))
    price_change = price - price.shift(24)
    residual_change = residual.shift(24) - residual.shift(48)
    beta = pl.rolling_cov(price_change, residual_change, window_size=24 * 30) / residual_change.rolling_var(24 * 30)
    # calendar
    calendar = {
        "weekday": LOCAL.dt.weekday(),
        "month": LOCAL.dt.month(),
        "holiday": pl.col("day")
        .is_in(list(holidays.country_holidays("ES", years=range(2018, 2028))))
        .cast(pl.Int8),  # national holiday flag
    }
    # input series as is, already lagged to when they are known
    raw = {
        c: pl.col(c)
        for c in [
            "price_es",  # yesterday's price
            "demand_forecast_d1",  # day-ahead forecasts for D
            "wind_forecast_d1",
            "solar_pv_forecast_d1",
            "imbalance_price_up",  # balancing prices, D-2
            "imbalance_price_down",
            "generation_actual|B04|in",  # gas, D-2
            "generation_actual|B05|in",  # coal
            "generation_actual|B10|out",  # pumped storage pumping
            "generation_actual|B11|in",  # run of river hydro
            "generation_actual|B12|in",  # reservoir hydro
            "transfer_capacity_da|FR|ES",  # import capacity from France
        ]
    }
    # one series vs its own recent past or seasonal profile
    single = {
        "price_z_recent_7d": z_recent(price, 24 * 7),  # price vs last week
        "price_z_hour_7d": z_season(price, 7, HOUR),  # vs same hour, recent days
        "price_z_weekhour_4w": z_season(price, 4, WEEK_HOUR),  # vs same weekday and hour
        "price_trend_1d_7d": trend(price, 24, 24 * 7),  # fast vs slow moving average
        "price_range_1d": price.max().over("day") - price.min().over("day"),  # yesterday's max - min
        "spread_1d": spread(price),  # yesterday's battery spread, the naive baseline
        "wind_z_hour_7d": z_season(wind, 7, HOUR),  # wind forecast vs normal for the hour
        "wind_z_recent_2d": z_recent(wind, 48),  # wind vs last 2 days
        "solar_z_hour_7d": z_season(solar, 7, HOUR),  # solar vs normal for the hour
        "demand_z_weekhour_4w": z_season(demand, 4, WEEK_HOUR),  # demand vs same weekday and hour
        "gas_z_hour_14d": z_season(gas, 14, HOUR),  # gas generation vs normal
        "hydro_z_recent_30d": z_recent(hydro, 24 * 30),  # reservoir hydro vs last month
    }
    # combinations of series
    cross = {
        "residual_load": residual,  # demand - wind - solar, what thermal and hydro must cover
        "residual_z_hour_7d": z_season(residual, 7, HOUR),
        "residual_change_1d": residual - residual.shift(24),
        "renewable_share": (wind + solar) / demand,
        "beta": beta,  # price move per MW of residual move, last 30 days
        "beta_adjusted_price": price
        + beta * (residual - residual.shift(24)),  # yesterday's price moved by today's residual change
        "es_pt_spread": price - pl.col("price_pt"),  # market splitting with Portugal
        "imbalance_premium": pl.col("imbalance_price_up") - price.shift(24),  # balancing vs day-ahead price
        "import_cover": pl.col("transfer_capacity_da|FR|ES") / residual,  # share of residual France could cover
        "gas_share": gas / residual.shift(48),  # gas as share of residual load
    }
    # yesterday's bid curves
    curve = {
        "curve_supply_le0": pl.col("curve|supply|0"),  # MW offered at 0 or below
        "curve_headroom_100": pl.col("curve|supply|100") - pl.col("curve|cleared"),  # spare MW up to 100 EUR
        "curve_demand_inelastic": pl.col("curve|demand|1000"),  # MW bought at any price
        "curve_implied_price": price_at(volume),  # price on yesterday's supply curve at today's expected volume
        "curve_implied_change": price_at(volume) - price,
        "curve_slope_1gw": price_at(volume + 1000) - price_at(volume),  # price impact of 1 GW more
    }
    return calendar | raw | single | cross | curve


def lag_to_available(hourly: pl.DataFrame) -> pl.DataFrame:
    # same local hour, lag calendar days earlier: shifting by 24 rows breaks on 23h and 25h DST days
    keyed = hourly.with_columns(day=LOCAL.dt.date(), hour=HOUR)
    lags = {c: AVAILABLE_LAG_DAYS[c.split("|")[0]] for c in hourly.columns if c != "datetime_utc"}
    out = keyed.select("datetime_utc", "day", "hour", *[c for c, lag in lags.items() if lag == 0])
    for lag in sorted(set(lags.values()) - {0}):
        cols = [c for c, n in lags.items() if n == lag]
        known = (
            keyed.group_by("day", "hour")
            .agg(pl.col(cols).mean())  # the repeated hour of a 25h day is averaged
            .with_columns(pl.col("day") + pl.duration(days=lag))
        )
        out = out.join(known, on=["day", "hour"], how="left", maintain_order="left")
    return out.select("datetime_utc", "day", *lags)


def build_daily(
    hourly: pl.DataFrame,
    start: date,
) -> pl.DataFrame:
    # one row per delivery day: daily mean features and y_spread
    features = feature_blocks()
    target = hourly.with_columns(day=LOCAL.dt.date()).select(y_spread=spread(pl.col(PRICE_ES)))
    data = pl.concat([lag_to_available(hourly), target], how="horizontal").with_columns(**features)
    return (
        data.group_by("day")
        .agg(pl.col(*features).mean(), pl.col("y_spread").first())
        .filter(pl.col("day") >= start)
        .drop_nulls(["y_spread", "spread_1d"])
        .sort("day")
    )
