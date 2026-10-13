#!/usr/bin/env bash
# Fetches ENTSO-E Transparency Platform data into
# data/entsoe/<zone>/<variable>/<variable>_YYYY-MM-DD.parquet.
# Re-running is adaptive: days already fetched are skipped, except from yesterday on (see src/data/entsoe.py).
# Needs ENTSOE_TOKEN in the environment or in .env (see .env-template).
#
# "core" = day-ahead load and wind/solar forecasts, actual load and generation
# by technology, day-ahead cross-border capacity, and generation-unit outages.
# "benchmark" = the intraday wind/solar forecast (a later vintage, upper bound only).
#
# Usage:
#   scripts/fetch_entsoe.sh                                 # Spain, core variables, last 5 years to tomorrow
#   scripts/fetch_entsoe.sh --zone ES --start 2024-09
#   scripts/fetch_entsoe.sh --variables benchmark
#   scripts/fetch_entsoe.sh --variables outages,transfer_capacity_da
#   scripts/fetch_entsoe.sh --force                          # refetch days even if their parquet exists
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run python -m src.data.entsoe "$@"
