#!/usr/bin/env bash
# Fetches REE ESIOS indicators into data/esios/<variable>/<variable>_YYYY-MM-DD.parquet.
# Re-running is adaptive: days already fetched are skipped, except from yesterday on (see src/data/esios.py).
# Needs ESIOS_TOKEN in the environment or in .env (see .env-template).
#
# "core" = demand/wind/solar forecasts issued the day before (hourly) plus
# imbalance prices and volume. "benchmark" = the revised 15-min forecasts:
# they leak post-auction information, use them only as an upper bound.
#
# Usage:
#   scripts/fetch_esios.sh                                  # core variables, last 5 years to tomorrow
#   scripts/fetch_esios.sh --variables benchmark
#   scripts/fetch_esios.sh --start 2024-09-01 --end 2026-09-30
#   scripts/fetch_esios.sh --force                           # refetch days even if their parquet exists
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run python -m src.data.esios "$@"
