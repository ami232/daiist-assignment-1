#!/usr/bin/env bash
# Fetches OMIE day-ahead files into data/omie/<variable>/<variable>_YYYY-MM-DD.parquet:
# curva_pbc (aggregate bid curves) and marginalpdbc
# (day-ahead marginal price). Re-running is adaptive: days already fetched are
# skipped, except from yesterday on (see src/data/omie.py). curva_pbc is ~20MB
# of download per day.
#
# Usage:
#   scripts/fetch_omie.sh                                   # both variables, last 5 years to tomorrow
#   scripts/fetch_omie.sh --variables marginalpdbc           # just prices
#   scripts/fetch_omie.sh --start 2024-09-01 --end 2026-09-30
#   scripts/fetch_omie.sh --force                            # refetch days even if their parquet exists
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run python -m src.data.omie "$@"
