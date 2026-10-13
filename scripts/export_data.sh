#!/usr/bin/env bash
# Exports the model's raw inputs from the scraper cache into data/raw/
# (series.parquet + curves.parquet), the files train.py reads. Re-running only adds
# rows/months the export is missing, plus yesterday's month (see src/data/export.py).
#
# Usage:
#   scripts/export_data.sh                    # full cached history (from 2019-01-01)
#   scripts/export_data.sh --start 2020-01    # shorter history, smaller files
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run python -m src.data.export "$@"
