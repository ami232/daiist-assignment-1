#!/usr/bin/env bash
# Launches the Gradio dashboard on http://127.0.0.1:7860 from the artifacts in models/.
# Run after `main.py train` has produced them; nothing is retrained here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run python main.py app
