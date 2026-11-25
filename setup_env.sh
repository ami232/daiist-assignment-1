#!/usr/bin/env bash
# One-time setup: builds/refreshes .venv (Python 3.12 + scikit-learn 1.9.1, etc.)
# and registers it as a Jupyter kernel so app.ipynb loads the saved models correctly.
# Usage (from the project folder):  bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1. Make sure uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create/sync .venv exactly as pinned in uv.lock (Python 3.12)
uv sync

# 3. Register the venv as a Jupyter kernel
.venv/bin/python -m ipykernel install --user \
  --name daiist-a1 --display-name "Python 3.12 (daiist-a1)"

# 4. Sanity check
.venv/bin/python -c "import sys, sklearn, torch, gradio; print('Python', sys.version.split()[0], '| sklearn', sklearn.__version__, '| torch', torch.__version__, '| gradio', gradio.__version__)"

echo
echo "Done. Activate with:   source .venv/bin/activate"
echo "In Jupyter/VS Code, select kernel: Python 3.12 (daiist-a1)"
