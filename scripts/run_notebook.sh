#!/usr/bin/env bash
set -euo pipefail

# Run a notebook in the comp4702 environment with repo-local Jupyter/IPython dirs.
# This avoids permission issues with ~/.ipython in restricted environments.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NOTEBOOK_PATH="${1:-pracs/week-02/week-02-prac.ipynb}"

if [[ ! -f "$NOTEBOOK_PATH" ]]; then
  echo "Notebook not found: $NOTEBOOK_PATH" >&2
  echo "Usage: scripts/run_notebook.sh [path/to/notebook.ipynb]" >&2
  exit 1
fi

mkdir -p .jupyter_tmp/.ipython .jupyter_tmp/.jupyter .jupyter_tmp/.runtime

IPYTHONDIR="$REPO_ROOT/.jupyter_tmp/.ipython" \
JUPYTER_CONFIG_DIR="$REPO_ROOT/.jupyter_tmp/.jupyter" \
JUPYTER_RUNTIME_DIR="$REPO_ROOT/.jupyter_tmp/.runtime" \
conda run -n comp4702 jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  "$NOTEBOOK_PATH"

echo "Executed notebook: $NOTEBOOK_PATH"
