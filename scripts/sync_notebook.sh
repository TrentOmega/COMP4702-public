#!/usr/bin/env bash
set -euo pipefail

# Sync a Jupytext-paired notebook triple (.ipynb, .md, .py) from any one path.
# Usage:
#   scripts/sync_notebook.sh path/to/notebook.ipynb
#   scripts/sync_notebook.sh path/to/notebook.md
#   scripts/sync_notebook.sh path/to/notebook.py

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_PATH="${1:-}"
if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: scripts/sync_notebook.sh path/to/notebook.{ipynb,md,py}" >&2
  exit 1
fi

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "File not found: $INPUT_PATH" >&2
  exit 1
fi

case "$INPUT_PATH" in
  *.ipynb|*.md|*.py) ;;
  *)
    echo "Unsupported file type: $INPUT_PATH" >&2
    echo "Expected one of: .ipynb, .md, .py" >&2
    exit 1
    ;;
esac

STEM="${INPUT_PATH%.*}"
IPYNB_PATH="${STEM}.ipynb"

if [[ ! -f "$IPYNB_PATH" ]]; then
  echo "Missing paired notebook: $IPYNB_PATH" >&2
  echo "Create or pair the notebook first, then run sync again." >&2
  exit 1
fi

FORMATS="ipynb,md,py:percent"
conda run -n comp4702 jupytext --set-formats "$FORMATS" "$IPYNB_PATH"
conda run -n comp4702 jupytext --sync "$IPYNB_PATH"

echo "Synced pairs for: $STEM"
