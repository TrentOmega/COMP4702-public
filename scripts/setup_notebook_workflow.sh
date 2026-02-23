#!/usr/bin/env bash
set -euo pipefail

# Establish Jupytext triple pairing for a notebook workflow:
#   ipynb + md + py:percent
#
# Usage:
#   scripts/setup_notebook_workflow.sh path/to/notebook.ipynb
#   scripts/setup_notebook_workflow.sh path/to/notebook.md
#   scripts/setup_notebook_workflow.sh path/to/notebook.py
#
# Behavior:
# - Accepts .ipynb/.md/.py input and resolves the notebook stem.
# - Creates parent directories as needed.
# - Creates a minimal .ipynb scaffold if missing.
# - Sets jupytext formats and syncs all paired files.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_PATH="${1:-}"
if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: scripts/setup_notebook_workflow.sh path/to/notebook.{ipynb,md,py}" >&2
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
MD_PATH="${STEM}.md"
PY_PATH="${STEM}.py"

mkdir -p "$(dirname "$IPYNB_PATH")"

if [[ ! -f "$IPYNB_PATH" ]]; then
  cat > "$IPYNB_PATH" <<'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "title-cell",
   "metadata": {},
   "source": [
    "# Notebook\n",
    "\n",
    "Created by scripts/setup_notebook_workflow.sh"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "seed-cell",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "import numpy as np\n",
    "\n",
    "SEED = 4702\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "print(f\"Seed set to {SEED}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF
  echo "Created scaffold notebook: $IPYNB_PATH"
fi

FORMATS="ipynb,md,py:percent"
conda run -n comp4702 jupytext --set-formats "$FORMATS" "$IPYNB_PATH"
conda run -n comp4702 jupytext --sync "$IPYNB_PATH"

echo "Workflow ready:"
echo "  $IPYNB_PATH"
echo "  $MD_PATH"
echo "  $PY_PATH"
echo
echo "Daily sync from any file:"
echo "  scripts/sync_notebook.sh ${STEM}.md"
