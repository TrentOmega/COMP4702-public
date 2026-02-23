#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to sync requirements.txt" >&2
  exit 1
fi

TMP_FILE="$(mktemp)"

conda env export -n comp4702 --from-history > "$TMP_FILE"

{
  echo "# Auto-generated from conda env 'comp4702' (from-history)"
  echo "# Regenerate with: ./scripts/sync_requirements.sh"
  while IFS= read -r line; do
    dep="${line#  - }"
    if [[ "$dep" == "python="* ]]; then
      continue
    fi
    if [[ "$dep" == *"="* ]]; then
      name="${dep%%=*}"
      version="${dep#*=}"
      echo "${name}==${version}"
    else
      echo "$dep"
    fi
  done < <(awk '
    /^dependencies:/ { in_deps=1; next }
    /^prefix:/ { in_deps=0 }
    in_deps && /^  - / { print }
  ' "$TMP_FILE")
} > requirements.txt

rm -f "$TMP_FILE"
echo "Synced requirements.txt from conda env history: comp4702"
