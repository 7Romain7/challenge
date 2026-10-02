#!/usr/bin/env bash
# One leave-one-family-out run: train, then score on the robustness suite.
#   bash challenge1/detection/lofo_job.sh <name> <train args...>
# Lines of `python -m detection.lofo jobs` are "name|args". Run from anywhere.
# Env: PY (default: .venv python), POOL (default: data/pool), ROBUST (default: data/robustness).
# A finished run (runs/<name>/DONE) is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
name=$1; shift
PY=${PY:-uv run python}
POOL=${POOL:-data/pool}
ROBUST=${ROBUST:-data/robustness}
mkdir -p "runs/$name"
[ -f "runs/$name/DONE" ] && exit 0
export PYTHONUNBUFFERED=1
{
  $PY -m detection.train --pool "$POOL" --out "runs/$name" "$@"
  $PY -m detection.evaluate --runs "runs/$name" --eval-dir "$ROBUST" --out "runs/$name/robustness"
  touch "runs/$name/DONE"
} > "runs/$name/stdout.log" 2>&1
