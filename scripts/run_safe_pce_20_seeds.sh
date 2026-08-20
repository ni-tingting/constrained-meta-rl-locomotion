#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19})
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/assets/plots}"

for seed in "${SEEDS[@]}"; do
  echo "Running test-time adaptation for seed ${seed}"
  "${PYTHON_BIN}" "${REPO_ROOT}/main.py" adapt \
    --seed "${seed}" \
    --save-path "${OUTPUT_DIR}/safe_pce_eval_seed${seed}.json"
done

echo "Saved outputs to ${OUTPUT_DIR}/safe_pce_eval_seed*.json"