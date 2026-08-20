#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"

ENV_NAME="${ENV_NAME:-Hopper}"
MODEL_PATH="${MODEL_PATH:-}"
ALGORITHMS_STR="${ALGORITHMS:-SafeMeta MAML_constraint CPOMeta CPO}"
SEEDS_STR="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a ALGORITHMS <<< "$ALGORITHMS_STR"
read -r -a SEEDS <<< "$SEEDS_STR"

COMMON_ARGS=(
  --env-name "$ENV_NAME"
  --is-meta-test False
  --time-horizon 200
  --max-constraint 5
)

if [[ -n "$MODEL_PATH" ]]; then
  COMMON_ARGS+=(--model-path "$MODEL_PATH")
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_ARGS_STR:-}" ]]; then
  # Provide extra args as a single string: EXTRA_ARGS_STR="--max-iter-num 300 --min-batch-size 8000"
  read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS_STR"
fi

for algo in "${ALGORITHMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running $algo seed=$seed"
    "$PYTHON_BIN" "$ROOT_DIR/main.py" \
      --algo-name "$algo" \
      --seed "$seed" \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"

    base_dir="$ROOT_DIR/assets/learned_models/$algo"
    latest_run=$(ls -td "$base_dir"/* 2>/dev/null | head -n 1 || true)
    if [[ -n "$latest_run" && "$latest_run" != *"-seed${seed}" ]]; then
      mv "$latest_run" "${latest_run}-seed${seed}"
      echo "Renamed run folder to ${latest_run}-seed${seed}"
    fi
  done
 done
