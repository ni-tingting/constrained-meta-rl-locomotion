#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"

ENV_NAME="${ENV_NAME:-Hopper}"
ALGORITHMS_STR="${ALGORITHMS:-CPOMeta SafeMeta MAML_constraint CPO}"
SEEDS_STR="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
MODEL_PATH="${MODEL_PATH:-}"

read -r -a ALGORITHMS <<< "$ALGORITHMS_STR"
read -r -a SEEDS <<< "$SEEDS_STR"

EXTRA_ARGS=()
if [[ -n "${EXTRA_ARGS_STR:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS_STR"
fi

for seed in "${SEEDS[@]}"; do
  echo "Running comparison for seed=$seed"
  cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/main.py" compare
    --env-name "$ENV_NAME"
    --seed "$seed"
    --skip-baseline
    --skip-plot
    --algos "${ALGORITHMS[@]}"
  )
  if [[ -n "$MODEL_PATH" ]]; then
    cmd+=(--model-path "$MODEL_PATH")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  "${cmd[@]}"

  for algo in "${ALGORITHMS[@]}"; do
    base_dir="$ROOT_DIR/assets/learned_models/$algo"
    latest_run=$(ls -td "$base_dir"/* 2>/dev/null | head -n 1 || true)
    if [[ -n "$latest_run" && "$latest_run" != *"-seed${seed}" ]]; then
      mv "$latest_run" "${latest_run}-seed${seed}"
      echo "Renamed run folder to ${latest_run}-seed${seed}"
    fi
  done
 done

echo "All seeds complete. Use plot_seeded_rewards.py with --log-type test for a single aggregate plot."