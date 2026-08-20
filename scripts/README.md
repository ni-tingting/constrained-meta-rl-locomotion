# Multi-seed launchers and aggregate plotting

Helpers for running the experiments over many seeds and collapsing the results into
figures. See the [top-level README](../README.md) for what the experiments are.

All launchers use `$REPO_ROOT/.venv/bin/python` by default; override with `PYTHON=...`.

## Meta-training over seeds

```bash
./scripts/run_10_seeds.sh              # main.py train, per algorithm per seed
./scripts/run_all_and_plot_20_seeds.sh # main.py compare, per seed
```

Configured with environment variables:

```bash
ENV_NAME=Hopper \
ALGORITHMS="SafeMeta MAML_constraint CPOMeta CPO" \
SEEDS="0 1 2 3 4 5 6 7 8 9" \
EXTRA_ARGS_STR="--max-iter-num 20 --min-batch-size 500 --max-batch-size 500" \
./scripts/run_10_seeds.sh
```

Each run folder is renamed to `...-seed<N>` after it finishes, so concurrent or
successive seeds never overwrite one another.

## Test-time adaptation over seeds

```bash
SEEDS="0 1 2 3" ./scripts/run_safe_pce_20_seeds.sh
```

Runs `main.py adapt` once per seed, writing one
`assets/plots/safe_pce_eval_seed<N>.json` each.

## Plotting

Run these from the repository root — they resolve `assets/` relative to the CWD.

| Script | Input | Output |
|---|---|---|
| `plot_seeded_rewards.py` | `training_log.csv` in each run folder | mean ± std meta-training curves |
| `plot_safe_pce_seeds.py` | `safe_pce_eval_seed*.json` | reward/cost with mean and p10–p90 bands |
| `plot_safe_pce_with_baselines.py` | the above + `test_log2.csv` + `shared_baseline.json` | adaptation curves against the baselines |
| `plot_test_metrics.py` | `test_log2.csv` + `shared_baseline.json` | per-algorithm test reward/cost curves (used by `main.py compare`) |

```bash
python scripts/plot_seeded_rewards.py \
  --base-dir assets/learned_models \
  --algorithms SafeMeta MAML_constraint CPOMeta CPO \
  --output assets/plots/hopper_meta_training.png

python scripts/plot_safe_pce_seeds.py --input-dir assets/plots
```
