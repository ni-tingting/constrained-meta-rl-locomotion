import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Safe PCE reward/cost across seeds.")
    parser.add_argument("--input-dir", default="assets/plots")
    parser.add_argument("--pattern", default="safe_pce_eval_seed*.json")
    parser.add_argument("--reward-output", default="assets/plots/safe_pce_reward_cost_20_seeds.png")
    parser.add_argument("--reward-percentile-output", default="assets/plots/safe_pce_reward_cost_20_seeds_p10_p90.png")
    return parser.parse_args()


def load_seed_file(path: Path) -> Dict[float, Dict[str, float]]:
    data = json.loads(path.read_text())
    result: Dict[float, Dict[str, float]] = {}
    for row in data:
        k = float(row["k"])
        result[k] = {
            "reward": float(row["reward_mean"]),
            "cost": float(row["cost_mean"]),
        }
    return result


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern {args.pattern} in {input_dir}")

    per_seed = [load_seed_file(path) for path in files]
    common_ks = sorted(set.intersection(*(set(seed.keys()) for seed in per_seed)))
    if not common_ks:
        raise ValueError("No common evaluation steps across seed files.")

    rewards: List[List[float]] = []
    costs: List[List[float]] = []
    for k in common_ks:
        rewards.append([seed[k]["reward"] for seed in per_seed])
        costs.append([seed[k]["cost"] for seed in per_seed])


    reward_p20 = np.quantile(rewards, 0.2, axis=1)
    reward_p80 = np.quantile(rewards, 0.8, axis=1)
    cost_p20 = np.quantile(costs, 0.2, axis=1)
    cost_p80 = np.quantile(costs, 0.8, axis=1)
    reward_mean = np.array(
        [
            np.mean(np.asarray(row)[(np.asarray(row) >= low) & (np.asarray(row) <= high)])
            for row, low, high in zip(rewards, reward_p20, reward_p80)
        ],
        dtype=float,
    )
    cost_mean = np.array(
        [
            np.mean(np.asarray(row)[(np.asarray(row) >= low) & (np.asarray(row) <= high)])
            for row, low, high in zip(costs, cost_p20, cost_p80)
        ],
        dtype=float,
    )

    x_vals = np.array(common_ks, dtype=float) / 300.0
    if len(x_vals) > 0:
        x_vals = x_vals - x_vals[0]



    fig_p, axes_p = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    axes_p[0].plot(x_vals, reward_mean, label="Cumulative reward")
    axes_p[0].fill_between(x_vals, reward_p20, reward_p80, alpha=0.2)
    axes_p[0].set_ylabel("Reward")
    axes_p[0].set_title("Safe PCE Reward/Cost (P20-P80)")
    axes_p[0].legend()

    axes_p[1].plot(x_vals, cost_mean, label="Cost Mean", color="tab:orange")
    axes_p[1].fill_between(x_vals, cost_p20, cost_p80, alpha=0.2, color="tab:orange")
    axes_p[1].set_xlabel("Iteration number")
    axes_p[1].set_xticks(np.arange(0, 20.1, 2.5))
    axes_p[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 300:g}"))
    axes_p[1].set_ylabel("Cost")
    axes_p[1].legend()

    plt.tight_layout()
    reward_p_output = Path(args.reward_percentile_output)
    reward_p_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(reward_p_output)

    print(f"Saved reward/cost percentile plot to {reward_p_output}")


if __name__ == "__main__":
    main()
