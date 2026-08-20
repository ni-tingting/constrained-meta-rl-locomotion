#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

TEST_LOG_FILES = ("test_log2.csv", "test_log.csv")


def read_test_log(csv_path: Path, value_index: int) -> Tuple[np.ndarray, np.ndarray]:
    rows_parsed: List[Tuple[int, float]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                rows_parsed.append((int(float(row[0])), float(row[value_index])))
            except (ValueError, IndexError):
                continue

    if not rows_parsed:
        return np.array([]), np.array([])

    segments: List[List[Tuple[int, float]]] = []
    current_segment: List[Tuple[int, float]] = []
    previous_iter: int | None = None
    for iteration, value in rows_parsed:
        if previous_iter is not None and iteration < previous_iter:
            if current_segment:
                segments.append(current_segment)
            current_segment = []
        current_segment.append((iteration, value))
        previous_iter = iteration

    if current_segment:
        segments.append(current_segment)

    latest_segment = segments[-1]
    latest_by_iter: Dict[int, float] = {iteration: value for iteration, value in latest_segment}
    ordered_iters = np.array(sorted(latest_by_iter.keys()), dtype=int)
    ordered_values = np.array([latest_by_iter[i] for i in ordered_iters], dtype=float)
    return ordered_iters, ordered_values


def collect_runs(base_dir: Path, algo: str, require_seed_tag: bool) -> List[Path]:
    algo_dir = base_dir / algo
    if not algo_dir.exists():
        return []
    run_dirs = [p for p in algo_dir.iterdir() if p.is_dir()]
    if require_seed_tag:
        run_dirs = [p for p in run_dirs if "-seed" in p.name]
    return [p for p in run_dirs if any((p / name).exists() for name in TEST_LOG_FILES)]


def align_runs(run_logs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not run_logs:
        return np.array([]), np.array([[]])

    run_maps: List[Dict[int, float]] = []
    for iterations, rewards in run_logs:
        run_maps.append({int(i): float(r) for i, r in zip(iterations, rewards)})

    common_iters = set(run_maps[0].keys())
    for run_map in run_maps[1:]:
        common_iters &= set(run_map.keys())

    if not common_iters:
        return np.array([]), np.array([[]])

    common_iter_sorted = np.array(sorted(common_iters), dtype=int)
    reward_matrix = np.vstack(
        [np.array([run_map[i] for i in common_iter_sorted], dtype=float) for run_map in run_maps]
    )
    return common_iter_sorted, reward_matrix


def load_safe_pce_seeds(input_dir: Path, pattern: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Safe PCE files found for pattern {pattern} in {input_dir}")

    per_seed: List[Dict[float, Dict[str, float]]] = []
    for path in files:
        data = json.loads(path.read_text())
        result: Dict[float, Dict[str, float]] = {}
        for row in data:
            k = float(row["k"])
            result[k] = {
                "reward": float(row["reward_mean"]),
                "cost": float(row["cost_mean"]),
            }
        per_seed.append(result)

    common_ks = sorted(set.intersection(*(set(seed.keys()) for seed in per_seed)))
    if not common_ks:
        raise ValueError("No common evaluation steps across Safe PCE seed files.")

    rewards = []
    costs = []
    for k in common_ks:
        rewards.append([seed[k]["reward"] for seed in per_seed])
        costs.append([seed[k]["cost"] for seed in per_seed])

    rewards = np.asarray(rewards, dtype=float)
    costs = np.asarray(costs, dtype=float)

    x_vals = np.asarray(common_ks, dtype=float) / 300.0
    if len(x_vals) > 0:
        x_vals = x_vals - x_vals[0]
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
    return x_vals, np.vstack([reward_mean, reward_p20, reward_p80, cost_mean, cost_p20, cost_p80])


def plot_algo(ax, iterations, matrix, label, color=None, linestyle="-", scale: float = 1.0):
    if matrix.size == 0:
        return
    value_p20 = np.quantile(matrix, 0.2, axis=0)
    value_p80 = np.quantile(matrix, 0.8, axis=0)
    value_mean = np.array([
        np.mean(col[(col >= low) & (col <= high)])
        for col, low, high in zip(matrix.T, value_p20, value_p80)
    ], dtype=float)
    # mean_values = matrix.mean(axis=0) * scale
    # std_values = matrix.std(axis=0) * scale
    # Use the same color for line and fill
    ax.plot(iterations, scale * value_mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(iterations, scale * value_p20, scale * value_p80, alpha=0.3, color=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Safe PCE with other algorithms.")
    parser.add_argument("--base-dir", default="assets/learned_models", type=Path)
    parser.add_argument("--safe-pce-dir", default="assets/plots", type=Path)
    parser.add_argument("--safe-pce-pattern", default="safe_pce_eval_seed*.json")
    parser.add_argument("--algorithms", nargs="+", default=["MAML_constraint", "CPOMeta", "CPO"])
    parser.add_argument("--require-seed-tag", action="store_true", default=True)
    parser.add_argument("--test-reward-col", type=int, default=6)
    parser.add_argument("--test-cost-col", type=int, default=4)
    parser.add_argument(
        "--safe-meta-iter-scale",
        type=float,
        default=1.0,
        help="Divide SafeMeta iterations by this factor when plotting (legacy option)",
    )
    parser.add_argument(
        "--safe-meta-step-multiplier",
        type=float,
        default=4.0,
        help="Multiply SafeMeta adaptation step index by this value (default: 4, so 0,1,2,3,4,5 -> 0,4,8,12,16,20)",
    )
    parser.add_argument(
        "--safe-meta-iter-offset",
        type=float,
        default=0.0,
        help="Add this offset to SafeMeta iterations before scaling (e.g., 1 moves 0→1 before scaling)",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional max x-axis value to truncate/align all curves (e.g., 19)",
    )
    parser.add_argument("--output", default="assets/plots/hopper_meta_test_with_safe_pce.png", type=Path)
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

    style_map = {
        "MAML_constraint": {"color": "tab:green", "linestyle": "-"},
         "SafeMeta": {"color": "tab:blue", "linestyle": "-"},
        "CPOMeta": {"color": "tab:orange", "linestyle": "-"},
        "CPO": {"color": "tab:purple", "linestyle": "-"}, 
    }

    safe_meta_iter_scale = args.safe_meta_iter_scale if args.safe_meta_iter_scale > 0 else 1.0
    safe_meta_iter_offset = args.safe_meta_iter_offset

    for algo in args.algorithms:
        run_dirs = collect_runs(args.base_dir, algo, args.require_seed_tag)
        if not run_dirs:
            print(f"No runs found for {algo}")
            continue

        reward_logs = []
        cost_logs = []
        for run_dir in sorted(run_dirs):
            log_path = next((run_dir / name for name in TEST_LOG_FILES if (run_dir / name).exists()), None)
            if log_path is None:
                continue
            reward_iters, reward_values = read_test_log(log_path, args.test_reward_col)
            cost_iters, cost_values = read_test_log(log_path, args.test_cost_col)
            if len(reward_iters) == 0 or len(cost_iters) == 0:
                continue
            reward_logs.append((reward_iters, reward_values))
            cost_logs.append((cost_iters, cost_values))

        reward_iters, reward_matrix = align_runs(reward_logs)
        cost_iters, cost_matrix = align_runs(cost_logs)
        if reward_matrix.size == 0 or cost_matrix.size == 0:
            print(f"No aligned iterations for {algo}")
            continue

        if algo == "SafeMeta":
            iter_scale = safe_meta_iter_scale
            iter_offset = safe_meta_iter_offset
        else:
            iter_scale = 1.0
            iter_offset = 0.0

        reward_iters_scaled = (reward_iters.astype(float) + iter_offset) / iter_scale
        cost_iters_scaled = (cost_iters.astype(float) + iter_offset) / iter_scale
        if algo == "SafeMeta":
            reward_iters_scaled = reward_iters_scaled * args.safe_meta_step_multiplier
            cost_iters_scaled = cost_iters_scaled * args.safe_meta_step_multiplier

        style = style_map.get(algo, {})
        plot_algo(axes[0], reward_iters_scaled, reward_matrix, algo, color=style.get("color"), linestyle=style.get("linestyle", "-"), scale=1.0)
        plot_algo(axes[1], cost_iters_scaled, cost_matrix, algo, color=style.get("color"), linestyle=style.get("linestyle", "-"))
        # plot_algo(axes[0], reward_iters, reward_matrix, algo, color=style.get("color"), scale=0.1)
        # plot_algo(axes[1], cost_iters, cost_matrix, algo, color=style.get("color"))

    safe_x, safe_stats = load_safe_pce_seeds(args.safe_pce_dir, args.safe_pce_pattern)
    reward_mean, reward_p20, reward_p80, cost_mean, cost_p20, cost_p80 = safe_stats
    axes[0].plot(safe_x, reward_mean, label="Our algorithm", color="tab:red")
    axes[0].fill_between(
        safe_x,
        (reward_p80),
        (reward_p20),
        alpha=0.2,
        color="tab:red",
    )
    axes[1].plot(safe_x, cost_mean, label="Our algorithm", color="tab:red")
    axes[1].fill_between(safe_x, cost_p20, cost_p80, alpha=0.2, color="tab:red")

    axes[0].set_ylabel("Reward value")
    axes[0].set_title("Reward value for Hopper")
    axes[1].set_ylabel("Constraint value")
    axes[1].set_title("Constraint value for Hopper")
    axes[1].set_xlabel("Iteration number")


    # Show iteration numbers as x*300 on the axis labels.
    formatter = FuncFormatter(lambda val, _pos: f"{int(round(val * 300))}")
    axes[0].xaxis.set_major_formatter(formatter)
    axes[1].xaxis.set_major_formatter(formatter)
    if args.x_max is not None:
        axes[0].set_xlim(right=args.x_max)
        axes[1].set_xlim(right=args.x_max)
    # Truncate displayed iterations to 5000 (x-axis labels show x*300).
    max_x_display = 5000.0 / 300.0
    axes[0].set_xlim(left=0.0, right=min(axes[0].get_xlim()[1], max_x_display))
    axes[1].set_xlim(left=0.0, right=min(axes[1].get_xlim()[1], max_x_display))
    axes[1].axhline(y=5, color='black', linewidth=1.5)
    axes[1].text(
        0.98,
        0.95,
        "Threshold = 5",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="black",
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, boxstyle='round,pad=0.2'),
    )

    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    # Remove threshold label from legend if present
    filtered = [(h, l) for h, l in list(zip(handles0, labels0)) + list(zip(handles1, labels1)) if l != "Threshold = 5"]
    unique = {}
    for handle, label in filtered:
        if label not in unique:
            unique[label] = handle
    legend_labels = list(unique.keys())
    legend_handles = [unique[label] for label in legend_labels]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()