#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


TEST_LOG_FILES = ("test_log2.csv", "test_log.csv")


def read_training_log(csv_path: Path, value_index: int) -> Tuple[np.ndarray, np.ndarray]:
    iterations: List[int] = []
    values: List[float] = []
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            try:
                iterations.append(int(float(row[0])))
                values.append(float(row[value_index]))
            except (ValueError, IndexError):
                continue
    return np.asarray(iterations, dtype=int), np.asarray(values, dtype=float)


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


def collect_runs(base_dir: Path, algo: str, require_seed_tag: bool, log_type: str) -> List[Path]:
    algo_dir = base_dir / algo
    if not algo_dir.exists():
        return []
    run_dirs = [p for p in algo_dir.iterdir() if p.is_dir()]
    if require_seed_tag:
        run_dirs = [p for p in run_dirs if "-seed" in p.name]
    if log_type == "train":
        return [p for p in run_dirs if (p / "training_log.csv").exists()]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="assets/learned_models", type=Path)
    parser.add_argument("--algorithms", nargs="+", default=["MAML_constraint", "CPOMeta", "CPO"])
    parser.add_argument("--require-seed-tag", action="store_true", default=True)
    parser.add_argument("--title", default="Hopper/Meta-training")
    parser.add_argument("--log-type", choices=["train", "test"], default="train")
    parser.add_argument("--metric", choices=["reward", "cost"], default="reward")
    parser.add_argument("--reward-col", type=int, default=-1, help="Column index for reward in training_log.csv (default: last column).")
    parser.add_argument("--cost-col", type=int, default=4, help="Column index for cost in training_log.csv (default: avg_cost column).")
    parser.add_argument("--test-reward-col", type=int, default=6, help="Column index for test reward in test_log*.csv.")
    parser.add_argument("--test-cost-col", type=int, default=4, help="Column index for test cost in test_log*.csv.")
    parser.add_argument("--output", default="assets/plots/hopper_meta_training.png", type=Path)
    args = parser.parse_args()

    if args.log_type == "train":
        value_index = args.reward_col if args.metric == "reward" else args.cost_col
    else:
        value_index = args.test_reward_col if args.metric == "reward" else args.test_cost_col

    fig, ax = plt.subplots(figsize=(7, 4))

    style_map = {
        "MAML_constraint": {"color": "tab:green", "linestyle": "-"},
        "CPOMeta": {"color": "tab:red", "linestyle": "--"},
        "CPO": {"color": "tab:purple", "linestyle": ":"},
    }

    for algo in args.algorithms:
        run_dirs = collect_runs(args.base_dir, algo, args.require_seed_tag, args.log_type)
        if not run_dirs:
            print(f"No runs found for {algo}")
            continue

        run_logs = []
        for run_dir in sorted(run_dirs):
            if args.log_type == "train":
                iterations, values = read_training_log(run_dir / "training_log.csv", value_index)
            else:
                log_path = next((run_dir / name for name in TEST_LOG_FILES if (run_dir / name).exists()), None)
                if log_path is None:
                    continue
                iterations, values = read_test_log(log_path, value_index)
            if len(iterations) == 0:
                continue
            run_logs.append((iterations, values))

        iterations, reward_matrix = align_runs(run_logs)
        if reward_matrix.size == 0:
            print(f"No aligned iterations for {algo}")
            continue

        p20 = np.quantile(reward_matrix, 0.3, axis=0)
        p80 = np.quantile(reward_matrix, 0.7, axis=0)
        band_means = []
        for idx in range(reward_matrix.shape[1]):
            column = reward_matrix[:, idx]
            mask = (column >= p20[idx]) & (column <= p80[idx])
            band_means.append(np.mean(column[mask]))
        mean_values = np.asarray(band_means, dtype=float)

        style = style_map.get(algo, {})
        ax.plot(iterations, mean_values, label=algo, **style)
        ax.fill_between(
            iterations,
            p20,
            p80,
            alpha=0.15,
            color=style.get("color"),
        )

    ax.set_title(args.title)
    ax.set_xlabel("Number of meta-training iterations")
    if args.log_type == "test":
        ylabel = "Test reward" if args.metric == "reward" else "Test cost"
    else:
        ylabel = "Average accumulated reward" if args.metric == "reward" else "Average accumulated cost"
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=True, bbox_to_anchor=(0.5, 1.02))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
