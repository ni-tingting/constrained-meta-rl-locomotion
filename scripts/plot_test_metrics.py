import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


TEST_LOG_FILES = ("test_log2.csv", "test_log.csv")
TITLE_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 10


def find_latest_run_dir(base_dir: Path, algo_name: str, env_name: Optional[str]) -> Optional[Path]:
    algo_dir = base_dir / "learned_models" / algo_name
    if not algo_dir.exists():
        return None

    candidates = [d for d in algo_dir.iterdir() if d.is_dir()]
    if env_name:
        candidates = [d for d in candidates if d.name.endswith(f"-{env_name}")]

    if not candidates:
        return None

    return max(candidates, key=lambda d: d.stat().st_mtime)


def find_test_log(run_dir: Path) -> Optional[Path]:
    for name in TEST_LOG_FILES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def read_test_log(csv_path: Path) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    rows_parsed: List[Tuple[int, float, float, float, float]] = []

    with csv_path.open("r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                rows_parsed.append(
                    (
                        int(float(row[0])),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                        float(row[6]),
                    )
                )
            except ValueError:
                continue

    if not rows_parsed:
        return [], [], [], [], []

    segments: List[List[Tuple[int, float, float, float, float]]] = []
    current_segment: List[Tuple[int, float, float, float, float]] = []
    previous_iter: Optional[int] = None

    for parsed_row in rows_parsed:
        iteration = parsed_row[0]
        if previous_iter is not None and iteration < previous_iter:
            if current_segment:
                segments.append(current_segment)
            current_segment = []
        current_segment.append(parsed_row)
        previous_iter = iteration

    if current_segment:
        segments.append(current_segment)

    latest_segment = segments[-1]

    latest_by_iteration: Dict[int, Tuple[int, float, float, float, float]] = {}
    for parsed_row in latest_segment:
        latest_by_iteration[parsed_row[0]] = parsed_row

    ordered_rows = [latest_by_iteration[key] for key in sorted(latest_by_iteration.keys())]

    m_iter = [row[0] for row in ordered_rows]
    avg_cost = [row[1] for row in ordered_rows]
    test_cost = [row[2] for row in ordered_rows]
    avg_reward = [row[3] for row in ordered_rows]
    test_reward = [row[4] for row in ordered_rows]

    return m_iter, avg_cost, test_cost, avg_reward, test_reward


def collect_runs(
    assets_dir: Path,
    algos: List[str],
    env_name: Optional[str],
    run_dirs: Optional[List[Path]],
) -> Dict[str, Tuple[Path, Path]]:
    resolved: Dict[str, Tuple[Path, Path]] = {}

    if run_dirs:
        for run_dir in run_dirs:
            parts = run_dir.parts
            algo = None
            if "learned_models" in parts:
                idx = parts.index("learned_models")
                if idx + 1 < len(parts):
                    algo = parts[idx + 1]
            if algo is None:
                algo = run_dir.name.split("-")[-2] if "-" in run_dir.name else run_dir.name

            log_path = find_test_log(run_dir)
            if log_path:
                resolved[algo] = (run_dir, log_path)
        return resolved

    for algo in algos:
        run_dir = find_latest_run_dir(assets_dir, algo, env_name)
        if run_dir is None:
            continue
        log_path = find_test_log(run_dir)
        if log_path is None:
            continue
        resolved[algo] = (run_dir, log_path)

    return resolved


def load_baseline(baseline_file: Optional[Path]) -> Optional[Dict[str, float]]:
    if baseline_file is None:
        return None
    if not baseline_file.exists():
        print(f"Baseline file not found, skipping baseline overlay: {baseline_file}")
        return None
    try:
        payload = json.loads(baseline_file.read_text())
        baseline_reward = float(payload["baseline_reward"])
        baseline_cost = float(payload["baseline_cost"])
        return {"baseline_reward": baseline_reward, "baseline_cost": baseline_cost}
    except Exception as error:
        print(f"Failed to parse baseline file {baseline_file}: {error}")
        return None


def plot_algorithms(
    runs: Dict[str, Tuple[Path, Path]],
    output: Optional[Path],
    env_name: Optional[str],
    baseline: Optional[Dict[str, float]] = None,
) -> None:
    if not runs:
        raise RuntimeError("No test logs found. Run meta-testing first or pass --run-dir explicitly.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_avg_reward = axes[0][0]
    ax_test_reward = axes[0][1]
    ax_avg_cost = axes[1][0]
    ax_test_cost = axes[1][1]

    for algo, (run_dir, log_path) in runs.items():
        m_iter, avg_cost, test_cost, avg_reward, test_reward = read_test_log(log_path)
        if not m_iter:
            print(f"Skipping {algo} (no valid rows): {log_path}")
            continue

        x_values = [iteration + 1 for iteration in m_iter] if baseline is not None else m_iter

        label = f"{algo}"
        ax_avg_reward.plot(x_values, avg_reward, marker="o", markersize=3, linewidth=1.5, label=label)
        ax_test_reward.plot(x_values, test_reward, marker="o", markersize=3, linewidth=1.5, label=label)
        ax_avg_cost.plot(x_values, avg_cost, marker="o", markersize=3, linewidth=1.5, label=label)
        ax_test_cost.plot(x_values, test_cost, marker="o", markersize=3, linewidth=1.5, label=label)

    if baseline is not None:
        baseline_reward = baseline["baseline_reward"]
        baseline_cost = baseline["baseline_cost"]
        ax_avg_reward.scatter([0], [baseline_reward], marker="*", s=160, color="black", label="Shared Init (step 0)")
        ax_test_reward.scatter([0], [baseline_reward], marker="*", s=160, color="black", label="Shared Init (step 0)")
        ax_avg_cost.scatter([0], [baseline_cost], marker="*", s=160, color="black", label="Shared Init (step 0)")
        ax_test_cost.scatter([0], [baseline_cost], marker="*", s=160, color="black", label="Shared Init (step 0)")

    ax_avg_reward.set_title("Avg Reward", fontsize=TITLE_FONTSIZE)
    ax_test_reward.set_title("Test Reward", fontsize=TITLE_FONTSIZE)
    ax_avg_cost.set_title("Avg Cost", fontsize=TITLE_FONTSIZE)
    ax_test_cost.set_title("Test Cost", fontsize=TITLE_FONTSIZE)

    ax_avg_cost.set_xlabel("Adaptation Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax_test_cost.set_xlabel("Adaptation Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax_avg_reward.set_ylabel("Value", fontsize=AXIS_LABEL_FONTSIZE)
    ax_avg_cost.set_ylabel("Value", fontsize=AXIS_LABEL_FONTSIZE)

    for ax in (ax_avg_reward, ax_test_reward, ax_avg_cost, ax_test_cost):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)

    if baseline is not None:
        fig.suptitle("Adaptation step 0 = shared pre-adaptation baseline", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        print(f"Saved plot: {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot test reward/cost curves for safe meta-RL runs.")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Assets directory root (default: ./assets)",
    )
    parser.add_argument(
        "--algo",
        nargs="+",
        default=["CPO", "CPOMeta", "SafeMeta", "MAML_constraint"],
        help="Algorithms to include when auto-discovering runs.",
    )
    parser.add_argument(
        "--env-name",
        default=None,
        help="Optional environment suffix filter (e.g., Hopper).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        default=None,
        help="One or more explicit run directories under assets/learned_models/...",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional image output path (e.g., assets/plots/test_metrics.png).",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="Optional baseline JSON for pre-adaptation marker overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.assets_dir, args.algo, args.env_name, args.run_dir)

    if runs:
        print("Using runs:")
        for algo, (run_dir, log_path) in runs.items():
            print(f"  {algo}: {run_dir} ({log_path.name})")

    baseline = load_baseline(args.baseline_file)
    plot_algorithms(runs, args.save, args.env_name, baseline=baseline)


if __name__ == "__main__":
    main()
