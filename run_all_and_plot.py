import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(command: List[str], cwd: Path) -> None:
    print("\n>>>", " ".join(command))
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def find_latest_model_dir(repo_root: Path, algo_name: str, env_name: str) -> Path | None:
    base = repo_root / "assets" / "learned_models" / algo_name
    if not base.exists():
        return None

    model_dirs = []
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.endswith(f"-{env_name}"):
            continue
        model_path = run_dir / "model.p"
        if model_path.exists():
            model_dirs.append(run_dir)

    if not model_dirs:
        return None

    return max(model_dirs, key=lambda p: (p / "model.p").stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all safe meta-RL algorithms with quick settings and plot test reward/cost curves."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to initial meta-policy directory containing model.p (default: latest SafeMeta checkpoint for env)",
    )
    parser.add_argument("--env-name", default="Hopper", help="Environment name used by main.py")
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["CPOMeta", "SafeMeta", "MAML_constraint", "CPO"],
        help="Algorithms to run",
    )
    parser.add_argument("--env-num", type=int, default=1, help="Tasks per meta-iteration for quick comparison")
    parser.add_argument("--meta-iter-num", type=int, default=6, help="Meta-test adaptation steps")
    parser.add_argument("--max-iter-num", type=int, default=20, help="Outer iterations (kept small for quick run)")
    parser.add_argument("--min-batch-size", type=int, default=1200, help="Rollout batch size")
    parser.add_argument("--max-batch-size", type=int, default=1200, help="Max batch size")
    parser.add_argument("--time-horizon", type=int, default=200, help="Episode horizon")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--max-constraint", type=float, default=2, help="Constraint threshold")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    parser.add_argument(
        "--save-plot",
        default="assets/plots/all_algorithms_test_metrics.png",
        help="Output image path for comparison plot",
    )
    parser.add_argument(
        "--save-baseline",
        default="assets/plots/shared_baseline.json",
        help="Output JSON path for shared pre-adaptation baseline",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip computing the shared baseline.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generating the comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    print("Running algorithms:", ", ".join(args.algos))
    print(f"Environment: {args.env_name}")

    safe_model_dir = Path(args.model_path).expanduser().resolve() if args.model_path else find_latest_model_dir(repo_root, "SafeMeta", args.env_name)
    if safe_model_dir is None:
        raise RuntimeError(
            "No initial meta-policy found. Provide --model-path or train one first using evaluate_meta_policy.py --train-first True."
        )
    if not (safe_model_dir / "model.p").exists():
        raise FileNotFoundError(f"model.p not found under initial policy path: {safe_model_dir}")

    print(f"Using initial policy from: {safe_model_dir}")

    if not args.skip_baseline:
        baseline_cmd = [
            args.python,
            "compute_shared_baseline.py",
            "--model-path",
            str(safe_model_dir),
            "--env-name",
            args.env_name,
            "--env-num",
            str(args.env_num),
            "--use-cover-set-tasks",
            "False",
            "--min-batch-size",
            str(args.min_batch_size),
            "--time-horizon",
            str(args.time_horizon),
            "--gamma",
            str(args.gamma),
            "--seed",
            str(args.seed),
            "--output",
            args.save_baseline,
        ]
        run_command(baseline_cmd, repo_root)

    for algo in args.algos:
        cmd = [
            args.python,
            "main.py",
            "--algo-name",
            algo,
            "--env-name",
            args.env_name,
            "--model-path",
            str(safe_model_dir),
            "--is-meta-test",
            "True",
            "--env-num",
            str(args.env_num),
            "--use-cover-set-tasks",
            "False",
            "--max-iter-num",
            str(args.max_iter_num),
            "--meta-iter-num",
            str(args.meta_iter_num),
            "--min-batch-size",
            str(args.min_batch_size),
            "--max-batch-size",
            str(args.max_batch_size),
            "--time-horizon",
            str(args.time_horizon),
            "--seed",
            str(args.seed),
            "--gamma",
            str(args.gamma),
            "--max-constraint",
            str(args.max_constraint),
        ]
        run_command(cmd, repo_root)

    if not args.skip_plot:
        plot_cmd = [
            args.python,
            "plot_test_metrics.py",
            "--env-name",
            args.env_name,
            "--algo",
            *args.algos,
            "--baseline-file",
            args.save_baseline,
            "--save",
            args.save_plot,
        ]
        run_command(plot_cmd, repo_root)

        print("\nDone.")
        print(f"Comparison plot: {args.save_plot}")


if __name__ == "__main__":
    main()
