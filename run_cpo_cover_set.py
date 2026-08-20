import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def load_cover_set(path: Path) -> List[float]:
    if not path.exists():
        raise FileNotFoundError(f"Cover set file not found: {path}")
    values = json.loads(path.read_text())
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"Cover set must be a non-empty JSON list: {path}")
    return [float(value) for value in values]


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


def sanitize_value(value: float) -> str:
    return str(value).replace(".", "p")


def run_command(command: List[str], cwd: Path, dry_run: bool) -> None:
    print("\n>>>", " ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def export_model(model_dir: Path, export_dir: Path, velocity: float) -> None:
    model_path = model_dir / "model.p"
    if not model_path.exists():
        raise FileNotFoundError(f"model.p not found in {model_dir}")

    velocity_dir = export_dir / f"velocity_{sanitize_value(velocity)}"
    velocity_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, velocity_dir / "model.p")
    (velocity_dir / "metadata.json").write_text(
        json.dumps({"velocity": velocity, "source": str(model_dir)}, indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPO on each cover-set task (Hopper goal velocity) and save best policies."
    )
    parser.add_argument("--cover-set-path", default="assets/cover_set.json")
    parser.add_argument("--export-dir", default="assets/learned_models/CPO_cover_set")
    parser.add_argument("--env-name", default="Hopper")
    parser.add_argument("--max-constraint", type=float, default=5.0)
    parser.add_argument("--max-iter-num", type=int, default=300)
    parser.add_argument("--meta-iter-num", type=int, default=20)
    parser.add_argument("--min-batch-size", type=int, default=500)
    parser.add_argument("--max-batch-size", type=int, default=500)
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--exp-name-prefix", default="CPO-cover")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    cover_set_path = Path(args.cover_set_path)
    if not cover_set_path.is_absolute():
        cover_set_path = repo_root / cover_set_path

    export_dir = Path(args.export_dir)
    if not export_dir.is_absolute():
        export_dir = repo_root / export_dir

    cover_values = load_cover_set(cover_set_path)
    tmp_dir = export_dir / "tmp_cover_sets"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for index, velocity in enumerate(cover_values):
        tmp_cover_file = tmp_dir / f"cover_{index}.json"
        tmp_cover_file.write_text(json.dumps([float(velocity)]))

        velocity_dir = export_dir / f"velocity_{sanitize_value(velocity)}"
        resume_model_path = velocity_dir / "model.p"

        exp_name = f"{args.exp_name_prefix}-idx{index}-v{sanitize_value(velocity)}"

        command = [
            args.python,
            "main.py",
            "--algo-name",
            "CPO",
            "--is-meta-test",
            "False",
            "--env-name",
            args.env_name,
            "--env-num",
            "1",
            "--use-cover-set-tasks",
            "True",
            "--cover-set-path",
            str(tmp_cover_file),
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
            "--exp-name",
            exp_name,
            "--exp-num",
            str(index + 1),
        ]

        if resume_model_path.exists():
            command.extend(["--model-path", str(velocity_dir)])

        run_command(command, repo_root, args.dry_run)

        if args.dry_run:
            continue

        latest_dir = find_latest_model_dir(repo_root, "CPO", args.env_name)
        if latest_dir is None:
            raise RuntimeError("No CPO checkpoint found after training.")
        export_model(latest_dir, export_dir, velocity)

    print("Done.")


if __name__ == "__main__":
    main()
