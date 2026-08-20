import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


SEED = 1
EPSILON = 0.1
MEAN = 0.5
STD = 0.1
LOW = 0.0
HIGH = 1.0
MAX_PHASES = 20
SAVE_PATH = Path("assets/cover_set.json")


def O_c(noise_i: float, noise_j: float, epsilon: float = EPSILON) -> int:
	return int(abs(noise_i - noise_j) < epsilon)


def truncated_gaussian(
	size: int,
	mean: float = MEAN,
	std: float = STD,
	low: float = LOW,
	high: float = HIGH,
	rng: np.random.Generator | None = None,
):
	generator = rng if rng is not None else np.random.default_rng(SEED)
	samples = generator.normal(mean, std, size=size)
	return np.clip(samples, low, high)


def policy_cover_subroutine(noises: List[float], delta: float, epsilon: float) -> Tuple[List[float], int]:
	N = len(noises)

	U = []
	T = set(range(N))
	A = np.zeros((N, N), dtype=int)

	for i in range(N):
		for j in range(N):
			A[i, j] = O_c(noises[i], noises[j], epsilon)

	cover_amounts = []

	for _ in range(N):
		if len(T) == 0:
			break

		best_j = None
		best_cover = -1

		for j in range(N):
			if j in U:
				continue
			cover_size = sum(A[i, j] for i in T)
			if cover_size > best_cover:
				best_cover = cover_size
				best_j = j

		if best_j is None:
			break

		U.append(best_j)

		newly_covered = {i for i in T if A[i, best_j] == 1}
		T = T - newly_covered

		cover_amounts.append(best_cover)

		if sum(cover_amounts) >= (1 - 3 * delta) * N:
			break

	U_cmdps = [noises[j] for j in U]
	return U_cmdps, len(U)


def pretrain_stage(delta: float, epsilon: float = EPSILON):
	if not (0.0 < delta < 1.0):
		raise ValueError("delta must be in (0, 1)")

	rng = np.random.default_rng(SEED)

	N_f = int(math.log(1 / delta) / (delta ** 2))
	N_f = max(N_f, 1)

	phase = 1
	noises = list(truncated_gaussian(size=N_f, rng=rng))
	N = N_f

	while phase <= MAX_PHASES:
		U, hat_Pi_size = policy_cover_subroutine(noises, delta, epsilon)

		denominator = max(N - hat_Pi_size, 1)
		lhs = math.sqrt((hat_Pi_size * math.log(2 * N / delta)) / denominator)
		if lhs <= delta:
			return U, hat_Pi_size

		N = 2 * N
		new_noises = list(truncated_gaussian(size=N_f, rng=rng))
		noises.extend(new_noises)
		phase += 1

	U, hat_Pi_size = policy_cover_subroutine(noises, delta, epsilon)
	return U, hat_Pi_size


def ensure_cover_set_exists(cover_set_path: Path, cover_delta: float) -> None:
	if cover_set_path.exists():
		return

	cover_set, _ = pretrain_stage(delta=cover_delta, epsilon=EPSILON)
	cover_set = [float(x) for x in cover_set]
	cover_set_path.parent.mkdir(parents=True, exist_ok=True)
	cover_set_path.write_text(json.dumps(cover_set))


def run_safemeta_training(args, cover_set_path: Path) -> None:
	latest_model_dir = find_latest_model_dir("SafeMeta", args.env_name)
	command = [
		args.python,
		"main.py",
		"--algo-name",
		"SafeMeta",
		"--is-meta-test",
		"False",
		"--env-name",
		args.env_name,
		"--use-cover-set-tasks",
		"True",
		"--cover-set-path",
		str(cover_set_path),
		"--max-iter-num",
		str(args.max_iter_num),
		"--meta-iter-num",
		str(args.meta_iter_num),
		"--env-num",
		str(args.env_num),
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
	if latest_model_dir is not None:
		command.extend(["--model-path", str(latest_model_dir)])
	print("Running training:", " ".join(command))
	completed = subprocess.run(command, cwd=Path(__file__).resolve().parent)
	if completed.returncode != 0:
		raise RuntimeError(f"SafeMeta training failed with exit code {completed.returncode}")


def find_latest_model_dir(algo_name: str, env_name: str) -> Path | None:
	base = Path(__file__).resolve().parent / "assets" / "learned_models" / algo_name
	if not base.exists():
		return None

	candidates = []
	for run_dir in base.iterdir():
		if not run_dir.is_dir():
			continue
		if not run_dir.name.endswith(f"-{env_name}"):
			continue
		model_path = run_dir / "model.p"
		if model_path.exists():
			candidates.append(run_dir)

	if not candidates:
		return None
	return max(candidates, key=lambda p: (p / "model.p").stat().st_mtime)


def parse_args():
	parser = argparse.ArgumentParser(description="Ensure cover set exists then train SafeMeta with fixed Hopper tasks.")
	parser.add_argument("--cover-delta", type=float, default=0.05, help="Delta used only when creating missing cover set.")
	parser.add_argument("--cover-set-path", type=str, default=str(SAVE_PATH), help="Cover set JSON path.")
	parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for training command.")
	parser.add_argument("--env-name", type=str, default="Hopper")
	parser.add_argument("--env-num", type=int, default=10)
	parser.add_argument("--max-iter-num", type=int, default=1000)
	parser.add_argument("--meta-iter-num", type=int, default=20)
	parser.add_argument("--min-batch-size", type=int, default=300)
	parser.add_argument("--max-batch-size", type=int, default=300)
	parser.add_argument("--time-horizon", type=int, default=200)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--max-constraint", type=float, default= 5.0)
	return parser.parse_args()


def main():
	args = parse_args()
	cover_set_path = Path(args.cover_set_path)
	if not cover_set_path.is_absolute():
		cover_set_path = Path(__file__).resolve().parent / cover_set_path

	ensure_cover_set_exists(cover_set_path=cover_set_path, cover_delta=args.cover_delta)
	run_safemeta_training(args, cover_set_path)


if __name__ == "__main__":
	main()
